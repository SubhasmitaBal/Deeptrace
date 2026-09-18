import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timezone

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.signal import resample_poly


st.set_page_config(
    page_title="DeepTrace | Multimodal Investigator",
    page_icon="🔎",
    layout="wide",
)

MAX_UPLOAD_MB = 100
MAX_ANALYSIS_SECONDS = 60
AUDIO_RATE = 16000


# ---------------------------------------------------------
# Model adapters
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_detector(task, model_reference):
    """
    Load compatible local or Hugging Face classification checkpoints.
    Remote custom Python code is disabled.
    Safetensors weights are requested.
    """
    from transformers import pipeline

    return pipeline(
        task=task,
        model=model_reference,
        device=-1,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )


def fake_label_score(predictions, fake_label):
    """
    Match a label explicitly supplied from the model's documented
    label mapping. Never guess what LABEL_0 or LABEL_1 means.
    """
    while (
        isinstance(predictions, list)
        and predictions
        and isinstance(predictions[0], list)
    ):
        predictions = predictions[0]

    if not isinstance(predictions, list):
        raise ValueError("Unexpected classification output.")

    available = []

    for prediction in predictions:
        label = str(prediction["label"])
        available.append(label)

        if label.strip().casefold() == fake_label.strip().casefold():
            value = float(prediction["score"])

            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Detector returned an invalid score.")

            return value

    raise ValueError(
        f"Fake label '{fake_label}' was not returned. "
        f"Available labels: {available}. Verify the model's label mapping."
    )


def visual_score(detector, frames, fake_label):
    scores = []

    for frame in frames:
        predictions = detector(
            Image.fromarray(frame),
            top_k=None,
        )
        scores.append(fake_label_score(predictions, fake_label))

    return float(np.mean(scores)) if scores else None


def audio_score(detector, audio, sample_rate, fake_label):
    target_rate = int(
        getattr(detector.feature_extractor, "sampling_rate", sample_rate)
    )

    samples = audio.astype(np.float32)

    if target_rate != sample_rate:
        divisor = math.gcd(target_rate, sample_rate)
        samples = resample_poly(
            samples,
            target_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

    predictions = detector(
        {"array": samples, "sampling_rate": target_rate},
        top_k=None,
    )

    return fake_label_score(predictions, fake_label)


# ---------------------------------------------------------
# Media utilities
# ---------------------------------------------------------

def extract_audio(video_path, directory, seconds):
    output_path = os.path.join(directory, "audio.wav")

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        video_path,
        "-t",
        str(seconds),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_RATE),
        "-c:a",
        "pcm_s16le",
        output_path,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=90,
            check=False,
        )

        if result.returncode != 0 or not os.path.exists(output_path):
            return None, (
                "Audio unavailable: the file may have no audio track, "
                "or decoding was unsuccessful."
            )

        with wave.open(output_path, "rb") as handle:
            samples = np.frombuffer(
                handle.readframes(handle.getnframes()),
                dtype="<i2",
            ).astype(np.float32) / 32768.0

        if samples.size == 0:
            return None, "The decoded audio track was empty."

        return samples, None

    except (subprocess.TimeoutExpired, wave.Error, OSError) as exc:
        return None, f"Audio extraction failed: {type(exc).__name__}."


def wav_bytes(samples, sample_rate):
    output = io.BytesIO()

    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)

        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
        handle.writeframes(pcm.tobytes())

    return output.getvalue()


def read_frame(capture, timestamp):
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = capture.read()

    if not success:
        return None

    height, width = frame.shape[:2]
    scale = min(1.0, 640 / max(height, width))

    if scale < 1:
        frame = cv2.resize(
            frame,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def image_quality(frames):
    if not frames:
        return None, None

    brightness = []
    sharpness = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness.append(float(gray.mean()))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return float(np.mean(brightness)), float(np.mean(sharpness))


# ---------------------------------------------------------
# Investigation orchestrator
# ---------------------------------------------------------

def investigate(
    data,
    suffix,
    window_seconds,
    visual_reference,
    visual_label,
    audio_reference,
    audio_label,
    progress,
):
    started = time.perf_counter()
    logs = []
    visual_detector = None
    audio_detector = None

    with tempfile.TemporaryDirectory(prefix="deeptrace_") as directory:
        video_path = os.path.join(directory, f"input{suffix}")

        with open(video_path, "wb") as handle:
            handle.write(data)

        capture = cv2.VideoCapture(video_path)

        try:
            if not capture.isOpened():
                raise ValueError("This video could not be opened.")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if (
                not math.isfinite(fps)
                or not math.isfinite(frame_count)
                or fps <= 0
                or frame_count <= 0
            ):
                raise ValueError(
                    "Video duration could not be determined. "
                    "Try exporting the clip as an MP4."
                )

            duration = frame_count / fps
            analyzed_seconds = min(duration, MAX_ANALYSIS_SECONDS)

        except Exception as exc:
            capture.release()
            raise ValueError(
                f"Could not read video metadata: {exc}"
            ) from exc

        logs.append(
            f"Planner: inspect {analyzed_seconds:.1f}s "
            f"in {window_seconds}s windows."
        )

        audio, audio_warning = extract_audio(
            video_path,
            directory,
            analyzed_seconds,
        )

        if audio_warning:
            logs.append(audio_warning)
        else:
            logs.append("Audio tool: mono 16 kHz audio extracted.")

        if visual_reference:
            try:
                visual_detector = load_detector(
                    "image-classification",
                    visual_reference,
                )
                logs.append("Visual detector loaded.")
            except Exception as exc:
                logs.append(
                    f"Visual detector unavailable: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
        else:
            logs.append(
                "Visual detector not configured; "
                "visual quality inspection only."
            )

        if audio_reference and audio is not None:
            try:
                audio_detector = load_detector(
                    "audio-classification",
                    audio_reference,
                )
                logs.append("Audio detector loaded.")
            except Exception as exc:
                logs.append(
                    f"Audio detector unavailable: "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
        elif not audio_reference:
            logs.append(
                "Audio detector not configured; "
                "audio quality inspection only."
            )

        rows = []
        thumbnails = []
        total_windows = math.ceil(analyzed_seconds / window_seconds)

        for index in range(total_windows):
            start = index * window_seconds
            end = min(
                start + window_seconds,
                analyzed_seconds,
            )
            span = end - start

            sample_times = [
                start + span * 0.2,
                start + span * 0.5,
                start + span * 0.8,
            ]

            frames = [
                frame
                for timestamp in sample_times
                if (
                    frame := read_frame(capture, timestamp)
                ) is not None
            ]

            brightness, sharpness = image_quality(frames)
            notes = []

            # These are quality warnings, NOT deepfake evidence.
            if not frames:
                notes.append("No video frames decoded.")
            else:
                if brightness < 30:
                    notes.append("Low-light warning.")
                if sharpness < 35:
                    notes.append("Low-sharpness warning.")

            visual_value = None

            if visual_detector is not None and frames:
                try:
                    visual_value = visual_score(
                        visual_detector,
                        frames,
                        visual_label,
                    )
                except Exception as exc:
                    notes.append(
                        f"Visual inference failed: {str(exc)[:250]}"
                    )

            rms_db = None
            clipping = None
            audio_value = None
            segment_audio = None

            if audio is not None:
                segment_audio = audio[
                    int(start * AUDIO_RATE):int(end * AUDIO_RATE)
                ]

                if segment_audio.size:
                    rms = float(
                        np.sqrt(np.mean(segment_audio ** 2))
                    )

                    rms_db = float(
                        20 * np.log10(max(rms, 1e-8))
                    )

                    clipping = float(
                        np.mean(np.abs(segment_audio) >= 0.999)
                    )

            if rms_db is not None and rms_db < -45:
                notes.append("Very quiet audio.")

            if clipping is not None and clipping > 0.01:
                notes.append("Audio clipping warning.")

            if (
                audio_detector is not None
                and segment_audio is not None
                and segment_audio.size >= AUDIO_RATE
            ):
                if rms_db is not None and rms_db >= -45:
                    try:
                        audio_value = audio_score(
                            audio_detector,
                            segment_audio,
                            AUDIO_RATE,
                            audio_label,
                        )
                    except Exception as exc:
                        notes.append(
                            f"Audio inference failed: {str(exc)[:250]}"
                        )
                else:
                    notes.append(
                        "Audio detector skipped: insufficient signal."
                    )
            elif audio_detector is not None:
                notes.append(
                    "Audio detector skipped: less than one second "
                    "of decoded audio."
                )

            rows.append(
                {
                    "segment": index + 1,
                    "start_s": round(start, 3),
                    "end_s": round(end, 3),
                    "sampled_frames": len(frames),
                    "visual_model_score": visual_value,
                    "audio_model_score": audio_value,
                    "brightness_0_255": brightness,
                    "sharpness_variance": sharpness,
                    "audio_rms_dbfs": rms_db,
                    "audio_clipping_fraction": clipping,
                    "notes": " ".join(notes)
                    or "No basic quality warnings.",
                }
            )

            thumbnails.append(
                frames[len(frames) // 2] if frames else None
            )

            progress.progress(
                (index + 1) / total_windows,
                text=(
                    f"Investigating segment "
                    f"{index + 1}/{total_windows}"
                ),
            )

        logs.append(
            "Reporter: quality measurements are not proof "
            "of manipulation."
        )
        logs.append(
            "Lip-sync analysis was not run; no synchronization "
            "claims are made."
        )

        capture.release()

        return {
            "duration_s": duration,
            "analyzed_seconds": analyzed_seconds,
            "processing_seconds": time.perf_counter() - started,
            "rows": rows,
            "thumbnails": thumbnails,
            "audio": audio,
            "logs": logs,
        }


# ---------------------------------------------------------
# Fusion and explanation
# ---------------------------------------------------------

def fuse(row, visual_weight, threshold, disagreement_limit):
    visual = row["visual_model_score"]
    audio = row["audio_model_score"]

    if visual is None or audio is None:
        return None, "Insufficient multimodal evidence"

    if abs(visual - audio) > disagreement_limit:
        return None, "Abstain: detectors disagree"

    score = visual_weight * visual + (1 - visual_weight) * audio

    if score >= threshold:
        return float(score), "Elevated model signal — review"

    return float(score), "Below review threshold — not verified"


def display_score(label, value):
    st.metric(
        label,
        "Unavailable" if value is None else f"{value:.3f}",
    )

    if value is not None:
        st.progress(float(value))


def explain_segment(row, question):
    visual = row["visual_model_score"]
    audio = row["audio_model_score"]
    question = question.lower()

    if any(word in question for word in ("real", "fake", "authentic")):
        return (
            "This prototype cannot establish authenticity. "
            f"The current segment status is: {row['status']}. "
            "Model outputs are uncalibrated investigation signals, "
            "not probabilities that the clip is fake."
        )

    if any(word in question for word in ("audio", "voice", "sound")):
        score = (
            "unavailable"
            if audio is None
            else f"{audio:.3f}"
        )
        return (
            f"Audio detector score: {score}. "
            f"Measured audio level: {row['audio_rms_dbfs']} dBFS. "
            "Quiet or clipped audio is a quality limitation, "
            "not evidence of a cloned voice."
        )

    if any(
        word in question
        for word in ("visual", "video", "frame", "face")
    ):
        score = (
            "unavailable"
            if visual is None
            else f"{visual:.3f}"
        )
        return (
            f"Visual detector score: {score}. "
            f"{row['sampled_frames']} frames were sampled. "
            "This baseline classifies whole frames; it does not "
            "localize manipulated pixels or automatically crop faces. "
            "The selected checkpoint must support this input."
        )

    if any(word in question for word in ("sync", "lip")):
        return (
            "Lip-sync analysis is not implemented. "
            "This report cannot assess mouth–speech alignment."
        )

    return (
        f"Segment {row['segment']} covers "
        f"{row['start_s']:.1f}–{row['end_s']:.1f}s. "
        f"Status: {row['status']}. "
        f"Quality observations: {row['notes']} "
        "Review this interval manually and compare it with "
        "a trusted original when available."
    )


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------

st.title("🔎 DeepTrace")
st.caption(
    "Evidence-grounded multimodal investigation • "
    "Local research prototype • No fabricated predictions"
)

st.info(
    "Model scores are not calibrated authenticity probabilities. "
    "Blur, darkness, silence, and clipping are quality observations—not "
    "deepfake indicators. Lip-sync detection is not included."
)


with st.sidebar:
    st.header("Investigation settings")

    window_seconds = st.select_slider(
        "Segment length",
        options=[2, 4, 6, 10],
        value=4,
        format_func=lambda value: f"{value} seconds",
    )

    with st.expander(
        "Optional pretrained detectors",
        expanded=True,
    ):
        st.caption(
            "Use compatible, trusted classification checkpoints. "
            "Enter a local model folder or Hugging Face model ID. "
            "Check the model card for preprocessing and fake-label mapping."
        )

        visual_reference = st.text_input(
            "Visual model ID or local folder",
            placeholder="Leave blank for quality-only inspection",
        ).strip()

        visual_label = st.text_input(
            "Exact visual fake-class label",
            placeholder="Use the model's documented output label",
        ).strip()

        audio_reference = st.text_input(
            "Audio model ID or local folder",
            placeholder="Leave blank for quality-only inspection",
        ).strip()

        audio_label = st.text_input(
            "Exact audio fake-class label",
            placeholder="Use the model's documented output label",
        ).strip()

    st.subheader("Experimental fusion")

    visual_weight = st.slider(
        "Visual weight",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

    st.caption(f"Audio weight: {1 - visual_weight:.2f}")

    threshold = st.slider(
        "Review threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.7,
        step=0.05,
    )

    disagreement_limit = st.slider(
        "Maximum detector disagreement",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

    st.caption(
        "Fusion assumes compatible score meanings. These settings are "
        "experimental and require validation on held-out data."
    )

    if st.button("Clear analysis and unload models"):
        st.session_state.pop("investigation", None)
        load_detector.clear()
        st.rerun()


uploaded = st.file_uploader(
    "Upload a consented or appropriately licensed video",
    type=["mp4", "mov", "avi", "webm", "mkv"],
)

if uploaded is None:
    st.markdown(
        "### Start an investigation\n"
        "Upload a short video, then select **Run investigation**. "
        "The app works without model checkpoints in quality-inspection mode."
    )
    st.stop()


if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(
        f"Please upload a file smaller than {MAX_UPLOAD_MB} MB."
    )
    st.stop()


data = uploaded.getvalue()
media_hash = hashlib.sha256(data).hexdigest()
suffix = os.path.splitext(uploaded.name)[1].lower()

st.video(data)

st.caption(
    f"File: {uploaded.name} · "
    f"Size: {len(data) / (1024 * 1024):.1f} MB · "
    f"Analysis capped at the first {MAX_ANALYSIS_SECONDS} seconds"
)


configuration = {
    "media_sha256": media_hash,
    "window_seconds": window_seconds,
    "visual_model": visual_reference,
    "visual_fake_label": visual_label,
    "audio_model": audio_reference,
    "audio_fake_label": audio_label,
}


run_key = hashlib.sha256(
    json.dumps(
        configuration,
        sort_keys=True,
    ).encode()
).hexdigest()


invalid_labels = (
    (bool(visual_reference) and not visual_label)
    or (bool(audio_reference) and not audio_label)
)


if invalid_labels:
    st.warning(
        "Enter the documented fake-class label for each configured model."
    )


if st.button(
    "Run investigation",
    type="primary",
    disabled=invalid_labels,
):
    progress = st.progress(
        0.0,
        text="Preparing investigation…",
    )

    try:
        result = investigate(
            data=data,
            suffix=suffix,
            window_seconds=window_seconds,
            visual_reference=visual_reference,
            visual_label=visual_label,
            audio_reference=audio_reference,
            audio_label=audio_label,
            progress=progress,
        )

        result["run_key"] = run_key
        result["created_utc"] = datetime.now(
            timezone.utc
        ).isoformat()

        st.session_state["investigation"] = result
        progress.empty()

    except Exception as exc:
        progress.empty()
        st.error(f"Investigation failed: {exc}")
        st.stop()


result = st.session_state.get("investigation")


if not result or result.get("run_key") != run_key:
    st.info(
        "Run the investigation for this file and detector configuration. "
        "Fusion settings can be adjusted afterward without rerunning models."
    )
    st.stop()


enriched_rows = []

for original in result["rows"]:
    row = dict(original)

    score, status = fuse(
        row,
        visual_weight,
        threshold,
        disagreement_limit,
    )

    row["fused_model_score"] = score
    row["status"] = status
    enriched_rows.append(row)


table = pd.DataFrame(enriched_rows)

elevated = sum(
    row["status"] == "Elevated model signal — review"
    for row in enriched_rows
)

abstained = sum(
    row["fused_model_score"] is None
    for row in enriched_rows
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Analyzed",
    f"{result['analyzed_seconds']:.1f}s",
)
col2.metric(
    "Segments",
    len(enriched_rows),
)
col3.metric(
    "Review flags",
    elevated,
)
col4.metric(
    "Abstained segments",
    abstained,
)

st.caption(
    f"Processing time: {result['processing_seconds']:.1f}s. "
    "A review flag is not a confirmed deepfake."
)


overview_tab, inspect_tab, agent_tab, export_tab = st.tabs(
    [
        "Evidence timeline",
        "Segment inspector",
        "Investigation agent",
        "Export",
    ]
)


with overview_tab:
    st.dataframe(
        table[
            [
                "segment",
                "start_s",
                "end_s",
                "visual_model_score",
                "audio_model_score",
                "fused_model_score",
                "status",
                "notes",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    if not visual_reference and not audio_reference:
        st.warning(
            "Quality-inspection mode: no deepfake detectors were configured. "
            "All authenticity-related decisions remain insufficient evidence."
        )

    st.caption(
        "Temporal localization is limited to analysis windows. "
        "This application does not identify exact manipulated frames."
    )


with inspect_tab:
    selected = st.selectbox(
        "Select a segment",
        options=range(len(enriched_rows)),
        format_func=lambda index: (
            f"Segment {index + 1}: "
            f"{enriched_rows[index]['start_s']:.1f}–"
            f"{enriched_rows[index]['end_s']:.1f}s"
        ),
    )

    row = enriched_rows[selected]
    media_col, evidence_col = st.columns([1.3, 1])

    with media_col:
        st.video(
            data,
            start_time=float(row["start_s"]),
            end_time=float(row["end_s"]),
        )

        thumbnail = result["thumbnails"][selected]

        if thumbnail is not None:
            st.image(
                thumbnail,
                caption="Sampled frame—not a manipulation heatmap",
                use_container_width=True,
            )

        if result["audio"] is not None:
            audio_slice = result["audio"][
                int(row["start_s"] * AUDIO_RATE):
                int(row["end_s"] * AUDIO_RATE)
            ]

            if audio_slice.size:
                st.audio(
                    wav_bytes(audio_slice, AUDIO_RATE),
                    format="audio/wav",
                )

    with evidence_col:
        st.subheader(row["status"])

        display_score(
            "Visual fake-class model score",
            row["visual_model_score"],
        )

        display_score(
            "Audio fake-class model score",
            row["audio_model_score"],
        )

        display_score(
            "Experimental fused score",
            row["fused_model_score"],
        )

        st.write(row["notes"])

        quality = pd.DataFrame(
            [
                {
                    "Measurement": "Brightness",
                    "Value": row["brightness_0_255"],
                },
                {
                    "Measurement": "Laplacian variance",
                    "Value": row["sharpness_variance"],
                },
                {
                    "Measurement": "Audio level, dBFS",
                    "Value": row["audio_rms_dbfs"],
                },
                {
                    "Measurement": "Audio clipping fraction",
                    "Value": row["audio_clipping_fraction"],
                },
            ]
        )

        st.dataframe(
            quality,
            hide_index=True,
            use_container_width=True,
        )


with agent_tab:
    st.subheader("Evidence-grounded assistant")
    st.caption(
        "This is a deterministic, rule-based assistant—not an LLM. "
        "It only explains measurements and detector outputs already obtained."
    )

    agent_segment = st.selectbox(
        "Segment to discuss",
        range(len(enriched_rows)),
        format_func=lambda index: f"Segment {index + 1}",
        key="agent_segment",
    )

    with st.form("question_form"):
        question = st.text_input(
            "Ask about this segment",
            placeholder="Why was this segment flagged?",
        )
        submitted = st.form_submit_button("Explain evidence")

    if submitted:
        if question.strip():
            st.write(
                explain_segment(
                    enriched_rows[agent_segment],
                    question,
                )
            )
        else:
            st.info("Enter a question about the selected segment.")

    with st.expander("Agent execution log"):
        st.code(
            "\n".join(result["logs"]),
            language="text",
        )


with export_tab:
    report = {
        "project": "DeepTrace",
        "version": "0.1-research-prototype",
        "created_utc": result["created_utc"],
        "exported_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "filename": uploaded.name,
        "configuration": configuration,
        "fusion": {
            "visual_weight": visual_weight,
            "audio_weight": 1 - visual_weight,
            "review_threshold": threshold,
            "maximum_disagreement": disagreement_limit,
            "calibrated": False,
        },
        "duration_s": result["duration_s"],
        "analyzed_seconds": result["analyzed_seconds"],
        "processing_seconds": result["processing_seconds"],
        "segments": enriched_rows,
        "execution_log": result["logs"],
        "limitations": [
            "Not a validated authenticity detector.",
            "Model outputs are not calibrated authenticity probabilities.",
            "No lip-sync detector is implemented.",
            "Visual baseline samples three whole frames per segment.",
            "Quality warnings are not evidence of manipulation.",
            "Analysis is limited to the first 60 seconds.",
            "Checkpoint compatibility and label meanings require verification.",
            "No authenticity conclusion should rely on this report alone.",
        ],
    }

    st.download_button(
        "Download investigation report · JSON",
        data=json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        file_name="deeptrace_report.json",
        mime="application/json",
    )

    st.download_button(
        "Download segment evidence · CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="deeptrace_segments.csv",
        mime="text/csv",
    )

    st.caption(
        "Reports contain measurements, model references, and a file hash; "
        "they do not embed the uploaded video."
    )


st.divider()
st.caption(
    "DeepTrace · Research and educational use · "
    "Temporary media files are deleted after analysis. "
    "Results remain in session memory until cleared or the session ends. "
    "Model downloads may contact Hugging Face; media is not sent there "
    "by this application."
)
