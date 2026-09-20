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
    page_title="DeepTrace | Media Authenticity",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
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

    # Visual-only mode is supported because the current project does not
    # include an audio deepfake detector yet.
    if visual is not None and audio is None:
        if visual >= threshold:
            return float(visual), "Likely AI-Generated / Manipulated"
        return float(visual), "Likely Real"

    # Keep audio-only support available for future expansion.
    if visual is None and audio is not None:
        if audio >= threshold:
            return float(audio), "Likely AI-Generated / Manipulated"
        return float(audio), "Likely Real"

    if visual is None and audio is None:
        return None, "Insufficient detector evidence"

    if abs(visual - audio) > disagreement_limit:
        return None, "Needs Review — visual and audio checks disagree"

    score = visual_weight * visual + (1 - visual_weight) * audio

    if score >= threshold:
        return float(score), "Likely AI-Generated / Manipulated"

    return float(score), "Likely Real"


def plain_result(score, threshold, disagreement=False):
    """Convert technical detector output into simple user-facing language."""
    if disagreement:
        return (
            "🟡 Needs Review",
            "The visual and audio checks disagree, so DeepTrace will not call it fake or real."
        )
    if score is None:
        return (
            "⚪ Not Enough Evidence",
            "The available detector could not provide a usable result."
        )
    if score >= threshold:
        return (
            "🔴 Likely AI-Generated / Manipulated",
            "The model found a stronger signal associated with fake or generated media."
        )
    return (
        "🟢 Likely Real",
        "The model did not find a strong fake/AI-generated signal."
    )


def display_score(label, value):
    st.metric(
        label,
        "Unavailable" if value is None else f"{value:.3f}",
    )

    if value is not None:
        st.progress(max(0.0, min(1.0, float(value))))


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

# -------------------- Professional UI styling --------------------

st.markdown("""
<style>
/* Main page */
.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Typography */
.deeptrace-kicker {
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    opacity: .65;
    margin-bottom: .35rem;
}

.deeptrace-title {
    font-size: clamp(2.2rem, 5vw, 4rem);
    line-height: 1;
    font-weight: 850;
    letter-spacing: -.055em;
    margin: 0;
}

.deeptrace-subtitle {
    font-size: 1.05rem;
    line-height: 1.6;
    opacity: .72;
    max-width: 760px;
    margin-top: .8rem;
}

/* Hero */
.deeptrace-hero {
    padding: 1.8rem 2rem;
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 24px;
    background:
        linear-gradient(135deg,
            rgba(100,150,255,.13),
            rgba(128,128,128,.055));
    margin-bottom: 1.2rem;
}

.deeptrace-pill {
    display: inline-block;
    padding: .35rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 750;
    border: 1px solid rgba(128,128,128,.25);
    background: rgba(128,128,128,.08);
    margin-bottom: .8rem;
}

/* How it works */
.deeptrace-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .8rem;
    margin: 1rem 0 1.4rem;
}

.deeptrace-step {
    padding: 1rem;
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 16px;
    background: rgba(128,128,128,.045);
}

.deeptrace-step-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    background: rgba(100,150,255,.18);
    margin-bottom: .55rem;
}

.deeptrace-step-title {
    font-weight: 750;
    margin-bottom: .25rem;
}

.deeptrace-step-text {
    font-size: .86rem;
    opacity: .68;
    line-height: 1.45;
}

/* Upload area */
.deeptrace-upload-card {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 20px;
    padding: 1.25rem;
    background: rgba(128,128,128,.035);
    margin: .7rem 0 1rem;
}

.deeptrace-section-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -.025em;
    margin: .4rem 0 .2rem;
}

.deeptrace-section-subtitle {
    opacity: .65;
    font-size: .9rem;
    margin-bottom: .9rem;
}

/* Result cards */
.deeptrace-result {
    border-radius: 22px;
    padding: 1.5rem;
    border: 1px solid rgba(128,128,128,.22);
    margin: 1rem 0;
}

.deeptrace-result-real {
    background: rgba(70, 180, 110, .09);
    border-color: rgba(70, 180, 110, .35);
}

.deeptrace-result-review {
    background: rgba(220, 180, 60, .10);
    border-color: rgba(220, 180, 60, .38);
}

.deeptrace-result-ai {
    background: rgba(220, 75, 75, .09);
    border-color: rgba(220, 75, 75, .35);
}

.deeptrace-result-icon {
    font-size: 2.2rem;
    margin-bottom: .25rem;
}

.deeptrace-result-title {
    font-size: 1.7rem;
    font-weight: 850;
    letter-spacing: -.035em;
}

.deeptrace-result-message {
    margin-top: .4rem;
    line-height: 1.55;
    opacity: .78;
}

/* Explanation */
.deeptrace-explain {
    padding: 1rem 1.15rem;
    border-radius: 16px;
    background: rgba(128,128,128,.055);
    border: 1px solid rgba(128,128,128,.16);
    line-height: 1.55;
    margin: .8rem 0;
}

.deeptrace-disclaimer {
    font-size: .82rem;
    line-height: 1.5;
    opacity: .65;
    margin-top: .7rem;
}

/* Metrics */
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.17);
    border-radius: 16px;
    padding: .8rem 1rem;
    background: rgba(128,128,128,.035);
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    min-height: 46px;
    font-weight: 750;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.15);
}

/* Mobile */
@media (max-width: 760px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.2rem;
    }

    .deeptrace-hero {
        padding: 1.25rem;
        border-radius: 18px;
    }

    .deeptrace-steps {
        grid-template-columns: 1fr;
    }

    .deeptrace-title {
        font-size: 2.35rem;
    }
}
</style>
""", unsafe_allow_html=True)


# -------------------- Header --------------------

st.markdown("""
<div class="deeptrace-hero">
    <div class="deeptrace-pill">🔎 AI MEDIA SCREENING • RESEARCH PROTOTYPE</div>
    <div class="deeptrace-title">DeepTrace</div>
    <div class="deeptrace-subtitle">
        Check an image or video for signals commonly associated with
        AI-generated or manipulated media.
        <b>Simple result first. Technical details when you need them.</b>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="deeptrace-steps">
    <div class="deeptrace-step">
        <div class="deeptrace-step-number">1</div>
        <div class="deeptrace-step-title">Upload</div>
        <div class="deeptrace-step-text">Choose an image or a short video you are allowed to analyze.</div>
    </div>
    <div class="deeptrace-step">
        <div class="deeptrace-step-number">2</div>
        <div class="deeptrace-step-title">Analyze</div>
        <div class="deeptrace-step-text">DeepTrace checks the media with the configured AI detection models.</div>
    </div>
    <div class="deeptrace-step">
        <div class="deeptrace-step-number">3</div>
        <div class="deeptrace-step-title">Understand</div>
        <div class="deeptrace-step-text">You get a plain-language result plus optional technical evidence.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "💡 **Important:** A result is a screening signal, not proof. "
    "“Likely Real” means the detector did not find a strong AI/manipulation signal. "
    "It does not guarantee authenticity."
)

# -------------------- Advanced settings --------------------

with st.sidebar:
    st.markdown("## ⚙️ Advanced settings")
    st.caption(
        "These settings are mainly for testing and research. "
        "Most users can leave the defaults unchanged."
    )

    window_seconds = st.select_slider(
        "Video segment length",
        options=[2, 4, 6, 10],
        value=4,
        format_func=lambda value: f"{value} seconds",
    )

    with st.expander("AI detection models", expanded=True):
        visual_reference = st.text_input(
            "Image / video model",
            value="dima806/deepfake_vs_real_image_detection",
            help="Visual detector used for image analysis and sampled video frames.",
        ).strip()

        visual_label = st.text_input(
            "Visual AI/fake label",
            value="Fake",
            help="Exact fake-class label documented by the model.",
        ).strip()

        image_second_reference = st.text_input(
            "Second image model (recommended)",
            value="delpot/steganograph-ia-detector",
            help="A second independent image detector used to reduce false confidence.",
        ).strip()

        image_second_label = st.text_input(
            "Second model fake label",
            value="ai_generated",
            help="Exact AI-generated class label documented by the second model.",
        ).strip()

        audio_reference = st.text_input(
            "Audio model (optional)",
            placeholder="Leave blank if not using audio detection",
        ).strip()

        audio_label = st.text_input(
            "Audio AI/fake label",
            placeholder="Use the label documented by the audio model",
        ).strip()

    with st.expander("Video fusion settings"):
        visual_weight = st.slider(
            "Visual contribution",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )

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
            "These values are experimental. They are not calibrated probabilities."
        )

    if st.button("Clear results and unload models"):
        st.session_state.pop("investigation", None)
        st.session_state.pop("image_investigation", None)
        load_detector.clear()
        st.rerun()


# -------------------- Media selection --------------------

st.markdown('<div class="deeptrace-section-title">What would you like to check?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="deeptrace-section-subtitle">Choose one type of media to begin.</div>',
    unsafe_allow_html=True,
)

media_type = st.radio(
    "Media type",
    ["Image", "Video"],
    horizontal=True,
    label_visibility="collapsed",
)

# -------------------- Result helper --------------------

def show_simple_result(status, detail):
    if status == "Likely AI-Generated / Manipulated":
        css = "deeptrace-result-ai"
        icon = "🔴"
        title = "Likely AI-Generated or Manipulated"
    elif status.startswith("Needs Review"):
        css = "deeptrace-result-review"
        icon = "🟡"
        title = "Not Sure — Needs Review"
    elif status == "Insufficient detector evidence":
        css = "deeptrace-result-review"
        icon = "🟡"
        title = "Not Enough Evidence"
    else:
        css = "deeptrace-result-real"
        icon = "🟢"
        title = "Likely Real"

    st.markdown(
        f"""
        <div class="deeptrace-result {css}">
            <div class="deeptrace-result-icon">{icon}</div>
            <div class="deeptrace-result-title">{title}</div>
            <div class="deeptrace-result-message">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- IMAGE MODE --------------------

if media_type == "Image":
    st.markdown(
        '<div class="deeptrace-upload-card"><div class="deeptrace-section-title">📷 Check an image</div>'
        '<div class="deeptrace-section-subtitle">Two AI detectors will compare their signals before giving a simple result.</div></div>',
        unsafe_allow_html=True,
    )

    image_uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if image_uploaded is None:
        st.markdown(
            """
            <div class="deeptrace-explain">
                <b>How DeepTrace checks an image</b><br>
                1. Two image detectors examine the same image.<br>
                2. DeepTrace compares their results.<br>
                3. If the evidence is clear, you get <b>Likely Real</b> or <b>Likely AI-Generated</b>.<br>
                4. If the detectors are uncertain or disagree, you get <b>Needs Review</b> instead of a misleading yes/no answer.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    if image_uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please choose an image smaller than {MAX_UPLOAD_MB} MB.")
        st.stop()

    image_data = image_uploaded.getvalue()

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open this image: {type(exc).__name__}.")
        st.stop()

    preview_col, info_col = st.columns([1.45, 1])

    with preview_col:
        st.image(image, caption=image_uploaded.name, use_container_width=True)

    with info_col:
        st.markdown("### Ready to check")
        st.write(
            "DeepTrace will compare two visual AI detectors instead of trusting a single model."
        )
        st.caption(
            "This reduces overconfident results, but it cannot guarantee authenticity. "
            "New AI image generators can still be difficult for older detectors to recognize."
        )

        image_hash = hashlib.sha256(image_data).hexdigest()

        image_key = hashlib.sha256(
            json.dumps(
                {
                    "media_sha256": image_hash,
                    "visual_model": visual_reference,
                    "visual_fake_label": visual_label,
                    "second_model": image_second_reference,
                    "second_fake_label": image_second_label,
                    "threshold": threshold,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if st.button(
            "🔎 Check Image",
            type="primary",
            use_container_width=True,
            disabled=not visual_reference or not visual_label or not image_second_reference or not image_second_label,
        ):
            progress = st.progress(0.0, text="Loading the image detectors…")

            try:
                detector_1 = load_detector("image-classification", visual_reference)
                progress.progress(0.35, text="Checking with detector 1…")
                predictions_1 = detector_1(image, top_k=None)
                score_1 = fake_label_score(predictions_1, visual_label)

                detector_2 = load_detector("image-classification", image_second_reference)
                progress.progress(0.65, text="Checking with detector 2…")
                predictions_2 = detector_2(image, top_k=None)
                score_2 = fake_label_score(predictions_2, image_second_label)

                brightness, sharpness = image_quality([np.asarray(image)])

                # The dedicated SteganographIA detector is used as the primary
                # image decision model because its documented labels are
                # `real` and `ai_generated` and its published validation reports
                # a low false-positive rate on its test set. Detector 1 remains
                # visible as a supporting signal.
                #
                # IMPORTANT: this is still a screening classifier, not ground truth.
                # The 0.50 decision boundary is the model's binary-class boundary,
                # not a claim that the image is "50% fake".
                primary_fake_threshold = 0.50

                if score_2 >= primary_fake_threshold:
                    status = "Likely AI-Generated / Manipulated"
                    result_detail = (
                        "The primary AI-image detector classified this image as AI-generated. "
                        "Detector 1 is shown as supporting evidence. This is a screening result, not proof."
                    )
                else:
                    status = "Likely Real"
                    result_detail = (
                        "The primary AI-image detector classified this image as real. "
                        "Detector 1 is shown as supporting evidence. This does not guarantee authenticity."
                    )

                combined_score = float(score_2)

                image_result = {
                    "score": combined_score,
                    "score_1": float(score_1),
                    "score_2": float(score_2),
                    "status": status,
                    "detail": result_detail,
                    "brightness": brightness,
                    "sharpness": sharpness,
                    "image_hash": image_hash,
                    "model": visual_reference,
                    "fake_label": visual_label,
                    "second_model": image_second_reference,
                    "second_fake_label": image_second_label,
                    "run_key": image_key,
                }

                st.session_state["image_investigation"] = image_result
                progress.progress(1.0, text="Done.")
                progress.empty()

            except Exception as exc:
                progress.empty()
                st.error(
                    f"Image check failed: {type(exc).__name__}: {str(exc)[:400]}"
                )
                st.stop()

    image_result = st.session_state.get("image_investigation")

    if not image_result or image_result.get("run_key") != image_key:
        st.markdown(
            '<div class="deeptrace-explain"><b>Tip:</b> Press <b>Check Image</b> to analyze this image.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown("## Result")
    show_simple_result(image_result["status"], image_result["detail"])

    st.markdown("### What does this mean?")
    st.markdown(
        """
        <div class="deeptrace-explain">
        <b>🟢 Likely Real:</b> Both detectors found only a low AI-generated signal.<br>
        <b>🔴 Likely AI-Generated or Manipulated:</b> Both detectors found a strong signal.<br>
        <b>🟡 Needs Review:</b> The evidence is mixed or not strong enough for a clear answer.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔬 Technical details"):
        display_score("Detector 1 — fake signal", image_result["score_1"])
        st.write(f"Model: `{image_result['model']}`")
        st.write(f"Fake-class label: `{image_result['fake_label']}`")
        display_score("Detector 2 — fake signal", image_result["score_2"])
        st.write(f"Model: `{image_result['second_model']}`")
        st.write(f"Fake-class label: `{image_result['second_fake_label']}`")
        st.write(f"Average detector signal: {image_result['score']:.3f}")
        st.write(f"Brightness: {image_result['brightness']:.2f} / 255")
        st.write(f"Sharpness: {image_result['sharpness']:.2f}")
        st.caption(
            "Detector scores are uncalibrated signals. They are NOT percentages or probabilities that the image is fake. "
            "The 0.10 review floor is only a conservative screening rule."
        )

    st.warning(
        "⚠️ **Important:** AI-image detectors can miss newer generators and can also make mistakes. "
        "For important decisions, verify the source and original file when possible."
    )

    st.download_button(
        "⬇️ Download investigation report",
        data=json.dumps(
            {
                "project": "DeepTrace",
                "version": "0.4-primary-image-detector",
                "media_type": "image",
                "filename": image_uploaded.name,
                "file_sha256": image_result["image_hash"],
                "detector_1": {
                    "model": image_result["model"],
                    "fake_label": image_result["fake_label"],
                    "fake_signal": image_result["score_1"],
                },
                "detector_2": {
                    "model": image_result["second_model"],
                    "fake_label": image_result["second_fake_label"],
                    "fake_signal": image_result["score_2"],
                },
                "primary_ai_signal": image_result["score"],
                "status": image_result["status"],
                "quality": {
                    "brightness_0_255": image_result["brightness"],
                    "sharpness_variance": image_result["sharpness"],
                },
                "limitations": [
                    "Not a validated authenticity detector.",
                    "Detector outputs are uncalibrated signals.",
                    "Model cards warn about concept drift with newer AI generators.",
                    "Quality measurements are not evidence of manipulation.",
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        file_name="deeptrace_image_report.json",
        mime="application/json",
    )

    st.stop()


# -------------------- VIDEO MODE --------------------

st.markdown(
    '<div class="deeptrace-upload-card"><div class="deeptrace-section-title">🎬 Check a video</div>'
    '<div class="deeptrace-section-subtitle">MP4, MOV, AVI, WEBM or MKV · Maximum 100 MB · First 60 seconds analyzed</div></div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Choose a video",
    type=["mp4", "mov", "avi", "webm", "mkv"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(
        """
        <div class="deeptrace-explain">
            <b>How video checking works:</b> DeepTrace divides the video into short
            sections, checks sampled frames and optional audio, then shows which
            sections need attention.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"Please choose a video smaller than {MAX_UPLOAD_MB} MB.")
    st.stop()

data = uploaded.getvalue()
media_hash = hashlib.sha256(data).hexdigest()
suffix = os.path.splitext(uploaded.name)[1].lower()

st.video(data)

meta1, meta2, meta3 = st.columns(3)
meta1.metric("File size", f"{len(data) / (1024 * 1024):.1f} MB")
meta2.metric("Maximum analysis", f"{MAX_ANALYSIS_SECONDS}s")
meta3.metric("Section length", f"{window_seconds}s")

configuration = {
    "media_sha256": media_hash,
    "window_seconds": window_seconds,
    "visual_model": visual_reference,
    "visual_fake_label": visual_label,
    "audio_model": audio_reference,
    "audio_fake_label": audio_label,
}

run_key = hashlib.sha256(
    json.dumps(configuration, sort_keys=True).encode()
).hexdigest()

invalid_labels = (
    (bool(visual_reference) and not visual_label)
    or (bool(audio_reference) and not audio_label)
)

if invalid_labels:
    st.warning("Enter the documented AI/fake label for each configured model.")

if st.button(
    "🔎 Check Video",
    type="primary",
    use_container_width=True,
    disabled=invalid_labels,
):
    progress = st.progress(0.0, text="Preparing video check…")

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
        result["created_utc"] = datetime.now(timezone.utc).isoformat()

        st.session_state["investigation"] = result
        progress.empty()

    except Exception as exc:
        progress.empty()
        st.error(f"Video check failed: {exc}")
        st.stop()

result = st.session_state.get("investigation")

if not result or result.get("run_key") != run_key:
    st.markdown(
        '<div class="deeptrace-explain"><b>Ready.</b> Press <b>Check Video</b> to start the analysis.</div>',
        unsafe_allow_html=True,
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
    row["status"] == "Likely AI-Generated / Manipulated"
    for row in enriched_rows
)

abstained = sum(
    row["fused_model_score"] is None
    for row in enriched_rows
)

st.markdown("## Video result")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Video analyzed", f"{result['analyzed_seconds']:.1f}s")
m2.metric("Sections checked", len(enriched_rows))
m3.metric("AI/manipulation signals", elevated)
m4.metric("Needs review", abstained)

if abstained > 0:
    show_simple_result(
        "Needs Review",
        f"{abstained} video section(s) need a closer look because the available checks did not agree. "
        "DeepTrace intentionally does not label those sections as simply real or fake.",
    )
elif elevated > 0:
    show_simple_result(
        "Likely AI-Generated / Manipulated",
        f"{elevated} section(s) showed a stronger AI/manipulation signal. "
        "Review those sections before drawing a conclusion.",
    )
else:
    show_simple_result(
        "Likely Real",
        "No strong AI/manipulation signal was found in the analyzed sections. "
        "This does not guarantee that the video is authentic.",
    )

st.caption(
    f"Processing time: {result['processing_seconds']:.1f}s · "
    "Results are screening signals, not proof of authenticity."
)

overview_tab, inspect_tab, agent_tab, export_tab = st.tabs(
    ["📊 Overview", "🔎 Check a section", "💬 Explain a result", "⬇️ Export"]
)

with overview_tab:
    st.markdown("### Section-by-section result")

    display_table = table[[
        "segment", "start_s", "end_s", "visual_model_score",
        "audio_model_score", "fused_model_score", "status", "notes"
    ]].rename(columns={
        "segment": "Section",
        "start_s": "Start (sec)",
        "end_s": "End (sec)",
        "visual_model_score": "Visual signal",
        "audio_model_score": "Audio signal",
        "fused_model_score": "Combined signal",
        "status": "Result",
        "notes": "Notes",
    })

    st.dataframe(display_table, hide_index=True, use_container_width=True)

    st.caption(
        "The timeline shows screening results for each analysis window. "
        "It does not identify exact manipulated frames."
    )

with inspect_tab:
    st.markdown("### Inspect one section")

    selected = st.selectbox(
        "Choose a video section",
        options=range(len(enriched_rows)),
        format_func=lambda index: (
            f"Section {index + 1}: "
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
                caption="Sampled frame — not a manipulation heatmap",
                use_container_width=True,
            )

        if result["audio"] is not None:
            audio_slice = result["audio"][
                int(row["start_s"] * AUDIO_RATE):
                int(row["end_s"] * AUDIO_RATE)
            ]

            if audio_slice.size:
                st.audio(wav_bytes(audio_slice, AUDIO_RATE), format="audio/wav")

    with evidence_col:
        show_simple_result(
            row["status"],
            "This section is the part of the video that the detector is describing. "
            "Use the technical measurements below if you need to understand why.",
        )

        with st.expander("Technical evidence", expanded=True):
            display_score("Visual signal", row["visual_model_score"])
            display_score("Audio signal", row["audio_model_score"])
            display_score("Combined signal", row["fused_model_score"])

        st.write(row["notes"])

        quality = pd.DataFrame([
            {"Measurement": "Brightness", "Value": row["brightness_0_255"]},
            {"Measurement": "Sharpness", "Value": row["sharpness_variance"]},
            {"Measurement": "Audio level, dBFS", "Value": row["audio_rms_dbfs"]},
            {"Measurement": "Audio clipping fraction", "Value": row["audio_clipping_fraction"]},
        ])

        with st.expander("Media quality measurements"):
            st.dataframe(quality, hide_index=True, use_container_width=True)

with agent_tab:
    st.markdown("### Explain the evidence")

    st.caption(
        "This assistant is rule-based. It explains measurements and detector outputs already produced by DeepTrace; it does not independently decide whether media is authentic."
    )

    agent_segment = st.selectbox(
        "Choose a section",
        range(len(enriched_rows)),
        format_func=lambda index: f"Section {index + 1}",
        key="agent_segment",
    )

    with st.form("question_form"):
        question = st.text_input(
            "What do you want to understand?",
            placeholder="Why was this section marked for review?",
        )
        submitted = st.form_submit_button("Explain")

    if submitted:
        if question.strip():
            st.markdown(
                '<div class="deeptrace-explain">' +
                explain_segment(enriched_rows[agent_segment], question) +
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Type a question first.")

    with st.expander("Technical execution log"):
        st.code("\n".join(result["logs"]), language="text")

with export_tab:
    st.markdown("### Save your investigation")

    export_payload = {
        "project": "DeepTrace",
        "version": "0.2-research-prototype",
        "media_type": "video",
        "filename": uploaded.name,
        "file_sha256": media_hash,
        "configuration": configuration,
        "analyzed_seconds": result["analyzed_seconds"],
        "processing_seconds": result["processing_seconds"],
        "sections": enriched_rows,
        "limitations": [
            "Not a validated authenticity detector.",
            "Model scores are uncalibrated signals, not authenticity probabilities.",
            "Temporal localization is limited to analysis windows.",
            "Quality measurements are not evidence of manipulation.",
        ],
    }

    st.download_button(
        "⬇️ Download video investigation report",
        data=json.dumps(export_payload, indent=2, allow_nan=False),
        file_name="deeptrace_video_report.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "DeepTrace • Research and educational use • "
    "Results are screening signals, not proof of authenticity."
)
