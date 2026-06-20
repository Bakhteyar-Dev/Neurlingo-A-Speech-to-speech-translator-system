import os
import uuid
import numpy as np
import scipy.io.wavfile
import streamlit as st
import torch
import tempfile

from datetime import datetime
from transformers import (
    pipeline,
    VitsModel,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

WHISPER_MODEL_ID = "openai/whisper-small"
MARIAN_MODEL_ID  = "Bakhteyar/Balochi-Model"
TTS_MODEL_ID     = "facebook/mms-tts-bcc-script_latin"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_device = 0 if torch.cuda.is_available() else -1

# ----------------------------------------------------------------------------
# CACHED MODEL LOADERS
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=whisper_device
    )


@st.cache_resource(show_spinner=False)
def load_translation():
    tok = MarianTokenizer.from_pretrained(MARIAN_MODEL_ID)
    model = MarianMTModel.from_pretrained(MARIAN_MODEL_ID).to(device)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_tts():
    tok = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    model = VitsModel.from_pretrained(TTS_MODEL_ID).to(device)
    model.eval()
    return tok, model

# ----------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# ----------------------------------------------------------------------------

def english_speech_to_text(audio_path):
    result = load_asr()(audio_path)
    return (result.get("text", "") if isinstance(result, dict) else str(result)).strip()


def english_to_balochi(text):
    tok, model = load_translation()
    inputs = tok(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=4
        )

    return tok.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def balochi_text_to_speech(text):
    tok, model = load_tts()
    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        waveform = model(**inputs).waveform

    waveform = waveform.squeeze().detach().cpu().numpy()
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = (waveform * 32767).astype(np.int16)

    sr = model.config.sampling_rate
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    path = os.path.join(OUTPUT_DIR, f"balochi_{ts}_{uid}.wav")

    scipy.io.wavfile.write(path, sr, waveform)
    return path

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------

if "audio_key_id" not in st.session_state:
    st.session_state.audio_key_id = 0

if "output_audio_bytes" not in st.session_state:
    st.session_state.output_audio_bytes = None

if "eng_text" not in st.session_state:
    st.session_state.eng_text = ""

if "bal_text" not in st.session_state:
    st.session_state.bal_text = ""


def clear_all():
    st.session_state.audio_key_id += 1
    st.session_state.output_audio_bytes = None
    st.session_state.eng_text = ""
    st.session_state.bal_text = ""

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Neurolingo", page_icon="🌐", layout="wide")

# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    font-family: 'Inter', sans-serif !important;
    background:
        radial-gradient(circle at 12% 10%, rgba(56,189,248,0.14), transparent 30%),
        radial-gradient(circle at 88% 85%, rgba(37,99,235,0.12), transparent 34%),
        linear-gradient(135deg, #ffffff 0%, #f7fbff 45%, #eaf6ff 100%) !important;
    color: #0f172a !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="block-container"] {
    padding: 190px 22px 20px 22px !important;
    max-width: 1200px !important;
}

section.main > div {
    padding-top: 0 !important;
}

#MainMenu,
footer,
[data-testid="stToolbar"] {
    visibility: hidden;
}

/* ---------- Fixed Header ---------- */
.st-key-nl_topbar {
    position: fixed !important;
    top: 18px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: calc(100% - 44px) !important;
    max-width: 1200px !important;
    z-index: 999999 !important;
    border-radius: 24px;
    border: 1px solid #cfe7fb;
    background: rgba(255,255,255,0.96);
    box-shadow: 0 14px 34px rgba(15,23,42,0.08);
    padding: 26px 30px 18px 30px;
}

.nl-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.nl-logo {
    width: 86px;
    height: 58px;
    border-radius: 14px;
    background: linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#7c3aed 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 900;
    font-size: 28px;
    box-shadow: 0 10px 24px rgba(14,165,233,.16);
    border: 1px solid #d8edfb;
    flex-shrink: 0;
}

.nl-title {
    font-size: 34px;
    font-weight: 900;
    color: #000;
    line-height: 1.05;
}

.nl-subtitle {
    font-size: 15px;
    color: #000;
    margin-top: 5px;
}

.nl-chip {
    display: inline-flex;
    background: #0b5cad;
    color: #fff;
    font-weight: 900;
    font-size: 13px;
    border-radius: 999px;
    padding: 9px 18px;
    box-shadow: 0 8px 18px rgba(11,92,173,.22);
    white-space: nowrap;
    margin-bottom: 14px;
}

.nl-toggle-wrap {
    display: flex;
    justify-content: flex-end;
}

/* ---------- Cards ---------- */
.st-key-main_card {
    border-radius: 24px;
    border: 1px solid #cfe7fb;
    background: rgba(255,255,255,0.84);
    box-shadow: 0 16px 40px rgba(15,23,42,.08);
    padding: 20px;
    margin-bottom: 16px;
}

.st-key-input_panel,
.st-key-output_panel {
    border: 1px solid #d7e7f7;
    border-radius: 18px;
    background: #fff;
    box-shadow: 0 12px 26px rgba(15,23,42,.06);
    padding: 18px;
    min-height: 150px;
}

.nl-panel-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg,#ffffff 0%,#eaf6ff 55%,#dbeafe 100%);
    color: #000;
    font-weight: 900;
    border: 1px solid #a8d8f5;
    border-radius: 999px;
    padding: 9px 18px;
    margin-bottom: 14px;
    box-shadow: 0 8px 18px rgba(14,165,233,.12);
    font-size: 14px;
}

/* ---------- Audio Alignment ---------- */
[data-testid="stAudioInput"],
[data-testid="stAudio"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    margin-top: 8px !important;
}

[data-testid="stAudioInput"] > div,
[data-testid="stAudio"] > div {
    width: 100% !important;
}

.st-key-output_panel audio {
    width: 100% !important;
}

.nl-placeholder {
    color: #64748b;
    font-size: 14px;
    margin-top: 10px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: #0b5cad !important;
    color: #fff !important;
    font-weight: 900 !important;
    font-size: 18px !important;
    border-radius: 22px !important;
    border: none !important;
    height: 52px !important;
    width: 100% !important;
    box-shadow: 0 10px 22px rgba(11,92,173,.28) !important;
}

.stButton > button:hover {
    background: #084b91 !important;
    box-shadow: 0 12px 26px rgba(11,92,173,.34) !important;
}

/* ---------- Clear Button ---------- */
.st-key-btn_clear button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 900 !important;
    box-shadow: 0 10px 22px rgba(34,197,94,.26) !important;
}

.st-key-btn_clear button:hover {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
}

/* ---------- Text Result ---------- */
.nl-text-result {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 14px;
    background: #f8fbff;
    border: 1px solid #d7e7f7;
    font-size: 14px;
    color: #0f172a;
}

/* ---------- Mobile ---------- */
@media(max-width: 700px) {
    [data-testid="block-container"] {
        padding: 215px 14px 20px 14px !important;
    }

    .st-key-nl_topbar {
        top: 10px !important;
        width: calc(100% - 24px) !important;
        padding: 16px !important;
        border-radius: 18px;
    }

    .nl-brand {
        gap: 12px;
    }

    .nl-logo {
        width: 60px;
        height: 44px;
        font-size: 20px;
    }

    .nl-title {
        font-size: 24px;
    }

    .nl-subtitle {
        font-size: 12px;
    }

    .nl-chip {
        font-size: 12px;
        margin-top: 12px;
    }

    .st-key-input_panel,
    .st-key-output_panel {
        padding: 14px;
        min-height: 120px;
    }

    .stButton > button {
        font-size: 15px !important;
        height: 48px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FIXED TOP HEADER
# ----------------------------------------------------------------------------

with st.container(key="nl_topbar"):
    top_left, top_right = st.columns([3, 2], vertical_alignment="center")

    with top_left:
        st.markdown("""
        <div class="nl-brand">
            <div class="nl-logo">NL</div>
            <div>
                <div class="nl-title">Neurolingo</div>
                <div class="nl-subtitle">نیچرل لینگوئج رجانکارے ۔ انگریزی ءَ چہ بلوچی ءَ</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top_right:
        st.markdown("""
        <div style="text-align:right;">
            <div class="nl-chip">گپ ءِ سرا گپ ءِ رجانکار</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MAIN CARD
# ----------------------------------------------------------------------------

with st.container(key="main_card"):
    col_in, col_out = st.columns(2, gap="medium")

    with col_in:
        with st.container(key="input_panel"):
            st.markdown(
                '<div class="nl-panel-label">♫ انگریزی تواربند</div>',
                unsafe_allow_html=True
            )

            audio_input = st.audio_input(
                "Record or upload English audio",
                label_visibility="collapsed",
                key=f"eng_audio_{st.session_state.audio_key_id}"
            )

    with col_out:
        with st.container(key="output_panel"):
            st.markdown(
                '<div class="nl-panel-label">♫ بلوچی تواربند</div>',
                unsafe_allow_html=True
            )

            if st.session_state.output_audio_bytes:
                st.audio(st.session_state.output_audio_bytes, format="audio/wav")

                if st.session_state.eng_text:
                    st.markdown(
                        f'<div class="nl-text-result"><b>English:</b> {st.session_state.eng_text}</div>',
                        unsafe_allow_html=True
                    )

                if st.session_state.bal_text:
                    st.markdown(
                        f'<div class="nl-text-result"><b>Balochi:</b> {st.session_state.bal_text}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="nl-placeholder">Balochi audio will appear here after translation.</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([3, 1])

    with btn_col1:
        translate_clicked = st.button(
            "بلوچی ءَ رجانک بِکن ئِے",
            use_container_width=True,
            key="btn_translate"
        )

    with btn_col2:
        st.button(
            "Clear",
            use_container_width=True,
            key="btn_clear",
            on_click=clear_all
        )

# ----------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------

if translate_clicked:
    if audio_input is None:
        st.warning("Please record or upload an English audio clip first.")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_input.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Translating…"):
                eng_text = english_speech_to_text(tmp_path)
                bal_text = english_to_balochi(eng_text)
                out_path = balochi_text_to_speech(bal_text)

            with open(out_path, "rb") as f:
                audio_bytes = f.read()

            st.session_state.eng_text = eng_text
            st.session_state.bal_text = bal_text
            st.session_state.output_audio_bytes = audio_bytes

            st.rerun()

        except Exception as e:
            st.error(f"Translation failed: {e}")

        finally:
            os.unlink(tmp_path)
