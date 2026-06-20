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

# ─── Config ───────────────────────────────────────────────────────────────────
WHISPER_MODEL_ID  = "openai/whisper-small"
MARIAN_MODEL_ID   = "Bakhteyar/Balochi-Model"
TTS_MODEL_ID      = "facebook/mms-tts-bcc-script_latin"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device        = "cuda" if torch.cuda.is_available() else "cpu"
whisper_device = 0    if torch.cuda.is_available() else -1

# ─── Model loaders (cached so they load only once) ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=whisper_device,
    )

@st.cache_resource(show_spinner=False)
def load_translation():
    tok   = MarianTokenizer.from_pretrained(MARIAN_MODEL_ID)
    model = MarianMTModel.from_pretrained(MARIAN_MODEL_ID).to(device)
    model.eval()
    return tok, model

@st.cache_resource(show_spinner=False)
def load_tts():
    tok   = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    model = VitsModel.from_pretrained(TTS_MODEL_ID).to(device)
    model.eval()
    return tok, model

# ─── Pipeline functions ───────────────────────────────────────────────────────
def english_speech_to_text(audio_path: str) -> str:
    asr    = load_asr()
    result = asr(audio_path)
    if isinstance(result, dict):
        return result.get("text", "").strip()
    return str(result).strip()

def english_to_balochi(english_text: str) -> str:
    tok, model = load_translation()
    inputs = tok(
        english_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        tokens = model.generate(**inputs, max_length=256, num_beams=4)
    return tok.batch_decode(tokens, skip_special_tokens=True)[0].strip()

def balochi_text_to_speech(balochi_text: str) -> str:
    tok, model = load_tts()
    inputs = tok(balochi_text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        waveform = model(**inputs).waveform
    waveform = waveform.squeeze().detach().cpu().numpy()
    waveform = np.clip(waveform, -1.0, 1.0)
    waveform = (waveform * 32767).astype(np.int16)
    sr       = model.config.sampling_rate
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid      = uuid.uuid4().hex[:6]
    path     = os.path.join(OUTPUT_DIR, f"balochi_{ts}_{uid}.wav")
    scipy.io.wavfile.write(path, sr, waveform)
    return path

def translate_speech(audio_path: str):
    english_text  = english_speech_to_text(audio_path)
    balochi_text  = english_to_balochi(english_text)
    output_audio  = balochi_text_to_speech(balochi_text)
    return english_text, balochi_text, output_audio

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neurolingo",
    page_icon="🌐",
    layout="centered",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #ffffff 0%, #f7fbff 45%, #eaf6ff 100%) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Hero card */
.nl-hero {
    display: flex;
    align-items: center;
    gap: 18px;
    background: rgba(255,255,255,0.92);
    border: 1px solid #cfe7fb;
    border-radius: 22px;
    padding: 24px 28px;
    box-shadow: 0 14px 34px rgba(15,23,42,0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 28px;
}

.nl-logo {
    width: 68px; height: 68px;
    border-radius: 16px;
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 50%, #7c3aed 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 900; font-size: 22px;
    flex-shrink: 0;
    box-shadow: 0 8px 20px rgba(14,165,233,0.22);
}

.nl-title { font-size: 30px; font-weight: 900; color: #0f172a; margin: 0; line-height: 1.1; }
.nl-sub   { font-size: 14px; color: #475569; margin-top: 4px; }

.nl-badge {
    margin-left: auto;
    background: #0b5cad;
    color: white;
    font-weight: 800;
    font-size: 13px;
    border-radius: 999px;
    padding: 8px 18px;
    white-space: nowrap;
    flex-shrink: 0;
}

/* Panel cards */
.nl-card {
    background: #ffffff;
    border: 1px solid #d7e7f7;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.06);
    margin-bottom: 16px;
}

.nl-card-label {
    font-size: 13px;
    font-weight: 800;
    color: #0b5cad;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    text-transform: uppercase;
}

/* Flow pills */
.nl-flow {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.nl-pill {
    padding: 6px 14px;
    border-radius: 999px;
    color: #075fb8;
    background: linear-gradient(135deg, #ffffff 0%, #eef8ff 55%, #dbeeff 100%);
    border: 1px solid #a8d8f5;
    font-size: 13px;
    font-weight: 700;
}

.nl-arrow { color: #94a3b8; font-size: 18px; font-weight: 700; }

/* Translate button */
.stButton > button {
    background: #0b5cad !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 17px !important;
    border-radius: 18px !important;
    border: none !important;
    height: 54px !important;
    width: 100% !important;
    box-shadow: 0 10px 22px rgba(11,92,173,0.28) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #084b91 !important;
    box-shadow: 0 12px 28px rgba(11,92,173,0.38) !important;
}

/* Text boxes */
.nl-textbox {
    background: #f7fbff;
    border: 1px solid #d7e7f7;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 15px;
    color: #0f172a;
    min-height: 64px;
    word-break: break-word;
    line-height: 1.6;
}

/* Footer */
.nl-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 28px;
    padding-bottom: 16px;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nl-hero">
    <div class="nl-logo">NL</div>
    <div>
        <div class="nl-title">Neurolingo</div>
        <div class="nl-sub">A Natural Language Translator from English to Balochi</div>
    </div>
    <div class="nl-badge">Speech-to-Speech Translator</div>
</div>
""", unsafe_allow_html=True)

# ─── Flow pills ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="nl-flow">
    <span class="nl-pill">🎤 English Speech</span>
    <span class="nl-arrow">→</span>
    <span class="nl-pill">📝 Whisper ASR</span>
    <span class="nl-arrow">→</span>
    <span class="nl-pill">🔄 MarianMT</span>
    <span class="nl-arrow">→</span>
    <span class="nl-pill">🔊 Balochi TTS</span>
</div>
""", unsafe_allow_html=True)

# ─── Model warm-up notice ─────────────────────────────────────────────────────
with st.expander("ℹ️ First-time setup info", expanded=False):
    st.info(
        "Models are downloaded from Hugging Face on first use and cached locally. "
        "This may take a few minutes the very first time. After that, everything is fast."
    )

# ─── Audio input ──────────────────────────────────────────────────────────────
st.markdown('<div class="nl-card"><div class="nl-card-label">🎤 English Speech Input</div>', unsafe_allow_html=True)
audio_input = st.audio_input("Record or upload your English audio")
st.markdown('</div>', unsafe_allow_html=True)

# ─── Translate button ─────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    translate_clicked = st.button("Translate to Balochi Speech", use_container_width=True)

# ─── Run pipeline ─────────────────────────────────────────────────────────────
if translate_clicked:
    if audio_input is None:
        st.warning("Please record or upload an English audio clip first.")
    else:
        # Save uploaded bytes to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_input.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Loading models & translating… this may take a moment on first run."):
                eng_text, bal_text, out_audio_path = translate_speech(tmp_path)

            # ── Transcription ──────────────────────────────────────────────
            st.markdown('<div class="nl-card"><div class="nl-card-label">📝 Transcribed English Text</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="nl-textbox">{eng_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Balochi text ───────────────────────────────────────────────
            st.markdown('<div class="nl-card"><div class="nl-card-label">🔄 Balochi Translation (Latin Script)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="nl-textbox">{bal_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Output audio ───────────────────────────────────────────────
            st.markdown('<div class="nl-card"><div class="nl-card-label">🔊 Balochi Speech Output</div>', unsafe_allow_html=True)
            with open(out_audio_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                label="⬇️ Download Balochi Audio",
                data=audio_bytes,
                file_name="balochi_speech.wav",
                mime="audio/wav",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Translation failed: {e}")
        finally:
            os.unlink(tmp_path)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nl-footer">
    Neurolingo · English → Balochi (Latin Script) · Powered by Whisper · MarianMT · MMS TTS
</div>
""", unsafe_allow_html=True)
