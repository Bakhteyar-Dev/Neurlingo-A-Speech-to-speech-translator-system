import os
import uuid
import numpy as np
import scipy.io.wavfile
import streamlit as st
import streamlit.components.v1 as components
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
WHISPER_MODEL_ID = "openai/whisper-small"
MARIAN_MODEL_ID  = "Bakhteyar/Balochi-Model"
TTS_MODEL_ID     = "facebook/mms-tts-bcc-script_latin"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_device = 0 if torch.cuda.is_available() else -1

# ─── Cached model loaders ─────────────────────────────────────────────────────
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

# ─── Pipeline functions ────────────────────────────────────────────────────────
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
        tokens = model.generate(**inputs, max_length=256, num_beams=4)

    return tok.batch_decode(tokens, skip_special_tokens=True)[0].strip()

def balochi_text_to_speech(text):
    tok, model = load_tts()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
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

# ─── Session State ────────────────────────────────────────────────────────────
if "audio_key_id" not in st.session_state:
    st.session_state.audio_key_id = 0

if "output_audio" not in st.session_state:
    st.session_state.output_audio = None

def clear_all():
    st.session_state.audio_key_id += 1
    st.session_state.output_audio = None

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neurolingo", page_icon="🌐", layout="wide")

# ─── Global page CSS ──────────────────────────────────────────────────────────
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
    padding: 195px 22px 20px 22px !important;
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

/* FIXED NEUROLINGO HEADER */
.st-key-hero_fixed {
    position: fixed !important;
    top: 18px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: calc(100% - 44px) !important;
    max-width: 1200px !important;
    z-index: 999999 !important;
}

.st-key-hero_fixed iframe {
    width: 100% !important;
    display: block !important;
}

/* MAIN CARD */
.st-key-main_card {
    border-radius: 24px;
    border: 1px solid #cfe7fb;
    background: rgba(255,255,255,0.84);
    box-shadow: 0 16px 40px rgba(15,23,42,.08);
    padding: 18px;
    margin-bottom: 16px;
}

/* AUDIO PANEL WRAPPERS */
.st-key-input_panel,
.st-key-output_panel {
    border: 1px solid #d7e7f7;
    border-radius: 18px;
    background: #fff;
    box-shadow: 0 12px 26px rgba(15,23,42,.06);
    padding: 18px;
    min-height: 210px;
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
    font-family: 'Inter', sans-serif;
}

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

.nl-output-empty {
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #94a3b8;
    font-size: 34px;
}

/* BUTTONS */
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

/* CLEAR BUTTON */
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

/* COLUMNS */
div[data-testid="column"] {
    padding: 0 6px !important;
}

/* MOBILE */
@media(max-width:700px) {
    [data-testid="block-container"] {
        padding: 245px 14px 20px 14px !important;
    }

    .st-key-hero_fixed {
        top: 10px !important;
        width: calc(100% - 24px) !important;
    }

    .st-key-main_card {
        padding: 14px;
    }

    .st-key-input_panel,
    .st-key-output_panel {
        padding: 14px;
        min-height: 170px;
    }

    .stButton > button {
        font-size: 15px !important;
        height: 48px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── HERO — original header and original toggle preserved ─────────────────────
with st.container(key="hero_fixed"):
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Inter', sans-serif;
        background: transparent;
        overflow: hidden;
    }
    .hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
        padding: 28px 32px;
        border-radius: 22px;
        border: 1px solid #cfe7fb;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 14px 34px rgba(15,23,42,0.08);
    }
    .brand { display: flex; align-items: center; gap: 16px; }
    .logo {
        width: 86px; height: 58px; border-radius: 14px;
        background: linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#7c3aed 100%);
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 900; font-size: 28px;
        box-shadow: 0 10px 24px rgba(14,165,233,.16);
        border: 1px solid #d8edfb; flex-shrink: 0;
    }
    .main-title { font-size: 34px; font-weight: 900; color: #000; line-height: 1.05; }
    .subtitle   { font-size: 15px; color: #000; margin-top: 5px; }
    .hero-right { display: flex; flex-direction: column; align-items: flex-end; gap: 16px; }
    .chip {
        background: #0b5cad; color: #fff; font-weight: 900; font-size: 13px;
        border-radius: 999px; padding: 9px 18px;
        box-shadow: 0 8px 18px rgba(11,92,173,.22); white-space: nowrap;
    }
    .toggle-box {
        display: flex; align-items: center; gap: 10px;
        background: #fff; padding: 7px 12px;
        border-radius: 999px; border: 1px solid #d6e4f0;
        box-shadow: 0 8px 20px rgba(15,23,42,.05); min-width: 260px;
    }
    .toggle-text { font-size: 14px; font-weight: 900; color: #0f172a; white-space: nowrap; }
    .switch { position: relative; display: inline-block; width: 64px; height: 34px; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
        position: absolute; cursor: pointer; inset: 0;
        background: #e2e8f0; border-radius: 999px;
        border: 1px solid #d6e4f0; transition: .25s;
    }
    .slider:before {
        position: absolute; content: "";
        height: 28px; width: 28px; left: 3px; top: 2px;
        background: #fff; border-radius: 50%;
        box-shadow: 0 3px 8px rgba(15,23,42,.16); transition: .25s;
    }
    .switch input:checked + .slider { background: linear-gradient(135deg,#38bdf8,#2563eb); }
    .switch input:checked + .slider:before { transform: translateX(30px); }

    @media(max-width:700px){
        .hero { flex-direction: column; align-items: flex-start; padding: 16px; }
        .hero-right { width: 100%; align-items: flex-start; }
        .main-title { font-size: 24px; }
        .logo { width: 60px; height: 44px; font-size: 20px; }
    }
    </style>
    </head>
    <body>
    <div class="hero">
      <div class="brand">
        <div class="logo">NL</div>
        <div>
          <div class="main-title">Neurolingo</div>
          <div class="subtitle" id="subtitle">A Natural Language Translator from English to Balochi</div>
        </div>
      </div>
      <div class="hero-right">
        <div class="chip" id="chip">Speech-to-Speech Translator</div>
        <div class="toggle-box">
          <span class="toggle-text">English</span>
          <label class="switch">
            <input type="checkbox" id="lang-toggle">
            <span class="slider"></span>
          </label>
          <span class="toggle-text">بلوچی</span>
        </div>
      </div>
    </div>

    <script>
    var EN = {
        subtitle : "A Natural Language Translator from English to Balochi",
        chip     : "Speech-to-Speech Translator",
        panelIn  : "&#9836; English Speech",
        panelOut : "&#9836; Balochi Speech",
        btn      : "Translate to Balochi Speech"
    };
    var BAL = {
        subtitle : "نیچرل لینگوئج رجانکارے ۔ انگریزی ءَ چہ بلوچی ءَ",
        chip     : "گپ ءِ سرا گپ ءِ رجانکار",
        panelIn  : "&#9836; انگریزی تواربند",
        panelOut : "&#9836; بلوچی تواربند",
        btn      : "بلوچی ءَ رجانک بِکن ئِے"
    };

    function getParentDoc() {
        try { return window.parent.document; } catch(e) { return null; }
    }

    function applyLang(map) {
        document.getElementById("subtitle").innerHTML = map.subtitle;
        document.getElementById("chip").innerHTML = map.chip;

        var pd = getParentDoc();
        if (!pd) return;

        var panelIn = pd.getElementById("nl-panel-in");
        if (panelIn) panelIn.innerHTML = map.panelIn;

        var panelOut = pd.getElementById("nl-panel-out");
        if (panelOut) panelOut.innerHTML = map.panelOut;

        var btns = pd.querySelectorAll("button");
        btns.forEach(function(b) {
            var t = b.innerText.trim();
            if (t === EN.btn || t === BAL.btn) b.innerText = map.btn;
        });
    }

    document.getElementById("lang-toggle").addEventListener("change", function() {
        applyLang(this.checked ? BAL : EN);
    });
    </script>
    </body>
    </html>
    """, height=150)

# ─── MAIN CARD ────────────────────────────────────────────────────────────────
with st.container(key="main_card"):
    col_in, col_out = st.columns(2, gap="medium")

    with col_in:
        with st.container(key="input_panel"):
            st.markdown(
                '<div class="nl-panel-label" id="nl-panel-in">♫ English Speech</div>',
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
                '<div class="nl-panel-label" id="nl-panel-out">♫ Balochi Speech</div>',
                unsafe_allow_html=True
            )

            output_area = st.empty()

            if st.session_state.output_audio:
                output_area.audio(st.session_state.output_audio, format="audio/wav")
            else:
                output_area.markdown(
                    '<div class="nl-output-empty">♫</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns([3, 1])

    with btn_col1:
        translate_clicked = st.button(
            "Translate to Balochi Speech",
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

# ─── Pipeline ─────────────────────────────────────────────────────────────────
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

            st.session_state.output_audio = audio_bytes
            output_area.audio(audio_bytes, format="audio/wav")

        except Exception as e:
            st.error(f"Translation failed: {e}")

        finally:
            os.unlink(tmp_path)
