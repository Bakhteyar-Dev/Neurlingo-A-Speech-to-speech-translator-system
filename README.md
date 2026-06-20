# Neurolingo — English to Balochi Speech-to-Speech Translator

A Streamlit app that translates English speech into Balochi speech (Latin script) using three AI models.

## Pipeline

```
🎤 English Audio
      ↓
 Whisper (openai/whisper-small)
 English Speech → English Text
      ↓
 MarianMT (Bakhteyar/Balochi-Model)
 English Text → Balochi Text (Latin script)
      ↓
 Facebook MMS TTS (facebook/mms-tts-bcc-script_latin)
 Balochi Text → Balochi Audio
      ↓
🔊 Balochi Audio Output
```

## Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/neurolingo.git
cd neurolingo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> **Note:** Models are downloaded from Hugging Face automatically on first run and cached locally. This may take a few minutes the first time depending on your internet speed.

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

## Models Used

| Model | Hugging Face ID | Purpose |
|---|---|---|
| Whisper Small | `openai/whisper-small` | English speech → English text |
| MarianMT | `Bakhteyar/Balochi-Model` | English text → Balochi text |
| MMS TTS | ` Balochi text → Balochi speech |

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies
- GPU recommended for faster inference (CPU works too)
