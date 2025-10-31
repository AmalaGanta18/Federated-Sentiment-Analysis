import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import whisper
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

FILE_PATH = "live_audio.wav"

# Load Whisper model
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("medium", device="cpu")

model = load_whisper_model()

# Load improved Sentiment model (CardiffNLP)
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze(text):
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            output = model(**encoded_input)
        scores = softmax(output.logits.numpy()[0])
        labels = ['Negative', 'Neutral', 'Positive']
        return dict(zip(labels, map(float, scores)))

    return analyze

sentiment_model = load_sentiment_model()

# Supported languages
LANGUAGES = {
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Malayalam": "ml",
    "Odia": "or",
    "Assamese": "as",
    "English": "en"
}

# Save audio
def record_audio(duration=5, samplerate=44100):
    st.info(f"🎙 Recording for {duration} seconds... Speak Now!")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()

    language = st.session_state.language
    lang_code = LANGUAGES[language]
    folder_path = os.path.join("data", "datasets", "audio", lang_code)
    os.makedirs(folder_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_{timestamp}.wav"
    file_path = os.path.join(folder_path, filename)

    write(file_path, samplerate, audio)  # Save to dataset
    write(FILE_PATH, samplerate, audio)  # Save for Whisper

    st.success(f"✅ Recording saved as '{filename}' in folder '{lang_code}'")

# Transcribe and translate to English
def transcribe_audio():
    if not os.path.exists(FILE_PATH):
        st.error("❌ No audio file found! Please record first.")
        return ""

    st.info("🔍 Translating speech to fluent English using Whisper...")
    result = model.transcribe(FILE_PATH, task="translate")  # Your original working code
    return result["text"]

# Analyze sentiment
def analyze_sentiment(text):
    if not text.strip():
        return None
    return sentiment_model(text)

# Page 1 - Language selection
def language_selection_page():
    st.title("🌐 Multi-Language Speech-to-English Translator")
    st.write("Select your input language below (for your info only):")
    language = st.selectbox("Choose language", list(LANGUAGES.keys()))
    if st.button("➡ Next: Record Audio"):
        st.session_state.language = language
        st.session_state.page = "record"

# Page 2 - Record and translate
def record_and_translate_page():
    st.title("🎤 Record Your Speech")
    language = st.session_state.language
    st.write(f"Input Language Selected: {language} (Whisper will auto-detect and translate)")

    duration = st.slider("Select Recording Duration (seconds)", min_value=2, max_value=15, value=5)

    if st.button("🎙 Record Audio"):
        record_audio(duration=duration)

    if st.button("📝 Transcribe & Translate to English"):
        translated_text = transcribe_audio()
        st.session_state.translated_text = translated_text
        st.session_state.page = "result"

    if st.button("⬅ Back to Language Selection"):
        st.session_state.page = "language_select"

# Page 3 - Results
def result_page():
    st.title("📜 Translated Text & Sentiment Analysis")
    translated_text = st.session_state.get("translated_text", "")
    if translated_text:
        st.subheader("Translated English Text:")
        st.write(translated_text)

        sentiment = analyze_sentiment(translated_text)
        if sentiment:
            st.subheader("Sentiment Analysis:")
            for label, score in sentiment.items():
                st.write(f"**{label}**: {score:.2f}")
    else:
        st.warning("No translated text available.")

    if st.button("⬅ Back to Recording"):
        st.session_state.page = "record"
    if st.button("🏠 Home"):
        st.session_state.page = "language_select"

# Page router
if "page" not in st.session_state:
    st.session_state.page = "language_select"

if st.session_state.page == "language_select":
    language_selection_page()
elif st.session_state.page == "record":
    record_and_translate_page()
elif st.session_state.page == "result":
    result_page()
