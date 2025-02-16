import os
import requests
import json
import streamlit as st
import nltk
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from groq import Groq
from pinecone import Pinecone
import sounddevice as sd
import numpy as np
import wave

# ✅ Streamlit page config (Must be first Streamlit command!)
st.set_page_config(page_title="Anu AI", page_icon="🧠")

# ✅ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "chatbot-memory"
CHAT_HISTORY_FILE = "/mnt/data/chat_history.json"  # ✅ Custom storage path

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your .env file!")

# ✅ Initialize Pinecone client (No index creation!)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ---------------------------- Helper Functions ----------------------------

def load_chat_history():
    """Load chat history from a JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_chat_history(history):
    """Save chat history to a JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file)

# ✅ Load chat history on startup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

def record_audio(filename="voice_input.wav", duration=5, samplerate=44100):
    """Records audio from the microphone and saves it as a WAV file."""
    st.info("🎤 Listening... Speak now!")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    wave_file = wave.open(filename, "wb")
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(samplerate)
    wave_file.writeframes(audio_data.tobytes())
    wave_file.close()
    return filename

def transcribe_audio_groq(audio_file):
    """Sends audio to Groq API for transcription using Whisper v3 Large."""
    with open(audio_file, "rb") as f:
        audio_data = f.read()

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    response = requests.post(
        "https://api.groq.com/v1/audio/transcriptions",
        headers=headers,
        files={"file": ("audio.wav", audio_data, "audio/wav")},
        data={"model": "whisper-large-v3"}
    )

    if response.status_code == 200:
        return response.json().get("text", "⚠️ No transcription available")
    else:
        return f"⚠️ Error: {response.json()}"

# ---------------------------- Streamlit UI ----------------------------

def display_chat_messages():
    """Display chat messages with styled bubbles."""
    for message in st.session_state.chat_history:
        bg_color = "#DCF8C6" if message["role"] == "assistant" else "#E0E0E0"
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(
                f"<div style='padding:10px; border-radius:8px; background-color:{bg_color}; margin-bottom:5px;'>{message['content']}</div>",
                unsafe_allow_html=True
            )

def main():
    st.title("🧠 Anu AI - Your Intelligent Assistant")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.divider()

        st.caption(f"Chat History is saved at: `{CHAT_HISTORY_FILE}`")

        if st.sidebar.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            save_chat_history([])  # Clear JSON file too
            st.success("Chat history cleared!")

    st.subheader("Chat with Anu AI")

    # ✅ Display existing chat history
    display_chat_messages()

    # ✅ Voice Input Button
    if st.button("🎙 Speak to Anu AI"):
        audio_file = record_audio()
        transcript = transcribe_audio_groq(audio_file)
        
        if transcript:
            st.session_state.chat_history.append({"role": "user", "content": transcript, "avatar": "👤"})
            response = query_chatbot(transcript, use_model_only=False)
            st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
            save_chat_history(st.session_state.chat_history)

    # ✅ Text Input for Chat
    if prompt := st.chat_input("Ask a question... 🎤"):
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        response = query_chatbot(prompt, use_model_only=False)
        st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
        save_chat_history(st.session_state.chat_history)

    display_chat_messages()

if __name__ == "__main__":
    main()
