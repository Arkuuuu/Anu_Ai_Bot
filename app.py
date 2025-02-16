import os
import streamlit as st
import boto3  # AWS Polly for TTS
import requests  # Needed for Groq API requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from io import BytesIO
from uuid import uuid4

# ✅ Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your Streamlit Secrets!")

# ✅ Initialize AWS Polly for TTS
polly_client = boto3.client("polly",
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=AWS_REGION)

# ✅ Speech-to-Text (STT) Function using Groq Whisper API
def speech_to_text(audio_file):
    """Convert spoken audio to text using Groq Whisper API."""
    try:
        url = "https://api.groq.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        files = {
            "file": audio_file,
            "model": (None, "whisper-large-v3")
        }

        response = requests.post(url, headers=headers, files=files)
        response_data = response.json()

        if "text" in response_data:
            return response_data["text"]
        else:
            return f"❌ STT Error: {response_data}"

    except Exception as e:
        return f"❌ STT Request Failed: {str(e)}"

# ✅ Text-to-Speech (TTS) Function
def text_to_speech(text):
    """Convert text to speech using AWS Polly."""
    try:
        response = polly_client.synthesize_speech(Text=text,
                                                  OutputFormat="mp3",
                                                  VoiceId="Joanna")
        audio_stream = response["AudioStream"].read()

        # Save and play audio
        audio_file = "response.mp3"
        with open(audio_file, "wb") as f:
            f.write(audio_stream)
        st.audio(audio_file)

    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")

# ✅ AI Chatbot Function using Groq API & LLaMA-3.3-70b-versatile
def query_chatbot(question):
    """Send user query to Groq API and return response."""
    url = "https://api.groq.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"]
        else:
            return "❌ No response received from Groq API."

    except Exception as e:
        return f"❌ Groq API Request Failed: {str(e)}"

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="Anu AI Bot", page_icon="🤖")
    st.title("🤖 Anu AI Bot - Now with Voice!")

    # Chat UI
    st.subheader("Chat with Anu AI Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # ✅ Voice Input (STT)
    audio_file = st.file_uploader("🎙️ Upload a voice query (MP3/WAV)", type=["mp3", "wav"])
    if audio_file and st.button("Convert Speech to Text"):
        with st.spinner("Converting speech to text..."):
            transcribed_text = speech_to_text(audio_file)
            st.text_area("Transcribed Text:", value=transcribed_text, height=100)
            prompt = transcribed_text
    else:
        prompt = st.chat_input("Ask a question...")

    # ✅ Process Query & Generate Response
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("🔍 Analyzing..."):
            response = query_chatbot(prompt)

            st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)

            # ✅ Text-to-Speech Output
            st.button("🔊 Listen", on_click=text_to_speech, args=(response,))

if __name__ == "__main__":
    main()
