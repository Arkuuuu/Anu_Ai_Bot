import os
import streamlit as st
import boto3  # AWS Polly for TTS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai  # Whisper API for STT
from io import BytesIO
from uuid import uuid4

# ✅ Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Whisper API
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not GROQ_API_KEY or not OPENAI_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your Streamlit Secrets!")

# ✅ Initialize AWS Polly
polly_client = boto3.client("polly",
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=AWS_REGION)

# ✅ Whisper API Client (for Speech-to-Text)
openai.api_key = OPENAI_API_KEY

# ✅ Speech-to-Text (STT) Function
def speech_to_text(audio_file):
    """Convert spoken audio to text using Whisper API."""
    try:
        with open(audio_file, "rb") as file:
            response = openai.Audio.transcribe("whisper-1", file)
            return response["text"]
    except Exception as e:
        return f"❌ STT Error: {str(e)}"

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

# ✅ Main Chatbot Function
def query_chatbot(question):
    """Retrieve relevant information from FAISS and generate a response."""
    if "vectorstore" not in st.session_state:
        return "❌ No knowledge base found. Upload a document first."

    relevant_docs = st.session_state.vectorstore.similarity_search(question, k=10)
    retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

    client = openai.ChatCompletion.create(
        model="llama-3.3-70b-specdec",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Relevant Information:\n\n{retrieved_text}\n\nUser's question: {question}"}
        ],
        stream=False,
    )

    return client.choices[0].message.content

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
