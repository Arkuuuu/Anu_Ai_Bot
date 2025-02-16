import os
import streamlit as st
import boto3  # AWS Polly for TTS
import requests  # Needed for Groq API requests
import tempfile
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import base64
from streamlit_mic_recorder import mic_recorder  # ✅ Web-Based Mic Input

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
def speech_to_text(audio_path):
    """Convert spoken audio to text using Groq Whisper API."""
    try:
        url = "https://api.groq.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        files = {"file": open(audio_path, "rb"), "model": (None, "whisper-large-v3")}

        response = requests.post(url, headers=headers, files=files)
        response_data = response.json()

        if "text" in response_data:
            return response_data["text"]
        else:
            return "❌ STT Error: No text received from Whisper API."

    except Exception as e:
        return f"❌ STT Request Failed: {str(e)}"

# ✅ Text-to-Speech (TTS) Function with Auto-Play
def text_to_speech(text):
    """Convert text to speech using AWS Polly and auto-play the response."""
    try:
        response = polly_client.synthesize_speech(Text=text,
                                                  OutputFormat="mp3",
                                                  VoiceId="Joanna")
        audio_stream = response["AudioStream"].read()

        # Save and auto-play audio
        audio_file_path = "response.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(audio_stream)

        st.audio(audio_file_path, format="audio/mp3")

    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")

# ✅ AI Chatbot Function using Groq API & LLaMA-3.3-70b-versatile
def query_chatbot(question):
    """Retrieve response from Groq API or PDF Knowledge Base."""
    if "vectorstore" in st.session_state and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

        retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

        if retrieved_text.strip():
            question = f"Using retrieved context:\n\n{retrieved_text}\n\nQuestion: {question}"

    url = "https://api.groq.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
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

# ✅ Process PDF and Store in FAISS
@st.cache_resource
def process_pdf(pdf_file):
    """Extracts text from a PDF file and stores embeddings in FAISS."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())  # Save uploaded PDF as temp file
            temp_pdf_path = temp_file.name

        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = FAISS.from_documents(text_chunks, embeddings)

        return vectorstore

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="Anu AI Bot", page_icon="🤖")
    st.title("🤖 Anu AI Bot - Now with Voice & PDFs!")

    # Sidebar - File Upload
    with st.sidebar:
        st.header("📂 Upload a PDF")
        pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
        
        if pdf_file:
            with st.spinner("Processing PDF..."):
                vectorstore = process_pdf(pdf_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ PDF processed successfully! You can now ask questions.")

    # Chat UI
    st.subheader("Chat with Anu AI Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    # ✅ Styled Chat Input with Mic Button 🎤
    mic_clicked = False
    st.markdown(
        """
        <style>
            .stChatInput { display: flex; align-items: center; }
            .stChatInput button { background: none; border: none; font-size: 20px; cursor: pointer; }
            .stChatInput input { flex-grow: 1; padding-left: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        mic_clicked = st.button("🎤", key="mic_button")  # Mic emoji button

    with col2:
        prompt = st.text_input("Type your message...")

    # ✅ Auto Speech-to-Text Processing After Recording
    if mic_clicked:
        st.info("🎙️ Recording... Speak now!")
        audio_data = mic_recorder()

        if audio_data:
            st.success("✅ Recording complete! Converting speech to text...")
            audio_path = "temp_voice_input.wav"
            audio_bytes = base64.b64decode(audio_data.split(",")[1])

            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            prompt = speech_to_text(audio_path)

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

            text_to_speech(response)  # ✅ Auto Play Response

if __name__ == "__main__":
    main()
