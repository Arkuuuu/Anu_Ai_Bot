import os
import requests
import json
import streamlit as st
import nltk
import boto3
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from groq import Groq
from pinecone import Pinecone
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings  # ✅ Alternative for audio recording
import av  # Required for handling audio streams

# ✅ Streamlit page config
st.set_page_config(page_title="Anu AI", page_icon="🧠")

# ✅ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
PINECONE_INDEX_NAME = "chatbot-memory"
CHAT_HISTORY_FILE = "/mnt/data/chat_history.json"  # ✅ Custom storage path

if not PINECONE_API_KEY or not GROQ_API_KEY or not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your .env file!")

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ Initialize AWS Polly client
polly_client = boto3.client(
    "polly",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

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

@st.cache_resource  # ✅ Cache to prevent reloading Pinecone every time
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

docsearch = load_vector_store()

def is_valid_url(url):
    """Check if a URL is valid."""
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text_from_webpage(url):
    """Extract text content from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([para.get_text() for para in paragraphs]).strip()

def store_embeddings(input_text, source_name):
    """Process and store embeddings from text data."""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if source_name in st.session_state.processed_files:
        return "✅ This document is already processed. You can now ask queries!"

    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(input_text)

    if not text_chunks:
        return "❌ Error: No text found in document."

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Store embeddings in Pinecone
    PineconeVectorStore.from_texts(text_chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

    st.session_state.processed_files.add(source_name)
    st.session_state.current_source_name = source_name

    return "✅ Data successfully processed and stored."

def transcribe_audio_groq(audio_bytes):
    """Sends recorded audio to Groq API for transcription using Whisper v3 Large."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    response = requests.post(
        "https://api.groq.com/v1/audio/transcriptions",
        headers=headers,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "whisper-large-v3"}
    )

    if response.status_code == 200:
        return response.json().get("text", "⚠️ No transcription available")
    else:
        return f"⚠️ Error: {response.json()}"

def synthesize_speech_aws(text):
    """Convert text to speech using AWS Polly."""
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joanna"  # You can change the voice
    )
    return response["AudioStream"].read()

def query_chatbot(question):
    """Retrieve relevant information from stored embeddings and generate a response."""
    relevant_docs = docsearch.similarity_search(question, k=10)

    if not relevant_docs:
        return "❌ No relevant information found."

    retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant."},
            {"role": "user", "content": f"Relevant Information:\n\n{retrieved_text}\n\nUser's question: {question}"}
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
    )

    return chat_completion.choices[0].message.content

# ---------------------------- Streamlit UI ----------------------------

def main():
    st.title("🧠 Anu AI - Your Intelligent Assistant")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.divider()

        option = st.radio("Select knowledge base:", ("Model", "College Data", "Upload PDF", "Enter URL"), index=0)

        if option == "Upload PDF":
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])
            if pdf_file:
                with st.spinner("Processing PDF..."):
                    text = PyPDFLoader(pdf_file).load()
                    processed_text = "\n".join([doc.page_content for doc in text])
                    st.success(store_embeddings(processed_text, pdf_file.name))

        elif option == "Enter URL":
            url = st.text_input("Enter website URL:")
            if st.button("Process URL") and url:
                with st.spinner("Analyzing website content..."):
                    if is_valid_url(url):
                        st.success(store_embeddings(extract_text_from_webpage(url), url))

    # ✅ Voice Input
    webrtc_ctx = webrtc_streamer(key="speech-to-text", mode=WebRtcMode.SENDRECV)
    if webrtc_ctx.audio_receiver:
        st.success("🎧 Recording captured! Processing...")
        audio_bytes = webrtc_ctx.audio_receiver.get_frames(timeout=1.0)[0].to_ndarray().tobytes()
        transcript = transcribe_audio_groq(audio_bytes)
        st.session_state.chat_history.append({"role": "user", "content": transcript, "avatar": "👤"})

    if prompt := st.chat_input("Ask a question... 🎤"):
        response = query_chatbot(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
        st.audio(synthesize_speech_aws(response), format="audio/mp3")

if __name__ == "__main__":
    main()
