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
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an advanced AI assistant."},
            {"role": "user", "content": question}
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

        option = st.radio("Select knowledge base:", ("Model", "Upload PDF", "Enter URL"), index=0)

        if option == "Upload PDF":
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])
            if pdf_file:
                with st.spinner("Processing PDF..."):
                    text = PyPDFLoader(pdf_file).load()
                    processed_text = "\n".join([doc.page_content for doc in text])
                    st.success("✅ PDF uploaded successfully!")

        elif option == "Enter URL":
            url = st.text_input("Enter website URL:")
            if st.button("Process URL") and url:
                with st.spinner("Analyzing website content..."):
                    if is_valid_url(url):
                        extracted_text = extract_text_from_webpage(url)
                        st.success("✅ Website processed successfully!")

    # ✅ HTML5 Voice Recorder
    st.write("🎙 **Record Your Voice and Ask a Question**")

    audio_html = """
    <script>
    var my_recorder;
    var audio_data;
    
    function startRecording() {
        my_recorder = new MediaRecorder(window.stream);
        my_recorder.start();
        audio_data = [];
        my_recorder.ondataavailable = function(event) {
            audio_data.push(event.data);
        };
    }

    function stopRecording() {
        my_recorder.stop();
        my_recorder.onstop = function() {
            var audioBlob = new Blob(audio_data, { type: "audio/wav" });
            var formData = new FormData();
            formData.append("file", audioBlob, "recorded_audio.wav");
            
            fetch("/upload_audio", { method: "POST", body: formData })
            .then(response => response.text())
            .then(data => { 
                Streamlit.setComponentValue(data);
            });
        };
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => { window.stream = stream; })
    .catch(error => { console.error("Error accessing microphone:", error); });
    </script>

    <button onclick="startRecording()">🎙 Start Recording</button>
    <button onclick="stopRecording()">🛑 Stop Recording</button>
    """

    st.markdown(audio_html, unsafe_allow_html=True)

    if st.session_state.get("audio_data"):
        st.success("🎧 Recording captured! Processing...")
        transcript = transcribe_audio_groq(st.session_state.audio_data)
        st.session_state.chat_history.append({"role": "user", "content": transcript, "avatar": "👤"})

    if prompt := st.chat_input("Ask a question... 🎤"):
        response = query_chatbot(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
        st.audio(synthesize_speech_aws(response), format="audio/mp3")

    for message in st.session_state.chat_history:
        st.markdown(f"**{message['role']}**: {message['content']}")

if __name__ == "__main__":
    main()
