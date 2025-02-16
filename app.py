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
CHAT_HISTORY_FILE = "/mnt/data/chat_history.json"

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
        VoiceId="Joanna"  # Change the voice as needed
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

    # Sidebar Configuration
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
                    extracted_text = extract_text_from_webpage(url)
                    st.success("✅ Website processed successfully!")

    # ✅ Mic Button in Input Field (Starts/Stops Recording)
    st.write(
        """
        <script>
        let recording = false;
        let mediaRecorder;
        let audioChunks = [];

        function toggleRecording() {
            if (!recording) {
                startRecording();
            } else {
                stopRecording();
            }
        }

        function startRecording() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();
                    audioChunks = [];

                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                        const formData = new FormData();
                        formData.append("file", audioBlob, "recorded_audio.wav");

                        fetch("/upload_audio", { method: "POST", body: formData })
                            .then(response => response.text())
                            .then(data => {
                                Streamlit.setComponentValue(data);
                            });
                    };

                    recording = true;
                    document.getElementById("mic-button").innerText = "🎤 Stop Recording";
                });
        }

        function stopRecording() {
            mediaRecorder.stop();
            recording = false;
            document.getElementById("mic-button").innerText = "🎤 Start Recording";
        }
        </script>
        <button id="mic-button" onclick="toggleRecording()">🎤 Start Recording</button>
        """,
        unsafe_allow_html=True
    )

    # ✅ Check if audio has been recorded
    if st.session_state.get("audio_data"):
        st.success("🎧 Recording captured! Processing...")
        transcript = transcribe_audio_groq(st.session_state.audio_data)
        st.session_state.chat_history.append({"role": "user", "content": transcript, "avatar": "👤"})

    # ✅ Chat Input with Mic Button
    col1, col2 = st.columns([9, 1])
    with col1:
        prompt = st.text_input("Ask a question...")
    with col2:
        st.markdown('<button id="mic-button" onclick="toggleRecording()">🎤</button>', unsafe_allow_html=True)

    if prompt:
        response = query_chatbot(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
        st.audio(synthesize_speech_aws(response), format="audio/mp3")

    # ✅ Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
