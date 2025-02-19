import os
import time
import requests
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
import pandas as pd
from audio_recorder_streamlit import audio_recorder
from utils import speech_to_text, text_to_speech, get_answer, store_embeddings

# ✅ Streamlit page config
st.set_page_config(page_title="Anu AI", page_icon="🧠")

# ✅ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "chatbot-memory"

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your .env file!")

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure nltk dependency
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
# ---------------------------- Helper Functions ----------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

@st.cache_resource
def load_vector_store():
    return PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

docsearch = load_vector_store()

def is_valid_url(url):
    """Check if a URL is valid and accessible."""
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text_from_webpage(url):
    """Extract text from a webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([para.get_text() for para in paragraphs]).strip()

def load_pdf(pdf_path):
    """Load and extract text from a PDF file."""
    documents = PyPDFLoader(pdf_path).load()
    if not documents:
        return "❌ Error: No readable text found in the PDF."
    return documents

def store_embeddings(input_path, source_name):
    """Store text embeddings into Pinecone for retrieval."""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if source_name in st.session_state.processed_files:
        return "✅ This document is already processed. You can now ask queries!"

    text_data = ""
    if input_path.startswith("http"):
        if not is_valid_url(input_path):
            return "❌ Error: URL is not accessible."
        text_data = extract_text_from_webpage(input_path)
    else:
        documents = load_pdf(input_path)
        text_data = "\n".join([doc.page_content for doc in documents])

    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)

    if not text_chunks:
        return "❌ Error: No text found in document."

    batch_size = 100
    for i in range(0, len(text_chunks), batch_size):
        PineconeVectorStore.from_texts(
            text_chunks[i : i + batch_size], embedding=embeddings, index_name=PINECONE_INDEX_NAME
        )

    st.session_state.processed_files.add(source_name)
    st.session_state.current_source_name = source_name

    return "✅ Data successfully processed and stored."

def query_chatbot(question, use_model_only=False):
    """Retrieve relevant data from Pinecone and generate a chatbot response."""
    retries = 3
    delay = 2  
    for attempt in range(retries):
        try:
            if use_model_only:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "system", "content": "You are an advanced AI assistant."},
                              {"role": "user", "content": question}],
                    model="llama-3.3-70b-versatile",
                    stream=False,
                )
                return chat_completion.choices[0].message.content

            relevant_docs = docsearch.max_marginal_relevance_search(question, k=10, fetch_k=20, lambda_mult=0.5)
            if not relevant_docs:
                return "❌ No relevant information found."

            retrieved_text = "\n".join(set(doc.page_content.strip() for doc in relevant_docs))

            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": "You are an advanced AI assistant."},
                          {"role": "user", "content": f"Relevant Information:\n\n{retrieved_text}\n\nUser's question: {question}"}],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            time.sleep(delay)
            delay *= 2  
            if attempt == retries - 1:
                return "⚠️ Sorry, I couldn't process your request. Please try again later."


# ---------------------------- Streamlit UI ----------------------------

def display_chat_messages():
    """Display chat messages with proper styling."""
    for message in st.session_state.chat_history:
        bg_color = "#DCF8C6" if message["role"] == "assistant" else "#E0E0E0"
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(
                f"<div style='padding:10px; border-radius:8px; background-color:{bg_color}; margin-bottom:5px;'>{message['content']}</div>",
                unsafe_allow_html=True
            )

def main():
    st.title("🧠 Anu AI - Your Intelligent Assistant")

    with st.sidebar:
        st.header("⚙️ Configuration")

        if "selected_option" not in st.session_state:
            st.session_state.selected_option = "Model"
        if "current_source_name" not in st.session_state:
            st.session_state.current_source_name = "Model (No external knowledge)"

        # ✅ Radio selection with session tracking
        selected_option = st.radio("Select knowledge base:", 
                                   ("Model", "College Data", "Upload PDF", "Enter URL"),
                                   index=("Model", "College Data", "Upload PDF", "Enter URL").index(st.session_state.selected_option),
                                   key="selected_option")

        # ✅ Update knowledge source based on selection
        if selected_option == "Model":
            st.session_state.current_source_name = "Model (No external knowledge)"
        elif selected_option == "College Data":
            st.session_state.current_source_name = "collegedata.pdf"
        elif selected_option == "Upload PDF":
            st.session_state.current_source_name = "Uploaded PDF (Pending Processing)"
        elif selected_option == "Enter URL":
            st.session_state.current_source_name = "Website URL (Pending Processing)"

        st.markdown(f"**📄 Current Knowledge Source:** `{st.session_state.current_source_name}`")

        with st.form("file_upload"):
            pdf_file = st.file_uploader("Choose file", type=["pdf", "txt", "csv"]) if selected_option == "Upload PDF" else None
            url = st.text_input("Enter website URL:") if selected_option == "Enter URL" else ""
            submitted = st.form_submit_button("Process")

        if submitted:
            with st.spinner("Processing..."):
                if pdf_file:
                    temp_path = f"temp_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    st.success(store_embeddings(temp_path, pdf_file.name))
                    st.session_state.current_source_name = pdf_file.name  # ✅ Update source
                elif url:
                    st.success(store_embeddings(url, url))
                    st.session_state.current_source_name = url  # ✅ Update source dynamically

    # ✅ Handle voice input
    st.subheader("Chat with Anu AI")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    display_chat_messages()

    # 🎤 Voice Input Handling
    audio_bytes = audio_recorder()
    if audio_bytes:
        with st.spinner("Transcribing..."):
            temp_audio_path = "temp_audio.mp3"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)

            transcript = speech_to_text(temp_audio_path)
            if transcript:
                st.session_state.chat_history.append({"role": "user", "content": transcript, "avatar": "👤"})
                with st.chat_message("user"):
                    st.write(transcript)

                with st.spinner("Thinking 🤔..."):
                    final_response = get_answer(st.session_state.chat_history)

                with st.spinner("Generating audio response..."):
                    audio_file = text_to_speech(final_response)

                st.session_state.chat_history.append({"role": "assistant", "content": final_response, "avatar": "🤖"})
                st.audio(audio_file, format="audio/mp3")

                os.remove(temp_audio_path)
                os.remove(audio_file)

    display_chat_messages()

if __name__ == "__main__":
    main()
