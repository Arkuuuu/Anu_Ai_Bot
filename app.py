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
import pandas as pd

# ✅ Streamlit page config
st.set_page_config(page_title="Anu AI", page_icon="🧠")

# ✅ Load environment variables (using st.secrets for production)
if os.path.exists('.env'):
    load_dotenv()
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "chatbot-memory"
PINECONE_ENVIRONMENT = "us-east-1"  # Ensure correct region

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your secrets or .env file!")

# ✅ Initialize Pinecone using the new API.
# For pinecone==3.0.0 (or above), import Client from pinecone.core.client.
from pinecone.core.client import Client, ServerlessSpec
pc = Client(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

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
    # Retrieve the list of indexes using the new API.
    indexes = pc.list_indexes().names()
    st.write("Pinecone indexes available:", indexes)
    if PINECONE_INDEX_NAME not in indexes:
        raise ValueError(f"❌ ERROR: Pinecone index '{PINECONE_INDEX_NAME}' does not exist. Please create it first!")
    # Get the index instance using the new API.
    pinecone_index = pc.index(PINECONE_INDEX_NAME)
    # Instantiate the vector store using the retrieved index.
    return PineconeVectorStore(pinecone_index, embedding=embeddings, text_key="text")

docsearch = load_vector_store()

def is_valid_url(url):
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def extract_text_from_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([para.get_text() for para in paragraphs]).strip()

def load_pdf(pdf_path):
    documents = PyPDFLoader(pdf_path).load()
    if not documents:
        return "❌ Error: No readable text found in the PDF."
    return documents

def store_embeddings(input_path, source_name):
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if source_name in st.session_state.processed_files:
        return "✅ This document is already processed."

    text_data = ""
    if input_path.startswith("http"):
        if not is_valid_url(input_path):
            return "❌ Error: URL is not accessible."
        text_data = extract_text_from_webpage(input_path)
    else:
        documents = load_pdf(input_path)
        text_data = "\n".join([doc.page_content for doc in documents])

    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(text_data)
    st.write("Number of text chunks:", len(text_chunks))

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
    retries = 3
    delay = 2  
    for attempt in range(retries):
        try:
            if use_model_only:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an advanced AI assistant."},
                        {"role": "user", "content": question}
                    ],
                    model="llama-3.3-70b-versatile",
                    stream=False,
                )
                return response.choices[0].message.content if response.choices else "⚠️ No response from AI."

            relevant_docs = docsearch.max_marginal_relevance_search(
                question, k=10, fetch_k=20, lambda_mult=0.5
            )
            if not relevant_docs:
                return "❌ No relevant information found."

            retrieved_text = "\n".join(set(doc.page_content.strip() for doc in relevant_docs))
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an advanced AI assistant."},
                    {"role": "user", "content": f"Relevant Information:\n\n{retrieved_text}\n\nUser's question: {question}"}
                ],
                model="llama-3.3-70b-versatile",
                stream=False,
            )
            return response.choices[0].message.content if response.choices else "⚠️ No response from AI."
        except Exception as e:
            st.error(f"Error encountered during query attempt {attempt + 1}: {e}")
            time.sleep(delay)
            delay *= 2  
            if attempt == retries - 1:
                return "⚠️ Sorry, I couldn't process your request. Please try again later."

def display_chat_messages():
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
        selected_option = st.radio(
            "Select knowledge base:", 
            ("Model", "College Data", "Upload PDF", "Enter URL"),
            index=("Model", "College Data", "Upload PDF", "Enter URL").index(st.session_state.selected_option),
            key="selected_option"
        )

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

        # ✅ Auto-processing on file upload
        pdf_file = st.file_uploader("Choose file", type=["pdf", "txt", "csv"]) if selected_option == "Upload PDF" else None
        if pdf_file:
            temp_path = f"temp_{pdf_file.name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            with st.spinner("Processing..."):
                st.success(store_embeddings(temp_path, pdf_file.name))
                st.session_state.current_source_name = pdf_file.name  # Update source dynamically

        url = st.text_input("Enter website URL:") if selected_option == "Enter URL" else ""
        if url:
            with st.spinner("Processing URL..."):
                st.success(store_embeddings(url, url))
                st.session_state.current_source_name = url  # Update source dynamically

    st.subheader("Chat with Anu AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    display_chat_messages()

    if prompt := st.chat_input("Ask a question... 🎤"):
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})
        with st.spinner("🔍 Analyzing..."):
            response = query_chatbot(prompt, use_model_only=(selected_option == "Model"))
            st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})

    display_chat_messages()

if __name__ == "__main__":
    main()
