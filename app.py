import os
import requests
import streamlit as st
import nltk
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatGroq
from io import BytesIO
from uuid import uuid4
import transformers

# ✅ Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing Groq API key. Check your Streamlit Secrets!")

# ✅ Initialize Groq client
llm = ChatGroq(model="llama-3.3-70b-specdec", api_key=GROQ_API_KEY)

# ✅ Ensure nltk dependency
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# ---------------------------- Helper Functions ----------------------------

def is_valid_url(url):
    """Check if the URL is valid and accessible."""
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

def load_pdf(pdf_path):
    """Load and extract text from a PDF."""
    return PyPDFLoader(pdf_path).load()

@st.cache_resource
def load_embeddings():
    """Loads and caches Hugging Face Embeddings model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def store_embeddings(input_text, source_name):
    """Store embeddings in FAISS and avoid redundant processing."""
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_text(input_text)
    
    if not text_chunks:
        return "❌ Error: No text found in document."

    # ✅ Initialize embedding model
    embeddings = load_embeddings()

    # ✅ Store embeddings in FAISS
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    return vectorstore

def process_input(input_source, source_name):
    """Process files or URLs and store embeddings."""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if source_name in st.session_state.processed_files:
        return "✅ This document has already been processed."

    if input_source.startswith("http"):
        if not is_valid_url(input_source):
            return "❌ Error: URL is not accessible."
        extracted_text = extract_text_from_webpage(input_source)
    else:
        documents = load_pdf(input_source)
        extracted_text = "\n".join([doc.page_content for doc in documents])

    vectorstore = store_embeddings(extracted_text, source_name)
    
    st.session_state.processed_files.add(source_name)
    st.session_state.vectorstore = vectorstore
    st.session_state.current_source_name = source_name

    return "✅ Data successfully processed and stored."

def query_chatbot(question):
    """Retrieve relevant information from FAISS and generate a response."""
    if "vectorstore" not in st.session_state:
        return "❌ No knowledge base found. Upload a document first."

    relevant_docs = st.session_state.vectorstore.similarity_search(question, k=10)

    if not relevant_docs:
        return "❌ No relevant information found."

    retrieved_text = "\n".join([doc.page_content for doc in relevant_docs])

    chat_completion = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Relevant Information:\n\n{retrieved_text}\n\nUser's question: {question}"}
        ],
        stream=False,
    )

    return chat_completion.choices[0].message.content

# ---------------------------- Streamlit UI ----------------------------

def main():
    st.set_page_config(page_title="Anu AI Bot", page_icon="🤖")
    st.title("🤖 Anu AI Bot - Now More Powerful!")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.divider()

        if "current_source_name" not in st.session_state:
            st.session_state.current_source_name = "No Data Loaded"

        st.caption(f"Current Knowledge Source: {st.session_state.current_source_name}")

        option = st.radio(
            "Select knowledge base:",
            ("Model Only", "Upload PDF", "Enter URL"),
            index=0
        )

        if option == "Upload PDF":
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])
            if pdf_file:
                temp_path = f"temp_{pdf_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                with st.spinner("Processing PDF..."):
                    result = process_input(temp_path, pdf_file.name)
                    st.success(result)

        elif option == "Enter URL":
            url = st.text_input("Enter website URL:")
            if st.button("Process URL") and url:
                with st.spinner("Analyzing website content..."):
                    result = process_input(url, url)
                    st.success(result)

    # Main chat interface
    st.subheader("Chat with Anu AI Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "avatar": "👤"
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("🔍 Analyzing..."):
            response = query_chatbot(prompt)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "avatar": "🤖"
            })

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)

if __name__ == "__main__":
    main()
