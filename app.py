import os
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

# ✅ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "chatbot-memory"

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your .env file!")

# ✅ Initialize Pinecone client (No index creation!)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Ensure nltk dependency
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# ✅ Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

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

@st.cache_resource  # ✅ Cache to prevent reloading Pinecone every time
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

docsearch = load_vector_store()

def store_embeddings(input_path, source_name):
    """Process and store embeddings only if not already stored."""
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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Efficiently store embeddings in Pinecone (Batch Processing)
    PineconeVectorStore.from_texts(text_chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

    st.session_state.processed_files.add(source_name)
    st.session_state.current_source_name = source_name

    return "✅ Data successfully processed and stored."

def query_chatbot(question, use_model_only=False):
    """Retrieve relevant information from stored embeddings and generate a response."""
    retries = 3  
    for attempt in range(retries):
        try:
            if use_model_only:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an advanced AI assistant."},
                        {"role": "user", "content": question}
                    ],
                    model="llama-3.3-70b-versatile",
                    stream=False,
                )
                return chat_completion.choices[0].message.content

            # ✅ Use cached docsearch for faster querying
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

        except Exception as e:
            error_message = str(e)
            print(f"❌ Groq API Error (Attempt {attempt+1}/{retries}):", error_message)
            st.error(f"⚠️ API Error: {error_message}")

    return "⚠️ Sorry, I couldn't process your request. Please try again later."

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
    st.set_page_config(page_title="Anu AI", page_icon="🧠")
    st.title("🧠 Anu AI - Your Intelligent Assistant")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.divider()

        if "current_source_name" not in st.session_state:
            st.session_state.current_source_name = "collegedata.pdf"

        st.caption(f"Current Knowledge Source: {st.session_state.current_source_name}")

        with st.form("file_upload"):
            option = st.radio("Select knowledge base:", ("Model", "College Data", "Upload PDF", "Enter URL"), index=0)
            
            pdf_file = st.file_uploader("Choose PDF file", type=["pdf"]) if option == "Upload PDF" else None
            url = st.text_input("Enter website URL:") if option == "Enter URL" else ""

            submitted = st.form_submit_button("Process")

        if submitted:
            with st.spinner("Processing..."):
                if pdf_file:
                    temp_path = f"temp_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    st.success(store_embeddings(temp_path, pdf_file.name))
                elif url:
                    st.success(store_embeddings(url, url))

    # Chat Interface
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
            response = query_chatbot(prompt, use_model_only=(option == "Model"))
            st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})
    
    display_chat_messages()

if __name__ == "__main__":
    main()
