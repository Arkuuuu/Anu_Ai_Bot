import os
import requests
import streamlit as st
import nltk
import pinecone
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

# ✅ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = "chatbot-memory"

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ ERROR: Missing API keys. Check your .env file!")

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Get existing indexes (Optimized)
existing_indexes = pc.list_indexes()

# ✅ Create index only if it doesn't exist
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"🔹 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists. Skipping creation.")

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
    vector_store = PineconeVectorStore.from_texts(text_chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME)

    st.session_state.processed_files.add(source_name)
    st.session_state.current_source_name = source_name

    return "✅ Data successfully processed and stored."

def query_chatbot(question, use_model_only=False):
    """Retrieve relevant information from stored embeddings and generate a response."""
    retries = 3  # ✅ Retry up to 3 times for API failures
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

        except Exception as e:
            print(f"❌ Groq API Error (Attempt {attempt+1}/{retries}):", str(e))
            st.error("⚠️ Error: API is not responding. Retrying...")
    
    return "⚠️ Error: Unable to process your request. Please try again later."

# ---------------------------- Streamlit UI ----------------------------

def display_chat_messages():
    """Display chat messages in a styled format."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(
                f"<div style='padding:10px; border-radius:8px; background-color:#f1f1f1; margin-bottom:5px;'>{message['content']}</div>",
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

        # ✅ Use form for better responsiveness
        with st.form("file_upload"):
            option = st.radio("Select knowledge base:", ("Model", "College Data", "Upload PDF", "Enter URL"), index=0)

            if option == "Upload PDF":
                pdf_file = st.file_uploader("Choose PDF file", type=["pdf"])

            elif option == "Enter URL":
                url = st.text_input("Enter website URL:")

            submitted = st.form_submit_button("Process")

        if submitted:
            with st.spinner("Processing..."):
                if option == "Upload PDF" and pdf_file:
                    temp_path = f"temp_{pdf_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    result = store_embeddings(temp_path, pdf_file.name)
                    st.success(result)

                elif option == "Enter URL" and url:
                    result = store_embeddings(url, url)
                    st.success(result)

    # Main chat interface
    st.subheader("Chat with Anu AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    display_chat_messages()  # ✅ Use function for structured chat display

    if prompt := st.chat_input("Ask a question... 🎤"):
        st.session_state.chat_history.append({"role": "user", "content": prompt, "avatar": "👤"})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("🔍 Analyzing..."):
            response = query_chatbot(prompt, use_model_only=(option == "Model"))

            st.session_state.chat_history.append({"role": "assistant", "content": response, "avatar": "🤖"})

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)

if __name__ == "__main__":
    main()
