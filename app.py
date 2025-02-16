#=================
# Import Libraries
#=================

import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.embeddings import HuggingFaceEmbeddings
from io import BytesIO
from uuid import uuid4
import transformers

#=================
# Fetch API Key from Streamlit Secrets
#=================

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_api_key
except KeyError:
    st.error("🚨 API Key Not Found! Please add 'GROQ_API_KEY' in Streamlit Secrets.")
    st.stop()

#=================
# UI Setup
#=================

st.title("Anu AI bot")

# Define the path for the logo in the root directory
image_path = "logo-new.png"

# Check if the image exists in the deployment environment
if os.path.exists(image_path):
   st.sidebar.image(image_path, caption="", use_container_width=True)
else:
    # Use a fallback online image if the logo is missing
    fallback_url = "https://static.vecteezy.com/system/resources/previews/010/794/341/non_2x/purple-artificial-intelligence-technology-circuit-file-free-png.png"
    st.sidebar.image(fallback_url, caption="", use_container_width=True)

# File Upload
file_format = st.sidebar.selectbox("Select File Format", ["CSV", "PDF", "TXT"])
uploaded_files = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "pdf"], accept_multiple_files=True)

#=================
# Helper Functions
#=================

def validate_format(file_format, uploaded_files):
    """Validates file format against uploaded files."""
    return all(str(file_format).lower() in str(file.type).lower() for file in uploaded_files)

def history_func(answer, q):
    """Maintains chat history in Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = ''

    st.session_state.history = f'Q: {q} \nA: {answer}\n{"-" * 100}\n{st.session_state.history}'
    st.text_area(label='Chat History', value=st.session_state.history, height=400)

@st.cache_resource
def load_embeddings():
    """Loads and caches the Hugging Face Embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

@st.cache_resource
def create_groq_llm():
    """Creates a Groq-based LLM instance."""
    return ChatGroq(temperature=0)

#=================
# File Processing Functions
#=================

def csv_analysis(uploaded_file):
    """Processes CSV files & allows AI-powered Q&A."""
    df = pd.read_csv(uploaded_file)
    st.subheader("CSV File Preview")
    st.write(df.head())

    user_query = st.text_input('Enter your query')
    agent = create_csv_agent(create_groq_llm(), uploaded_file, verbose=True, max_iterations=100)

    if st.button("Answer My Question"):
        response = agent.run(user_query)
        st.text_area('LLM Answer:', value=response, height=400)
        history_func(response, user_query)

def process_pdf(uploaded_files):
    """Processes PDFs and allows AI-powered Q&A."""
    raw_text = ''
    
    # Extract text from PDFs
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            raw_text += text if text else ''

    # Chunk text for better retrieval
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    # Load cached embeddings and create FAISS index
    embeddings = load_embeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    # Question input
    question = st.text_input("Enter your question")

    if st.button("Answer My Question"):
        docs = docsearch.similarity_search(question)
        chain = load_qa_chain(create_groq_llm(), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        st.text_area('LLM Answer:', value=answer, height=400)
        history_func(answer, question)

def compare_pdf_analysis(uploaded_files):
    """Compares multiple PDFs and allows AI-powered Q&A."""
    tools = []
    llm = create_groq_llm()

    for file in uploaded_files:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)

        embeddings = load_embeddings()
        retriever = FAISS.from_documents(docs, embeddings).as_retriever()

        tool_name = file.name.replace('.pdf', '').replace(' ', '_')[:64]
        tools.append(Tool(name=tool_name, description=f"Q&A for {tool_name}", func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever)))

    agent = initialize_agent(agent=AgentType.OPENAI_FUNCTIONS, tools=tools, llm=llm, verbose=True)
    question = st.text_input("Enter your question")

    if st.button("Answer My Question"):
        response = agent.run(question)
        st.text_area('LLM Answer:', value=response, height=400)
        history_func(response, question)

def text_analysis(uploaded_files):
    """Processes text files and allows AI-powered Q&A."""
    raw_text = ''.join(file.read().decode("utf-8") for file in uploaded_files)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    embeddings = load_embeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    question = st.text_input("Enter your question")

    if st.button("Answer My Question"):
        docs = docsearch.similarity_search(question)
        chain = load_qa_chain(create_groq_llm(), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        st.text_area('LLM Answer:', value=answer, height=400)
        history_func(answer, question)

#=================
# File Analysis Handling
#=================

if uploaded_files:
    if validate_format(file_format, uploaded_files):
        if file_format == "CSV":
            csv_analysis(uploaded_files[0])
        elif file_format == "PDF":
            if len(uploaded_files) > 1:
                analysis_choice = st.selectbox("Select PDF Analysis Type", ["Compare", "Merge"])
                if analysis_choice == "Compare":
                    compare_pdf_analysis(uploaded_files)
                else:
                    process_pdf(uploaded_files)
            else:
                process_pdf(uploaded_files)
        else:  # Text file processing
            text_analysis(uploaded_files)
    else:
        st.error("Uploaded files do not match the selected format.")
