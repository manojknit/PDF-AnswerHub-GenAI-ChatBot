import os
import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Path to save/load FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# Initialize Streamlit app
st.header("My first Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Check if FAISS index exists
vector_store = None
if os.path.exists(FAISS_INDEX_PATH):
    # Load the existing FAISS index
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    st.write("Loaded existing FAISS index.")

# Process uploaded PDF
if file:
    text = extract_text_from_pdf(file)
    splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    st.write(chunks)
    st.write(f"Total chunks created: {len(chunks)}")

    # Create new FAISS index if not already loaded
    if vector_store is None:
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        st.write("Created and saved new FAISS index with uploaded PDF.")

# Allow question input if vector store is available
if vector_store is not None:
    question = st.text_input("Ask a question")

    # Perform similarity search when user asks a question
    if question:
        question_embedding = embeddings.embed_query(question)
        match = vector_store.similarity_search_by_vector(question_embedding)

        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000, model="gpt-3.5-turbo")
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        answer = qa_chain.run(input_documents=match, question=question)
        st.write(answer)
else:
    st.write("Please upload a PDF to create or load the FAISS index.")
