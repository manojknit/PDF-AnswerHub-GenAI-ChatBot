import PyPDF2
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "sk-proj-edummygA"

#Upload PDF files
st.header("My first Chatbot")
with st.sidebar:
  st.title("Your Documents")
  file = st.file_uploader(" Upload a PDf file and start asking questions",
  type="pdf")

# extract text from pdf
def extract_text_from_pdf(file):
  # Create a PDF reader object
  reader = PyPDF2.PdfReader(file)
  
  # Initialize a variable to store the extracted text
  text = ""
  
  # Iterate through all the pages and extract text
  for page_num in range(len(reader.pages)):
      page = reader.pages[page_num]
      text += page.extract_text()
  
  return text

if file:
  text = extract_text_from_pdf(file)
  # st.write(text)
  # Break it into chunks
  splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  # Break the text into chunks
  chunks = splitter.split_text(text)
  st.write(chunks)
  st.write(len(chunks))

  # generating embeddings - openAI
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

  # creating vector store - facebook faiss
  vector_store = FAISS.from_texts(chunks, embeddings) # stores chunks and embeddings

  # get user question
  question = st.text_input("Ask a question")

  # perform similarity search
  if question:
    # Create embedding for the question
    question_embedding = embeddings.embed_query(question)
    # Perform similarity search using the embedded query
    match = vector_store.similarity_search_by_vector(question_embedding)
    # st.write(match)

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=1000, model="gpt-3.5-turbo")
    # output results
    # chain -> take a question, get a relevent document, pass it to LLM, generate output
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    answer = qa_chain.run(input_documents=match, question=question)
    st.write(answer)

  