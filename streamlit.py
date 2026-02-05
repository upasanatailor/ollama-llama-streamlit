import streamlit as st
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

#constants

DOC_PATH = "./data/Nursery-Rhymes-Book.pdf"
model = "gemma3"
EMBEDDING_MODEL="nomic-embed-text:v1.5"
VECTOR_STORE_NAME="nursery_rhymes"
PERSIST_DIRECTORY = "./chroma_db"

def ingest_pdf(DOC_PATH):
    # Load PDF document
    if DOC_PATH:
        loader = PyPDFLoader(DOC_PATH)
        data = loader.load()
        print("PDF document loaded successfully.")
    else:
        print("No document loaded.") 
    return data


# Split the document into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print("splitting done")
    return chunks
# print("Number of text chunks:", len(chunks))
# print(chunks[0])

#print("Vector store created with Ollama embeddings.")

@st.cache_resource
def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db


#retrieval example

def create_retriever(vector_db,llm):
    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="you are a an AI generator . you will answer the question in five different language: {question}",
)

    retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),llm,prompt=QUERY_PROMPT,

)
    return retriever

#setup model to use
def create_chain(retriever,llm):
    #RAG prompt template
    template = """Use the following pieces of context to answer the question at the end.{context}
Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm  | StrOutputParser())
    return chain


def main():
    st.title("Document Assistant")
    # User input
    user_input = st.text_input("Enter your question:", "")
    if user_input:
        with st.spinner("Processing..."):
            try:
                llm = ChatOllama(model=model)
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return  
                retriever = create_retriever(vector_db,llm)
                chain = create_chain(retriever,llm)     
                response = chain.invoke(input={"question": user_input})
                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")    
    else:
        st.info("Please enter a question to get started.") 

    



if __name__ == "__main__":
    main()









#res = chain.invoke(input={"question": "tell me a short poem about a cat and a dog"})
#print(res)
#res = chain.invoke(input={"question": "Does the poem 'Twinkle, Twinkle, Little Star' mention anything about the moon?"})
#print(res)