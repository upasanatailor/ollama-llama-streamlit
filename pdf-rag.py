
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

# Configure logging
logging.basicConfig(level=logging.INFO)

#constants

doc_paths = "./data/Nursery-Rhymes-Book.pdf"
model = "gemma3"
EMBEDDING_MODEL="nomic-embed-text:v1.5"
VECTOR_STORE_NAME="nursery_rhymes"

def ingest_pdf(doc_paths):
    # Load PDF document
    if doc_paths:
        loader = PyPDFLoader(doc_paths)
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

def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
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
    data = ingest_pdf(doc_paths)
    chunks = split_documents(data)
    vector_db = create_vector_db(chunks)
    llm = ChatOllama(model=model)
    retriever = create_retriever(vector_db,llm)
    chain = create_chain(retriever,llm)

    #example question
    question = "What are the names of the characters in the nursery rhymes book?"

    #get the response
    res = chain.invoke(input={"question": question})
    print(res)


if __name__ == "__main__":
    main()









#res = chain.invoke(input={"question": "tell me a short poem about a cat and a dog"})
#print(res)
#res = chain.invoke(input={"question": "Does the poem 'Twinkle, Twinkle, Little Star' mention anything about the moon?"})
#print(res)