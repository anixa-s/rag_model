from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
import os
import shutil
import argparse

# Setting up environment 
load_dotenv()
OLLAMA_API_KEY = os.getenv("")
llm = Ollama(model="llama3", temperature = 0.0)
  
# Load an embedding model, define data path, and database path
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DATA_PATH = "env\data"
CHROMA_PATH = "chroma_data"

# Prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Functions 
def main():
    docs = load_documents()
    chunks = generate_embeddings(docs)
    save_to_chroma(chunks)
    query_text = input("Enter your query: ")
    query_data(query_text)



def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()   
    print(f"The length of the document is {len(documents)}")
    return documents

def generate_embeddings(documents: list[Document]):
    # Splitting the PDFs into chunks of text 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents) # With metadata 
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # To remove metadata to use for embeddings 
    counter = 0 
    for chunk in chunks:
        text_chunks = chunk.page_content
        metadata_chunks = chunk.metadata
        if counter == 1:
            print(text_chunks)
            print(metadata_chunks)
            print(metadata_chunks.get("source", None))
            print('++++++++++++++++++++++++++++++++++++++++++++++') 
        counter += 1
    return chunks 

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH) 

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedding_model, persist_directory=CHROMA_PATH
    )
    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_data(query_text):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if not results or results[0][1] < 0.355:
        print(results[0][1])
        print(f"Unable to find matching results.")
        return

    filtered_results = [doc.page_content for doc, score in results if score >= 0.5]
    if not filtered_results:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join(filtered_results)
    
    # Format prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    response_text = llm.invoke(prompt)
    print("*********************************************************")
    print(response_text)

# Output  
main()