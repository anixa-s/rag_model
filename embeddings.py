# Packages I had to import and install to create the embeddings 
import ollama 
import chromadb 
import numpy 
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.vectorstores import Chroma

DATA_PATH = "C:\Anika Singh\Glenforest Secondary School\Grade 10 MYP5\Personal Project\Documents\.venv\data"
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()
print(f"The length of the document is {len(documents)}")

# Splitting the PDF(s) into chunks of text 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Change chunk size 
    chunk_overlap=100,
    length_function=len,
    add_start_index=False,
)
chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
text_chunks = [chunk.page_content for chunk in chunks]
print(text_chunks)

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient

# Print the final sentences list to verify
# print(f"Sentences List: {chunks}")
for element in text_chunks:
    embeddings = model.encode(element)
    print(element, embeddings)
    print("*******************************************")

""" 
Display the embedding for the first sentence
print(embeddings[0])
print()
print(embeddings[2])
"""

""" Another trial error for embeddings to see if it works 
documents = [chunks]
print(documents)

client = chromadb.Client()
collection = client.create_collection(name="docs")
"""

# For adding the chunks to the ChromaDB, use variable text_chunks (chunk content), and concatenate with chunk_number variable 
# (counter should go up, use enumerate?) and use embeddings (embedding content)
# Code from GPT
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,  # Embedding function used
    persist_directory="./chroma_data"
)
print(vectorstore)

# Create a collection for storing vectors
collection = vectorstore.create_collection(name="my_rag_collection")

collection.add(
    embeddings=embeddings,
)

# Convert Chroma vector store into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Example usage
query = "Your question here"
docs = retriever.get_relevant_documents(query)
print(docs)