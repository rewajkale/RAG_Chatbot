from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

documents = []

data_path = "data"

for file in os.listdir(data_path):

    path = os.path.join(data_path, file)

    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)

    elif file.endswith(".txt"):
        loader = TextLoader(path)

    documents.extend(loader.load())


# chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)


# embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# save locally
vectorstore.save_local("vector_db")

print("Vector DB created")