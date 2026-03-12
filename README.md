# RAG Document QA with LangChain + FAISS + Gemini

This project implements a simple **Retrieval-Augmented Generation (RAG)** pipeline using:

- LangChain
- FAISS vector database
- HuggingFace sentence-transformer embeddings
- Google Gemini (via langchain-google-genai)

It allows you to:

1. Load documents (.pdf and .txt)
2. Split them into chunks
3. Generate embeddings
4. Store them in a FAISS vector database
5. Ask questions about the documents using Gemini

---

# Project Structure
project/
│
├── data/ # Input documents
│ ├── example.pdf
│ └── notes.txt
│
├── ingest.py # Creates FAISS vector database
├── rag_chatbot.py # Query the vector database
|___app.py # streamlit app
│
├── vector_db/ # Generated FAISS database
│
├── requirements.txt
└── README.md