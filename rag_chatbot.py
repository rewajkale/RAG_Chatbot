from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="your_api_key",
    temperature=0
)

def ask_question(query):

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content