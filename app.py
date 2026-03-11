import streamlit as st
from rag_chatbot import ask_question

st.title("RAG Chatbot")

query = st.text_input("Ask a question:")

if st.button("Submit"):

    if query:
        answer = ask_question(query)

        st.write("### Answer")
        st.write(answer)