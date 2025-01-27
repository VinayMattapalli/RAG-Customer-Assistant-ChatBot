import streamlit as st
from langchain.chains import RetrievalQA
import pandas as pd
import faiss
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# Load FAQ data and FAISS index
faq_data = pd.read_csv('faq_data.csv')
index = faiss.read_index('faq_index.faiss')

# Create FAISS vector store
vector_store = FAISS(
    embeddings=None,  # FAISS index already has embeddings
    index=index,
    texts=faq_data['Answer'].tolist()
)

# Initialize GPT model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Build RAG pipeline
retriever = vector_store.as_retriever()
rag_pipeline = RetrievalQA(llm=llm, retriever=retriever)

# Streamlit interface
st.title("RAG Chatbot")
st.write("Ask questions about the FAQs, and I'll provide the answers!")

query = st.text_input("Enter your question:")
if query:
    response = rag_pipeline.run(query)
    st.write("Answer:", response)
