import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Load FAISS index and FAQ data
faiss_index = faiss.read_index("faq_index.faiss")
with open("faq_data.pkl", "rb") as f:
    faq_data = pickle.load(f)

# Load SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Streamlit App
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("RAG Chatbot")
st.sidebar.markdown("""
**Navigation**
- Home
- About
- Contact
""")
st.sidebar.info("This is a Retrieval-Augmented Generation (RAG) chatbot.")

# Main App
st.title("RAG Chatbot")
st.subheader("Ask a question to the FAQ knowledge base:")

# User Input
user_question = st.text_input("Your Question", placeholder="Type your question here...")
if user_question:
    st.markdown(f"**You asked:** {user_question}")

    # Generate embeddings and search
    question_embedding = model.encode([user_question])
    distances, indices = faiss_index.search(question_embedding, k=3)

    # Results Display
    results = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] >= 1.0:  # Adjust threshold as needed
            break
        faq_entry = faq_data.iloc[idx]
        results.append({
            "question": faq_entry["Question"],
            "answer": faq_entry["Answer"],
            "score": float(distances[0][i]),
        })

    # Render Results
    if results:
        st.markdown("### Top Results:")
        for result in results:
            with st.expander(f"Q: {result['question']}"):
                st.write(f"**A:** {result['answer']}")
                st.caption(f"Confidence Score: {result['score']:.2f}")
    else:
        st.warning("I'm sorry, I couldn't find a relevant answer. Can you try rephrasing?")
