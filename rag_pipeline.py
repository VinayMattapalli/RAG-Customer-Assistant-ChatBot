import pandas as pd
import faiss
from langchain.chains import RetrievalQA
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

# Test the pipeline
if __name__ == "__main__":
    query = input("Ask your question: ")
    response = rag_pipeline.run(query)
    print("Answer:", response)
