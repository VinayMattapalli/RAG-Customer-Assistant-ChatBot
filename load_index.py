import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Load FAISS index
print("Loading FAISS index...")
faiss_index = faiss.read_index("faq_index.faiss")

# Load FAQ data
print("Loading FAQ data...")
with open("faq_data.pkl", "rb") as f:
    faq_data = pickle.load(f)

# Load Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Query function
def query_faq(question, top_k=3):
    query_embedding = model.encode([question])
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = []
    for i, index in enumerate(indices[0]):
        if index == -1:
            continue
        results.append((faq_data.iloc[index]['Question'], faq_data.iloc[index]['Answer'], distances[0][i]))
    return results

print("Ready to answer queries!")

# Continuous input loop for testing
while True:
    user_input = input("\nAsk a question (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    results = query_faq(user_input)
    if results:
        print("\nTop Results:")
        for question, answer, score in results:
            print(f"Q: {question}\nA: {answer}\nScore: {score:.4f}\n")
    else:
        print("No relevant answers found!")
