from flask import Flask, request, jsonify
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index and FAQ data
try:
    faiss_index = faiss.read_index("faq_index.faiss")
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit()

try:
    with open("faq_data.pkl", "rb") as f:
        faq_data = pickle.load(f)
    print("FAQ data loaded successfully.")
except Exception as e:
    print(f"Error loading FAQ data: {e}")
    exit()

# Load SentenceTransformer model
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    exit()

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")
    print(f"Received question: {question}")  # Debug log

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Generate embeddings for the question
    question_embedding = model.encode([question])

    # Search the FAISS index
    try:
        distances, indices = faiss_index.search(np.array(question_embedding, dtype=np.float32), k=3)
    except Exception as e:
        print(f"Error during FAISS index search: {e}")
        return jsonify({"error": "An error occurred during the search process"}), 500

    # Prepare results
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

    # Fallback response for unknown questions
    if not results:
        return jsonify({"message": "I'm sorry, I couldn't find a relevant answer. Can you try rephrasing?"}), 200

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
