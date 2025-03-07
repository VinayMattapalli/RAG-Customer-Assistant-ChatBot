import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
# Load your cleaned FAQ data
faq_data = pd.read_csv(r'/filepath.csv', encoding='ISO-8859-1')


  # Update with your file name

# Check for required columns
assert 'Question' in faq_data.columns and 'Answer' in faq_data.columns, "The file must contain 'Question' and 'Answer' columns"

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the questions
print("Generating embeddings...")
embeddings = np.vstack(faq_data['Question'].apply(lambda x: model.encode(x)).values)

# Create FAISS index
print("Creating FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index and FAQ data for reuse
faiss.write_index(index, 'faq_index.faiss')
faq_data.to_csv('faq_data.csv', index=False)
print("Embeddings and FAISS index saved successfully!")
# Save the FAQ data for later use
print("Saving FAQ data...")
with open('faq_data.pkl', 'wb') as f:
    pickle.dump(faq_data, f)
print("FAQ data saved as 'faq_data.pkl'!")

