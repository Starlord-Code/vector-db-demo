# Install required packages:
# pip install chromadb sentence-transformers

import chromadb
from sentence_transformers import SentenceTransformer

# Step 1: Initialize DB and embedding model
client = chromadb.Client()
collection = client.create_collection("my_collection")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Documents to embed
docs = [
    "How to learn Python",
    "Best books for data science",
    "Understanding vector databases",
    "Top destinations to visit in Europe",
    "Python libraries for machine learning"
]

# Step 3: Generate embeddings and add to DB
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"id{i}" for i in range(len(docs))]
)

# Step 4: Query
query = "AI tools to study"
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=2)
print("🔍 Top results:")
for result in results['documents'][0]:
    print("-", result)
