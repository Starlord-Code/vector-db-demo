from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_api_key)

# Initialize Chroma DB

chroma_client = chromadb.PersistentClient(path="db")

collection = chroma_client.get_or_create_collection(name="demo_collection")

def embed_text(text):
    response = client.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def add_documents_to_vector_db(texts):
    for i, text in enumerate(texts):
        embedding = embed_text(text)
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(i)]
        )

def query_vector_db(query_text, n_results=3):
    embedding = embed_text(query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results
