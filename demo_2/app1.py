import torch
from sentence_transformers import SentenceTransformer
import streamlit as st
import chromadb



# --- STEP 1: Initialize the model and vector DB ---
st.title("🔎 Vector Database Demo using ChromaDB")

# Ensure that you're not using a meta device. Instead, try to load the model correctly
try:
    # Initialize the model, but don't move it to a device just yet.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Move the model to the correct device after initialization
    #model.to_empty("cpu")  # This is the key fix: move from meta device to actual device
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

# ChromaDB client and collection
try:
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="demo_collection")
except Exception as e:
    st.error(f"Error initializing ChromaDB client: {e}")
    st.stop()

# Store input documents
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

st.subheader("📄 Add Documents")
docs = st.text_area("Enter documents (one per line):", height=200)

if st.button("➕ Add to Vector DB"):
    if docs.strip():
        doc_list = [line.strip() for line in docs.strip().split("\n") if line.strip()]
        embeddings = model.encode(doc_list).tolist()
        ids = [f"id{st.session_state.doc_count + i}" for i in range(len(doc_list))]
        
        try:
            collection.add(documents=doc_list, embeddings=embeddings, ids=ids)
            st.session_state.doc_count += len(doc_list)
            st.success(f"{len(doc_list)} documents added to vector database.")
        except Exception as e:
            st.error(f"Error adding documents to ChromaDB: {e}")
    else:
        st.warning("Please enter some documents.")

# --- STEP 2: Querying the vector DB ---
st.subheader("🔍 Search Similar Documents")
query = st.text_input("Enter your query:")

if st.button("🔎 Search"):
    if query:
        query_embedding = model.encode([query]).tolist()
        try:
            results = collection.query(query_embeddings=query_embedding, n_results=3)
            st.write("### 🔗 Top Matches:")
            for idx, doc in enumerate(results['documents'][0]):
                st.markdown(f"**{idx+1}.** {doc}")
        except Exception as e:
            st.error(f"Error querying ChromaDB: {e}")
    else:
        st.warning("Please enter a query to search.")
