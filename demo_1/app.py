import streamlit as st
from helper import add_documents_to_vector_db, query_vector_db
import os

# Load and add data
if "initialized" not in st.session_state:
    with open("sample_data.txt") as f:
        data = f.readlines()
    add_documents_to_vector_db([line.strip() for line in data])
    st.session_state["initialized"] = True

# UI
st.title("🧠 Vector DB Demo with ChromaDB + OpenAI + Streamlit")

st.markdown("""
This demo lets you:
1. Store sentences as vectors.
2. Query similar meanings (semantic search).
""")

query = st.text_input("Ask something:", placeholder="e.g. Which city is the capital of France?")

if query:
    with st.spinner("Searching..."):
        results = query_vector_db(query)
        st.subheader("🔍 Most Similar Results")
        for i, doc in enumerate(results['documents'][0]):
            st.write(f"{i+1}. {doc}")
