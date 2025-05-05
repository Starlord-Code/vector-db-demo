import streamlit as st
import torch
from PIL import Image
import chromadb
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# Title
st.title("🖼️ Text-to-Image Search using CLIP + ChromaDB")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ChromaDB collection
client = chromadb.Client()
collection = client.get_or_create_collection(name="clip_image_demo")

# Track count
if "image_count" not in st.session_state:
    st.session_state.image_count = 0

# Step 1: Upload images and store embeddings
st.subheader("📤 Upload Images to Store")
uploaded_files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("➕ Add Images to DB"):
    if uploaded_files:
        for img_file in uploaded_files:
            img = Image.open(img_file).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_embedding = image_features.squeeze().numpy().tolist()

            collection.add(
                documents=[img_file.name],
                embeddings=[image_embedding],
                ids=[f"img{st.session_state.image_count}"]
            )
            st.session_state.image_count += 1

        st.success(f"{len(uploaded_files)} images added.")
    else:
        st.warning("Please upload at least one image.")

# Step 2: Text query
st.subheader("🔎 Search Images by Text")
text_query = st.text_input("Enter search text (e.g., 'a cat', 'sunset', 'red car')")

if st.button("🔍 Search"):
    if text_query:
        inputs = processor(text=[text_query], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            query_embedding = text_features.squeeze().numpy().tolist()

        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            st.markdown("### 🔗 Top Matches:")
            for doc in results["documents"][0]:
                st.write(f"Matched Image: **{doc}**")
        except Exception as e:
            st.error(f"Error querying DB: {e}")
    else:
        st.warning("Please enter a query.")
