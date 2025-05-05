import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import chromadb
import numpy as np
import io

# --- STEP 1: Setup ---
st.title("🖼️ Image Semantic Search with ChromaDB")

# Load a pre-trained model (ResNet18) without classification head
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final layer
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(name="image_demo")

# Track added image count
if "image_count" not in st.session_state:
    st.session_state.image_count = 0

# --- STEP 2: Upload and store images ---
st.subheader("📥 Upload Images to Add to Database")
uploaded_files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("➕ Add to Vector DB"):
    if uploaded_files:
        for img_file in uploaded_files:
            img = Image.open(img_file).convert("RGB")
            tensor = transform(img).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                embedding = model(tensor).squeeze().numpy().tolist()

            # Save to ChromaDB
            collection.add(
                documents=[img_file.name],
                embeddings=[embedding],
                ids=[f"img{st.session_state.image_count}"]
            )
            st.session_state.image_count += 1

        st.success(f"✅ Added {len(uploaded_files)} images to vector DB.")
    else:
        st.warning("Please upload at least one image.")

# --- STEP 3: Upload query image ---
st.subheader("🔍 Upload Query Image")
query_file = st.file_uploader("Choose a query image", type=["png", "jpg", "jpeg"], key="query_img")

if st.button("🔎 Search Similar Images"):
    if query_file:
        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, caption="Query Image", width=200)

        # Embed query
        tensor = transform(query_img).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model(tensor).squeeze().numpy().tolist()

        # Query ChromaDB
        try:
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            st.markdown("### 🔗 Top Similar Images:")
            for doc in results["documents"][0]:
                st.write(doc)
        except Exception as e:
            st.error(f"Error querying vector DB: {e}")
    else:
        st.warning("Please upload a query image.")
