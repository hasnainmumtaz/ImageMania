import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity

# Load CLIP model
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess

# Embed an image
def get_image_embedding(image, model, preprocess):
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    return embedding.cpu().numpy().flatten()

# Load and embed all dataset images
@st.cache_data
def embed_dataset_images(image_folder, _model, _preprocess):
    embeddings = []
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, fname)
                image = Image.open(path).convert("RGB")
                emb = get_image_embedding(image, _model, _preprocess)
                embeddings.append(emb)
                image_paths.append(path)
    return np.array(embeddings), image_paths

st.title("Reverse Image Search with CLIP")

model, preprocess = load_model()

image_folder = "images"

# Embed dataset images
with st.spinner("Embedding dataset images..."):
    dataset_embeddings, dataset_paths = embed_dataset_images(image_folder, model, preprocess)

# Upload query image
uploaded_file = st.file_uploader("Upload a query image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", use_column_width=True)
    query_emb = get_image_embedding(query_image, model, preprocess)
    # Compute similarities
    sims = cosine_similarity([query_emb], dataset_embeddings)[0]
    # Get top 5 matches
    top_indices = np.argsort(sims)[::-1][:5]
    st.subheader("Top 5 Similar Images:")
    for idx in top_indices:
        st.image(dataset_paths[idx], caption=f"Score: {sims[idx]:.3f}", use_column_width=True) 