import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import shopify
import requests
from io import BytesIO
import json
import concurrent.futures
import time
from functools import partial
import psutil
import sys

# Configure Shopify API
def configure_shopify(shop_url, access_token):
    """Configure Shopify API with store credentials"""
    shopify.ShopifyResource.set_site(f"https://{shop_url}/admin/api/2023-10")
    shopify.ShopifyResource.set_headers({'X-Shopify-Access-Token': access_token})

# Resource monitoring functions
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_storage_size(data):
    """Estimate storage size of data in MB"""
    if isinstance(data, np.ndarray):
        return data.nbytes / 1024 / 1024  # Convert to MB
    elif isinstance(data, list):
        return sum(get_storage_size(item) for item in data)
    elif isinstance(data, dict):
        return sum(get_storage_size(value) for value in data.values())
    else:
        return sys.getsizeof(data) / 1024 / 1024  # Convert to MB

def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"

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

# Download and embed a single image
def process_single_image(product, _model, _preprocess, timeout_seconds=5):
    """Download and embed a single image with error handling"""
    try:
        # Download image with timeout and retry
        for attempt in range(2):  # Try twice
            try:
                response = requests.get(product['image_url'], timeout=timeout_seconds)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    emb = get_image_embedding(image, _model, _preprocess)
                    return emb, product, True, response.content  # Return image data for caching
                else:
                    time.sleep(0.1)  # Brief pause before retry
            except requests.exceptions.Timeout:
                if attempt == 0:  # Only retry once
                    time.sleep(0.2)
                    continue
                break
            except Exception:
                break
        return None, product, False, None
    except Exception as e:
        return None, product, False, None

# Fetch products from Shopify
@st.cache_data
def fetch_shopify_products(shop_url, access_token, limit=250):
    """Fetch products from Shopify store"""
    try:
        configure_shopify(shop_url, access_token)
        products = shopify.Product.find(limit=limit)
        
        product_data = []
        for product in products:
            for image in product.images:
                if image.src:  # Only include products with images
                    product_data.append({
                        'id': product.id,
                        'title': product.title,
                        'image_url': image.src,
                        'product_type': product.product_type,
                        'vendor': product.vendor,
                        'handle': product.handle
                    })
        
        return product_data
    except Exception as e:
        st.error(f"Error fetching products from Shopify: {str(e)}")
        return []

# Load and embed Shopify product images with concurrent processing
@st.cache_data
def embed_shopify_images(products, _model, _preprocess):
    """Embed images from Shopify products with concurrent processing"""
    embeddings = []
    product_info = []
    image_cache = {}  # Cache for downloaded images
    
    # Get performance settings
    max_workers = getattr(st.session_state, 'max_workers', 10)
    timeout_seconds = getattr(st.session_state, 'timeout_seconds', 5)
    
    # Start resource monitoring
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process images concurrently
    max_workers = min(max_workers, len(products))  # Limit concurrent workers
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with timeout setting
        future_to_product = {
            executor.submit(process_single_image, product, _model, _preprocess, timeout_seconds): product 
            for product in products
        }
        
        completed = 0
        successful = 0
        
        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_product):
            completed += 1
            embedding, product, success, image_data = future.result()
            
            if success and embedding is not None:
                embeddings.append(embedding)
                product_info.append(product)
                if image_data:
                    image_cache[product['image_url']] = image_data
                successful += 1
            
            # Update progress
            progress = completed / len(products)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {completed}/{len(products)} images ({successful} successful)")
    
    # End resource monitoring
    end_time = time.time()
    end_memory = get_memory_usage()
    
    processing_time = end_time - start_time
    memory_used = end_memory - start_memory
    current_memory = end_memory
    
    progress_bar.empty()
    status_text.empty()
    
    if successful < len(products):
        st.warning(f"⚠️ Successfully processed {successful}/{len(products)} images. Some images failed to load.")
    
    # Calculate storage costs
    embeddings_size = get_storage_size(np.array(embeddings)) if embeddings else 0
    image_cache_size = get_storage_size(image_cache)
    total_storage = embeddings_size + image_cache_size
    
    # Display resource usage
    with st.expander("📊 Resource Usage & Costs", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
            st.metric("🖥️ Memory Used", f"{memory_used:.1f} MB")
            st.metric("💾 Current Memory", f"{current_memory:.1f} MB")
        
        with col2:
            st.metric("📦 Embeddings Size", f"{embeddings_size:.2f} MB")
            st.metric("🖼️ Image Cache Size", f"{image_cache_size:.2f} MB")
            st.metric("💿 Total Storage", f"{total_storage:.2f} MB")
        
        with col3:
            st.metric("🚀 Processing Speed", f"{successful/processing_time:.1f} images/s")
            st.metric("✅ Success Rate", f"{(successful/len(products)*100):.1f}%")
            st.metric("💰 Estimated Cost", f"${total_storage * 0.0001:.4f}")
        
        # Detailed breakdown
        st.markdown("### 📈 Cost Breakdown:")
        st.markdown(f"""
        - **Embeddings Storage**: {format_bytes(embeddings_size * 1024 * 1024)} (CLIP vectors)
        - **Image Cache**: {format_bytes(image_cache_size * 1024 * 1024)} (downloaded images)
        - **Memory Overhead**: {format_bytes(memory_used * 1024 * 1024)} (processing memory)
        - **Total Estimated Cost**: ~${total_storage * 0.0001:.4f} (based on typical cloud storage rates)
        """)
        
        st.markdown("### 💡 Performance Tips:")
        st.markdown("""
        - **Reduce concurrent workers** if memory usage is high
        - **Lower timeout** for faster processing (if network is stable)
        - **Clear cache** periodically to free memory
        - **Monitor success rate** to optimize settings
        """)
    
    # Store image cache in session state for reuse
    st.session_state.image_cache = image_cache
    
    return np.array(embeddings), product_info

# Main application
st.title("🛍️ Shopify Product Image Search with CLIP")
st.markdown("Upload an image to find similar products in your Shopify store!")

# Sidebar for Shopify configuration
with st.sidebar:
    st.header("🔧 Shopify Configuration")
    
    # Shopify credentials
    shop_url = st.text_input("Shop URL (e.g., your-store.myshopify.com)", 
                            help="Your Shopify store URL without https://")
    access_token = st.text_input("Access Token", type="password",
                                help="Your Shopify private app access token")
    
    # Performance settings
    st.subheader("⚡ Performance Settings")
    max_workers = st.slider("Max concurrent workers", 1, 20, 10, 
                           help="Higher values = faster processing but more memory usage")
    timeout_seconds = st.slider("Image download timeout (seconds)", 3, 15, 5,
                               help="Timeout for downloading each image")
    
    if st.button("🔍 Load Products"):
        if shop_url and access_token:
            with st.spinner("Fetching products from Shopify..."):
                products = fetch_shopify_products(shop_url, access_token)
                if products:
                    st.success(f"✅ Loaded {len(products)} products from Shopify!")
                    st.session_state.products = products
                    st.session_state.max_workers = max_workers
                    st.session_state.timeout_seconds = timeout_seconds
                else:
                    st.error("❌ No products found or error occurred")
        else:
            st.error("Please provide both Shop URL and Access Token")

# Load CLIP model
model, preprocess = load_model()

# Main content area
if 'products' in st.session_state and st.session_state.products:
    products = st.session_state.products

    # Embed Shopify images
    with st.spinner("Embedding Shopify product images..."):
        dataset_embeddings, product_info = embed_shopify_images(products, model, preprocess)
    
    st.success(f"✅ Ready! {len(product_info)} products embedded for search")

# Upload query image
    st.header("📸 Upload Query Image")
    uploaded_file = st.file_uploader("Choose an image to find similar products", 
                                   type=["png", "jpg", "jpeg"])

    if uploaded_file and len(product_info) > 0:
    query_image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(query_image, caption="Query Image", use_container_width=True)
        
        with col2:
            st.subheader("🔍 Searching for similar products...")
    query_emb = get_image_embedding(query_image, model, preprocess)
            
    # Compute similarities
    sims = cosine_similarity([query_emb], dataset_embeddings)[0]
            
            # Get top matches
            top_k = st.slider("Number of results to show", 3, 10, 5)
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            # Group results by product ID
            product_groups = {}
    for idx in top_indices:
                product = product_info[idx]
                similarity_score = sims[idx]
                product_id = product['id']
                
                if product_id not in product_groups:
                    product_groups[product_id] = {
                        'product': product,
                        'images': [],
                        'best_score': similarity_score
                    }
                
                product_groups[product_id]['images'].append({
                    'image_url': product['image_url'],
                    'score': similarity_score
                })
                
                # Update best score if this image has higher similarity
                if similarity_score > product_groups[product_id]['best_score']:
                    product_groups[product_id]['best_score'] = similarity_score
            
            # Sort products by their best similarity score
            sorted_products = sorted(product_groups.values(), 
                                   key=lambda x: x['best_score'], reverse=True)
            
            # Apply two-stage ranking: similarity score first, then number of matching images
            def two_stage_ranking(product_group):
                best_score = product_group['best_score']
                num_matching_images = len(product_group['images'])
                
                # Create a composite score that prioritizes similarity but considers image count
                # Multiply by 1000 to ensure similarity is the primary factor
                # Add a small bonus for having more matching images (0.001 per image)
                composite_score = (best_score * 1000) + (num_matching_images * 0.001)
                return composite_score
            
            # Re-sort with two-stage ranking
            sorted_products = sorted(product_groups.values(), 
                                   key=two_stage_ranking, reverse=True)
            
            # Two-stage ranking: First by similarity, then by number of matching images
            # Step 1: Get top products by similarity score
            similarity_threshold = 0.1  # Threshold to consider products as "similar enough"
            top_similar_products = []
            
            for product_group in sorted_products:
                if product_group['best_score'] >= similarity_threshold:
                    top_similar_products.append(product_group)
            
            # Step 2: Among top similar products, rank purely by number of matching images
            final_ranked_products = sorted(top_similar_products, 
                                         key=lambda x: len(x['images']), reverse=True)
            
            st.subheader(f"🎯 Top {len(final_ranked_products)} Similar Products:")
            
            # Display results grouped by product
            for i, product_group in enumerate(final_ranked_products):
                product = product_group['product']
                best_score = product_group['best_score']
                images = product_group['images']
                num_images = len(images)
                
                st.markdown(f"### {i+1}. {product['title']}")
                st.markdown(f"**Best Similarity: {best_score:.3f}** | *{product['product_type']}* | **{num_images} matching image{'s' if num_images > 1 else ''}**")
                
                # Display all matching images for this product
                if len(images) > 1:
                    st.markdown(f"*Found {len(images)} similar images for this product*")
                
                # Create columns for images
                cols = st.columns(min(3, num_images))
                
                for j, img_data in enumerate(images):
                    col_idx = j % 3
                    with cols[col_idx]:
                        try:
                            # Use cached image if available
                            if img_data['image_url'] in st.session_state.image_cache:
                                product_image_data = st.session_state.image_cache[img_data['image_url']]
                                product_image = Image.open(BytesIO(product_image_data)).convert("RGB")
                            else:
                                response = requests.get(img_data['image_url'], timeout=5)
                                if response.status_code == 200:
                                    product_image = Image.open(BytesIO(response.content)).convert("RGB")
                                else:
                                    st.error("Failed to load product image")
                                    continue
                            
                            st.image(product_image, use_container_width=True)
                            st.markdown(f"**Score: {img_data['score']:.3f}**")
                            
                        except Exception as e:
                            st.error(f"Failed to load image: {str(e)}")
                
                # Product information and link
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Vendor:** {product['vendor']}")
                with col2:
                    product_url = f"https://{shop_url}/products/{product['handle']}"
                    st.markdown(f"[🛍️ View Product]({product_url})")
                
                st.divider()

else:
    st.info("👆 Please configure your Shopify credentials in the sidebar and load products to get started!")
    
    # Instructions
    with st.expander("📖 How to set up Shopify integration"):
        st.markdown("""
        ### Setting up Shopify API Access:
        
        1. **Create a Private App in Shopify:**
           - Go to your Shopify admin → Apps → Develop apps
           - Click "Create an app"
           - Give it a name (e.g., "Image Search App")
           - Under "Admin API access scopes", enable:
             - `read_products`
             - `read_product_listings`
        
        2. **Get your credentials:**
           - Copy your Shop URL (e.g., `your-store.myshopify.com`)
           - Copy the Admin API access token
        
        3. **Enter credentials in the sidebar and click 'Load Products'**
        
        ### How it works:
        - The app fetches all products from your Shopify store
        - Each product image is embedded using CLIP (AI model)
        - When you upload a query image, it finds the most similar products
        - Results are ranked by similarity score
        """)

# Footer
st.markdown("---")
st.markdown("🔧 Built with Streamlit, CLIP, and Shopify API") 