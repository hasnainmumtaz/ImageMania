import streamlit as st
import requests
import json
from PIL import Image
import io
import time
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Shopify Image Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ShopifyImageSearchClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def check_health(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def search_products(self, shop_url, access_token, image_data, top_k=5, similarity_threshold=0.1, include_product_details=False):
        """Search for similar products"""
        try:
            url = f"{self.api_url}/search"
            
            files = {'image': ('image.jpg', image_data, 'image/jpeg')}
            data = {
                'shop_url': shop_url,
                'access_token': access_token,
                'top_k': top_k,
                'similarity_threshold': similarity_threshold,
                'include_product_details': include_product_details
            }
            
            response = requests.post(url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"API Error {response.status_code}: {response.text}"}
                
        except Exception as e:
            return False, {"error": f"Request failed: {str(e)}"}
    
    def get_cache_info(self, shop_url, access_token):
        """Get cache information for a store"""
        try:
            url = f"{self.api_url}/cache/{shop_url}/info"
            data = {'access_token': access_token}
            response = requests.get(url, params=data, timeout=10)
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Cache info error: {response.text}"}
                
        except Exception as e:
            return False, {"error": f"Cache info failed: {str(e)}"}
    
    def get_resources(self):
        """Get resource usage information"""
        try:
            response = requests.get(f"{self.api_url}/resources", timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Resources error: {response.text}"}
        except Exception as e:
            return False, {"error": f"Resources failed: {str(e)}"}
    
    def get_storage_info(self):
        """Get storage information"""
        try:
            response = requests.get(f"{self.api_url}/storage/info", timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Storage info error: {response.text}"}
        except Exception as e:
            return False, {"error": f"Storage info failed: {str(e)}"}
    
    def get_config(self):
        """Get API configuration"""
        try:
            response = requests.get(f"{self.api_url}/config", timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Config error: {response.text}"}
        except Exception as e:
            return False, {"error": f"Config failed: {str(e)}"}
    
    def clear_cache(self, shop_url, access_token):
        """Clear cache for a specific store"""
        try:
            url = f"{self.api_url}/cache/{shop_url}"
            data = {'access_token': access_token}
            response = requests.delete(url, data=data, timeout=10)
            return True, response.json()
        except Exception as e:
            return False, {"error": f"Clear cache failed: {str(e)}"}
    
    def refresh_cache(self, shop_url, access_token):
        """Force refresh cache for a specific store"""
        try:
            url = f"{self.api_url}/cache/{shop_url}/refresh"
            data = {'access_token': access_token}
            response = requests.post(url, data=data, timeout=30)
            return True, response.json()
        except Exception as e:
            return False, {"error": f"Cache refresh failed: {str(e)}"}

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛍️ Shopify Image Search</h1>', unsafe_allow_html=True)
    st.markdown("### AI-powered product search using CLIP embeddings and FastAPI backend")
    
    # Initialize client
    client = ShopifyImageSearchClient()
    
    # Check API health
    with st.spinner("Checking API server status..."):
        is_healthy, health_data = client.check_health()
    
    if not is_healthy:
        st.error("❌ **API Server Not Available**")
        st.markdown("""
        <div class="error-box">
            <strong>Please start the FastAPI server first:</strong><br>
            <code>python api_server.py</code><br><br>
            The server should be running on <code>http://localhost:8000</code>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.success("✅ **API Server is running and healthy**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API URL configuration
        api_url = st.text_input(
            "API Server URL",
            value="http://localhost:8000",
            help="URL of the FastAPI server"
        )
        
        # Update client if URL changes
        if api_url != client.api_url:
            client = ShopifyImageSearchClient(api_url)
        
        # Shopify credentials
        st.subheader("🏪 Shopify Store")
        shop_url = st.text_input(
            "Shop URL",
            placeholder="your-store.myshopify.com",
            help="Your Shopify store URL without https://"
        )
        
        access_token = st.text_input(
            "Access Token",
            type="password",
            help="Your Shopify private app access token"
        )
        
        # Search parameters
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Number of results", 3, 15, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.1, 0.05)
        include_product_details = st.checkbox(
            "Include detailed product information", 
            value=False,
            help="Fetch complete product details including variants, pricing, inventory, etc. (slower but more comprehensive)"
        )
        
        # Cache management
        st.subheader("💾 Cache Management")
        if st.button("📊 Cache Info", help="Get cache information for the store"):
            if shop_url and access_token:
                with st.spinner("Getting cache info..."):
                    success, cache_info = client.get_cache_info(shop_url, access_token)
                    if success:
                        st.session_state.cache_info = cache_info
                    else:
                        st.error(f"Failed to get cache info: {cache_info.get('error', 'Unknown error')}")
            else:
                st.error("Please provide shop URL and access token")
        
        if st.button("🗑️ Clear Cache", help="Clear cache for this store"):
            if shop_url and access_token:
                with st.spinner("Clearing cache..."):
                    success, result = client.clear_cache(shop_url, access_token)
                    if success:
                        st.success("Cache cleared successfully!")
                    else:
                        st.error(f"Failed to clear cache: {result.get('error', 'Unknown error')}")
            else:
                st.error("Please provide shop URL and access token")
        
        if st.button("🔄 Refresh Cache", help="Force refresh cache from Shopify"):
            if shop_url and access_token:
                with st.spinner("Refreshing cache from Shopify..."):
                    success, result = client.refresh_cache(shop_url, access_token)
                    if success:
                        st.success(f"Cache refreshed! {result.get('cached_products', 0)} products cached")
                    else:
                        st.error(f"Failed to refresh cache: {result.get('error', 'Unknown error')}")
            else:
                st.error("Please provide shop URL and access token")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image to find similar products",
            type=["png", "jpg", "jpeg"],
            help="Upload a product image to find similar items in your Shopify store"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Search button
            if st.button("🔍 Search Products", type="primary"):
                if not shop_url or not access_token:
                    st.error("Please provide Shopify credentials in the sidebar")
                else:
                    # Perform search
                    with st.spinner("Searching for similar products..."):
                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()
                        
                        success, results = client.search_products(
                            shop_url, access_token, img_byte_arr, top_k, similarity_threshold, include_product_details
                        )
                        
                        if success:
                            st.session_state.search_results = results
                            st.success(f"✅ Found {len(results['results'])} similar products!")
                        else:
                            st.error(f"❌ Search failed: {results.get('error', 'Unknown error')}")
    
    with col2:
        st.header("📊 System Status")
        
        # Resource usage
        if st.button("🔄 Refresh Status"):
            with st.spinner("Getting system status..."):
                success, resources = client.get_resources()
                if success:
                    st.session_state.resources = resources
                else:
                    st.error(f"Failed to get resources: {resources.get('error', 'Unknown error')}")
        
        # Display resources if available
        if 'resources' in st.session_state:
            resources = st.session_state.resources
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Memory Usage", f"{resources['memory_usage_mb']:.1f} MB")
            st.metric("Memory Cache", f"{resources['memory_cache_size_mb']:.2f} MB")
            st.metric("Disk Cache", f"{resources['disk_cache_size_mb']:.2f} MB")
            st.metric("Total Cache", f"{resources['total_cache_size_mb']:.2f} MB")
            st.metric("Stores in Memory", resources['cached_stores_memory'])
            st.metric("Stores on Disk", resources['cached_stores_disk'])
            st.metric("Total Products", resources['total_cached_products'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Get and display configuration
        if st.button("⚙️ Show Config", help="Show API configuration"):
            with st.spinner("Getting configuration..."):
                success, config = client.get_config()
                if success:
                    st.session_state.config = config
                else:
                    st.error(f"Failed to get config: {config.get('error', 'Unknown error')}")
        
        # Display configuration if available
        if 'config' in st.session_state:
            config = st.session_state.config
            st.subheader("⚙️ API Configuration")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Pagination Limit:** {config['pagination_limit']} products per page")
            st.write(f"**Max Pages:** {config['max_pages']} pages")
            st.write(f"**Max Products:** {config['max_products']:,} products")
            st.write(f"**Cache Age Limit:** {config['cache_age_limit_hours']} hours")
            st.write(f"**Embedding Model:** {config['embedding_model']}")
            st.write(f"**Similarity Metric:** {config['similarity_metric']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cache info if available
        if 'cache_info' in st.session_state:
            cache_info = st.session_state.cache_info
            st.subheader("📋 Cache Information")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Total Products:** {cache_info['total_products']}")
            st.write(f"**Cached Products:** {cache_info['cached_products']}")
            st.write(f"**New Products:** {cache_info['new_products']}")
            st.write(f"**Updated Products:** {cache_info['updated_products']}")
            st.write(f"**Cache Size:** {cache_info['cache_size_mb']:.2f} MB")
            st.write(f"**Last Updated:** {cache_info['last_updated']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display search results
    if 'search_results' in st.session_state:
        results = st.session_state.search_results
        
        st.header("🎯 Search Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processing Time", f"{results['processing_time']:.2f}s")
        with col2:
            st.metric("Total Products", results['total_products'])
        with col3:
            st.metric("Successful Embeddings", results['successful_embeddings'])
        with col4:
            st.metric("Results Found", len(results['results']))
        
        # Resource usage from search
        if 'resource_usage' in results:
            usage = results['resource_usage']
            st.subheader("📈 Search Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Memory Used", f"{usage['memory_used_mb']:.1f} MB")
            with col2:
                st.metric("New Products Processed", usage['new_products_processed'])
            with col3:
                st.metric("Updated Products Processed", usage['updated_products_processed'])
        
        # Display results
        if results['results']:
            for i, result in enumerate(results['results']):
                st.markdown(f"### {i+1}. {result['title']}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Try to display the first image
                    if result['images']:
                        try:
                            img_response = requests.get(result['images'][0]['image_url'], timeout=5)
                            if img_response.status_code == 200:
                                product_image = Image.open(io.BytesIO(img_response.content))
                                st.image(product_image, use_container_width=True)
                            else:
                                st.error("Failed to load product image")
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")
                
                with col2:
                    st.write(f"**Product Type:** {result['product_type']}")
                    st.write(f"**Vendor:** {result['vendor']}")
                    st.write(f"**Best Similarity Score:** {result['best_score']:.3f}")
                    st.write(f"**Matching Images:** {result['matching_images']}")
                    
                    # Product link
                    st.markdown(f"[🛍️ View Product]({result['product_url']})")
                    
                    # Show all matching images if more than one
                    if len(result['images']) > 1:
                        st.write(f"**All matching images ({len(result['images'])}):**")
                        cols = st.columns(min(3, len(result['images'])))
                        for j, img_data in enumerate(result['images']):
                            with cols[j % 3]:
                                st.write(f"Score: {img_data['score']:.3f}")
                
                # Display detailed product information if available
                if 'detailed_info' in result and result['detailed_info']:
                    st.subheader("📋 Detailed Product Information")
                    
                    detailed = result['detailed_info']
                    
                    # Basic info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Status:** {detailed.get('status', 'N/A')}")
                        st.write(f"**Created:** {detailed.get('created_at', 'N/A')[:10] if detailed.get('created_at') else 'N/A'}")
                    with col2:
                        st.write(f"**Updated:** {detailed.get('updated_at', 'N/A')[:10] if detailed.get('updated_at') else 'N/A'}")
                        st.write(f"**Published:** {detailed.get('published_at', 'N/A')[:10] if detailed.get('published_at') else 'N/A'}")
                    with col3:
                        st.write(f"**SEO Title:** {detailed.get('seo_title', 'N/A')}")
                        st.write(f"**Template:** {detailed.get('template_suffix', 'N/A')}")
                    
                    # Tags
                    if detailed.get('tags'):
                        st.write(f"**Tags:** {', '.join(detailed['tags'])}")
                    
                    # Description
                    if detailed.get('body_html'):
                        st.write("**Description:**")
                        st.markdown(detailed['body_html'], unsafe_allow_html=True)
                    
                    # Variants
                    if detailed.get('variants'):
                        st.subheader("💰 Product Variants")
                        for variant in detailed['variants']:
                            with st.expander(f"Variant: {variant.get('title', 'Default')}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Price:** ${variant.get('price', 'N/A')}")
                                    st.write(f"**Compare At:** ${variant.get('compare_at_price', 'N/A')}")
                                with col2:
                                    st.write(f"**SKU:** {variant.get('sku', 'N/A')}")
                                    st.write(f"**Inventory:** {variant.get('inventory_quantity', 'N/A')}")
                                with col3:
                                    st.write(f"**Weight:** {variant.get('weight', 'N/A')} {variant.get('weight_unit', '')}")
                                    st.write(f"**Barcode:** {variant.get('barcode', 'N/A')}")
                                
                                # Options
                                options_text = []
                                if variant.get('option1'):
                                    options_text.append(f"Option 1: {variant['option1']}")
                                if variant.get('option2'):
                                    options_text.append(f"Option 2: {variant['option2']}")
                                if variant.get('option3'):
                                    options_text.append(f"Option 3: {variant['option3']}")
                                
                                if options_text:
                                    st.write("**Options:** " + " | ".join(options_text))
                    
                    # All Images
                    if detailed.get('images'):
                        st.subheader("🖼️ All Product Images")
                        image_cols = st.columns(min(4, len(detailed['images'])))
                        for j, img in enumerate(detailed['images']):
                            with image_cols[j % 4]:
                                try:
                                    img_response = requests.get(img['src'], timeout=5)
                                    if img_response.status_code == 200:
                                        product_img = Image.open(io.BytesIO(img_response.content))
                                        st.image(product_img, caption=f"Image {j+1}", use_container_width=True)
                                except Exception as e:
                                    st.write(f"Image {j+1}: Error loading")
                
                st.divider()
        else:
            st.info("No similar products found. Try adjusting the similarity threshold or upload a different image.")
    
    # Footer
    st.markdown("---")
    st.markdown("🔧 Built with Streamlit, FastAPI, CLIP, and Shopify API")

if __name__ == "__main__":
    main() 