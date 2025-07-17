# 🛍️ Shopify Product Image Search with CLIP

A powerful reverse image search application that connects to your Shopify store to find similar products using AI-powered image embeddings. Now with **local storage** for persistent caching and improved performance!

## ✨ Features

- **Shopify Integration**: Directly fetch products and images from your Shopify store
- **AI-Powered Search**: Uses OpenAI's CLIP model for accurate image similarity matching
- **Real-time Results**: Upload any image and find similar products instantly
- **Product Links**: Direct links to view products on your Shopify store
- **Beautiful UI**: Modern Streamlit interface with progress tracking
- **🆕 Local Storage**: Persistent disk caching for faster startup and reduced API calls
- **🆕 Smart Caching**: Only embeds new or updated products
- **🆕 Multi-store Support**: Search across multiple Shopify stores
- **🆕 FastAPI Backend**: Scalable API server with caching and monitoring
- **🆕 Dual Interface**: Both standalone Streamlit app and FastAPI + Streamlit combo
- **🆕 Easy Startup**: One-click startup script for both services

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Shopify API Access

1. **Create a Private App in Shopify:**
   - Go to your Shopify admin → Apps → Develop apps
   - Click "Create an app"
   - Give it a name (e.g., "Image Search App")
   - Under "Admin API access scopes", enable:
     - `read_products`
     - `read_product_listings`
   - Save and copy your credentials

2. **Get your credentials:**
   - Shop URL (e.g., `your-store.myshopify.com`)
   - Admin API access token

### 3. Run the Application

#### Option A: All-in-One Streamlit App (Simple)
```bash
streamlit run app.py
```

#### Option B: FastAPI + Streamlit (Recommended)
```bash
# Use the startup script (easiest)
python start_app.py

# Or start manually:
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Start Streamlit app
streamlit run streamlit_app.py
```

#### Option C: API Only (For Developers)
```bash
# Start the API server
python api_server.py

# In another terminal, run the client example
python client_example.py

# Or run the storage demo
python storage_demo.py
```

### 4. Configure and Use

1. Open the app in your browser (Streamlit) or use the API endpoints
2. Enter your Shopify credentials
3. Load products (embeddings are cached locally for faster subsequent runs)
4. Upload an image to find similar products
5. Browse results with similarity scores and direct product links

## 🔧 How It Works

1. **Product Fetching**: The app connects to your Shopify store via API and retrieves all products with images
2. **Smart Caching**: Checks for existing embeddings on disk and only processes new/updated products
3. **Image Embedding**: Each product image is processed through CLIP to create high-dimensional embeddings
4. **Local Storage**: Embeddings are saved to disk for persistent caching across server restarts
5. **Similarity Search**: When you upload a query image, it's embedded and compared against all product embeddings
6. **Results Ranking**: Products are ranked by cosine similarity score and displayed with relevant information

## 💾 Local Storage Features

### **Smart Caching System**
- **Persistent Storage**: Embeddings saved to `embeddings_cache/` directory
- **Change Detection**: Only embeds new or updated products using metadata hashing
- **Store Isolation**: Each store's embeddings are stored separately
- **Automatic Updates**: Detects product changes and updates embeddings incrementally

### **Storage Management**
- **Cache Info**: Monitor cache size, hit rates, and update frequency
- **Storage Monitoring**: Track disk usage and file management
- **Cache Clearing**: Clear specific store caches or all storage
- **Performance Metrics**: Monitor processing time and memory usage

### **Benefits**
- ⚡ **Faster Startup**: No need to re-embed all products on restart
- 🔄 **Reduced API Calls**: Only fetch new/updated products from Shopify
- 💾 **Persistent Cache**: Survives server restarts and updates
- 📊 **Efficient Updates**: Incremental embedding updates
- 🏪 **Multi-store Support**: Isolated caching per store

## 📋 Requirements

- Python 3.7+
- Shopify store with products
- Shopify private app with API access
- Internet connection for image processing

## 🛠️ Technical Details

- **CLIP Model**: ViT-B/32 for image embeddings
- **Similarity Metric**: Cosine similarity
- **Image Processing**: PIL for image handling
- **API Integration**: Shopify Python API
- **UI Framework**: Streamlit (dual interface)
- **Backend API**: FastAPI with async support
- **Storage**: Pickle-based local storage with numpy array serialization
- **Caching**: Smart change detection with MD5 hashing
- **Concurrency**: ThreadPoolExecutor for parallel image processing
- **Architecture**: Microservices with FastAPI backend + Streamlit frontend

## 🔒 Security

- Access tokens are stored securely in session state
- No credentials are saved to disk
- HTTPS connections for all API calls

## 📝 Usage Tips

- **Better Results**: Use high-quality, clear product images for queries
- **Product Types**: The app works best with similar product categories
- **Performance**: First run may take longer as it embeds all products
- **Caching**: Results are cached for faster subsequent searches
- **Storage Management**: Monitor cache size and clear old caches periodically
- **Multi-store**: Use the API server for searching across multiple stores
- **Cache Efficiency**: Check cache info to monitor embedding efficiency

## 🚀 API Endpoints

### Search Endpoints
- `POST /search` - Search products in a specific store
- `POST /search/multi-store` - Search across multiple stores

### Cache Management
- `GET /cache/{shop_url}/info` - Get cache information for a store
- `DELETE /cache/{shop_url}` - Clear cache for a specific store
- `GET /storage/info` - Get detailed storage information
- `DELETE /storage/clear` - Clear all storage files

### Monitoring
- `GET /health` - API health check
- `GET /resources` - Resource usage information
- `GET /stores/validate` - Validate store isolation

## 📊 Performance Monitoring

Monitor these key metrics:
- **Cache Hit Rate**: Ratio of cached vs total products
- **Processing Time**: Time to embed new/updated products
- **Storage Usage**: Disk space used by cache files
- **Memory Usage**: RAM usage during operations
- **API Efficiency**: Number of Shopify API calls made

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License. 