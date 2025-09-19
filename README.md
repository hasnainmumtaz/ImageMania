# 🛍️ Shopify Product Image Search with CLIP

A powerful reverse image search application that connects to your Shopify store to find similar products using AI-powered image embeddings. Features a FastAPI backend with local storage for persistent caching and a modern Streamlit frontend.

## ✨ Features

- **Shopify Integration**: Directly fetch products and images from your Shopify store
- **AI-Powered Search**: Uses OpenAI's CLIP model for accurate image similarity matching
- **Real-time Results**: Upload any image and find similar products instantly
- **Product Links**: Direct links to view products on your Shopify store
- **Beautiful UI**: Modern Streamlit interface with progress tracking
- **🆕 Local Storage**: Persistent disk caching for faster startup and reduced API calls
- **🆕 Smart Caching**: Only embeds new or updated products using metadata hashing
- **🆕 Multi-store Support**: Search across multiple Shopify stores simultaneously
- **🆕 FastAPI Backend**: Scalable API server with comprehensive monitoring
- **🆕 Dual Interface**: Both standalone Streamlit app and FastAPI + Streamlit combo
- **🆕 Store Isolation**: Secure separation of embeddings between different stores
- **🆕 Resource Monitoring**: Real-time memory, storage, and performance metrics

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/hasnainmumtaz/ImageMania
cd ImageMania
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Shopify API Access

1. **Create a Private App in Shopify:**
   - Go to your Shopify admin → Apps → Develop apps
   - Click "Create an app"
   - Give it a name (e.g., "Image Search App")
   - Under "Admin API access scopes", enable:
     - `read_products`
     - `read_product_listings`
   - Click **"Save"**
   - Go to **"API credentials"** tab
   - Click **"Install app"** to generate the access token
   - Copy the **Admin API access token**

2. **Get your credentials:**
   - Shop URL (e.g., `your-store.myshopify.com`)
   - Admin API access token (not the API key)

3. **Test your credentials:**
   ```bash
   python test_shopify_auth.py
   ```

### 4. Run the Application

#### Option A: FastAPI + Streamlit (Recommended)
```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Start Streamlit app
streamlit run streamlit_app.py
```

#### Option B: API Only (For Developers)
```bash
# Start the API server
python api_server.py

# In another terminal, run the client example
python client_example.py

# Or run the storage demo
python storage_demo.py
```

### 5. Configure and Use

1. Open the Streamlit app in your browser (usually `http://localhost:8501`)
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
- **Store Isolation**: Each store's embeddings are stored separately with secure token hashing
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
- 🔒 **Secure Isolation**: Store-specific token hashing prevents data leakage

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
- **UI Framework**: Streamlit with custom styling
- **Backend API**: FastAPI with async support and comprehensive endpoints
- **Storage**: Pickle-based local storage with numpy array serialization
- **Caching**: Smart change detection with MD5 hashing
- **Concurrency**: ThreadPoolExecutor for parallel image processing
- **Architecture**: Microservices with FastAPI backend + Streamlit frontend
- **Monitoring**: Real-time resource usage and performance metrics

## 🔒 Security

- Access tokens are stored securely in session state
- No credentials are saved to disk
- HTTPS connections for all API calls
- Store isolation with token hashing
- Secure file naming for cache storage

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
- `POST /cache/{shop_url}/refresh` - Force refresh cache for a store
- `GET /storage/info` - Get detailed storage information
- `DELETE /storage/clear` - Clear all storage files

### Monitoring & Health
- `GET /health` - API health check
- `GET /resources` - Resource usage information
- `GET /stores/validate` - Validate store isolation
- `GET /config` - Get API configuration

## 📊 Performance Monitoring

Monitor these key metrics:
- **Cache Hit Rate**: Ratio of cached vs total products
- **Processing Time**: Time to embed new/updated products
- **Storage Usage**: Disk space used by cache files
- **Memory Usage**: RAM usage during operations
- **API Efficiency**: Number of Shopify API calls made
- **Store Isolation**: Validation of secure data separation

## 🧪 Testing & Examples

### Test Shopify Credentials
```bash
python test_shopify_auth.py
```
Test your Shopify API credentials before using the main application.

### Client Example
```bash
python client_example.py
```
Demonstrates API usage with example store configurations.

### Storage Demo
```bash
python storage_demo.py
```
Shows local storage features, cache management, and monitoring.

## 🔧 Troubleshooting

### Shopify API Authentication Issues

If you get a "401: Invalid API key or access token" error:

1. **Verify App Type**: Make sure you created a **Private App**, not a Public App
2. **Check Installation**: Ensure you clicked "Install app" after creating the app
3. **Verify Permissions**: Confirm `read_products` and `read_product_listings` are enabled
4. **Use Correct Token**: Use the **Admin API access token**, not the API key
5. **Check Shop URL**: Use format `your-store.myshopify.com` (without https://)
6. **Test Credentials**: Run `python test_shopify_auth.py` to verify your setup

### Common Error Messages

- **"Invalid API key or access token"**: Check your access token and app installation
- **"No product embeddings found"**: Ensure your store has products with images
- **"Shopify API error"**: Verify your shop URL format and permissions

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License. 
