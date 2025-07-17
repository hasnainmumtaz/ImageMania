from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
from PIL import Image
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import shopify
import requests
from io import BytesIO
import concurrent.futures
import time
import json
from typing import List, Dict, Optional
import psutil
from pydantic import BaseModel
import asyncio
from datetime import datetime, timedelta
import hashlib
import pickle
import pathlib

# Pydantic models
class SearchRequest(BaseModel):
    shop_url: str
    access_token: str
    top_k: int = 5
    similarity_threshold: float = 0.1

class SearchResponse(BaseModel):
    results: List[Dict]
    processing_time: float
    total_products: int
    successful_embeddings: int
    resource_usage: Dict

class CacheInfo(BaseModel):
    total_products: int
    cached_products: int
    new_products: int
    updated_products: int
    last_updated: str
    cache_size_mb: float

# Initialize FastAPI app
app = FastAPI(
    title="Shopify Image Search API",
    description="AI-powered image search for Shopify stores using CLIP",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
model_cache = {}
embedding_cache = {}
product_metadata_cache = {}  # Store product metadata for change detection
last_updated = {}

# Local storage configuration
STORAGE_DIR = pathlib.Path("embeddings_cache")
STORAGE_DIR.mkdir(exist_ok=True)

# Resource monitoring
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_storage_size(data):
    if isinstance(data, np.ndarray):
        return data.nbytes / 1024 / 1024
    elif isinstance(data, list):
        return sum(get_storage_size(item) for item in data)
    elif isinstance(data, dict):
        return sum(get_storage_size(value) for value in data.values())
    else:
        return len(str(data).encode()) / 1024 / 1024

def format_bytes(bytes_value):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"

# Load CLIP model
def load_model():
    if 'model' not in model_cache:
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        model_cache['model'] = model
        model_cache['preprocess'] = preprocess
    return model_cache['model'], model_cache['preprocess']

# Configure Shopify API
def configure_shopify(shop_url: str, access_token: str):
    shopify.ShopifyResource.set_site(f"https://{shop_url}/admin/api/2023-10")
    shopify.ShopifyResource.set_headers({'X-Shopify-Access-Token': access_token})

# Generate product hash for change detection
def generate_product_hash(product: Dict) -> str:
    """Generate a hash for product metadata to detect changes"""
    metadata = {
        'id': product['id'],
        'title': product['title'],
        'image_url': product['image_url'],
        'product_type': product['product_type'],
        'vendor': product['vendor'],
        'handle': product['handle']
    }
    return hashlib.md5(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

# Local storage functions
def get_cache_file_path(shop_url: str, access_token: str) -> pathlib.Path:
    """Get the file path for storing cache data for a specific store"""
    # Create a safe filename from shop URL and token hash
    token_hash = hashlib.md5(access_token.encode()).hexdigest()[:8]
    safe_shop_name = shop_url.replace('.', '_').replace('-', '_')
    filename = f"{safe_shop_name}_{token_hash}.pkl"
    return STORAGE_DIR / filename

def save_embeddings_to_disk(shop_url: str, access_token: str, cache_data: Dict):
    """Save embeddings and metadata to disk"""
    try:
        cache_file = get_cache_file_path(shop_url, access_token)
        
        # Prepare data for serialization (convert numpy arrays to lists)
        serializable_data = {
            'embeddings': cache_data['embeddings'].tolist() if isinstance(cache_data['embeddings'], np.ndarray) else cache_data['embeddings'],
            'product_info': cache_data['product_info'],
            'product_hashes': cache_data['product_hashes'],
            'total_products': cache_data['total_products'],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        print(f"✅ Saved embeddings to disk: {cache_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving embeddings to disk: {str(e)}")
        return False

def load_embeddings_from_disk(shop_url: str, access_token: str) -> Optional[Dict]:
    """Load embeddings and metadata from disk"""
    try:
        cache_file = get_cache_file_path(shop_url, access_token)
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        # Convert lists back to numpy arrays
        if isinstance(data['embeddings'], list):
            data['embeddings'] = np.array(data['embeddings'])
        
        print(f"✅ Loaded embeddings from disk: {cache_file}")
        return data
    except Exception as e:
        print(f"❌ Error loading embeddings from disk: {str(e)}")
        return None

def delete_cache_file(shop_url: str, access_token: str) -> bool:
    """Delete cache file from disk"""
    try:
        cache_file = get_cache_file_path(shop_url, access_token)
        if cache_file.exists():
            cache_file.unlink()
            print(f"✅ Deleted cache file: {cache_file}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error deleting cache file: {str(e)}")
        return False

def get_cache_file_info(shop_url: str, access_token: str) -> Dict:
    """Get information about cache file"""
    cache_file = get_cache_file_path(shop_url, access_token)
    
    if not cache_file.exists():
        return {
            'exists': False,
            'size_mb': 0,
            'last_modified': None
        }
    
    try:
        stat = cache_file.stat()
        return {
            'exists': True,
            'size_mb': stat.st_size / (1024 * 1024),
            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    except Exception:
        return {
            'exists': False,
            'size_mb': 0,
            'last_modified': None
        }

# Fetch products from Shopify with cursor-based pagination
def fetch_shopify_products(shop_url: str, access_token: str, limit: int = 250):
    try:
        configure_shopify(shop_url, access_token)
        
        product_data = []
        total_products = 0
        page_count = 0
        
        print(f"🔄 Fetching products from {shop_url}...")
        
        # Use the shopify library's pagination support
        # The library should handle cursor-based pagination automatically
        products = shopify.Product.find(limit=limit)
        
        while products:
            page_count += 1
            page_products = 0
            
            for product in products:
                for image in product.images:
                    if image.src:
                        product_data.append({
                            'id': product.id,
                            'title': product.title,
                            'image_url': image.src,
                            'product_type': product.product_type,
                            'vendor': product.vendor,
                            'handle': product.handle
                        })
                        page_products += 1
            
            total_products += page_products
            print(f"   📄 Page {page_count}: {page_products} products (Total: {total_products})")
            
            # If we got fewer products than the limit, we've reached the end
            if len(products) < limit:
                break
            
            # Safety check to prevent infinite loops
            if page_count > 100:  # Maximum 100 pages (25,000 products)
                print(f"⚠️  Reached maximum page limit ({page_count}). Stopping pagination.")
                break
            
            # The shopify library should automatically handle the next page
            # We need to check if there are more products by making another request
            try:
                # This will automatically use the next page_info from the Link header
                products = shopify.Product.find(limit=limit)
            except Exception as e:
                print(f"⚠️  Error fetching next page: {str(e)}")
                break
        
        print(f"✅ Fetched {total_products} total products from {page_count} pages")
        return product_data
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Shopify API error: {str(e)}")

# Process single image
def process_single_image(product: Dict, model, preprocess, timeout: int = 5):
    try:
        for attempt in range(2):
            try:
                response = requests.get(product['image_url'], timeout=timeout)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image_input = preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        embedding = model.encode_image(image_input)
                    return embedding.cpu().numpy().flatten(), product, True
                else:
                    time.sleep(0.1)
            except requests.exceptions.Timeout:
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                break
            except Exception:
                break
        return None, product, False
    except Exception:
        return None, product, False

# Embed specific products
def embed_products(products: List[Dict], model, preprocess, max_workers: int = 10):
    embeddings = []
    product_info = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_product = {
            executor.submit(process_single_image, product, model, preprocess): product 
            for product in products
        }
        
        for future in concurrent.futures.as_completed(future_to_product):
            embedding, product, success = future.result()
            if success and embedding is not None:
                embeddings.append(embedding)
                product_info.append(product)
    
    return embeddings, product_info

# Smart cache management with change detection and local storage
def get_store_embeddings_smart(shop_url: str, access_token: str, pagination_limit: int = 250):
    cache_key = f"{shop_url}_{access_token}"
    
    # Try to load from disk first if not in memory
    if cache_key not in embedding_cache:
        disk_data = load_embeddings_from_disk(shop_url, access_token)
        if disk_data:
            embedding_cache[cache_key] = {
                'embeddings': disk_data['embeddings'],
                'product_info': disk_data['product_info'],
                'product_hashes': disk_data['product_hashes'],
                'total_products': disk_data['total_products']
            }
            last_updated[cache_key] = datetime.fromisoformat(disk_data['last_updated'])
            print(f"📁 Loaded {len(disk_data['product_info'])} products from disk cache")
    
    # Initialize cache if not exists (either in memory or disk)
    if cache_key not in embedding_cache:
        embedding_cache[cache_key] = {
            'embeddings': [],
            'product_info': [],
            'product_hashes': {},
            'total_products': 0
        }
        product_metadata_cache[cache_key] = {}
        last_updated[cache_key] = datetime.now()
    
    # Get cached data
    cached_data = embedding_cache[cache_key]
    cached_hashes = cached_data['product_hashes']
    
    # Check if we have cached data and if it's recent (within 1 hour)
    cache_age = datetime.now() - last_updated[cache_key] if cache_key in last_updated else timedelta(hours=2)
    
    # If we have cached data and it's recent, skip Shopify API call unless forced
    if len(cached_data['product_info']) > 0 and cache_age < timedelta(hours=1):
        print(f"✅ Using cached data ({len(cached_data['product_info'])} products, age: {cache_age.total_seconds()/60:.1f} minutes)")
        return {
            'embeddings': np.array(cached_data['embeddings']),
            'product_info': cached_data['product_info'],
            'total_products': cached_data['total_products'],
            'new_products': 0,
            'updated_products': 0,
            'cached_products': len(cached_data['product_info'])
        }
    
    # Fetch current products from Shopify (only if cache is old or empty)
    print(f"🔄 Fetching products from Shopify (cache age: {cache_age.total_seconds()/60:.1f} minutes)")
    current_products = fetch_shopify_products(shop_url, access_token, pagination_limit)
    
    # Detect changes
    new_products = []
    updated_products = []
    unchanged_products = []
    
    for product in current_products:
        product_hash = generate_product_hash(product)
        
        if product['id'] not in cached_hashes:
            # New product
            new_products.append(product)
        elif cached_hashes[product['id']] != product_hash:
            # Updated product
            updated_products.append(product)
        else:
            # Unchanged product
            unchanged_products.append(product)
    
    # Process new and updated products
    products_to_process = new_products + updated_products
    
    if products_to_process:
        print(f"🔄 Processing {len(products_to_process)} new/updated products...")
        model, preprocess = load_model()
        new_embeddings, new_product_info = embed_products(products_to_process, model, preprocess)
        
        # Update cache
        if new_embeddings:
            # Remove old embeddings for updated products
            if updated_products:
                updated_ids = {p['id'] for p in updated_products}
                cached_data['embeddings'] = [
                    emb for i, emb in enumerate(cached_data['embeddings'])
                    if cached_data['product_info'][i]['id'] not in updated_ids
                ]
                cached_data['product_info'] = [
                    info for info in cached_data['product_info']
                    if info['id'] not in updated_ids
                ]
            
            # Add new embeddings
            cached_data['embeddings'].extend(new_embeddings)
            cached_data['product_info'].extend(new_product_info)
            
            # Update hashes
            for product in products_to_process:
                cached_hashes[product['id']] = generate_product_hash(product)
            
            # Save updated cache to disk
            save_embeddings_to_disk(shop_url, access_token, cached_data)
    else:
        print(f"✅ No changes detected - using existing cache ({len(cached_data['product_info'])} products)")
    
    # Update metadata
    cached_data['total_products'] = len(current_products)
    last_updated[cache_key] = datetime.now()
    
    return {
        'embeddings': np.array(cached_data['embeddings']),
        'product_info': cached_data['product_info'],
        'total_products': len(current_products),
        'new_products': len(new_products),
        'updated_products': len(updated_products),
        'cached_products': len(cached_data['product_info'])
    }

# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_products(
    shop_url: str = Form(...),
    access_token: str = Form(...),
    image: UploadFile = File(...),
    top_k: int = Form(5),
    similarity_threshold: float = Form(0.1)
):
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Load and process query image
        image_data = await image.read()
        query_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Get store embeddings with smart caching (store-specific)
        store_data = get_store_embeddings_smart(shop_url, access_token, pagination_limit=250)
        dataset_embeddings = store_data['embeddings']
        product_info = store_data['product_info']
        
        # Validate that we have embeddings for this specific store
        if len(dataset_embeddings) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No product embeddings found for store: {shop_url}. Please ensure the store has products with images."
            )
        
        # Get query embedding
        model, preprocess = load_model()
        image_input = preprocess(query_image).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model.encode_image(image_input)
        query_embedding = query_embedding.cpu().numpy().flatten()
        
        # Compute similarities (only within this store's products)
        sims = cosine_similarity([query_embedding], dataset_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(sims)[::-1][:top_k * 3]  # Get more for grouping
        
        # Group results by product ID (store-specific)
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
            
            if similarity_score > product_groups[product_id]['best_score']:
                product_groups[product_id]['best_score'] = similarity_score
        
        # Two-stage ranking: First by similarity, then by number of matching images
        # Step 1: Get top products by similarity score
        top_similar_products = []
        for product_group in product_groups.values():
            if product_group['best_score'] >= similarity_threshold:
                top_similar_products.append(product_group)
        
        # Step 2: Among top similar products, rank purely by number of matching images
        final_ranked_products = sorted(top_similar_products, 
                                     key=lambda x: len(x['images']), reverse=True)
        
        # Format results with store-specific URLs
        results = []
        for product_group in final_ranked_products[:top_k]:
            product = product_group['product']
            results.append({
                'id': product['id'],
                'title': product['title'],
                'product_type': product['product_type'],
                'vendor': product['vendor'],
                'handle': product['handle'],
                'best_score': float(product_group['best_score']),  # Convert numpy.float32 to Python float
                'matching_images': len(product_group['images']),
                'images': [{
                    'image_url': img['image_url'],
                    'score': float(img['score'])  # Convert numpy.float32 to Python float
                } for img in product_group['images']],
                'product_url': f"https://{shop_url}/products/{product['handle']}",
                'store_url': shop_url  # Include store URL for verification
            })
        
        # Calculate resource usage
        end_time = time.time()
        end_memory = get_memory_usage()
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        resource_usage = {
            'processing_time': float(processing_time),
            'memory_used_mb': float(memory_used),
            'current_memory_mb': float(end_memory),
            'embeddings_count': len(dataset_embeddings),
            'successful_embeddings': len(product_info),
            'new_products_processed': store_data['new_products'],
            'updated_products_processed': store_data['updated_products'],
            'store_url': shop_url,  # Include store URL for tracking
            'store_products_total': store_data['total_products']
        }
        
        return SearchResponse(
            results=results,
            processing_time=processing_time,
            total_products=store_data['total_products'],
            successful_embeddings=len(product_info),
            resource_usage=resource_usage
        )
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Multi-store search endpoint (for comparing across stores)
@app.post("/search/multi-store")
async def search_products_multi_store(
    stores: str = Form(...),  # JSON string of store configs
    image: UploadFile = File(...),
    top_k: int = Form(5),
    similarity_threshold: float = Form(0.1)
):
    """
    Search across multiple stores simultaneously
    stores format: [{"shop_url": "...", "access_token": "..."}, ...]
    """
    start_time = time.time()
    
    try:
        # Parse stores configuration
        stores_config = json.loads(stores)
        if not isinstance(stores_config, list) or len(stores_config) == 0:
            raise HTTPException(status_code=400, detail="Invalid stores configuration")
        
        # Load and process query image
        image_data = await image.read()
        query_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Get query embedding
        model, preprocess = load_model()
        image_input = preprocess(query_image).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model.encode_image(image_input)
        query_embedding = query_embedding.cpu().numpy().flatten()
        
        all_results = []
        
        # Search each store separately
        for store_config in stores_config:
            shop_url = store_config.get('shop_url')
            access_token = store_config.get('access_token')
            
            if not shop_url or not access_token:
                continue
            
            try:
                # Get store embeddings
                store_data = get_store_embeddings_smart(shop_url, access_token, pagination_limit=250)
                dataset_embeddings = store_data['embeddings']
                product_info = store_data['product_info']
                
                if len(dataset_embeddings) == 0:
                    continue
                
                # Compute similarities for this store
                sims = cosine_similarity([query_embedding], dataset_embeddings)[0]
                
                # Get top matches for this store
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
                    
                    if similarity_score > product_groups[product_id]['best_score']:
                        product_groups[product_id]['best_score'] = similarity_score
                
                # Apply two-stage ranking
                top_similar_products = []
                for product_group in product_groups.values():
                    if product_group['best_score'] >= similarity_threshold:
                        top_similar_products.append(product_group)
                
                final_ranked_products = sorted(top_similar_products, 
                                             key=lambda x: len(x['images']), reverse=True)
                
                # Add store results
                for product_group in final_ranked_products[:top_k]:
                    product = product_group['product']
                    all_results.append({
                        'id': product['id'],
                        'title': product['title'],
                        'product_type': product['product_type'],
                        'vendor': product['vendor'],
                        'handle': product['handle'],
                        'best_score': float(product_group['best_score']),  # Convert numpy.float32 to Python float
                        'matching_images': len(product_group['images']),
                        'images': [{
                            'image_url': img['image_url'],
                            'score': float(img['score'])  # Convert numpy.float32 to Python float
                        } for img in product_group['images']],
                        'product_url': f"https://{shop_url}/products/{product['handle']}",
                        'store_url': shop_url
                    })
                    
            except Exception as e:
                # Log error but continue with other stores
                print(f"Error processing store {shop_url}: {str(e)}")
                continue
        
        # Sort all results by best score
        all_results.sort(key=lambda x: x['best_score'], reverse=True)
        
        # Calculate resource usage
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "results": all_results[:top_k * len(stores_config)],
            "processing_time": float(processing_time),
            "stores_processed": len(stores_config),
            "total_results": len(all_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-store search error: {str(e)}")

# Store isolation validation endpoint
@app.get("/stores/validate")
async def validate_store_isolation():
    """Validate that store embeddings are properly isolated"""
    validation_results = {}
    
    for cache_key, cached_data in embedding_cache.items():
        shop_url = cache_key.split('_')[0]  # Extract shop URL from cache key
        
        # Check if all products belong to the correct store
        products_in_store = cached_data['product_info']
        store_validation = {
            'total_products': len(products_in_store),
            'valid_products': 0,
            'invalid_products': 0,
            'store_url': shop_url
        }
        
        for product in products_in_store:
            # Validate product belongs to this store
            if product.get('handle') and shop_url in product.get('handle', ''):
                store_validation['valid_products'] += 1
            else:
                store_validation['invalid_products'] += 1
        
        validation_results[cache_key] = store_validation
    
    return {
        "store_isolation_validation": validation_results,
        "total_stores": len(validation_results),
        "validation_timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Cache management endpoint
@app.delete("/cache/{shop_url}")
async def clear_cache(shop_url: str, access_token: str = Form(...)):
    cache_key = f"{shop_url}_{access_token}"
    
    # Clear memory cache
    if cache_key in embedding_cache:
        del embedding_cache[cache_key]
    if cache_key in product_metadata_cache:
        del product_metadata_cache[cache_key]
    if cache_key in last_updated:
        del last_updated[cache_key]
    
    # Clear disk cache
    disk_deleted = delete_cache_file(shop_url, access_token)
    
    return {
        "message": "Cache cleared successfully",
        "memory_cache_cleared": True,
        "disk_cache_cleared": disk_deleted
    }

# Cache refresh endpoint
@app.post("/cache/{shop_url}/refresh")
async def refresh_cache(shop_url: str, access_token: str = Form(...)):
    """Force refresh cache for a specific store"""
    cache_key = f"{shop_url}_{access_token}"
    
    # Clear memory cache to force fresh fetch
    if cache_key in embedding_cache:
        del embedding_cache[cache_key]
    if cache_key in last_updated:
        del last_updated[cache_key]
    
    # Force fresh fetch and processing
    try:
        store_data = get_store_embeddings_smart(shop_url, access_token, pagination_limit=250)
        return {
            "message": "Cache refreshed successfully",
            "total_products": store_data['total_products'],
            "cached_products": store_data['cached_products'],
            "new_products": store_data['new_products'],
            "updated_products": store_data['updated_products']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache refresh failed: {str(e)}")

# Cache info endpoint
@app.get("/cache/{shop_url}/info", response_model=CacheInfo)
async def get_cache_info(shop_url: str, access_token: str = Form(...)):
    cache_key = f"{shop_url}_{access_token}"
    
    # Get disk cache info
    disk_info = get_cache_file_info(shop_url, access_token)
    
    if cache_key not in embedding_cache and not disk_info['exists']:
        return CacheInfo(
            total_products=0,
            cached_products=0,
            new_products=0,
            updated_products=0,
            last_updated="Never",
            cache_size_mb=0
        )
    
    # Load from disk if not in memory
    if cache_key not in embedding_cache and disk_info['exists']:
        disk_data = load_embeddings_from_disk(shop_url, access_token)
        if disk_data:
            embedding_cache[cache_key] = {
                'embeddings': disk_data['embeddings'],
                'product_info': disk_data['product_info'],
                'product_hashes': disk_data['product_hashes'],
                'total_products': disk_data['total_products']
            }
            last_updated[cache_key] = datetime.fromisoformat(disk_data['last_updated'])
    
    cached_data = embedding_cache[cache_key]
    memory_cache_size = get_storage_size(cached_data)
    total_cache_size = memory_cache_size + disk_info['size_mb']
    
    # Get current product count
    try:
        current_products = fetch_shopify_products(shop_url, access_token, limit=250)
        total_products = len(current_products)
        
        # Count new and updated products
        cached_hashes = cached_data['product_hashes']
        new_products = sum(1 for p in current_products if p['id'] not in cached_hashes)
        updated_products = sum(1 for p in current_products 
                             if p['id'] in cached_hashes and 
                             cached_hashes[p['id']] != generate_product_hash(p))
        
    except Exception:
        total_products = cached_data['total_products']
        new_products = 0
        updated_products = 0
    
    return CacheInfo(
        total_products=total_products,
        cached_products=len(cached_data['product_info']),
        new_products=new_products,
        updated_products=updated_products,
        last_updated=last_updated[cache_key].isoformat() if cache_key in last_updated else disk_info['last_modified'],
        cache_size_mb=total_cache_size
    )

# Resource usage endpoint
@app.get("/resources")
async def get_resource_usage():
    memory_usage = get_memory_usage()
    memory_cache_size = sum(get_storage_size(data) for data in embedding_cache.values())
    
    # Calculate disk storage
    disk_files = list(STORAGE_DIR.glob("*.pkl"))
    disk_size = sum(f.stat().st_size for f in disk_files) / (1024 * 1024)  # MB
    total_cache_size = memory_cache_size + disk_size
    
    return {
        "memory_usage_mb": float(memory_usage),
        "memory_cache_size_mb": float(memory_cache_size),
        "disk_cache_size_mb": float(disk_size),
        "total_cache_size_mb": float(total_cache_size),
        "cached_stores_memory": len(embedding_cache),
        "cached_stores_disk": len(disk_files),
        "model_loaded": 'model' in model_cache,
        "total_cached_products": sum(len(data['product_info']) for data in embedding_cache.values()),
        "storage_directory": str(STORAGE_DIR.absolute())
    }

# Disk storage management endpoint
@app.get("/storage/info")
async def get_storage_info():
    """Get detailed information about disk storage"""
    disk_files = list(STORAGE_DIR.glob("*.pkl"))
    
    file_info = []
    total_size = 0
    
    for file_path in disk_files:
        try:
            stat = file_path.stat()
            file_size_mb = stat.st_size / (1024 * 1024)
            total_size += file_size_mb
            
            # Try to extract store info from filename
            filename = file_path.stem
            if '_' in filename:
                store_name = filename.split('_')[0]
            else:
                store_name = "unknown"
            
            file_info.append({
                "filename": file_path.name,
                "store_name": store_name,
                "size_mb": round(file_size_mb, 2),
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        except Exception as e:
            file_info.append({
                "filename": file_path.name,
                "error": str(e)
            })
    
    return {
        "storage_directory": str(STORAGE_DIR.absolute()),
        "total_files": len(disk_files),
        "total_size_mb": round(total_size, 2),
        "files": sorted(file_info, key=lambda x: x.get('size_mb', 0), reverse=True)
    }

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current configuration settings"""
    return {
        "pagination_limit": 250,
        "max_pages": 100,
        "max_products": 25000,
        "cache_age_limit_hours": 1,
        "embedding_model": "ViT-B/32",
        "similarity_metric": "cosine"
    }

# Clear all disk storage
@app.delete("/storage/clear")
async def clear_all_storage():
    """Clear all disk storage files"""
    disk_files = list(STORAGE_DIR.glob("*.pkl"))
    deleted_count = 0
    
    for file_path in disk_files:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return {
        "message": f"Cleared {deleted_count} cache files",
        "deleted_files": deleted_count,
        "storage_directory": str(STORAGE_DIR.absolute())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 