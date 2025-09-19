import requests
import json
from PIL import Image
import io

class ShopifyImageSearchClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def search_products(self, shop_url, access_token, image_path, top_k=5, similarity_threshold=0.1, include_product_details=False):
        """
        Search for similar products in a specific Shopify store
        
        Args:
            shop_url (str): Shopify store URL (e.g., 'your-store.myshopify.com')
            access_token (str): Shopify admin API access token
            image_path (str): Path to the query image
            top_k (int): Number of results to return
            similarity_threshold (float): Minimum similarity score
            include_product_details (bool): Whether to include detailed product information
            
        Returns:
            dict: Search results (only from the specified store)
        """
        try:
            # Prepare the request
            url = f"{self.api_url}/search"
            
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'shop_url': shop_url,
                    'access_token': access_token,
                    'top_k': top_k,
                    'similarity_threshold': similarity_threshold,
                    'include_product_details': include_product_details
                }
                
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    return None
                    
        except Exception as e:
            print(f"Error making request: {str(e)}")
            return None
    
    def search_products_multi_store(self, stores_config, image_path, top_k=5, similarity_threshold=0.1):
        """
        Search for similar products across multiple Shopify stores
        
        Args:
            stores_config (list): List of store configurations
                [{"shop_url": "...", "access_token": "..."}, ...]
            image_path (str): Path to the query image
            top_k (int): Number of results per store
            similarity_threshold (float): Minimum similarity score
            
        Returns:
            dict: Search results from all stores
        """
        try:
            url = f"{self.api_url}/search/multi-store"
            
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'stores': json.dumps(stores_config),
                    'top_k': top_k,
                    'similarity_threshold': similarity_threshold
                }
                
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)
                    return None
                    
        except Exception as e:
            print(f"Error making multi-store request: {str(e)}")
            return None
    
    def get_cache_info(self, shop_url, access_token):
        """Get cache information for a specific store"""
        try:
            url = f"{self.api_url}/cache/{shop_url}/info"
            data = {'access_token': access_token}
            response = requests.get(url, params=data)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"Error getting cache info: {str(e)}")
            return None
    
    def validate_store_isolation(self):
        """Validate that store embeddings are properly isolated"""
        try:
            response = requests.get(f"{self.api_url}/stores/validate")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"Error validating store isolation: {str(e)}")
            return None
    
    def get_health(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.api_url}/health")
            return response.json()
        except Exception as e:
            print(f"Error checking health: {str(e)}")
            return None
    
    def get_resources(self):
        """Get resource usage information"""
        try:
            response = requests.get(f"{self.api_url}/resources")
            return response.json()
        except Exception as e:
            print(f"Error getting resources: {str(e)}")
            return None
    
    def clear_cache(self, shop_url, access_token):
        """Clear cache for a specific store"""
        try:
            url = f"{self.api_url}/cache/{shop_url}"
            data = {'access_token': access_token}
            response = requests.delete(url, data=data)
            return response.json()
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return None
    
    def get_storage_info(self):
        """Get detailed information about disk storage"""
        try:
            response = requests.get(f"{self.api_url}/storage/info")
            return response.json()
        except Exception as e:
            print(f"Error getting storage info: {str(e)}")
            return None
    
    def clear_all_storage(self):
        """Clear all disk storage files"""
        try:
            response = requests.delete(f"{self.api_url}/storage/clear")
            return response.json()
        except Exception as e:
            print(f"Error clearing all storage: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize client
    client = ShopifyImageSearchClient("http://localhost:8000")
    
    # Check API health
    health = client.get_health()
    print("API Health:", health)
    
    # Example credentials (replace with your actual credentials)
    STORE_1 = {
        "shop_url": "store1.myshopify.com",
        "access_token": "your-access-token-1"
    }
    
    STORE_2 = {
        "shop_url": "store2.myshopify.com", 
        "access_token": "your-access-token-2"
    }
    
    # Validate store isolation
    isolation_validation = client.validate_store_isolation()
    if isolation_validation:
        print("\n🔒 Store Isolation Validation:")
        for cache_key, validation in isolation_validation['store_isolation_validation'].items():
            print(f"Store: {validation['store_url']}")
            print(f"  Total Products: {validation['total_products']}")
            print(f"  Valid Products: {validation['valid_products']}")
            print(f"  Invalid Products: {validation['invalid_products']}")
            if validation['invalid_products'] > 0:
                print(f"  ⚠️  WARNING: {validation['invalid_products']} products may not belong to this store!")
    
    # Get cache information for each store
    for store_config in [STORE_1, STORE_2]:
        cache_info = client.get_cache_info(store_config['shop_url'], store_config['access_token'])
        if cache_info:
            print(f"\n📊 Cache Information for {store_config['shop_url']}:")
            print(f"Total Products: {cache_info['total_products']}")
            print(f"Cached Products: {cache_info['cached_products']}")
            print(f"New Products: {cache_info['new_products']}")
            print(f"Updated Products: {cache_info['updated_products']}")
            print(f"Cache Size: {cache_info['cache_size_mb']:.2f} MB")
            print(f"Last Updated: {cache_info['last_updated']}")
    
    # Example single store search (basic)
    # results = client.search_products(
    #     shop_url=STORE_1['shop_url'],
    #     access_token=STORE_1['access_token'],
    #     image_path="path/to/your/image.jpg",
    #     top_k=5
    # )
    
    # Example single store search with detailed product information
    # results_with_details = client.search_products(
    #     shop_url=STORE_1['shop_url'],
    #     access_token=STORE_1['access_token'],
    #     image_path="path/to/your/image.jpg",
    #     top_k=5,
    #     include_product_details=True  # This will fetch complete product details
    # )
    # 
    # if results:
    #     print(f"\n🎯 Search Results from {STORE_1['shop_url']}:")
    #     print(f"Processing Time: {results['processing_time']:.2f}s")
    #     print(f"Total Products: {results['total_products']}")
    #     print(f"Successful Embeddings: {results['successful_embeddings']}")
    #     
    #     # Verify store isolation
    #     for result in results['results']:
    #         if result.get('store_url') != STORE_1['shop_url']:
    #             print(f"⚠️  WARNING: Product from wrong store: {result['store_url']}")
    #     
    #     for i, result in enumerate(results['results']):
    #         print(f"\n{i+1}. {result['title']} (Store: {result['store_url']})")
    #         print(f"   Similarity: {result['best_score']:.3f}")
    #         print(f"   Matching images: {result['matching_images']}")
    #         print(f"   Product URL: {result['product_url']}")
    #         
    #         # If detailed information was requested, show it
    #         if result.get('detailed_info'):
    #             detailed = result['detailed_info']
    #             print(f"   Status: {detailed.get('status')}")
    #             print(f"   Variants: {len(detailed.get('variants', []))}")
    #             print(f"   Images: {len(detailed.get('images', []))}")
    #             print(f"   Tags: {', '.join(detailed.get('tags', []))}")
    #             
    #             # Show variant pricing
    #             for variant in detailed.get('variants', [])[:2]:  # Show first 2 variants
    #                 print(f"     Variant: {variant.get('title')} - ${variant.get('price')}")
    
    # Example multi-store search
    # multi_results = client.search_products_multi_store(
    #     stores_config=[STORE_1, STORE_2],
    #     image_path="path/to/your/image.jpg",
    #     top_k=3
    # )
    # 
    # if multi_results:
    #     print(f"\n🌐 Multi-Store Search Results:")
    #     print(f"Processing Time: {multi_results['processing_time']:.2f}s")
    #     print(f"Stores Processed: {multi_results['stores_processed']}")
    #     print(f"Total Results: {multi_results['total_results']}")
    #     
    #     # Group results by store
    #     results_by_store = {}
    #     for result in multi_results['results']:
    #         store_url = result['store_url']
    #         if store_url not in results_by_store:
    #             results_by_store[store_url] = []
    #         results_by_store[store_url].append(result)
    #     
    #     for store_url, store_results in results_by_store.items():
    #         print(f"\n📦 Results from {store_url}:")
    #         for i, result in enumerate(store_results):
    #             print(f"  {i+1}. {result['title']} (Score: {result['best_score']:.3f})")
    
    # Get resource usage
    resources = client.get_resources()
    if resources:
        print("\n💾 Resource Usage:")
        print(f"Memory Usage: {resources['memory_usage_mb']:.1f} MB")
        print(f"Memory Cache Size: {resources['memory_cache_size_mb']:.2f} MB")
        print(f"Disk Cache Size: {resources['disk_cache_size_mb']:.2f} MB")
        print(f"Total Cache Size: {resources['total_cache_size_mb']:.2f} MB")
        print(f"Cached Stores (Memory): {resources['cached_stores_memory']}")
        print(f"Cached Stores (Disk): {resources['cached_stores_disk']}")
        print(f"Total Cached Products: {resources['total_cached_products']}")
        print(f"Model Loaded: {resources['model_loaded']}")
        print(f"Storage Directory: {resources['storage_directory']}")
    
    # Get storage information
    storage_info = client.get_storage_info()
    if storage_info:
        print(f"\n📁 Disk Storage Information:")
        print(f"Storage Directory: {storage_info['storage_directory']}")
        print(f"Total Files: {storage_info['total_files']}")
        print(f"Total Size: {storage_info['total_size_mb']:.2f} MB")
        
        if storage_info['files']:
            print("\n📋 Cache Files:")
            for file_info in storage_info['files'][:5]:  # Show top 5 files
                if 'error' not in file_info:
                    print(f"  {file_info['filename']} ({file_info['store_name']})")
                    print(f"    Size: {file_info['size_mb']:.2f} MB")
                    print(f"    Modified: {file_info['last_modified']}")
                else:
                    print(f"  {file_info['filename']} - Error: {file_info['error']}")

# Example monitoring script for store isolation
def monitor_store_isolation():
    """Monitor store isolation and cache efficiency"""
    client = ShopifyImageSearchClient("http://localhost:8000")
    
    STORES = [
        {"shop_url": "store1.myshopify.com", "access_token": "token1"},
        {"shop_url": "store2.myshopify.com", "access_token": "token2"}
    ]
    
    print("🔒 Monitoring Store Isolation...")
    
    # Validate isolation
    isolation = client.validate_store_isolation()
    if isolation:
        for cache_key, validation in isolation['store_isolation_validation'].items():
            if validation['invalid_products'] > 0:
                print(f"⚠️  Store {validation['store_url']} has {validation['invalid_products']} invalid products!")
            else:
                print(f"✅ Store {validation['store_url']} is properly isolated")
    
    # Test search for each store
    for store_config in STORES:
        print(f"\n🔍 Testing search for {store_config['shop_url']}...")
        
        # Get cache info before search
        cache_before = client.get_cache_info(store_config['shop_url'], store_config['access_token'])
        
        # Perform search
        results = client.search_products(
            shop_url=store_config['shop_url'],
            access_token=store_config['access_token'],
            image_path="test_image.jpg",
            top_k=3
        )
        
        if results:
            # Verify all results belong to this store
            store_url = store_config['shop_url']
            correct_store_results = [r for r in results['results'] if r.get('store_url') == store_url]
            
            if len(correct_store_results) == len(results['results']):
                print(f"✅ All results belong to {store_url}")
            else:
                print(f"⚠️  {len(results['results']) - len(correct_store_results)} results from wrong store!")
        
        # Get cache info after search
        cache_after = client.get_cache_info(store_config['shop_url'], store_config['access_token'])
        
        if cache_before and cache_after:
            new_products = cache_after['cached_products'] - cache_before['cached_products']
            print(f"📈 New products cached: {new_products}")

if __name__ == "__main__":
    # Run basic example
    pass
    
    # Uncomment to run store isolation monitoring
    # monitor_store_isolation() 