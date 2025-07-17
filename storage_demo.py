#!/usr/bin/env python3
"""
Local Storage Demo for Shopify Image Search API

This script demonstrates the local storage features of the API server,
including disk caching, cache management, and storage monitoring.
"""

import requests
import json
import time
from client_example import ShopifyImageSearchClient

def demo_local_storage():
    """Demonstrate local storage features"""
    print("🚀 Local Storage Demo for Shopify Image Search API")
    print("=" * 60)
    
    # Initialize client
    client = ShopifyImageSearchClient("http://localhost:8000")
    
    # Check API health
    health = client.get_health()
    if not health:
        print("❌ API server is not running. Please start the server first.")
        return
    
    print("✅ API server is running")
    
    # Example store configuration (replace with your actual credentials)
    STORE_CONFIG = {
        "shop_url": "your-store.myshopify.com",
        "access_token": "your-access-token"
    }
    
    print(f"\n📊 Initial Storage State:")
    print("-" * 40)
    
    # Get initial resource usage
    resources = client.get_resources()
    if resources:
        print(f"Memory Usage: {resources['memory_usage_mb']:.1f} MB")
        print(f"Memory Cache: {resources['memory_cache_size_mb']:.2f} MB")
        print(f"Disk Cache: {resources['disk_cache_size_mb']:.2f} MB")
        print(f"Total Cache: {resources['total_cache_size_mb']:.2f} MB")
        print(f"Stores in Memory: {resources['cached_stores_memory']}")
        print(f"Stores on Disk: {resources['cached_stores_disk']}")
    
    # Get initial storage info
    storage_info = client.get_storage_info()
    if storage_info:
        print(f"Disk Files: {storage_info['total_files']}")
        print(f"Disk Size: {storage_info['total_size_mb']:.2f} MB")
        print(f"Storage Directory: {storage_info['storage_directory']}")
    
    print(f"\n🔍 Cache Information for Store:")
    print("-" * 40)
    
    # Get cache info for the store
    cache_info = client.get_cache_info(STORE_CONFIG['shop_url'], STORE_CONFIG['access_token'])
    if cache_info:
        print(f"Total Products: {cache_info['total_products']}")
        print(f"Cached Products: {cache_info['cached_products']}")
        print(f"New Products: {cache_info['new_products']}")
        print(f"Updated Products: {cache_info['updated_products']}")
        print(f"Cache Size: {cache_info['cache_size_mb']:.2f} MB")
        print(f"Last Updated: {cache_info['last_updated']}")
        
        if cache_info['cached_products'] == 0:
            print("💡 No cached products found. This is normal for a new store.")
    else:
        print("❌ Could not get cache information")
    
    print(f"\n🔄 Simulating Product Search (this will create/update cache):")
    print("-" * 40)
    
    # Note: This is a simulation since we don't have actual credentials
    print("Note: This is a simulation. Replace STORE_CONFIG with your actual credentials to test.")
    print("The search would:")
    print("1. Load existing cache from disk (if available)")
    print("2. Fetch current products from Shopify")
    print("3. Detect new/updated products")
    print("4. Embed only new/updated products")
    print("5. Save updated cache to disk")
    print("6. Return search results")
    
    print(f"\n📁 Detailed Storage Information:")
    print("-" * 40)
    
    # Show detailed storage info
    storage_info = client.get_storage_info()
    if storage_info and storage_info['files']:
        print("Cache Files:")
        for i, file_info in enumerate(storage_info['files'], 1):
            if 'error' not in file_info:
                print(f"  {i}. {file_info['filename']}")
                print(f"     Store: {file_info['store_name']}")
                print(f"     Size: {file_info['size_mb']:.2f} MB")
                print(f"     Modified: {file_info['last_modified']}")
                print(f"     Created: {file_info['created']}")
            else:
                print(f"  {i}. {file_info['filename']} - Error: {file_info['error']}")
    else:
        print("No cache files found on disk")
    
    print(f"\n🧹 Cache Management Options:")
    print("-" * 40)
    print("1. Clear specific store cache:")
    print("   client.clear_cache(shop_url, access_token)")
    print("   - Removes from memory and disk")
    print()
    print("2. Clear all storage:")
    print("   client.clear_all_storage()")
    print("   - Removes all cache files from disk")
    print()
    print("3. Monitor cache efficiency:")
    print("   - Check cache_info before/after searches")
    print("   - Monitor new_products and updated_products")
    print("   - Track cache size growth")
    
    print(f"\n💡 Benefits of Local Storage:")
    print("-" * 40)
    print("✅ Faster startup - no need to re-embed all products")
    print("✅ Reduced API calls - only fetch new/updated products")
    print("✅ Persistent across server restarts")
    print("✅ Automatic change detection")
    print("✅ Store-specific isolation")
    print("✅ Memory and disk caching")
    
    print(f"\n🔧 Storage Configuration:")
    print("-" * 40)
    print("Storage Directory: embeddings_cache/")
    print("File Format: {shop_name}_{token_hash}.pkl")
    print("Data Stored:")
    print("  - Product embeddings (numpy arrays)")
    print("  - Product metadata")
    print("  - Product hashes (for change detection)")
    print("  - Last update timestamp")
    
    print(f"\n📈 Performance Monitoring:")
    print("-" * 40)
    print("Monitor these metrics:")
    print("  - Cache hit rate (cached vs total products)")
    print("  - New products per search")
    print("  - Updated products per search")
    print("  - Cache size growth")
    print("  - Processing time improvements")
    print("  - Memory usage patterns")

def demo_cache_operations():
    """Demonstrate cache operations"""
    print(f"\n🔧 Cache Operations Demo:")
    print("=" * 60)
    
    client = ShopifyImageSearchClient("http://localhost:8000")
    
    # Example store
    STORE_CONFIG = {
        "shop_url": "demo-store.myshopify.com",
        "access_token": "demo-token"
    }
    
    print("1. Getting cache info...")
    cache_info = client.get_cache_info(STORE_CONFIG['shop_url'], STORE_CONFIG['access_token'])
    if cache_info:
        print(f"   Cached products: {cache_info['cached_products']}")
        print(f"   Cache size: {cache_info['cache_size_mb']:.2f} MB")
    
    print("\n2. Clearing cache...")
    clear_result = client.clear_cache(STORE_CONFIG['shop_url'], STORE_CONFIG['access_token'])
    if clear_result:
        print(f"   Memory cache cleared: {clear_result['memory_cache_cleared']}")
        print(f"   Disk cache cleared: {clear_result['disk_cache_cleared']}")
    
    print("\n3. Getting storage info...")
    storage_info = client.get_storage_info()
    if storage_info:
        print(f"   Total files: {storage_info['total_files']}")
        print(f"   Total size: {storage_info['total_size_mb']:.2f} MB")
    
    print("\n4. Clearing all storage...")
    clear_all_result = client.clear_all_storage()
    if clear_all_result:
        print(f"   Deleted files: {clear_all_result['deleted_files']}")
        print(f"   Message: {clear_all_result['message']}")

if __name__ == "__main__":
    try:
        demo_local_storage()
        demo_cache_operations()
        
        print(f"\n🎉 Demo completed!")
        print("To use with your store:")
        print("1. Update STORE_CONFIG with your credentials")
        print("2. Run a search to create initial cache")
        print("3. Monitor cache efficiency over time")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Make sure the API server is running on http://localhost:8000") 