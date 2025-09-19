#!/usr/bin/env python3
"""
Shopify API Authentication Test Script

This script helps you test your Shopify API credentials before using the main application.
"""

import shopify
import requests
import json

def test_shopify_credentials(shop_url, access_token):
    """Test Shopify API credentials"""
    print(f"🔍 Testing Shopify API credentials...")
    print(f"Shop URL: {shop_url}")
    print(f"Access Token: {access_token[:10]}...{access_token[-10:] if len(access_token) > 20 else '***'}")
    print("-" * 50)
    
    # Test 1: Basic API connection
    try:
        shopify.ShopifyResource.set_site(f"https://{shop_url}/admin/api/2023-10")
        shopify.ShopifyResource.set_headers({'X-Shopify-Access-Token': access_token})
        
        # Try to get shop info
        shop = shopify.Shop.current()
        print(f"✅ Shop connection successful!")
        print(f"   Shop name: {shop.name}")
        print(f"   Shop domain: {shop.domain}")
        print(f"   Shop email: {shop.email}")
        
    except Exception as e:
        print(f"❌ Shop connection failed: {str(e)}")
        return False
    
    # Test 2: Products API access
    try:
        products = shopify.Product.find(limit=5)
        print(f"✅ Products API access successful!")
        print(f"   Found {len(products)} products (showing first 5)")
        
        for i, product in enumerate(products[:3]):
            print(f"   {i+1}. {product.title} (ID: {product.id})")
            if product.images:
                print(f"      Images: {len(product.images)}")
        
    except Exception as e:
        print(f"❌ Products API access failed: {str(e)}")
        return False
    
    # Test 3: Direct HTTP request
    try:
        url = f"https://{shop_url}/admin/api/2023-10/products.json"
        headers = {
            'X-Shopify-Access-Token': access_token,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Direct HTTP request successful!")
            print(f"   Status code: {response.status_code}")
            print(f"   Products in response: {len(data.get('products', []))}")
        else:
            print(f"❌ Direct HTTP request failed!")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Direct HTTP request failed: {str(e)}")
        return False
    
    print("-" * 50)
    print("🎉 All tests passed! Your credentials are working correctly.")
    return True

def main():
    print("🛍️ Shopify API Authentication Test")
    print("=" * 50)
    
    # Get credentials from user
    shop_url = input("Enter your shop URL (e.g., your-store.myshopify.com): ").strip()
    access_token = input("Enter your Admin API access token: ").strip()
    
    if not shop_url or not access_token:
        print("❌ Please provide both shop URL and access token.")
        return
    
    # Clean up shop URL
    if shop_url.startswith('https://'):
        shop_url = shop_url.replace('https://', '')
    if shop_url.startswith('http://'):
        shop_url = shop_url.replace('http://', '')
    if shop_url.endswith('/'):
        shop_url = shop_url.rstrip('/')
    
    print(f"\n🧪 Testing with cleaned shop URL: {shop_url}")
    
    # Run tests
    success = test_shopify_credentials(shop_url, access_token)
    
    if success:
        print("\n✅ Your credentials are ready to use with the ImageMania app!")
        print("\nNext steps:")
        print("1. Start the API server: python api_server.py")
        print("2. Start the Streamlit app: streamlit run streamlit_app.py")
        print("3. Use these same credentials in the app")
    else:
        print("\n❌ Please check your credentials and try again.")
        print("\nCommon issues:")
        print("- Make sure you created a Private App (not Public)")
        print("- Ensure you clicked 'Install app' after creating it")
        print("- Check that read_products and read_product_listings permissions are enabled")
        print("- Verify the shop URL format (your-store.myshopify.com)")

if __name__ == "__main__":
    main() 