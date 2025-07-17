# 🚀 Deployment Guide - Shopify Image Search API

This guide covers multiple deployment options for the Shopify Image Search API.

## 📋 Prerequisites

- Python 3.9+ (for local development)
- Shopify store with API access

## 🚀 Quick Deployment Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python api_server.py

# Test with client
python client_example.py
```

## 📞 Support

For deployment issues:
1. Verify health endpoint: `curl http://localhost:8000/health`
2. Check resource usage: `curl http://localhost:8000/resources`
3. Review the API documentation: `http://localhost:8000/docs` 