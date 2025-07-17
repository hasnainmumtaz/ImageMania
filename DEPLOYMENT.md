# 🚀 Deployment Guide - Shopify Image Search API

This guide covers multiple deployment options for the Shopify Image Search API.

## 📋 Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for local development)
- Shopify store with API access

## 🐳 Docker Deployment (Recommended)

### 1. Quick Start with Docker Compose

```bash
# Clone the repository
git clone <your-repo-url>
cd ImageMania

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### 2. Manual Docker Build

```bash
# Build the image
docker build -t shopify-image-search .

# Run the container
docker run -d -p 8000:8000 --name shopify-search-api shopify-image-search

# Check logs
docker logs shopify-search-api
```

## ☁️ Cloud Platform Deployments

### AWS ECS/Fargate

1. **Create ECR Repository:**
```bash
aws ecr create-repository --repository-name shopify-image-search
```

2. **Build and Push Image:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag shopify-image-search:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/shopify-image-search:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/shopify-image-search:latest
```

3. **Create ECS Task Definition:**
```json
{
  "family": "shopify-image-search",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "shopify-search-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/shopify-image-search:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/shopify-image-search",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run

1. **Enable APIs:**
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

2. **Build and Deploy:**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/<project-id>/shopify-image-search

# Deploy to Cloud Run
gcloud run deploy shopify-image-search \
  --image gcr.io/<project-id>/shopify-image-search \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure Container Instances

1. **Build and Push to Azure Container Registry:**
```bash
# Login to Azure
az login

# Create ACR
az acr create --resource-group myResourceGroup --name myacr --sku Basic

# Build and push
az acr build --registry myacr --image shopify-image-search .

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name shopify-search-api \
  --image myacr.azurecr.io/shopify-image-search:latest \
  --dns-name-label shopify-search-api \
  --ports 8000 \
  --memory 2 \
  --cpu 2
```

## 🐙 Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace shopify-search
```

### 2. Deploy Application
```bash
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-ingress.yaml
```

### 3. Check Deployment
```bash
kubectl get pods -n shopify-search
kubectl get services -n shopify-search
kubectl logs -f deployment/shopify-image-search -n shopify-search
```

## 🔧 Environment Configuration

### Environment Variables

Create a `.env` file for local development:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Resource Limits
MAX_WORKERS=10
MEMORY_LIMIT_MB=2048

# Cache Configuration
CACHE_TTL_HOURS=1
MAX_CACHE_SIZE_MB=1000

# Shopify Configuration
SHOPIFY_API_VERSION=2023-10
```

### Production Configuration

For production, set these environment variables:

```bash
# Security
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
API_KEY_REQUIRED=true

# Performance
MAX_WORKERS=20
MEMORY_LIMIT_MB=4096

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

## 📊 Monitoring and Scaling

### Health Checks

The API provides health check endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Resource usage
curl http://localhost:8000/resources

# API documentation
curl http://localhost:8000/docs
```

### Auto-scaling Configuration

For Kubernetes, add HorizontalPodAutoscaler:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: shopify-search-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: shopify-image-search
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔒 Security Considerations

### 1. API Authentication

Add API key authentication:

```python
# In api_server.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-api-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

### 2. Rate Limiting

Implement rate limiting:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/search")
@limiter.limit("10/minute")
async def search_products(request: Request, ...):
    # Your search logic
```

### 3. CORS Configuration

Update CORS for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## 💰 Cost Optimization

### Resource Optimization

1. **Memory Management:**
   - Set appropriate memory limits
   - Implement cache eviction policies
   - Monitor memory usage

2. **CPU Optimization:**
   - Use appropriate CPU limits
   - Implement connection pooling
   - Optimize image processing

3. **Storage Costs:**
   - Implement cache TTL
   - Use compressed storage
   - Monitor cache size

### Cost Estimation

| Scale | Requests/Month | Estimated Cost |
|-------|----------------|----------------|
| Small | 1,000 | $30-60 |
| Medium | 10,000 | $120-240 |
| Large | 100,000 | $600-1200 |

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

### Production Deployment
```bash
# Docker Compose
docker-compose up -d

# Kubernetes
kubectl apply -f k8s/

# Cloud Run
gcloud run deploy shopify-image-search --source .
```

## 📞 Support

For deployment issues:
1. Check the logs: `docker-compose logs api`
2. Verify health endpoint: `curl http://localhost:8000/health`
3. Check resource usage: `curl http://localhost:8000/resources`
4. Review the API documentation: `http://localhost:8000/docs` 