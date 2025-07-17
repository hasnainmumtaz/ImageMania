#!/usr/bin/env python3
"""
Startup script for Shopify Image Search Application

This script provides options to start either:
1. The FastAPI server only
2. The Streamlit app only  
3. Both the FastAPI server and Streamlit app
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'fastapi', 'uvicorn', 'requests', 'pillow', 
        'torch', 'clip', 'scikit-learn', 'shopify', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server started successfully
        if process.poll() is None:
            print("✅ FastAPI server is running on http://localhost:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start API server:")
            print(stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")
        return None

def start_streamlit_app():
    """Start the Streamlit app"""
    print("🌐 Starting Streamlit app...")
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for app to start
        time.sleep(5)
        
        # Check if app started successfully
        if process.poll() is None:
            print("✅ Streamlit app is running on http://localhost:8501")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start Streamlit app:")
            print(stderr)
            return None
            
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {str(e)}")
        return None

def wait_for_api_server():
    """Wait for API server to be ready"""
    import requests
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ API server is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(f"⏳ Waiting for API server... ({attempt + 1}/{max_attempts})")
    
    print("❌ API server did not start within expected time")
    return False

def main():
    print("🛍️ Shopify Image Search Application")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\nChoose an option:")
    print("1. Start FastAPI server only")
    print("2. Start Streamlit app only")
    print("3. Start both (recommended)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                # Start API server only
                api_process = start_api_server()
                if api_process:
                    print("\n📋 API Server Information:")
                    print("   - URL: http://localhost:8000")
                    print("   - Health check: http://localhost:8000/health")
                    print("   - API docs: http://localhost:8000/docs")
                    print("\nPress Ctrl+C to stop the server")
                    
                    try:
                        api_process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping API server...")
                        api_process.terminate()
                        api_process.wait()
                        print("✅ API server stopped")
                break
                
            elif choice == "2":
                # Start Streamlit app only
                print("⚠️  Note: You need to start the API server separately first")
                print("   Run: python api_server.py")
                print("   Then start this Streamlit app")
                
                streamlit_process = start_streamlit_app()
                if streamlit_process:
                    print("\n📋 Streamlit App Information:")
                    print("   - URL: http://localhost:8501")
                    print("   - Make sure API server is running on http://localhost:8000")
                    print("\nPress Ctrl+C to stop the app")
                    
                    try:
                        streamlit_process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping Streamlit app...")
                        streamlit_process.terminate()
                        streamlit_process.wait()
                        print("✅ Streamlit app stopped")
                break
                
            elif choice == "3":
                # Start both
                print("\n🚀 Starting both API server and Streamlit app...")
                
                # Start API server first
                api_process = start_api_server()
                if not api_process:
                    print("❌ Failed to start API server. Exiting.")
                    return
                
                # Wait for API server to be ready
                if not wait_for_api_server():
                    print("❌ API server is not responding. Exiting.")
                    api_process.terminate()
                    return
                
                # Start Streamlit app
                streamlit_process = start_streamlit_app()
                if not streamlit_process:
                    print("❌ Failed to start Streamlit app. Stopping API server.")
                    api_process.terminate()
                    return
                
                print("\n🎉 Both services are running!")
                print("\n📋 Application Information:")
                print("   - API Server: http://localhost:8000")
                print("   - API Docs: http://localhost:8000/docs")
                print("   - Streamlit App: http://localhost:8501")
                print("   - Health Check: http://localhost:8000/health")
                
                print("\n📖 Usage:")
                print("   1. Open http://localhost:8501 in your browser")
                print("   2. Enter your Shopify credentials")
                print("   3. Upload an image to search for similar products")
                
                print("\nPress Ctrl+C to stop both services")
                
                try:
                    # Wait for either process to finish
                    while api_process.poll() is None and streamlit_process.poll() is None:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n🛑 Stopping services...")
                    api_process.terminate()
                    streamlit_process.terminate()
                    api_process.wait()
                    streamlit_process.wait()
                    print("✅ All services stopped")
                break
                
            elif choice == "4":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 