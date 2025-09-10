#!/bin/bash
# RunPod Deployment Script
# This script helps set up the UI-Venus Mobile Crawler on RunPod

set -e

echo "🚀 UI-Venus Mobile Crawler - RunPod Deployment"
echo "=============================================="

# Check if running on RunPod
if [[ -d "/workspace" ]]; then
    echo "✅ Detected RunPod environment"
    WORKSPACE="/workspace"
else
    echo "⚠️ Not running on RunPod, using current directory"
    WORKSPACE="."
fi

# Update system and install required tools
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install essential tools
echo "🔧 Installing essential tools..."
apt install -y unzip wget curl

# Note: ADB functionality provided by adbutils Python package
echo "📱 ADB functionality will be provided by adbutils Python package"

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

# Set up workspace
echo "📁 Setting up workspace..."
mkdir -p $WORKSPACE/crawl_results
mkdir -p $WORKSPACE/ui_venus_cache

# Test setup
echo "🧪 Testing setup..."
python scripts/test_runpod_setup.py

# Test adbutils integration
echo "🧪 Testing adbutils integration..."
python scripts/test_adbutils.py

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Set your Hugging Face token: export HUGGINGFACE_HUB_TOKEN='your_token'"
echo "2. On your LOCAL machine, start ADB server: adb start-server"
echo "3. On your LOCAL machine, enable TCP/IP: adb tcpip 5555"
echo "4. Update the crawler config with your local IP address"
echo "5. Run the crawler: python scripts/runpod_crawler.py"
echo ""
echo "Note: adbutils will handle the ADB connection automatically"
echo "For detailed instructions, see RUNPOD_SETUP.md"
