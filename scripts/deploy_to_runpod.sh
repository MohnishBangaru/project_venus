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

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Android SDK
echo "📱 Installing Android SDK..."
if ! command -v adb &> /dev/null; then
    echo "Installing Android SDK..."
    
    # Download and install command line tools
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
    unzip -q commandlinetools-linux-11076708_latest.zip
    mkdir -p /opt/android-sdk/cmdline-tools/latest
    mv cmdline-tools/* /opt/android-sdk/cmdline-tools/latest/
    rm -rf cmdline-tools commandlinetools-linux-11076708_latest.zip
    
    # Add to PATH
    echo 'export ANDROID_HOME=/opt/android-sdk' >> ~/.bashrc
    echo 'export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools' >> ~/.bashrc
    source ~/.bashrc
    
    # Install platform tools
    yes | /opt/android-sdk/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-34"
    
    echo "✅ Android SDK installed"
else
    echo "✅ Android SDK already installed"
fi

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

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Set your Hugging Face token: export HUGGINGFACE_HUB_TOKEN='your_token'"
echo "2. Connect your emulator via ADB"
echo "3. Run the crawler: python scripts/runpod_crawler.py"
echo ""
echo "For detailed instructions, see RUNPOD_SETUP.md"
