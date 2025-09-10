#!/bin/bash
# Quick fix for RunPod unzip issue
# Run this script if you encounter "unzip: command not found" error

set -e

echo "🔧 Fixing RunPod unzip issue..."
echo "==============================="

# Install missing tools
echo "📦 Installing essential tools..."
apt update
apt install -y unzip wget curl

echo "✅ Essential tools installed successfully!"
echo ""
echo "You can now continue with the deployment:"
echo "bash scripts/deploy_to_runpod.sh"
echo ""
echo "Note: ADB functionality is provided by the adbutils Python package."
echo "Make sure your local ADB server is running and accessible."
