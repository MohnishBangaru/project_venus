#!/usr/bin/env python3
"""
Quick Emulator Test

This script provides a quick test to verify your Android emulator
is properly connected and ready for crawling.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.automation.device_controller import DeviceController
from config import DeviceConfig


def check_adb():
    """Check if ADB is available."""
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ADB is available")
            return True
        else:
            print("❌ ADB not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ADB not found in PATH")
        print("💡 Please install Android SDK and add ADB to your PATH")
        return False


def check_devices():
    """Check for connected Android devices."""
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split('\n')[1:]  # Skip header
            
            devices = []
            for line in lines:
                if line.strip() and '\tdevice' in line:
                    device_id = line.split('\t')[0]
                    devices.append(device_id)
            
            if devices:
                print(f"✅ Found {len(devices)} device(s): {', '.join(devices)}")
                return devices
            else:
                print("❌ No Android devices found")
                print("💡 Make sure your emulator is running and fully booted")
                return []
        else:
            print("❌ Failed to list devices")
            return []
    except subprocess.TimeoutExpired:
        print("❌ ADB devices command timed out")
        return []


def test_device_connection():
    """Test device controller connection."""
    try:
        print("📱 Testing device controller...")
        
        device_config = DeviceConfig(
            device_id=None,  # Auto-detect
            connection_timeout=10,
            enable_debug_logging=False
        )
        
        device_controller = DeviceController(device_config)
        
        if device_controller.connect():
            print("✅ Device controller connected successfully")
            
            # Get device info
            device_info = device_controller.get_device_info()
            model = device_info.get('model', 'Unknown')
            android_version = device_info.get('android_version', 'Unknown')
            print(f"📊 Device: {model} - Android {android_version}")
            
            device_controller.cleanup()
            return True
        else:
            print("❌ Device controller connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Device controller test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Android Emulator Quick Test")
    print("=" * 40)
    
    # Check ADB
    if not check_adb():
        return False
    
    # Check devices
    devices = check_devices()
    if not devices:
        return False
    
    # Test device controller
    if not test_device_connection():
        return False
    
    print("=" * 40)
    print("🎉 All tests passed! Your emulator is ready for crawling.")
    print("🚀 You can now run: python scripts/local_crawler.py")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
