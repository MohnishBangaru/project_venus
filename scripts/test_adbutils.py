#!/usr/bin/env python3
"""
Test script for adbutils integration

This script tests the adbutils integration to ensure it works correctly
with the updated device controller.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import adbutils
from src.automation.device_controller import DeviceController
from config.device_config import DeviceConfig

def test_adbutils_basic():
    """Test basic adbutils functionality."""
    print("🧪 Testing basic adbutils functionality...")
    
    try:
        # Test adbutils client creation
        adb = adbutils.AdbClient()
        print("✅ AdbClient created successfully")
        
        # Test device listing
        devices = adb.device_list()
        print(f"📱 Found {len(devices)} device(s)")
        
        for device in devices:
            print(f"   - Device: {device.serial}")
            try:
                # Try to get device state, fallback if not available
                state = getattr(device, 'state', 'unknown')
                print(f"     State: {state}")
            except:
                print(f"     State: unknown")
        
        return len(devices) > 0
        
    except Exception as e:
        print(f"❌ Basic adbutils test failed: {e}")
        return False

def test_device_controller():
    """Test device controller with adbutils."""
    print("\n🧪 Testing DeviceController with adbutils...")
    
    try:
        # Create device config
        config = DeviceConfig()
        
        # Create device controller
        controller = DeviceController(config)
        print("✅ DeviceController created successfully")
        
        # Test connection
        if controller.connect():
            print("✅ Device connection successful")
            
            # Test device info
            info = controller.get_device_info()
            print(f"📱 Device info: {info.get('model', 'Unknown')} - {info.get('android_version', 'Unknown')}")
            
            # Test device status
            status = controller.get_device_status()
            print(f"📊 Device status: {status}")
            
            # Disconnect
            controller.disconnect()
            print("✅ Device disconnection successful")
            
            return True
        else:
            print("❌ Device connection failed")
            return False
            
    except Exception as e:
        print(f"❌ DeviceController test failed: {e}")
        return False

def test_shell_commands():
    """Test shell command execution."""
    print("\n🧪 Testing shell command execution...")
    
    try:
        adb = adbutils.AdbClient()
        devices = adb.device_list()
        
        if not devices:
            print("⚠️ No devices available for shell command test")
            return False
        
        device = devices[0]
        
        # Test basic shell command
        result = device.shell("echo test")
        print(f"✅ Shell command result: {result}")
        
        # Test device property
        result = device.shell("getprop ro.product.model")
        print(f"📱 Device model: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Shell command test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing adbutils integration")
    print("=" * 50)
    
    tests = [
        test_adbutils_basic,
        test_device_controller,
        test_shell_commands
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! adbutils integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
