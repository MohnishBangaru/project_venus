#!/usr/bin/env python3
"""
ADB Utils Testing Script

This script tests the adbutils integration for ADB device control,
including device listing, communication, and basic operations.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_adbutils_import():
    """Test adbutils import."""
    logger.info("🔍 Testing adbutils import...")
    
    try:
        import adbutils
        logger.info("✅ adbutils imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import adbutils: {e}")
        return False


def test_adb_client_creation():
    """Test ADB client creation."""
    logger.info("🔍 Testing ADB client creation...")
    
    try:
        import adbutils
        adb = adbutils.AdbClient()
        logger.info("✅ ADB client created successfully")
        return adb
    except Exception as e:
        logger.error(f"❌ Failed to create ADB client: {e}")
        return None


def test_device_listing(adb):
    """Test device listing."""
    logger.info("🔍 Testing device listing...")
    
    try:
        devices = adb.device_list()
        logger.info(f"✅ Found {len(devices)} device(s)")
        
        for i, device in enumerate(devices):
            logger.info(f"  Device {i+1}: {device.serial}")
        
        return devices
    except Exception as e:
        logger.error(f"❌ Failed to list devices: {e}")
        return []


def test_device_communication(devices):
    """Test device communication."""
    logger.info("🔍 Testing device communication...")
    
    if not devices:
        logger.warning("⚠️ No devices available for communication test")
        return False
    
    try:
        device = devices[0]
        logger.info(f"Testing communication with device: {device.serial}")
        
        # Test basic shell command
        result = device.shell("echo 'Hello from adbutils'")
        logger.info(f"✅ Shell command result: {result.strip()}")
        
        # Test device info
        device_info = device.shell("getprop ro.build.version.release")
        logger.info(f"✅ Android version: {device_info.strip()}")
        
        # Test device model
        device_model = device.shell("getprop ro.product.model")
        logger.info(f"✅ Device model: {device_model.strip()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Device communication failed: {e}")
        return False


def test_screenshot_capability(devices):
    """Test screenshot capability."""
    logger.info("🔍 Testing screenshot capability...")
    
    if not devices:
        logger.warning("⚠️ No devices available for screenshot test")
        return False
    
    try:
        device = devices[0]
        logger.info(f"Testing screenshot with device: {device.serial}")
        
        # Take screenshot
        screenshot = device.screenshot()
        logger.info(f"✅ Screenshot taken: {screenshot.size}")
        
        # Save screenshot for testing
        test_screenshot_path = "test_screenshot.png"
        screenshot.save(test_screenshot_path)
        logger.info(f"✅ Screenshot saved to: {test_screenshot_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Screenshot test failed: {e}")
        return False


def test_touch_operations(devices):
    """Test touch operations."""
    logger.info("🔍 Testing touch operations...")
    
    if not devices:
        logger.warning("⚠️ No devices available for touch test")
        return False
    
    try:
        device = devices[0]
        logger.info(f"Testing touch operations with device: {device.serial}")
        
        # Get screen size
        screen_size = device.window_size()
        logger.info(f"✅ Screen size: {screen_size}")
        
        # Test tap in center of screen
        center_x = screen_size[0] // 2
        center_y = screen_size[1] // 2
        
        device.click(center_x, center_y)
        logger.info(f"✅ Tap test successful at ({center_x}, {center_y})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Touch operations test failed: {e}")
        return False


def run_adbutils_tests():
    """Run all adbutils tests."""
    logger.info("🚀 Starting adbutils integration tests...")
    logger.info("=" * 50)
    
    # Test 1: Import
    if not test_adbutils_import():
        logger.error("❌ Cannot proceed without adbutils")
        return False
    
    # Test 2: Client creation
    adb = test_adb_client_creation()
    if not adb:
        logger.error("❌ Cannot proceed without ADB client")
        return False
    
    # Test 3: Device listing
    devices = test_device_listing(adb)
    
    # Test 4: Device communication
    comm_success = test_device_communication(devices)
    
    # Test 5: Screenshot capability
    screenshot_success = test_screenshot_capability(devices)
    
    # Test 6: Touch operations
    touch_success = test_touch_operations(devices)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 ADB Utils Test Results:")
    logger.info("=" * 50)
    
    tests = [
        ("Import", True),
        ("Client Creation", adb is not None),
        ("Device Listing", len(devices) > 0),
        ("Device Communication", comm_success),
        ("Screenshot Capability", screenshot_success),
        ("Touch Operations", touch_success)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All adbutils tests passed!")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_adbutils_tests()
    sys.exit(0 if success else 1)