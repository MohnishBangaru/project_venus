#!/usr/bin/env python3
"""
RunPod Setup Testing Script

This script tests the RunPod environment setup for the UI-Venus Mobile Crawler,
including GPU availability, ADB connection, and environment configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.device_config import DeviceConfig
from config.crawler_config import CrawlerConfig
from config.ui_venus_config import UIVenusConfig
from config import ProjectConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_runpod_environment():
    """Test if running in RunPod environment."""
    logger.info("🔍 Testing RunPod environment detection...")
    
    # Check for RunPod-specific directories
    runpod_indicators = [
        "/workspace",
        "/runpod",
        "/opt/runpod"
    ]
    
    found_indicators = []
    for indicator in runpod_indicators:
        if os.path.exists(indicator):
            found_indicators.append(indicator)
    
    if found_indicators:
        logger.info(f"✅ RunPod environment detected. Found: {', '.join(found_indicators)}")
        return True
    else:
        logger.warning("⚠️ RunPod environment not detected. Running in local mode.")
        return False


def test_gpu_availability():
    """Test GPU availability and CUDA setup."""
    logger.info("🔍 Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"✅ CUDA available with {gpu_count} GPU(s)")
            logger.info(f"✅ Current device: {device_name}")
            logger.info(f"✅ CUDA version: {torch.version.cuda}")
            
            # Test GPU memory
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            logger.info(f"✅ GPU memory: {gpu_memory_gb:.2f} GB")
            
            return True
        else:
            logger.warning("⚠️ CUDA not available. GPU acceleration disabled.")
            return False
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False


def test_adb_connection():
    """Test ADB connection using adbutils."""
    logger.info("🔍 Testing ADB connection with adbutils...")
    
    try:
        import adbutils
        
        # Create adbutils client
        adb = adbutils.AdbClient()
        logger.info("✅ adbutils client created successfully")
        
        # Check for devices
        devices = adb.device_list()
        
        if devices:
            device_list = [device.serial for device in devices]
            logger.info(f"✅ Found {len(devices)} device(s): {', '.join(device_list)}")
            
            # Test basic communication with first device
            if devices:
                device = devices[0]
                try:
                    result = device.shell("echo test")
                    logger.info(f"✅ Device communication test successful: {result}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Device communication failed: {e}")
                    return False
        else:
            logger.warning("⚠️ No devices found")
            logger.info("💡 Make sure your emulator is running and connected")
            logger.info("💡 For RunPod, ensure your local ADB server is accessible")
            return False
            
    except ImportError:
        logger.error("❌ adbutils not installed")
        return False
    except Exception as e:
        logger.error(f"❌ ADB test failed: {e}")
        return False


def test_huggingface_token():
    """Test Hugging Face token."""
    logger.info("🔍 Testing Hugging Face token...")
    
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        logger.info("✅ Hugging Face token found")
        return True
    else:
        logger.warning("⚠️ Hugging Face token not found")
        logger.info("💡 Set HUGGINGFACE_HUB_TOKEN environment variable")
        return False


def test_workspace_setup():
    """Test workspace setup and permissions."""
    logger.info("🔍 Testing workspace setup...")
    
    try:
        # Check workspace directory
        workspace_path = "/workspace" if os.path.exists("/workspace") else "."
        logger.info(f"✅ Workspace path: {workspace_path}")
        
        # Test write permissions
        test_file = Path(workspace_path) / "test_write_permission.txt"
        test_file.write_text("test")
        test_file.unlink()
        logger.info("✅ Write permissions confirmed")
        
        # Check required directories
        required_dirs = ["crawl_results", "ui_venus_cache"]
        for dir_name in required_dirs:
            dir_path = Path(workspace_path) / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Directory created/verified: {dir_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workspace setup failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration loading."""
    logger.info("🔍 Testing configuration loading...")
    
    try:
        # Test device configuration
        device_config = DeviceConfig()
        logger.info("✅ Device configuration loaded")
        
        # Test RunPod-specific settings
        device_config.is_runpod_environment = True
        device_config.enable_remote_adb = True
        device_config.remote_adb_host = "localhost"
        device_config.remote_adb_port = 5037
        
        logger.info("✅ RunPod device configuration set")
        
        # Test crawler configuration
        crawler_config = CrawlerConfig()
        logger.info("✅ Crawler configuration loaded")
        
        # Test UI-Venus configuration
        ui_venus_config = UIVenusConfig()
        logger.info("✅ UI-Venus configuration loaded")
        
        # Test project configuration
        project_config = ProjectConfig(
            device=device_config,
            crawler=crawler_config,
            ui_venus=ui_venus_config
        )
        logger.info("✅ Project configuration loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False


def test_dependencies():
    """Test required dependencies."""
    logger.info("🔍 Testing dependencies...")
    
    required_packages = [
        "torch",
        "transformers",
        "accelerate",
        "adbutils",
        "opencv-python",
        "Pillow",
        "numpy",
        "pydantic",
        "click",
        "rich"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} not available")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        logger.info("✅ All required packages available")
        return True


def run_all_tests():
    """Run all tests and provide summary."""
    logger.info("🚀 Starting RunPod setup tests...")
    logger.info("=" * 50)
    
    tests = [
        ("RunPod Environment", test_runpod_environment),
        ("GPU Availability", test_gpu_availability),
        ("ADB Connection", test_adb_connection),
        ("Hugging Face Token", test_huggingface_token),
        ("Workspace Setup", test_workspace_setup),
        ("Configuration Loading", test_configuration_loading),
        ("Dependencies", test_dependencies)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    logger.info("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RunPod setup is ready.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Please check the setup.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)