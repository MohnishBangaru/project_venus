#!/usr/bin/env python3
"""
RunPod Setup Test

This script tests the RunPod environment setup including:
- GPU availability and CUDA
- ADB connection
- UI-Venus model loading
- Device connectivity
"""

import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_gpu_availability():
    """Test GPU and CUDA availability."""
    logger.info("🔍 Testing GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"✅ GPU available: {gpu_name}")
            logger.info(f"✅ GPU count: {gpu_count}")
            logger.info(f"✅ GPU memory: {gpu_memory:.1f} GB")
            return True
        else:
            logger.warning("⚠️ CUDA not available, will use CPU")
            return False
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False


def test_adb_connection():
    """Test ADB connection."""
    logger.info("🔍 Testing ADB connection...")
    
    try:
        # Check ADB version
        result = subprocess.run(["adb", "version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ ADB is available")
        else:
            logger.error("❌ ADB not working properly")
            return False
        
        # Check for devices
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            devices = []
            for line in result.stdout.strip().split('\n')[1:]:
                if line.strip() and '\tdevice' in line:
                    device_id = line.split('\t')[0]
                    devices.append(device_id)
            
            if devices:
                logger.info(f"✅ Found {len(devices)} device(s): {', '.join(devices)}")
                return True
            else:
                logger.warning("⚠️ No devices found")
                logger.info("💡 Make sure your emulator is running and connected")
                return False
        else:
            logger.error("❌ Failed to list devices")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ ADB commands timed out")
        return False
    except FileNotFoundError:
        logger.error("❌ ADB not found in PATH")
        return False
    except Exception as e:
        logger.error(f"❌ ADB test failed: {e}")
        return False


def test_huggingface_token():
    """Test Hugging Face token."""
    logger.info("🔍 Testing Hugging Face token...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        logger.info(f"✅ HF token valid for user: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"❌ HF token test failed: {e}")
        logger.info("💡 Set your token with: export HUGGINGFACE_HUB_TOKEN='your_token'")
        return False


def test_ui_venus_model():
    """Test UI-Venus model availability."""
    logger.info("🔍 Testing UI-Venus model availability...")
    
    try:
        from huggingface_hub import list_models
        models = list(list_models(filter="inclusionAI/UI-Venus-Ground-7B"))
        
        if models:
            logger.info("✅ UI-Venus model found on Hugging Face")
            return True
        else:
            logger.error("❌ UI-Venus model not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ UI-Venus model test failed: {e}")
        return False


def test_workspace_setup():
    """Test workspace setup."""
    logger.info("🔍 Testing workspace setup...")
    
    try:
        # Check workspace directory
        workspace = Path("/workspace")
        if workspace.exists():
            logger.info("✅ Workspace directory exists")
        else:
            logger.warning("⚠️ Workspace directory not found, using current directory")
            workspace = Path(".")
        
        # Check write permissions
        test_file = workspace / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        logger.info("✅ Write permissions OK")
        
        # Check available space
        import shutil
        total, used, free = shutil.disk_usage(workspace)
        free_gb = free / 1024**3
        logger.info(f"✅ Available space: {free_gb:.1f} GB")
        
        if free_gb < 20:
            logger.warning("⚠️ Low disk space, UI-Venus model requires ~15GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workspace test failed: {e}")
        return False


def test_imports():
    """Test required imports."""
    logger.info("🔍 Testing required imports...")
    
    try:
        # Test core imports
        from src.crawler import CrawlerEngine
        from src.automation.device_controller import DeviceController
        from src.automation.action_executor import ActionExecutor
        from src.automation.screenshot_manager import ScreenshotManager
        from config import CrawlerConfig, UIVenusConfig, DeviceConfig
        
        logger.info("✅ All core imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting RunPod Setup Tests")
    logger.info("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("ADB Connection", test_adb_connection),
        ("Hugging Face Token", test_huggingface_token),
        ("UI-Venus Model", test_ui_venus_model),
        ("Workspace Setup", test_workspace_setup),
        ("Required Imports", test_imports)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RunPod setup is ready.")
        logger.info("🚀 You can now run: python scripts/runpod_crawler.py")
    else:
        logger.warning("⚠️ Some tests failed. Please check the issues above.")
        logger.info("📖 See RUNPOD_SETUP.md for detailed setup instructions")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
