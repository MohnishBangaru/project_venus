#!/usr/bin/env python3
"""
Simple Test Script for Mac

This script runs basic tests without requiring pytest.
"""

import sys
from pathlib import Path

# Add src and config to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_import():
    """Test configuration import."""
    try:
        from config import UIVenusConfig, ProjectConfig
        print("✅ Configuration import successful")
        return True
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    try:
        from config import UIVenusConfig
        
        config = UIVenusConfig(
            model_name="test-model",
            device="cpu",
            max_tokens=256
        )
        
        print(f"✅ Configuration created: {config.model_name}")
        return True
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False

def test_ui_venus_import():
    """Test UI-Venus module import."""
    try:
        from src.ui_venus import UIVenusModelClient, UIVenusElementDetector, UIVenusActionSuggester
        print("✅ UI-Venus module import successful")
        return True
    except Exception as e:
        print(f"❌ UI-Venus module import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Simple Tests on Mac")
    print("=" * 40)
    
    tests = [
        test_config_import,
        test_config_creation,
        test_ui_venus_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
