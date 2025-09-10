#!/usr/bin/env python3
"""
Configuration System Demo

This script demonstrates how to use the configuration system
for the UI-Venus Mobile Crawler.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ProjectConfig, UIVenusConfig, CrawlerConfig, DeviceConfig


def demo_basic_config():
    """Demonstrate basic configuration usage."""
    print("🔧 Basic Configuration Demo")
    print("=" * 50)
    
    # Create default configuration
    config = ProjectConfig.load_default()
    
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Debug Mode: {config.debug}")
    print(f"Log Level: {config.log_level}")
    print()
    
    # UI-Venus configuration
    print("🤖 UI-Venus Configuration:")
    print(f"  Model: {config.ui_venus.model_name}")
    print(f"  Device: {config.ui_venus.device}")
    print(f"  Max Tokens: {config.ui_venus.max_tokens}")
    print(f"  Temperature: {config.ui_venus.temperature}")
    print(f"  Image Size: {config.ui_venus.image_size}")
    print()
    
    # Crawler configuration
    print("🕷️ Crawler Configuration:")
    print(f"  Strategy: {config.crawler.strategy}")
    print(f"  Max Depth: {config.crawler.max_depth}")
    print(f"  Max Actions: {config.crawler.max_actions}")
    print(f"  Coverage Threshold: {config.crawler.coverage_threshold}")
    print(f"  Action Delay: {config.crawler.action_delay_ms}ms")
    print()
    
    # Device configuration
    print("📱 Device Configuration:")
    print(f"  ADB Host: {config.device.adb_host}:{config.device.adb_port}")
    print(f"  Screen Resolution: {config.device.screen_resolution}")
    print(f"  Orientation: {config.device.orientation}")
    print(f"  Touch Duration: {config.device.touch_duration_ms}ms")
    print()


def demo_custom_config():
    """Demonstrate custom configuration creation."""
    print("🎨 Custom Configuration Demo")
    print("=" * 50)
    
    # Create custom UI-Venus config
    ui_venus_config = UIVenusConfig(
        model_name="ui-venus-7b",
        device="cuda",
        temperature=0.2,  # Slightly higher temperature
        max_tokens=1024,  # More tokens
        image_size=(512, 512),  # Smaller images for faster processing
    )
    
    # Create custom crawler config
    crawler_config = CrawlerConfig(
        strategy="breadth_first",
        max_depth=5,  # Shallow crawling
        max_actions=500,  # Fewer actions
        coverage_threshold=0.6,  # Lower coverage target
        action_delay_ms=2000,  # Slower actions
    )
    
    # Create custom device config
    device_config = DeviceConfig(
        device_id="emulator-5554",
        screen_resolution=(720, 1280),  # Smaller screen
        orientation="portrait",
        touch_duration_ms=150,  # Longer touch
    )
    
    # Combine into project config
    custom_config = ProjectConfig(
        ui_venus=ui_venus_config,
        crawler=crawler_config,
        device=device_config,
        debug=True,
        log_level="DEBUG"
    )
    
    print("Custom Configuration Created:")
    print(f"  UI-Venus Temperature: {custom_config.ui_venus.temperature}")
    print(f"  Crawler Strategy: {custom_config.crawler.strategy}")
    print(f"  Device ID: {custom_config.device.device_id}")
    print(f"  Debug Mode: {custom_config.debug}")
    print()


def demo_config_validation():
    """Demonstrate configuration validation."""
    print("✅ Configuration Validation Demo")
    print("=" * 50)
    
    # Test valid configuration
    try:
        config = ProjectConfig.load_default()
        is_valid = config.validate_configuration()
        print(f"Default config validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    except Exception as e:
        print(f"Default config validation: ❌ FAILED - {e}")
    
    # Test invalid configuration
    try:
        invalid_config = UIVenusConfig(temperature=3.0)  # Invalid temperature
        print("Invalid config validation: ❌ FAILED (as expected)")
    except ValueError as e:
        print(f"Invalid config validation: ✅ CAUGHT ERROR - {e}")
    print()


def demo_config_helpers():
    """Demonstrate configuration helper methods."""
    print("🛠️ Configuration Helper Methods Demo")
    print("=" * 50)
    
    config = ProjectConfig.load_default()
    
    # UI-Venus helpers
    print("UI-Venus Helpers:")
    print(f"  Model Path: {config.ui_venus.get_model_path()}")
    print(f"  Cache Dir: {config.ui_venus.get_cache_dir()}")
    print(f"  Is Remote API: {config.ui_venus.is_remote_api()}")
    print()
    
    # Crawler helpers
    print("Crawler Helpers:")
    print(f"  Button Priority: {config.crawler.get_element_priority('button')}")
    print(f"  Text Priority: {config.crawler.get_element_priority('text')}")
    print(f"  Is Button High Priority: {config.crawler.is_high_priority_element('button')}")
    print(f"  Action Delay: {config.crawler.get_action_delay_seconds()}s")
    print()
    
    # Device helpers
    print("Device Helpers:")
    print(f"  Screen Width: {config.device.get_screen_width()}")
    print(f"  Screen Height: {config.device.get_screen_height()}")
    print(f"  Screen Center: {config.device.get_screen_center()}")
    print(f"  Is Portrait: {config.device.is_portrait_orientation()}")
    print(f"  Touch Duration: {config.device.get_touch_duration_seconds()}s")
    print()


def demo_config_summary():
    """Demonstrate configuration summary."""
    print("📊 Configuration Summary Demo")
    print("=" * 50)
    
    config = ProjectConfig.load_default()
    summary = config.get_summary()
    
    print("Configuration Summary:")
    for section, values in summary.items():
        print(f"\n{section.upper()}:")
        for key, value in values.items():
            print(f"  {key}: {value}")
    print()


def main():
    """Run all configuration demos."""
    print("🚀 UI-Venus Mobile Crawler - Configuration System Demo")
    print("=" * 60)
    print()
    
    demo_basic_config()
    demo_custom_config()
    demo_config_validation()
    demo_config_helpers()
    demo_config_summary()
    
    print("🎉 Configuration system demo completed!")
    print("\nNext steps:")
    print("1. Create your own configuration file using config/sample_config.py as a template")
    print("2. Use ProjectConfig.load_from_file() to load your custom configuration")
    print("3. Override settings using environment variables with the appropriate prefixes")
    print("4. Validate your configuration using config.validate_configuration()")


if __name__ == "__main__":
    main()
