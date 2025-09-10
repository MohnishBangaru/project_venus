"""
Sample Configuration File

This is an example configuration file showing how to customize
the UI-Venus Mobile Crawler settings for different use cases.
"""

from config import UIVenusConfig, CrawlerConfig, DeviceConfig

# UI-Venus Configuration - Optimized for 7B model on GPU
ui_venus = UIVenusConfig(
    model_name="ui-venus-7b",
    device="cuda",
    max_tokens=512,
    temperature=0.1,  # Low temperature for more deterministic results
    top_p=0.9,
    batch_size=1,
    max_memory_usage=0.8,  # Use 80% of GPU memory
    use_half_precision=True,  # Use FP16 for better performance
    image_size=(1024, 1024),
    normalize_images=True,
    trust_remote_code=True,
)

# Crawler Configuration - Balanced approach for comprehensive coverage
crawler = CrawlerConfig(
    strategy="priority_based",  # Use priority-based strategy for intelligent navigation
    max_depth=10,
    max_actions=1000,
    max_time_minutes=60,
    coverage_threshold=0.8,  # Target 80% coverage
    action_delay_ms=1000,  # 1 second between actions
    screenshot_delay_ms=500,  # 0.5 seconds before screenshot
    retry_attempts=3,
    enable_backtracking=True,
    enable_swipe_navigation=True,
    enable_form_filling=False,  # Disable form filling for now
    save_screenshots=True,
    save_action_logs=True,
    save_ui_venus_responses=True,
    output_directory="./crawl_results",
    state_similarity_threshold=0.95,
    max_state_history=100,
)

# Device Configuration - Standard Android device settings
device = DeviceConfig(
    device_id=None,  # Use first available device
    adb_host="localhost",
    adb_port=5037,
    connection_timeout=30,
    screen_resolution=(1080, 1920),  # Standard phone resolution
    dpi=420,
    orientation="portrait",
    touch_duration_ms=100,
    swipe_duration_ms=300,
    key_delay_ms=50,
    long_press_duration_ms=1000,
    screenshot_format="PNG",
    screenshot_quality=95,
    compress_screenshots=True,
    screenshot_scale=1.0,
    target_package=None,  # Will be set dynamically
    launch_activity=None,  # Will be set dynamically
    clear_app_data=False,
    force_stop_app=True,
    grant_permissions=True,
    enable_auto_recovery=True,
    recovery_actions=["back", "home", "recent_apps", "restart_app"],
    max_recovery_attempts=3,
    recovery_delay_ms=2000,
    enable_gesture_optimization=True,
    use_hardware_acceleration=True,
    max_concurrent_operations=1,
    enable_debug_logging=False,
    save_device_logs=True,
    log_level="INFO",
)

# Global Settings
debug = False
log_level = "INFO"
project_name = "UI-Venus Mobile Crawler"
version = "1.0.0"
