# Android Emulator Setup Guide

This guide will help you set up the UI-Venus Mobile Crawler to run on your local Android emulator.

## Prerequisites

### 1. Android SDK
- Download and install [Android Studio](https://developer.android.com/studio)
- Or install just the [Android SDK command-line tools](https://developer.android.com/studio#command-tools)

### 2. ADB (Android Debug Bridge)
- ADB is included with Android SDK
- Add `$ANDROID_HOME/platform-tools` to your PATH
- Verify installation: `adb version`

### 3. Android Emulator
- Create an Android Virtual Device (AVD) using Android Studio
- Recommended: API level 28+ (Android 9.0+)
- Screen resolution: 1080x1920 (or similar)
- Enable hardware acceleration for better performance

### 4. Python Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start Your Emulator
```bash
# List available emulators
emulator -list-avds

# Start an emulator (replace 'your_avd_name' with actual name)
emulator -avd your_avd_name
```

### 2. Verify ADB Connection
```bash
# Check connected devices
adb devices

# Should show your emulator (e.g., emulator-5554)
```

### 3. Run the Crawler
```bash
# Basic usage (crawls Android Settings app)
python scripts/local_crawler.py

# Custom app and parameters
python scripts/local_crawler.py --app com.example.myapp --max-actions 100 --max-time 15
```

## Configuration

### Device Configuration
The crawler automatically detects your emulator, but you can customize settings in the script:

```python
device_config = DeviceConfig(
    device_id=None,  # Auto-detect first device
    screen_resolution=(1080, 1920),
    target_package="com.android.settings",
    enable_debug_logging=True
)
```

### Crawler Configuration
```python
crawler_config = CrawlerConfig(
    strategy="priority_based",  # or "breadth_first", "depth_first", "random"
    max_actions=50,
    max_time_minutes=10,
    coverage_threshold=0.3,
    action_delay_ms=1000
)
```

## Supported Apps

The crawler works with any Android app, but here are some good test apps:

### System Apps (Pre-installed)
- `com.android.settings` - Android Settings
- `com.android.calculator2` - Calculator
- `com.android.calendar` - Calendar
- `com.android.contacts` - Contacts

### Popular Apps (Install from Play Store)
- `com.whatsapp` - WhatsApp
- `com.instagram.android` - Instagram
- `com.twitter.android` - Twitter
- `com.facebook.katana` - Facebook

## Troubleshooting

### ADB Connection Issues
```bash
# Restart ADB server
adb kill-server
adb start-server

# Check device status
adb devices -l
```

### Emulator Not Detected
1. Ensure emulator is fully booted (not just started)
2. Check that USB debugging is enabled
3. Try restarting the emulator
4. Verify ADB is in your PATH

### Permission Issues
```bash
# Grant all permissions to target app
adb shell pm grant com.android.settings android.permission.CAMERA
adb shell pm grant com.android.settings android.permission.ACCESS_FINE_LOCATION
# ... add other permissions as needed
```

### Performance Issues
1. Enable hardware acceleration in emulator settings
2. Increase emulator RAM (4GB+ recommended)
3. Use x86_64 system images
4. Close unnecessary applications on host machine

## Advanced Usage

### Custom Action Sequences
```python
# Define custom actions
custom_actions = [
    {"type": "click", "coordinates": [100, 200]},
    {"type": "swipe", "start_coordinates": [500, 1000], "end_coordinates": [500, 500]},
    {"type": "input", "text": "test input"}
]

# Execute actions
for action in custom_actions:
    result = action_executor.execute_action(action)
    print(f"Action result: {result.success}")
```

### Screenshot Analysis
```python
# Capture and analyze screenshots
screenshot = screenshot_manager.capture_screenshot()
if screenshot:
    info = screenshot_manager.get_screenshot_info(screenshot)
    print(f"Screenshot info: {info}")
```

### State Tracking
```python
# Track app states
coverage_tracker = CoverageTracker(config)
state_id = coverage_tracker.add_state(screenshot)
metrics = coverage_tracker.get_coverage_metrics()
print(f"Coverage: {metrics['coverage_percentage']}%")
```

## Output Files

The crawler generates several output files:

- `crawler.log` - Detailed execution log
- `crawl_results/` - Screenshots and action logs
- `coverage_data.json` - State coverage data (if exported)

## Performance Tips

1. **Emulator Settings**:
   - Use hardware acceleration
   - Allocate sufficient RAM (4GB+)
   - Use x86_64 system images

2. **Crawler Settings**:
   - Adjust `action_delay_ms` for app responsiveness
   - Use appropriate `max_actions` for your testing needs
   - Enable screenshot compression for storage efficiency

3. **System Resources**:
   - Close unnecessary applications
   - Ensure sufficient disk space for screenshots
   - Monitor CPU and memory usage

## Integration with UI-Venus Model

To use the actual UI-Venus model instead of mock responses:

1. Set up UI-Venus model server
2. Update configuration:
```python
ui_venus_config = UIVenusConfig(
    model_type="remote_api",
    api_url="http://your-ui-venus-server:8000",
    api_key="your-api-key"
)
```

## Support

For issues and questions:
1. Check the logs in `crawler.log`
2. Verify emulator and ADB setup
3. Test with simple apps first (like Settings)
4. Review the troubleshooting section above
