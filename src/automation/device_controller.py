"""
Device Controller

This module provides Android device control capabilities using ADB and UIAutomator2,
including device connection management, app lifecycle control, and system operations.
"""

import logging
import subprocess
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

from config.device_config import DeviceConfig, DeviceStatus, DeviceOrientation


logger = logging.getLogger(__name__)


class DeviceController:
    """
    Android device controller using ADB and UIAutomator2.
    
    Provides comprehensive device management including connection, app control,
    system operations, and device information retrieval.
    """
    
    def __init__(self, config: DeviceConfig):
        """
        Initialize the device controller.
        
        Args:
            config: Device configuration
        """
        self.config = config
        self._device_id: Optional[str] = None
        self._connected = False
        self._device_info: Dict[str, Any] = {}
        self._current_app: Optional[str] = None
        
        logger.info("Device controller initialized")
    
    def connect(self) -> bool:
        """
        Connect to the Android device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check if ADB is available
            if not self._check_adb_available():
                logger.error("ADB not found in PATH")
                return False
            
            # Start ADB server
            self._start_adb_server()
            
            # Get available devices
            devices = self._get_available_devices()
            if not devices:
                logger.error("No Android devices found")
                return False
            
            # Select device
            if self.config.device_id:
                if self.config.device_id in devices:
                    self._device_id = self.config.device_id
                else:
                    logger.error(f"Specified device {self.config.device_id} not found")
                    return False
            else:
                # Use first available device
                self._device_id = devices[0]
            
            # Test connection
            if self._test_connection():
                self._connected = True
                self._device_info = self._get_device_info()
                logger.info(f"Connected to device: {self._device_id}")
                return True
            else:
                logger.error("Failed to establish connection with device")
                return False
                
        except Exception as e:
            logger.error(f"Device connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        try:
            if self._connected:
                # Stop current app if running
                if self._current_app:
                    self.stop_app(self._current_app)
                
                self._connected = False
                self._device_id = None
                self._device_info = {}
                self._current_app = None
                logger.info("Disconnected from device")
                
        except Exception as e:
            logger.error(f"Device disconnection failed: {e}")
    
    def get_device_status(self) -> DeviceStatus:
        """
        Get current device status.
        
        Returns:
            Device status
        """
        try:
            if not self._connected:
                return DeviceStatus.OFFLINE
            
            # Check if device is still responsive
            result = self._run_adb_command("shell", "echo", "test")
            if result.returncode == 0:
                return DeviceStatus.ONLINE
            else:
                return DeviceStatus.OFFLINE
                
        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return DeviceStatus.UNKNOWN
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information.
        
        Returns:
            Device information dictionary
        """
        if not self._connected:
            return {}
        
        try:
            info = {}
            
            # Basic device info
            info["device_id"] = self._device_id
            info["model"] = self._get_device_property("ro.product.model")
            info["brand"] = self._get_device_property("ro.product.brand")
            info["android_version"] = self._get_device_property("ro.build.version.release")
            info["api_level"] = self._get_device_property("ro.build.version.sdk")
            
            # Screen info
            info["screen_resolution"] = self._get_screen_resolution()
            info["screen_density"] = self._get_device_property("ro.sf.lcd_density")
            
            # Memory info
            info["total_memory"] = self._get_total_memory()
            info["available_memory"] = self._get_available_memory()
            
            # Storage info
            info["total_storage"] = self._get_total_storage()
            info["available_storage"] = self._get_available_storage()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
    
    def launch_app(self, package_name: str, activity_name: Optional[str] = None) -> bool:
        """
        Launch an Android app.
        
        Args:
            package_name: App package name
            activity_name: Specific activity to launch (optional)
            
        Returns:
            True if app launched successfully, False otherwise
        """
        try:
            if not self._connected:
                logger.error("Device not connected")
                return False
            
            # Force stop app if configured
            if self.config.force_stop_app:
                self.stop_app(package_name)
                time.sleep(1)
            
            # Clear app data if configured
            if self.config.clear_app_data:
                self._run_adb_command("shell", "pm", "clear", package_name)
                time.sleep(2)
            
            # Grant permissions if configured
            if self.config.grant_permissions:
                self._grant_all_permissions(package_name)
            
            # Launch app
            if activity_name:
                launch_command = f"am start -n {package_name}/{activity_name}"
            else:
                launch_command = f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
            
            result = self._run_adb_command("shell", *launch_command.split())
            
            if result.returncode == 0:
                self._current_app = package_name
                logger.info(f"Launched app: {package_name}")
                time.sleep(3)  # Wait for app to load
                return True
            else:
                logger.error(f"Failed to launch app: {package_name}")
                return False
                
        except Exception as e:
            logger.error(f"App launch failed: {e}")
            return False
    
    def stop_app(self, package_name: str) -> bool:
        """
        Stop an Android app.
        
        Args:
            package_name: App package name
            
        Returns:
            True if app stopped successfully, False otherwise
        """
        try:
            if not self._connected:
                return False
            
            result = self._run_adb_command("shell", "am", "force-stop", package_name)
            
            if result.returncode == 0:
                if self._current_app == package_name:
                    self._current_app = None
                logger.info(f"Stopped app: {package_name}")
                return True
            else:
                logger.error(f"Failed to stop app: {package_name}")
                return False
                
        except Exception as e:
            logger.error(f"App stop failed: {e}")
            return False
    
    def get_current_app(self) -> Optional[str]:
        """
        Get the currently running app package name.
        
        Returns:
            Current app package name or None
        """
        try:
            if not self._connected:
                return None
            
            result = self._run_adb_command("shell", "dumpsys", "window", "windows")
            if result.returncode == 0:
                # Parse the output to find current app
                output = result.stdout.decode('utf-8')
                # Look for mCurrentFocus or similar patterns
                match = re.search(r'mCurrentFocus.*?(\w+\.\w+\.\w+)', output)
                if match:
                    return match.group(1)
            
            return self._current_app
            
        except Exception as e:
            logger.error(f"Failed to get current app: {e}")
            return None
    
    def get_installed_apps(self) -> List[Dict[str, str]]:
        """
        Get list of installed apps.
        
        Returns:
            List of app information dictionaries
        """
        try:
            if not self._connected:
                return []
            
            result = self._run_adb_command("shell", "pm", "list", "packages", "-3")
            if result.returncode == 0:
                apps = []
                output = result.stdout.decode('utf-8')
                for line in output.strip().split('\n'):
                    if line.startswith('package:'):
                        package_name = line.replace('package:', '').strip()
                        # Get app label
                        label_result = self._run_adb_command(
                            "shell", "pm", "dump", package_name
                        )
                        label = package_name  # Default to package name
                        if label_result.returncode == 0:
                            label_output = label_result.stdout.decode('utf-8')
                            label_match = re.search(r'applicationLabel=(.+)', label_output)
                            if label_match:
                                label = label_match.group(1)
                        
                        apps.append({
                            "package": package_name,
                            "label": label
                        })
                
                return apps
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get installed apps: {e}")
            return []
    
    def press_key(self, key_code: str) -> bool:
        """
        Press a system key.
        
        Args:
            key_code: Android key code (e.g., 'KEYCODE_BACK', 'KEYCODE_HOME')
            
        Returns:
            True if key press successful, False otherwise
        """
        try:
            if not self._connected:
                return False
            
            result = self._run_adb_command("shell", "input", "keyevent", key_code)
            
            if result.returncode == 0:
                logger.debug(f"Pressed key: {key_code}")
                time.sleep(self.config.get_key_delay_seconds())
                return True
            else:
                logger.error(f"Failed to press key: {key_code}")
                return False
                
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False
    
    def press_back(self) -> bool:
        """Press the back button."""
        return self.press_key("KEYCODE_BACK")
    
    def press_home(self) -> bool:
        """Press the home button."""
        return self.press_key("KEYCODE_HOME")
    
    def press_recent_apps(self) -> bool:
        """Press the recent apps button."""
        return self.press_key("KEYCODE_APP_SWITCH")
    
    def set_orientation(self, orientation: DeviceOrientation) -> bool:
        """
        Set device orientation.
        
        Args:
            orientation: Target orientation
            
        Returns:
            True if orientation set successfully, False otherwise
        """
        try:
            if not self._connected:
                return False
            
            orientation_map = {
                DeviceOrientation.PORTRAIT: "0",
                DeviceOrientation.LANDSCAPE: "1"
            }
            
            orientation_value = orientation_map.get(orientation, "0")
            result = self._run_adb_command(
                "shell", "settings", "put", "system", "user_rotation", orientation_value
            )
            
            if result.returncode == 0:
                logger.info(f"Set orientation to: {orientation}")
                time.sleep(1)  # Wait for orientation change
                return True
            else:
                logger.error(f"Failed to set orientation: {orientation}")
                return False
                
        except Exception as e:
            logger.error(f"Orientation change failed: {e}")
            return False
    
    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        Get current screen resolution.
        
        Returns:
            Screen resolution as (width, height)
        """
        try:
            if not self._connected:
                return self.config.screen_resolution
            
            result = self._run_adb_command("shell", "wm", "size")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                match = re.search(r'Physical size: (\d+)x(\d+)', output)
                if match:
                    width = int(match.group(1))
                    height = int(match.group(2))
                    return (width, height)
            
            return self.config.screen_resolution
            
        except Exception as e:
            logger.error(f"Failed to get screen resolution: {e}")
            return self.config.screen_resolution
    
    def _check_adb_available(self) -> bool:
        """Check if ADB is available in PATH."""
        try:
            result = subprocess.run(["adb", "version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _start_adb_server(self) -> None:
        """Start ADB server."""
        try:
            subprocess.run(["adb", "start-server"], 
                          capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("ADB server start timed out")
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available devices."""
        try:
            result = subprocess.run(["adb", "devices"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                devices = []
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line.strip() and '\tdevice' in line:
                        device_id = line.split('\t')[0]
                        devices.append(device_id)
                return devices
            return []
        except subprocess.TimeoutExpired:
            logger.error("ADB devices command timed out")
            return []
    
    def _test_connection(self) -> bool:
        """Test connection to the device."""
        try:
            result = self._run_adb_command("shell", "echo", "test")
            return result.returncode == 0
        except Exception:
            return False
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get basic device information."""
        try:
            info = {
                "device_id": self._device_id,
                "model": self._get_device_property("ro.product.model"),
                "brand": self._get_device_property("ro.product.brand"),
                "android_version": self._get_device_property("ro.build.version.release"),
                "api_level": self._get_device_property("ro.build.version.sdk")
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {}
    
    def _get_device_property(self, property_name: str) -> str:
        """Get a device property value."""
        try:
            result = self._run_adb_command("shell", "getprop", property_name)
            if result.returncode == 0:
                return result.stdout.decode('utf-8').strip()
            return ""
        except Exception:
            return ""
    
    def _get_screen_resolution(self) -> Tuple[int, int]:
        """Get screen resolution from device."""
        try:
            result = self._run_adb_command("shell", "wm", "size")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                match = re.search(r'Physical size: (\d+)x(\d+)', output)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
            return self.config.screen_resolution
        except Exception:
            return self.config.screen_resolution
    
    def _get_total_memory(self) -> int:
        """Get total device memory in MB."""
        try:
            result = self._run_adb_command("shell", "cat", "/proc/meminfo")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                match = re.search(r'MemTotal:\s+(\d+)', output)
                if match:
                    return int(match.group(1)) // 1024  # Convert KB to MB
            return 0
        except Exception:
            return 0
    
    def _get_available_memory(self) -> int:
        """Get available device memory in MB."""
        try:
            result = self._run_adb_command("shell", "cat", "/proc/meminfo")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                match = re.search(r'MemAvailable:\s+(\d+)', output)
                if match:
                    return int(match.group(1)) // 1024  # Convert KB to MB
            return 0
        except Exception:
            return 0
    
    def _get_total_storage(self) -> int:
        """Get total device storage in MB."""
        try:
            result = self._run_adb_command("shell", "df", "/data")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                lines = output.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) > 1:
                        return int(parts[1]) // 1024  # Convert KB to MB
            return 0
        except Exception:
            return 0
    
    def _get_available_storage(self) -> int:
        """Get available device storage in MB."""
        try:
            result = self._run_adb_command("shell", "df", "/data")
            if result.returncode == 0:
                output = result.stdout.decode('utf-8')
                lines = output.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) > 3:
                        return int(parts[3]) // 1024  # Convert KB to MB
            return 0
        except Exception:
            return 0
    
    def _grant_all_permissions(self, package_name: str) -> None:
        """Grant all permissions to an app."""
        try:
            # Common permissions to grant
            permissions = [
                "android.permission.CAMERA",
                "android.permission.RECORD_AUDIO",
                "android.permission.ACCESS_FINE_LOCATION",
                "android.permission.ACCESS_COARSE_LOCATION",
                "android.permission.READ_EXTERNAL_STORAGE",
                "android.permission.WRITE_EXTERNAL_STORAGE",
                "android.permission.READ_CONTACTS",
                "android.permission.WRITE_CONTACTS",
                "android.permission.READ_PHONE_STATE",
                "android.permission.CALL_PHONE",
                "android.permission.SEND_SMS",
                "android.permission.READ_SMS"
            ]
            
            for permission in permissions:
                self._run_adb_command(
                    "shell", "pm", "grant", package_name, permission
                )
                
        except Exception as e:
            logger.warning(f"Failed to grant permissions: {e}")
    
    def _run_adb_command(self, *args) -> subprocess.CompletedProcess:
        """Run an ADB command."""
        cmd = ["adb"]
        if self._device_id:
            cmd.extend(["-s", self._device_id])
        cmd.extend(args)
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.connection_timeout
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.disconnect()
        logger.info("Device controller cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
