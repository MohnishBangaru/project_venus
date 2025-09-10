"""
Device Configuration

This module defines the configuration for Android device control,
including connection settings, device parameters, and action configurations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum


class DeviceOrientation(str, Enum):
    """Device orientation options."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    AUTO = "auto"


class DeviceStatus(str, Enum):
    """Device connection status."""
    ONLINE = "online"
    OFFLINE = "offline"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"


class DeviceConfig(BaseModel):
    """Configuration for Android device control."""
    
    # Device Connection
    device_id: Optional[str] = Field(
        default=None, 
        description="Specific device ID (if None, will use first available device)"
    )
    adb_host: str = Field(
        default="localhost", 
        description="ADB host address"
    )
    adb_port: int = Field(
        default=5037, 
        description="ADB port number"
    )
    connection_timeout: int = Field(
        default=30, 
        description="Connection timeout in seconds"
    )
    auto_connect: bool = Field(
        default=True, 
        description="Automatically connect to device on startup"
    )
    
    # Device Settings
    screen_resolution: Tuple[int, int] = Field(
        default=(1080, 1920), 
        description="Screen resolution (width, height)"
    )
    dpi: int = Field(
        default=420, 
        description="Device DPI (dots per inch)"
    )
    orientation: DeviceOrientation = Field(
        default=DeviceOrientation.PORTRAIT, 
        description="Screen orientation"
    )
    screen_density: float = Field(
        default=3.0, 
        description="Screen density multiplier"
    )
    
    # Action Settings
    touch_duration_ms: int = Field(
        default=100, 
        description="Touch duration in milliseconds"
    )
    swipe_duration_ms: int = Field(
        default=300, 
        description="Swipe duration in milliseconds"
    )
    key_delay_ms: int = Field(
        default=50, 
        description="Key press delay in milliseconds"
    )
    long_press_duration_ms: int = Field(
        default=1000, 
        description="Long press duration in milliseconds"
    )
    
    # Screenshot Settings
    screenshot_format: str = Field(
        default="PNG", 
        description="Screenshot format (PNG/JPEG)"
    )
    screenshot_quality: int = Field(
        default=95, 
        description="Screenshot quality (1-100, for JPEG)"
    )
    compress_screenshots: bool = Field(
        default=True, 
        description="Compress screenshots to save space"
    )
    screenshot_scale: float = Field(
        default=1.0, 
        description="Screenshot scaling factor"
    )
    
    # App Management
    target_package: Optional[str] = Field(
        default=None, 
        description="Target app package name"
    )
    launch_activity: Optional[str] = Field(
        default=None, 
        description="Launch activity name"
    )
    clear_app_data: bool = Field(
        default=False, 
        description="Clear app data before launch"
    )
    force_stop_app: bool = Field(
        default=True, 
        description="Force stop app before launch"
    )
    grant_permissions: bool = Field(
        default=True, 
        description="Grant all permissions to app"
    )
    
    # Recovery Settings
    enable_auto_recovery: bool = Field(
        default=True, 
        description="Enable automatic recovery from errors"
    )
    recovery_actions: List[str] = Field(
        default=["back", "home", "recent_apps", "restart_app"], 
        description="Recovery actions to try in order"
    )
    max_recovery_attempts: int = Field(
        default=3, 
        description="Maximum recovery attempts"
    )
    recovery_delay_ms: int = Field(
        default=2000, 
        description="Delay between recovery attempts in milliseconds"
    )
    
    # Performance Settings
    enable_gesture_optimization: bool = Field(
        default=True, 
        description="Enable gesture optimization for better performance"
    )
    use_hardware_acceleration: bool = Field(
        default=True, 
        description="Use hardware acceleration when available"
    )
    max_concurrent_operations: int = Field(
        default=1, 
        description="Maximum concurrent device operations"
    )
    
    # Debug Settings
    enable_debug_logging: bool = Field(
        default=False, 
        description="Enable debug logging for device operations"
    )
    save_device_logs: bool = Field(
        default=True, 
        description="Save device logs for debugging"
    )
    log_level: str = Field(
        default="INFO", 
        description="Logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    
    # Advanced Settings
    custom_adb_commands: Dict[str, str] = Field(
        default={}, 
        description="Custom ADB commands for specific operations"
    )
    device_specific_settings: Dict[str, Any] = Field(
        default={}, 
        description="Device-specific configuration overrides"
    )
    
    @validator('adb_port')
    def validate_adb_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('adb_port must be between 1 and 65535')
        return v
    
    @validator('connection_timeout')
    def validate_connection_timeout(cls, v):
        if v <= 0:
            raise ValueError('connection_timeout must be positive')
        return v
    
    @validator('screen_resolution')
    def validate_screen_resolution(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError('screen_resolution must be a tuple of two positive integers')
        return v
    
    @validator('dpi')
    def validate_dpi(cls, v):
        if v <= 0:
            raise ValueError('dpi must be positive')
        return v
    
    @validator('screenshot_quality')
    def validate_screenshot_quality(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('screenshot_quality must be between 1 and 100')
        return v
    
    @validator('screenshot_scale')
    def validate_screenshot_scale(cls, v):
        if v <= 0:
            raise ValueError('screenshot_scale must be positive')
        return v
    
    @validator('touch_duration_ms')
    def validate_touch_duration(cls, v):
        if v < 0:
            raise ValueError('touch_duration_ms must be non-negative')
        return v
    
    @validator('swipe_duration_ms')
    def validate_swipe_duration(cls, v):
        if v < 0:
            raise ValueError('swipe_duration_ms must be non-negative')
        return v
    
    @validator('key_delay_ms')
    def validate_key_delay(cls, v):
        if v < 0:
            raise ValueError('key_delay_ms must be non-negative')
        return v
    
    @validator('long_press_duration_ms')
    def validate_long_press_duration(cls, v):
        if v < 0:
            raise ValueError('long_press_duration_ms must be non-negative')
        return v
    
    @validator('max_recovery_attempts')
    def validate_max_recovery_attempts(cls, v):
        if v < 0:
            raise ValueError('max_recovery_attempts must be non-negative')
        return v
    
    @validator('recovery_delay_ms')
    def validate_recovery_delay(cls, v):
        if v < 0:
            raise ValueError('recovery_delay_ms must be non-negative')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = "DEVICE_"
        case_sensitive = False
        validate_assignment = True
    
    def get_adb_connection_string(self) -> str:
        """Get ADB connection string."""
        if self.device_id:
            return f"{self.adb_host}:{self.adb_port}"
        return f"{self.adb_host}:{self.adb_port}"
    
    def get_touch_duration_seconds(self) -> float:
        """Get touch duration in seconds."""
        return self.touch_duration_ms / 1000.0
    
    def get_swipe_duration_seconds(self) -> float:
        """Get swipe duration in seconds."""
        return self.swipe_duration_ms / 1000.0
    
    def get_key_delay_seconds(self) -> float:
        """Get key delay in seconds."""
        return self.key_delay_ms / 1000.0
    
    def get_long_press_duration_seconds(self) -> float:
        """Get long press duration in seconds."""
        return self.long_press_duration_ms / 1000.0
    
    def get_recovery_delay_seconds(self) -> float:
        """Get recovery delay in seconds."""
        return self.recovery_delay_ms / 1000.0
    
    def get_screenshot_quality_for_format(self) -> Optional[int]:
        """Get screenshot quality if format supports it."""
        if self.screenshot_format.upper() == "JPEG":
            return self.screenshot_quality
        return None
    
    def is_portrait_orientation(self) -> bool:
        """Check if device is in portrait orientation."""
        return self.orientation == DeviceOrientation.PORTRAIT
    
    def is_landscape_orientation(self) -> bool:
        """Check if device is in landscape orientation."""
        return self.orientation == DeviceOrientation.LANDSCAPE
    
    def get_screen_width(self) -> int:
        """Get screen width."""
        return self.screen_resolution[0]
    
    def get_screen_height(self) -> int:
        """Get screen height."""
        return self.screen_resolution[1]
    
    def get_screen_center(self) -> Tuple[int, int]:
        """Get screen center coordinates."""
        return (self.screen_resolution[0] // 2, self.screen_resolution[1] // 2)
