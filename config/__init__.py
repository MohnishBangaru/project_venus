"""
Configuration Management

This module provides the main configuration system for the UI-Venus mobile crawler.
It combines all configuration modules and provides a unified interface for
loading and managing configuration settings.
"""

from .ui_venus_config import UIVenusConfig
from .crawler_config import CrawlerConfig
from .device_config import DeviceConfig
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import json
from pathlib import Path
import importlib.util


class ProjectConfig(BaseModel):
    """Main configuration class that combines all configs."""
    
    ui_venus: UIVenusConfig = Field(default_factory=UIVenusConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    
    # Global Settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Global logging level")
    config_file: Optional[str] = Field(default=None, description="Path to configuration file")
    
    # Project Settings
    project_name: str = Field(default="UI-Venus Mobile Crawler", description="Project name")
    version: str = Field(default="1.0.0", description="Project version")
    
    class Config:
        env_prefix = "PROJECT_"
        case_sensitive = False
        validate_assignment = True
    
    @classmethod
    def load_from_file(cls, config_path: str) -> "ProjectConfig":
        """Load configuration from Python file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix == '.py':
            return cls._load_from_python_file(config_path)
        elif config_path.suffix == '.json':
            return cls._load_from_json_file(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    @classmethod
    def _load_from_python_file(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from Python file."""
        try:
            # Load the Python module
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract configuration
            config_data = {}
            
            # Look for configuration variables
            for attr_name in dir(config_module):
                if not attr_name.startswith('_'):
                    attr_value = getattr(config_module, attr_name)
                    if isinstance(attr_value, (UIVenusConfig, CrawlerConfig, DeviceConfig)):
                        config_data[attr_name] = attr_value
                    elif attr_name in ['debug', 'log_level', 'project_name', 'version']:
                        config_data[attr_name] = attr_value
            
            return cls(**config_data)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from Python file: {e}")
    
    @classmethod
    def _load_from_json_file(cls, config_path: Path) -> "ProjectConfig":
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Convert nested dictionaries to config objects
            if 'ui_venus' in config_data:
                config_data['ui_venus'] = UIVenusConfig(**config_data['ui_venus'])
            if 'crawler' in config_data:
                config_data['crawler'] = CrawlerConfig(**config_data['crawler'])
            if 'device' in config_data:
                config_data['device'] = DeviceConfig(**config_data['device'])
            
            return cls(**config_data)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from JSON file: {e}")
    
    @classmethod
    def load_from_env(cls) -> "ProjectConfig":
        """Load configuration from environment variables."""
        return cls(
            ui_venus=UIVenusConfig(),
            crawler=CrawlerConfig(),
            device=DeviceConfig()
        )
    
    @classmethod
    def load_default(cls) -> "ProjectConfig":
        """Load default configuration."""
        return cls()
    
    def save_to_file(self, config_path: str, format: str = "python") -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "python":
            self._save_to_python_file(config_path)
        elif format.lower() == "json":
            self._save_to_json_file(config_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_to_python_file(self, config_path: Path) -> None:
        """Save configuration to Python file."""
        content = f'''"""
Generated Configuration File

This file was automatically generated by the UI-Venus Mobile Crawler.
Modify the values below to customize your configuration.
"""

from config import UIVenusConfig, CrawlerConfig, DeviceConfig

# UI-Venus Configuration
ui_venus = UIVenusConfig(
    model_name="{self.ui_venus.model_name}",
    device="{self.ui_venus.device}",
    max_tokens={self.ui_venus.max_tokens},
    temperature={self.ui_venus.temperature},
    top_p={self.ui_venus.top_p},
    batch_size={self.ui_venus.batch_size},
    max_memory_usage={self.ui_venus.max_memory_usage},
    use_half_precision={self.ui_venus.use_half_precision},
    image_size={self.ui_venus.image_size},
    normalize_images={self.ui_venus.normalize_images},
)

# Crawler Configuration
crawler = CrawlerConfig(
    strategy="{self.crawler.strategy}",
    max_depth={self.crawler.max_depth},
    max_actions={self.crawler.max_actions},
    max_time_minutes={self.crawler.max_time_minutes},
    coverage_threshold={self.crawler.coverage_threshold},
    action_delay_ms={self.crawler.action_delay_ms},
    screenshot_delay_ms={self.crawler.screenshot_delay_ms},
    retry_attempts={self.crawler.retry_attempts},
    enable_backtracking={self.crawler.enable_backtracking},
    enable_swipe_navigation={self.crawler.enable_swipe_navigation},
    enable_form_filling={self.crawler.enable_form_filling},
    save_screenshots={self.crawler.save_screenshots},
    save_action_logs={self.crawler.save_action_logs},
    output_directory="{self.crawler.output_directory}",
)

# Device Configuration
device = DeviceConfig(
    device_id="{self.device.device_id or ''}",
    adb_host="{self.device.adb_host}",
    adb_port={self.device.adb_port},
    connection_timeout={self.device.connection_timeout},
    screen_resolution={self.device.screen_resolution},
    dpi={self.device.dpi},
    orientation="{self.device.orientation}",
    touch_duration_ms={self.device.touch_duration_ms},
    swipe_duration_ms={self.device.swipe_duration_ms},
    screenshot_format="{self.device.screenshot_format}",
    screenshot_quality={self.device.screenshot_quality},
    compress_screenshots={self.device.compress_screenshots},
    enable_auto_recovery={self.device.enable_auto_recovery},
    recovery_actions={self.device.recovery_actions},
    max_recovery_attempts={self.device.max_recovery_attempts},
)

# Global Settings
debug = {self.debug}
log_level = "{self.log_level}"
project_name = "{self.project_name}"
version = "{self.version}"
'''
        
        with open(config_path, 'w') as f:
            f.write(content)
    
    def _save_to_json_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = self.dict()
        
        # Convert config objects to dictionaries
        config_dict['ui_venus'] = self.ui_venus.dict()
        config_dict['crawler'] = self.crawler.dict()
        config_dict['device'] = self.device.dict()
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in ['ui_venus', 'crawler', 'device']:
                config_obj = getattr(self, key)
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
    
    def validate_configuration(self) -> bool:
        """Validate the entire configuration."""
        try:
            # Validate each config section
            self.ui_venus.validate()
            self.crawler.validate()
            self.device.validate()
            
            # Additional cross-validation
            if self.crawler.max_actions <= 0:
                raise ValueError("max_actions must be positive")
            
            if self.crawler.max_time_minutes <= 0:
                raise ValueError("max_time_minutes must be positive")
            
            return True
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            "project": {
                "name": self.project_name,
                "version": self.version,
                "debug": self.debug,
                "log_level": self.log_level
            },
            "ui_venus": {
                "model_name": self.ui_venus.model_name,
                "device": self.ui_venus.device,
                "max_tokens": self.ui_venus.max_tokens,
                "temperature": self.ui_venus.temperature,
                "is_remote_api": self.ui_venus.is_remote_api()
            },
            "crawler": {
                "strategy": self.crawler.strategy,
                "max_depth": self.crawler.max_depth,
                "max_actions": self.crawler.max_actions,
                "max_time_minutes": self.crawler.max_time_minutes,
                "coverage_threshold": self.crawler.coverage_threshold
            },
            "device": {
                "device_id": self.device.device_id,
                "screen_resolution": self.device.screen_resolution,
                "orientation": self.device.orientation,
                "enable_auto_recovery": self.device.enable_auto_recovery
            }
        }


# Convenience functions for common operations
def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """Load configuration from file or environment."""
    if config_path and Path(config_path).exists():
        return ProjectConfig.load_from_file(config_path)
    else:
        return ProjectConfig.load_from_env()


def create_default_config(config_path: str) -> ProjectConfig:
    """Create and save a default configuration file."""
    config = ProjectConfig.load_default()
    config.save_to_file(config_path)
    return config


def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file."""
    try:
        config = ProjectConfig.load_from_file(config_path)
        return config.validate_configuration()
    except Exception:
        return False
