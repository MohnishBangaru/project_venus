"""
Configuration System Tests

This module tests the configuration system to ensure it works correctly.
"""

import pytest
import tempfile
import os
from pathlib import Path
from config import ProjectConfig, UIVenusConfig, CrawlerConfig, DeviceConfig


class TestUIVenusConfig:
    """Test UI-Venus configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = UIVenusConfig()
        assert config.model_name == "ui-venus-7b"
        assert config.device == "cuda"
        assert config.max_tokens == 512
        assert config.temperature == 0.1
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = UIVenusConfig(temperature=0.5, top_p=0.8)
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        
        # Invalid temperature
        with pytest.raises(ValueError):
            UIVenusConfig(temperature=3.0)
        
        # Invalid top_p
        with pytest.raises(ValueError):
            UIVenusConfig(top_p=1.5)
    
    def test_helper_methods(self):
        """Test helper methods."""
        config = UIVenusConfig()
        assert config.get_model_path() == "inclusionAI/ui-venus-7b"
        assert config.is_remote_api() == False
        
        # Test with API endpoint
        config.api_endpoint = "https://api.example.com"
        assert config.is_remote_api() == True


class TestCrawlerConfig:
    """Test crawler configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = CrawlerConfig()
        assert config.strategy == "priority_based"
        assert config.max_depth == 10
        assert config.max_actions == 1000
        assert config.coverage_threshold == 0.8
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = CrawlerConfig(coverage_threshold=0.9, max_depth=5)
        assert config.coverage_threshold == 0.9
        assert config.max_depth == 5
        
        # Invalid coverage threshold
        with pytest.raises(ValueError):
            CrawlerConfig(coverage_threshold=1.5)
        
        # Invalid max depth
        with pytest.raises(ValueError):
            CrawlerConfig(max_depth=0)
    
    def test_helper_methods(self):
        """Test helper methods."""
        config = CrawlerConfig()
        assert config.get_element_priority("button") == "high"
        assert config.get_element_priority("unknown") == "low"
        assert config.is_high_priority_element("button") == True
        assert config.is_high_priority_element("text") == False
        assert config.get_action_delay_seconds() == 1.0


class TestDeviceConfig:
    """Test device configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = DeviceConfig()
        assert config.adb_host == "localhost"
        assert config.adb_port == 5037
        assert config.screen_resolution == (1080, 1920)
        assert config.orientation == "portrait"
    
    def test_validation(self):
        """Test configuration validation."""
        # Valid config
        config = DeviceConfig(adb_port=5555, screenshot_quality=80)
        assert config.adb_port == 5555
        assert config.screenshot_quality == 80
        
        # Invalid port
        with pytest.raises(ValueError):
            DeviceConfig(adb_port=70000)
        
        # Invalid screenshot quality
        with pytest.raises(ValueError):
            DeviceConfig(screenshot_quality=150)
    
    def test_helper_methods(self):
        """Test helper methods."""
        config = DeviceConfig()
        assert config.get_screen_width() == 1080
        assert config.get_screen_height() == 1920
        assert config.get_screen_center() == (540, 960)
        assert config.is_portrait_orientation() == True
        assert config.is_landscape_orientation() == False


class TestProjectConfig:
    """Test main project configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ProjectConfig()
        assert isinstance(config.ui_venus, UIVenusConfig)
        assert isinstance(config.crawler, CrawlerConfig)
        assert isinstance(config.device, DeviceConfig)
        assert config.debug == False
        assert config.log_level == "INFO"
    
    def test_load_from_env(self):
        """Test loading from environment variables."""
        config = ProjectConfig.load_from_env()
        assert isinstance(config, ProjectConfig)
    
    def test_load_default(self):
        """Test loading default configuration."""
        config = ProjectConfig.load_default()
        assert isinstance(config, ProjectConfig)
    
    def test_save_and_load_python_file(self):
        """Test saving and loading Python configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            config_path = f.name
        
        try:
            # Create and save config
            config = ProjectConfig.load_default()
            config.save_to_file(config_path, format="python")
            
            # Load config back
            loaded_config = ProjectConfig.load_from_file(config_path)
            assert loaded_config.project_name == config.project_name
            assert loaded_config.version == config.version
            
        finally:
            os.unlink(config_path)
    
    def test_save_and_load_json_file(self):
        """Test saving and loading JSON configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Create and save config
            config = ProjectConfig.load_default()
            config.save_to_file(config_path, format="json")
            
            # Load config back
            loaded_config = ProjectConfig.load_from_file(config_path)
            assert loaded_config.project_name == config.project_name
            assert loaded_config.version == config.version
            
        finally:
            os.unlink(config_path)
    
    def test_validation(self):
        """Test configuration validation."""
        config = ProjectConfig.load_default()
        assert config.validate_configuration() == True
    
    def test_get_summary(self):
        """Test getting configuration summary."""
        config = ProjectConfig.load_default()
        summary = config.get_summary()
        
        assert "project" in summary
        assert "ui_venus" in summary
        assert "crawler" in summary
        assert "device" in summary
        
        assert summary["project"]["name"] == config.project_name
        assert summary["ui_venus"]["model_name"] == config.ui_venus.model_name
        assert summary["crawler"]["strategy"] == config.crawler.strategy
        assert summary["device"]["adb_host"] == config.device.adb_host


if __name__ == "__main__":
    pytest.main([__file__])
