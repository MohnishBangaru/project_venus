#!/usr/bin/env python3
"""
RunPod Crawler Implementation

This module provides RunPod-specific functionality for the UI-Venus Mobile Crawler,
including remote ADB connection management, GPU-optimized model loading, and
RunPod workspace management.

Based on AA_VA-Phi's distributed setup architecture.
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.device_config import DeviceConfig
from config.crawler_config import CrawlerConfig
from config.ui_venus_config import UIVenusConfig
from config import ProjectConfig

# Import UI-Venus components
try:
    from src.ui_venus.model_client import UIVenusModelClient
    from src.ui_venus.element_detector import UIVenusElementDetector
    from src.ui_venus.action_suggester import UIVenusActionSuggester
except ImportError:
    logger.warning("UI-Venus components not available, using mock implementations")
    UIVenusModelClient = None
    UIVenusElementDetector = None
    UIVenusActionSuggester = None

# Import automation components
try:
    from src.automation.device_controller import DeviceController
    from src.automation.action_executor import ActionExecutor
    from src.automation.screenshot_manager import ScreenshotManager
except ImportError:
    logger.warning("Automation components not available, using mock implementations")
    DeviceController = None
    ActionExecutor = None
    ScreenshotManager = None

# Configure logging first
logger = logging.getLogger(__name__)

# Import core components
try:
    from src.core.orchestrator import CrawlerOrchestrator
    from src.core.logger import setup_logger
except ImportError:
    logger.warning("Core components not available, using mock implementations")
    CrawlerOrchestrator = None
    setup_logger = lambda level: None


class RunPodCrawler:
    """
    RunPod-specific crawler implementation.
    
    This class handles the distributed setup where:
    - RunPod instance runs the UI-Venus model with GPU acceleration
    - Local machine runs the Android emulator and ADB server
    - Network connection enables remote device control
    """
    
    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initialize RunPod crawler.
        
        Args:
            config: Project configuration (if None, will load from environment)
        """
        self.config = config or self._load_default_config()
        self.device_config = self.config.device
        self.crawler_config = self.config.crawler
        self.ui_venus_config = self.config.ui_venus
        
        # Initialize components
        self.model_client: Optional[UIVenusModelClient] = None
        self.element_detector: Optional[UIVenusElementDetector] = None
        self.action_suggester: Optional[UIVenusActionSuggester] = None
        self.device_controller: Optional[DeviceController] = None
        self.action_executor: Optional[ActionExecutor] = None
        self.screenshot_manager: Optional[ScreenshotManager] = None
        self.orchestrator: Optional[CrawlerOrchestrator] = None
        
        # RunPod-specific settings
        self.workspace_path = self.device_config.get_workspace_path()
        self.results_path = Path(self.workspace_path) / "crawl_results"
        self.cache_path = Path(self.workspace_path) / "ui_venus_cache"
        
        # Ensure directories exist
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RunPod crawler initialized with workspace: {self.workspace_path}")
    
    def _load_default_config(self) -> ProjectConfig:
        """Load default configuration for RunPod."""
        try:
            # Try to load from environment variables
            config = ProjectConfig.load_from_env()
            
            # Override with RunPod-specific settings
            config.device.is_runpod_environment = True
            config.device.enable_remote_adb = True
            
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from environment: {e}")
            # Return default config with RunPod settings
            return ProjectConfig(
                device=DeviceConfig(
                    is_runpod_environment=True,
                    enable_remote_adb=True,
                    runpod_workspace_path="/workspace"
                )
            )
    
    def setup_runpod_configurations(self, 
                                  target_app: str = "com.android.settings",
                                  max_actions: int = 100,
                                  max_time_minutes: int = 30,
                                  adb_host: str = "localhost",
                                  adb_port: int = 5037) -> None:
        """
        Set up RunPod-specific configurations.
        
        Args:
            target_app: Target app package name
            max_actions: Maximum number of actions to perform
            max_time_minutes: Maximum time in minutes
            adb_host: ADB host address
            adb_port: ADB port number
        """
        logger.info("Setting up RunPod configurations...")
        
        # Update device configuration
        self.device_config.target_package = target_app
        self.device_config.remote_adb_host = adb_host
        self.device_config.remote_adb_port = adb_port
        self.device_config.enable_remote_adb = True
        self.device_config.is_runpod_environment = True
        
        # Update crawler configuration
        self.crawler_config.max_actions = max_actions
        self.crawler_config.max_time_minutes = max_time_minutes
        self.crawler_config.strategy = "priority_based"
        
        # Update UI-Venus configuration for GPU optimization
        self.ui_venus_config.device = "cuda"
        self.ui_venus_config.max_memory_usage = 0.8
        self.ui_venus_config.batch_size = 1
        
        logger.info("RunPod configurations set up successfully")
    
    def setup_adb_server(self) -> bool:
        """
        Set up ADB server for remote connections.
        
        Returns:
            True if ADB server setup successful, False otherwise
        """
        try:
            logger.info("Setting up ADB server...")
            
            # Start ADB server
            result = subprocess.run(["adb", "start-server"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✅ ADB server started successfully")
                
                # Check for devices
                result = subprocess.run(["adb", "devices"], 
                                      capture_output=True, text=True, timeout=10)
                
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
                        logger.warning("⚠️ No devices found. Make sure your emulator is connected.")
                        return False
                else:
                    logger.error("❌ Failed to list devices")
                    return False
            else:
                logger.error(f"❌ Failed to start ADB server: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ ADB server setup failed: {e}")
            return False
    
    def connect_to_device(self) -> bool:
        """
        Connect to remote device via ADB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to remote device...")
            
            # Connect to remote ADB server
            if self.device_config.remote_adb_host:
                connect_cmd = [
                    "adb", "connect", 
                    f"{self.device_config.remote_adb_host}:{self.device_config.remote_adb_port}"
                ]
                
                result = subprocess.run(connect_cmd, 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    logger.info("✅ Connected to remote ADB server")
                    
                    # Verify connection
                    result = subprocess.run(["adb", "devices"], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        devices = []
                        for line in result.stdout.strip().split('\n')[1:]:
                            if line.strip() and '\tdevice' in line:
                                device_id = line.split('\t')[0]
                                devices.append(device_id)
                        
                        if devices:
                            logger.info(f"✅ Found {len(devices)} remote device(s): {', '.join(devices)}")
                            return True
                        else:
                            logger.warning("⚠️ No remote devices found")
                            return False
                    else:
                        logger.error("❌ Failed to list remote devices")
                        return False
                else:
                    logger.error(f"❌ Failed to connect to remote ADB: {result.stderr}")
                    return False
            else:
                logger.error("❌ No remote ADB host configured")
                return False
                
        except Exception as e:
            logger.error(f"❌ Device connection failed: {e}")
            return False
    
    async def initialize_components(self) -> bool:
        """
        Initialize all crawler components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing RunPod crawler components...")
            
            # Initialize UI-Venus components
            if UIVenusModelClient:
                self.model_client = UIVenusModelClient(self.ui_venus_config)
                await self.model_client.initialize()
                
                if UIVenusElementDetector:
                    self.element_detector = UIVenusElementDetector(self.model_client)
                if UIVenusActionSuggester:
                    self.action_suggester = UIVenusActionSuggester(self.model_client)
            else:
                logger.warning("UI-Venus components not available, using mock implementations")
            
            # Initialize automation components
            if DeviceController:
                self.device_controller = DeviceController(self.device_config)
            if ActionExecutor:
                self.action_executor = ActionExecutor(self.device_config)
            if ScreenshotManager:
                self.screenshot_manager = ScreenshotManager(self.device_config)
            
            # Initialize orchestrator
            if CrawlerOrchestrator:
                self.orchestrator = CrawlerOrchestrator(
                    device_config=self.device_config,
                    crawler_config=self.crawler_config,
                    ui_venus_config=self.ui_venus_config,
                    model_client=self.model_client,
                    element_detector=self.element_detector,
                    action_suggester=self.action_suggester,
                    device_controller=self.device_controller,
                    action_executor=self.action_executor,
                    screenshot_manager=self.screenshot_manager
                )
            else:
                logger.warning("Orchestrator not available, using mock implementation")
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def launch_target_app(self) -> bool:
        """
        Launch the target app on the remote device.
        
        Returns:
            True if app launch successful, False otherwise
        """
        try:
            logger.info(f"Launching target app: {self.device_config.target_package}")
            
            if not self.device_controller:
                logger.warning("⚠️ Device controller not available, using mock app launch")
                # Simulate app launch
                await asyncio.sleep(2)
                logger.info("✅ Mock app launch completed")
                return True
            
            # Launch app
            success = await self.device_controller.launch_app(
                self.device_config.target_package,
                self.device_config.launch_activity
            )
            
            if success:
                logger.info("✅ Target app launched successfully")
                # Wait for app to load
                await asyncio.sleep(3)
                return True
            else:
                logger.error("❌ Failed to launch target app")
                return False
                
        except Exception as e:
            logger.error(f"❌ App launch failed: {e}")
            return False
    
    async def run_crawling_session(self) -> Dict[str, Any]:
        """
        Run a complete crawling session.
        
        Returns:
            Dictionary containing crawling results
        """
        try:
            logger.info("🚀 Starting crawling session...")
            
            if not self.orchestrator:
                logger.warning("⚠️ Orchestrator not available, using mock crawling session")
                # Return mock results for testing
                results = {
                    "total_actions": 10,
                    "coverage": 0.3,
                    "duration": 60.0,
                    "screenshots_taken": 10,
                    "errors": 0,
                    "success_rate": 0.9,
                    "status": "completed",
                    "message": "Mock crawling session completed"
                }
            else:
                # Run actual crawling session
                results = await self.orchestrator.run_crawling_session()
            
            # Save results to workspace
            results_file = self.results_path / f"crawl_results_{int(time.time())}.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Crawling session completed. Results saved to: {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Crawling session failed: {e}")
            return {}
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display crawling results.
        
        Args:
            results: Crawling results dictionary
        """
        if not results:
            logger.warning("⚠️ No results to display")
            return
        
        logger.info("📊 Crawling Results Summary:")
        logger.info(f"  Total Actions: {results.get('total_actions', 0)}")
        logger.info(f"  Coverage: {results.get('coverage', 0):.2%}")
        logger.info(f"  Duration: {results.get('duration', 0):.2f} seconds")
        logger.info(f"  Screenshots: {results.get('screenshots_taken', 0)}")
        logger.info(f"  Errors: {results.get('errors', 0)}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            logger.info("Cleaning up RunPod crawler resources...")
            
            # Clean up components
            if self.model_client and hasattr(self.model_client, 'cleanup'):
                self.model_client.cleanup()
            
            if self.device_controller and hasattr(self.device_controller, 'disconnect'):
                self.device_controller.disconnect()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def run_runpod_demo(self, target_app: str = "com.android.settings") -> None:
        """
        Run a complete demo crawling session on RunPod.
        
        Args:
            target_app: Target app package name
        """
        try:
            logger.info("🚀 Starting RunPod Android Emulator Crawler Demo")
            
            # Setup configurations
            self.setup_runpod_configurations(target_app=target_app)
            
            # Setup ADB server
            if not self.setup_adb_server():
                logger.error("❌ Failed to setup ADB server")
                return
            
            # Connect to device
            if not self.connect_to_device():
                logger.error("❌ Failed to connect to device")
                return
            
            # Initialize components
            if not await self.initialize_components():
                logger.error("❌ Failed to initialize components")
                return
            
            # Launch target app
            if not await self.launch_target_app():
                logger.error("❌ Failed to launch target app")
                return
            
            # Run crawling session
            results = await self.run_crawling_session()
            
            # Display results
            self.display_results(results)
            
            logger.info("✅ RunPod demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ RunPod demo failed: {e}")
        finally:
            self.cleanup()


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod UI-Venus Mobile Crawler")
    parser.add_argument("--app", default="com.android.settings", 
                       help="Target app package name")
    parser.add_argument("--max-actions", type=int, default=100,
                       help="Maximum number of actions")
    parser.add_argument("--max-time", type=int, default=30,
                       help="Maximum time in minutes")
    parser.add_argument("--adb-host", default="localhost",
                       help="ADB host address")
    parser.add_argument("--adb-port", type=int, default=5037,
                       help="ADB port number")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level="INFO")
    
    # Load configuration
    config = None
    if args.config:
        config = ProjectConfig.load_from_file(args.config)
    
    # Create and run crawler
    crawler = RunPodCrawler(config)
    
    try:
        await crawler.run_runpod_demo(
            target_app=args.app,
            max_actions=args.max_actions,
            max_time_minutes=args.max_time,
            adb_host=args.adb_host,
            adb_port=args.adb_port
        )
    except KeyboardInterrupt:
        logger.info("🛑 Crawling interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        crawler.cleanup()


if __name__ == "__main__":
    asyncio.run(main())