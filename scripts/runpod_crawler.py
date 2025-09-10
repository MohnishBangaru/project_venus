#!/usr/bin/env python3
"""
RunPod Android Emulator Crawler

This script is optimized for RunPod deployment with local ADB server,
providing remote emulator access and GPU-accelerated UI-Venus inference.
"""

import asyncio
import logging
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.crawler import CrawlerEngine, CrawlerMode, CrawlerStatus
from src.automation.device_controller import DeviceController
from src.automation.action_executor import ActionExecutor
from src.automation.screenshot_manager import ScreenshotManager
from config import CrawlerConfig, UIVenusConfig, DeviceConfig


# Configure logging for RunPod
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/workspace/crawler.log')
    ]
)
logger = logging.getLogger(__name__)


class RunPodCrawler:
    """RunPod-optimized Android emulator crawler."""
    
    def __init__(self):
        """Initialize the RunPod crawler."""
        self.device_controller: Optional[DeviceController] = None
        self.action_executor: Optional[ActionExecutor] = None
        self.screenshot_manager: Optional[ScreenshotManager] = None
        self.crawler_engine: Optional[CrawlerEngine] = None
        
        # Configuration
        self.crawler_config = None
        self.ui_venus_config = None
        self.device_config = None
        
        # State
        self.is_connected = False
        self.current_app = None
        
        logger.info("RunPod crawler initialized")
    
    def setup_runpod_configurations(self, 
                                  target_app: str = "com.android.settings",
                                  max_actions: int = 100,
                                  max_time_minutes: int = 30,
                                  adb_host: str = "localhost",
                                  adb_port: int = 5037) -> None:
        """
        Set up RunPod-optimized configurations.
        
        Args:
            target_app: Target app package name
            max_actions: Maximum actions to perform
            max_time_minutes: Maximum crawling time in minutes
            adb_host: ADB server host (localhost for local ADB server)
            adb_port: ADB server port
        """
        logger.info("Setting up RunPod configurations...")
        
        # Crawler configuration - optimized for RunPod
        self.crawler_config = CrawlerConfig(
            strategy="priority_based",
            max_actions=max_actions,
            max_time_minutes=max_time_minutes,
            coverage_threshold=0.4,  # Higher threshold for better coverage
            action_delay_ms=800,  # Faster actions for RunPod
            screenshot_delay_ms=300,
            save_screenshots=True,
            save_action_logs=True,
            output_directory="/workspace/crawl_results"  # RunPod workspace
        )
        
        # UI-Venus configuration - GPU-optimized for RunPod
        self.ui_venus_config = UIVenusConfig(
            model_name="UI-Venus-Ground-7B",
            device="cuda",  # Use GPU on RunPod
            api_endpoint=None,
            cache_dir="/workspace/ui_venus_cache",  # RunPod workspace
            use_half_precision=True,  # Enable for GPU efficiency
            max_memory_usage=0.8,  # Use 80% of GPU memory
            batch_size=1,
            max_tokens=512,
            temperature=0.1
        )
        
        # Device configuration - optimized for remote ADB
        self.device_config = DeviceConfig(
            device_id=None,  # Auto-detect
            adb_host=adb_host,
            adb_port=adb_port,
            connection_timeout=60,  # Longer timeout for remote connections
            auto_connect=True,
            screen_resolution=(1080, 1920),
            target_package=target_app,
            enable_debug_logging=True,
            save_device_logs=True,
            log_level="INFO",
            # Performance optimizations
            enable_gesture_optimization=True,
            use_hardware_acceleration=True,
            max_concurrent_operations=1
        )
        
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
            import subprocess
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
        Connect to the Android device via ADB.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to Android device...")
            
            # Initialize device controller
            self.device_controller = DeviceController(self.device_config)
            
            # Connect to device
            if not self.device_controller.connect():
                logger.error("Failed to connect to device")
                return False
            
            # Initialize action executor
            self.action_executor = ActionExecutor(self.device_controller, self.device_config)
            
            # Initialize screenshot manager
            self.screenshot_manager = ScreenshotManager(self.device_controller, self.device_config)
            
            # Initialize crawler engine
            self.crawler_engine = CrawlerEngine(
                self.crawler_config,
                self.ui_venus_config,
                self.device_config
            )
            
            # Set up callbacks
            self.crawler_engine.set_callback("state_change", self._on_state_change)
            self.crawler_engine.set_callback("action", self._on_action)
            self.crawler_engine.set_callback("error", self._on_error)
            
            self.is_connected = True
            logger.info("Successfully connected to Android device")
            
            # Display device info
            device_info = self.device_controller.get_device_info()
            logger.info(f"Device: {device_info.get('model', 'Unknown')} - {device_info.get('android_version', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Device connection failed: {e}")
            return False
    
    async def launch_target_app(self) -> bool:
        """
        Launch the target app.
        
        Returns:
            True if app launched successfully, False otherwise
        """
        try:
            if not self.is_connected:
                logger.error("Not connected to device")
                return False
            
            target_app = self.device_config.target_package
            logger.info(f"Launching target app: {target_app}")
            
            # Launch the app
            if self.device_controller.launch_app(target_app):
                self.current_app = target_app
                logger.info(f"Successfully launched {target_app}")
                
                # Wait for app to load
                await asyncio.sleep(3)
                
                # Take initial screenshot
                screenshot = self.screenshot_manager.capture_screenshot()
                if screenshot:
                    logger.info("Initial screenshot captured")
                    return True
                else:
                    logger.warning("Failed to capture initial screenshot")
                    return True  # Still consider successful
            else:
                logger.error(f"Failed to launch {target_app}")
                return False
                
        except Exception as e:
            logger.error(f"App launch failed: {e}")
            return False
    
    async def run_crawling_session(self) -> Dict[str, Any]:
        """
        Run a complete crawling session.
        
        Returns:
            Session results and metrics
        """
        try:
            if not self.is_connected:
                logger.error("Not connected to device")
                return {}
            
            logger.info("Starting crawling session...")
            
            # Start crawling
            session = await self.crawler_engine.start_crawling(
                target_app=self.current_app,
                mode=CrawlerMode.EXPLORATION,
                max_actions=self.crawler_config.max_actions,
                max_time_minutes=self.crawler_config.max_time_minutes
            )
            
            # Get session results
            session_info = self.crawler_engine.get_session_info()
            coverage_report = self.crawler_engine.get_coverage_report()
            
            logger.info("Crawling session completed")
            
            return {
                "session": session_info,
                "coverage_report": coverage_report,
                "device_info": self.device_controller.get_device_info(),
                "action_stats": self.action_executor.get_success_rate(),
                "screenshot_count": self.screenshot_manager.get_screenshot_count()
            }
            
        except Exception as e:
            logger.error(f"Crawling session failed: {e}")
            return {"error": str(e)}
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display crawling results."""
        try:
            logger.info("=" * 60)
            logger.info("RUNPOD CRAWLING SESSION RESULTS")
            logger.info("=" * 60)
            
            if "error" in results:
                logger.error(f"Session failed: {results['error']}")
                return
            
            # Session info
            session = results.get("session", {})
            logger.info(f"Session ID: {session.get('session_id', 'N/A')}")
            logger.info(f"Duration: {session.get('duration', 0):.2f} seconds")
            logger.info(f"Actions performed: {session.get('action_count', 0)}")
            logger.info(f"Status: {session.get('status', 'N/A')}")
            
            # Coverage metrics
            coverage_metrics = results.get("coverage_report", {}).get("coverage_metrics", {})
            logger.info(f"States discovered: {coverage_metrics.get('total_states', 0)}")
            logger.info(f"States explored: {coverage_metrics.get('explored_states', 0)}")
            logger.info(f"Coverage percentage: {coverage_metrics.get('coverage_percentage', 0):.1f}%")
            
            # Action statistics
            action_stats = results.get("action_stats", 0)
            logger.info(f"Action success rate: {action_stats:.1%}")
            
            # Screenshot count
            screenshot_count = results.get("screenshot_count", 0)
            logger.info(f"Screenshots captured: {screenshot_count}")
            
            # Device info
            device_info = results.get("device_info", {})
            logger.info(f"Device: {device_info.get('model', 'Unknown')}")
            logger.info(f"Android version: {device_info.get('android_version', 'Unknown')}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to display results: {e}")
    
    def _on_state_change(self, state_id: str, state_analysis: Any) -> None:
        """Callback for state changes."""
        logger.info(f"🔄 State changed: {state_id} ({state_analysis.state_type})")
    
    def _on_action(self, action: Dict[str, Any], success: bool) -> None:
        """Callback for actions."""
        action_type = action.get("type", "unknown")
        status = "✅" if success else "❌"
        logger.info(f"{status} Action: {action_type}")
    
    def _on_error(self, error: Exception) -> None:
        """Callback for errors."""
        logger.error(f"🚨 Crawler error: {error}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.crawler_engine:
                self.crawler_engine.cleanup()
            
            if self.screenshot_manager:
                self.screenshot_manager.cleanup()
            
            if self.action_executor:
                self.action_executor.cleanup()
            
            if self.device_controller:
                self.device_controller.cleanup()
            
            logger.info("RunPod crawler cleaned up")
            
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
    
    parser = argparse.ArgumentParser(description="RunPod Android Emulator Crawler")
    parser.add_argument(
        "--app", 
        default="com.android.settings",
        help="Target app package name (default: com.android.settings)"
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=100,
        help="Maximum actions to perform (default: 100)"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=30,
        help="Maximum crawling time in minutes (default: 30)"
    )
    parser.add_argument(
        "--adb-host",
        default="localhost",
        help="ADB server host (default: localhost)"
    )
    parser.add_argument(
        "--adb-port",
        type=int,
        default=5037,
        help="ADB server port (default: 5037)"
    )
    
    args = parser.parse_args()
    
    # Create and run crawler
    crawler = RunPodCrawler()
    crawler.setup_runpod_configurations(
        target_app=args.app,
        max_actions=args.max_actions,
        max_time_minutes=args.max_time,
        adb_host=args.adb_host,
        adb_port=args.adb_port
    )
    await crawler.run_runpod_demo(target_app=args.app)


if __name__ == "__main__":
    # Check if ADB is available
    import subprocess
    try:
        subprocess.run(["adb", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ADB not found. Please install Android SDK and add ADB to your PATH.")
        print("   You can download Android SDK from: https://developer.android.com/studio")
        sys.exit(1)
    
    # Run the crawler
    asyncio.run(main())
