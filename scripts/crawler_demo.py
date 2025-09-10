#!/usr/bin/env python3
"""
Crawler Engine Demonstration

This script demonstrates the capabilities of the crawler engine by showing
how to initialize, configure, and run a crawling session.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import CrawlerEngine, CrawlerMode, CrawlerStatus
from config import CrawlerConfig, UIVenusConfig, DeviceConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrawlerDemo:
    """Demonstration of crawler engine capabilities."""
    
    def __init__(self):
        """Initialize the demo."""
        self.crawler_engine: CrawlerEngine = None
        self.session_stats = {}
    
    async def run_demo(self):
        """Run the complete crawler demonstration."""
        try:
            logger.info("🚀 Starting Crawler Engine Demonstration")
            
            # Step 1: Initialize crawler engine
            await self._initialize_crawler()
            
            # Step 2: Demonstrate configuration
            self._demonstrate_configuration()
            
            # Step 3: Run exploration session
            await self._run_exploration_session()
            
            # Step 4: Generate coverage report
            self._generate_coverage_report()
            
            # Step 5: Demonstrate different modes
            await self._demonstrate_modes()
            
            logger.info("✅ Crawler Engine Demonstration Completed Successfully")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            # Cleanup
            if self.crawler_engine:
                self.crawler_engine.cleanup()
    
    async def _initialize_crawler(self):
        """Initialize the crawler engine with demo configuration."""
        logger.info("📋 Step 1: Initializing Crawler Engine")
        
        # Create demo configurations
        crawler_config = CrawlerConfig(
            strategy="priority_based",
            max_actions=50,  # Small number for demo
            max_time_minutes=5,  # Short demo
            coverage_threshold=0.3,  # Low threshold for demo
            action_delay_ms=500,  # Fast actions for demo
            save_screenshots=True,
            save_action_logs=True
        )
        
        ui_venus_config = UIVenusConfig(
            model_type="remote_api",
            api_url="http://localhost:8000",  # Demo API URL
            max_retries=3,
            timeout_seconds=30
        )
        
        device_config = DeviceConfig(
            device_id="demo_device",
            platform="android",
            screen_width=1080,
            screen_height=1920
        )
        
        # Initialize crawler engine
        self.crawler_engine = CrawlerEngine(
            crawler_config=crawler_config,
            ui_venus_config=ui_venus_config,
            device_config=device_config
        )
        
        # Set up callbacks
        self.crawler_engine.set_callback("state_change", self._on_state_change)
        self.crawler_engine.set_callback("action", self._on_action)
        self.crawler_engine.set_callback("error", self._on_error)
        
        logger.info("✅ Crawler Engine Initialized")
    
    def _demonstrate_configuration(self):
        """Demonstrate crawler configuration capabilities."""
        logger.info("⚙️ Step 2: Demonstrating Configuration")
        
        # Show current configuration
        config_info = {
            "Strategy": self.crawler_engine.crawler_config.strategy,
            "Max Actions": self.crawler_engine.crawler_config.max_actions,
            "Max Time": f"{self.crawler_engine.crawler_config.max_time_minutes} minutes",
            "Coverage Threshold": f"{self.crawler_engine.crawler_config.coverage_threshold * 100}%",
            "Action Delay": f"{self.crawler_engine.crawler_config.action_delay_ms}ms"
        }
        
        logger.info("📊 Current Configuration:")
        for key, value in config_info.items():
            logger.info(f"  {key}: {value}")
        
        # Demonstrate different strategies
        strategies = ["breadth_first", "depth_first", "priority_based", "random"]
        logger.info("🎯 Available Strategies:")
        for strategy in strategies:
            logger.info(f"  - {strategy}")
        
        logger.info("✅ Configuration Demonstration Complete")
    
    async def _run_exploration_session(self):
        """Run an exploration crawling session."""
        logger.info("🔍 Step 3: Running Exploration Session")
        
        try:
            # Start crawling session
            session = await self.crawler_engine.start_crawling(
                target_app="com.example.demoapp",
                mode=CrawlerMode.EXPLORATION,
                max_actions=20,  # Small number for demo
                max_time_minutes=2  # Short demo
            )
            
            # Store session info
            self.session_stats = {
                "session_id": session.session_id,
                "duration": session.end_time - session.start_time,
                "status": session.status,
                "metrics": session.metrics
            }
            
            logger.info(f"📈 Session Completed: {session.session_id}")
            logger.info(f"  Duration: {self.session_stats['duration']:.2f} seconds")
            logger.info(f"  Status: {session.status}")
            
        except Exception as e:
            logger.error(f"❌ Exploration session failed: {e}")
            raise
    
    def _generate_coverage_report(self):
        """Generate and display coverage report."""
        logger.info("📊 Step 4: Generating Coverage Report")
        
        try:
            # Get comprehensive coverage report
            coverage_report = self.crawler_engine.get_coverage_report()
            
            # Display key metrics
            logger.info("📈 Coverage Metrics:")
            coverage_metrics = coverage_report["coverage_metrics"]
            logger.info(f"  Total States: {coverage_metrics.get('total_states', 0)}")
            logger.info(f"  Explored States: {coverage_metrics.get('explored_states', 0)}")
            logger.info(f"  Coverage Percentage: {coverage_metrics.get('coverage_percentage', 0):.1f}%")
            
            # Display exploration progress
            exploration_progress = coverage_report["exploration_progress"]
            logger.info("🔍 Exploration Progress:")
            logger.info(f"  Unexplored States: {exploration_progress.get('unexplored_states', 0)}")
            logger.info(f"  Partially Explored: {exploration_progress.get('partially_explored_states', 0)}")
            logger.info(f"  Fully Explored: {exploration_progress.get('fully_explored_states', 0)}")
            
            # Display navigation metrics
            navigation_metrics = coverage_report["navigation_metrics"]
            logger.info("🧭 Navigation Metrics:")
            logger.info(f"  Total Navigations: {navigation_metrics.get('total_navigations', 0)}")
            logger.info(f"  Successful Navigations: {navigation_metrics.get('successful_navigations', 0)}")
            logger.info(f"  Exploration Efficiency: {navigation_metrics.get('exploration_efficiency', 0):.2f}")
            
            # Display analysis metrics
            analysis_metrics = coverage_report["analysis_metrics"]
            logger.info("🔬 Analysis Metrics:")
            logger.info(f"  Total Analyses: {analysis_metrics.get('total_analyses', 0)}")
            logger.info(f"  State Changes Detected: {analysis_metrics.get('state_changes_detected', 0)}")
            logger.info(f"  Average Analysis Time: {analysis_metrics.get('analysis_time_avg_ms', 0):.1f}ms")
            
            logger.info("✅ Coverage Report Generated")
            
        except Exception as e:
            logger.error(f"❌ Coverage report generation failed: {e}")
    
    async def _demonstrate_modes(self):
        """Demonstrate different crawler modes."""
        logger.info("🎭 Step 5: Demonstrating Different Modes")
        
        modes = [
            (CrawlerMode.EXPLORATION, "Systematic app exploration"),
            (CrawlerMode.TESTING, "Automated testing scenarios"),
            (CrawlerMode.COVERAGE_ANALYSIS, "Coverage analysis and reporting"),
            (CrawlerMode.DEBUGGING, "Debug mode with detailed logging")
        ]
        
        for mode, description in modes:
            logger.info(f"🎯 Mode: {mode.value}")
            logger.info(f"  Description: {description}")
            
            # Show mode-specific capabilities
            if mode == CrawlerMode.EXPLORATION:
                logger.info("  - Maximizes app coverage")
                logger.info("  - Systematic exploration strategies")
                logger.info("  - State tracking and analysis")
            elif mode == CrawlerMode.TESTING:
                logger.info("  - Automated test execution")
                logger.info("  - Test scenario validation")
                logger.info("  - Bug detection and reporting")
            elif mode == CrawlerMode.COVERAGE_ANALYSIS:
                logger.info("  - Coverage metrics calculation")
                logger.info("  - Gap analysis")
                logger.info("  - Optimization recommendations")
            elif mode == CrawlerMode.DEBUGGING:
                logger.info("  - Detailed logging")
                logger.info("  - State inspection")
                logger.info("  - Performance monitoring")
        
        logger.info("✅ Mode Demonstration Complete")
    
    def _on_state_change(self, state_id: str, state_analysis: Any):
        """Callback for state changes."""
        logger.debug(f"🔄 State changed: {state_id} ({state_analysis.state_type})")
    
    def _on_action(self, action: Dict[str, Any], success: bool):
        """Callback for actions."""
        action_type = action.get("type", "unknown")
        status = "✅" if success else "❌"
        logger.debug(f"{status} Action: {action_type}")
    
    def _on_error(self, error: Exception):
        """Callback for errors."""
        logger.error(f"🚨 Error: {error}")


async def main():
    """Main demonstration function."""
    demo = CrawlerDemo()
    await demo.run_demo()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
