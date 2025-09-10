"""
Crawler Engine

This module provides the main crawler engine that orchestrates all crawler components
for intelligent Android app exploration and testing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import time
import asyncio
from dataclasses import dataclass
from enum import Enum

from src.crawler.action_planner import ActionPlanner, ExplorationGoal
from src.crawler.coverage_tracker import CoverageTracker, StateType, ExplorationStatus
from src.crawler.navigation_engine import NavigationEngine, NavigationStrategy, NavigationGoal
from src.crawler.state_analyzer import StateAnalyzer, StateChangeType
from config.crawler_config import CrawlerConfig
from config.ui_venus_config import UIVenusConfig
from config.device_config import DeviceConfig


logger = logging.getLogger(__name__)


class CrawlerStatus(str, Enum):
    """Crawler execution status."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CrawlerMode(str, Enum):
    """Crawler operation modes."""
    EXPLORATION = "exploration"
    TESTING = "testing"
    COVERAGE_ANALYSIS = "coverage_analysis"
    DEBUGGING = "debugging"


@dataclass
class CrawlerSession:
    """Represents a crawler session."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    status: CrawlerStatus
    mode: CrawlerMode
    target_app: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]


class CrawlerEngine:
    """
    Main crawler engine that orchestrates all crawler components.
    
    Provides a unified interface for intelligent Android app exploration,
    combining UI-Venus model capabilities with systematic navigation strategies.
    """
    
    def __init__(self, 
                 crawler_config: CrawlerConfig,
                 ui_venus_config: UIVenusConfig,
                 device_config: DeviceConfig):
        """
        Initialize the crawler engine.
        
        Args:
            crawler_config: Crawler configuration
            ui_venus_config: UI-Venus model configuration
            device_config: Device configuration
        """
        self.crawler_config = crawler_config
        self.ui_venus_config = ui_venus_config
        self.device_config = device_config
        
        # Initialize components
        self.action_planner = ActionPlanner(crawler_config, ui_venus_config)
        self.coverage_tracker = CoverageTracker(crawler_config)
        self.navigation_engine = NavigationEngine(crawler_config, self.coverage_tracker)
        self.state_analyzer = StateAnalyzer(crawler_config)
        
        # Engine state
        self._status = CrawlerStatus.IDLE
        self._mode = CrawlerMode.EXPLORATION
        self._current_session: Optional[CrawlerSession] = None
        self._current_state_id: Optional[str] = None
        self._current_image: Optional[Image.Image] = None
        
        # Execution state
        self._action_count = 0
        self._start_time: Optional[float] = None
        self._last_action_time: Optional[float] = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        
        # Callbacks
        self._on_state_change_callback: Optional[callable] = None
        self._on_action_callback: Optional[callable] = None
        self._on_error_callback: Optional[callable] = None
        
        logger.info("Crawler engine initialized")
    
    async def start_crawling(self, 
                           target_app: str,
                           mode: CrawlerMode = CrawlerMode.EXPLORATION,
                           max_actions: Optional[int] = None,
                           max_time_minutes: Optional[int] = None) -> CrawlerSession:
        """
        Start crawling session.
        
        Args:
            target_app: Target app package name
            mode: Crawler mode
            max_actions: Maximum actions (overrides config)
            max_time_minutes: Maximum time in minutes (overrides config)
            
        Returns:
            Crawler session
        """
        try:
            if self._status != CrawlerStatus.IDLE:
                raise RuntimeError(f"Crawler is not idle (current status: {self._status})")
            
            self._status = CrawlerStatus.INITIALIZING
            self._mode = mode
            
            # Create session
            session_id = f"crawl_{int(time.time())}"
            self._current_session = CrawlerSession(
                session_id=session_id,
                start_time=time.time(),
                end_time=None,
                status=CrawlerStatus.INITIALIZING,
                mode=mode,
                target_app=target_app,
                config={
                    "max_actions": max_actions or self.crawler_config.max_actions,
                    "max_time_minutes": max_time_minutes or self.crawler_config.max_time_minutes,
                    "strategy": self.crawler_config.strategy
                },
                metrics={}
            )
            
            # Initialize components
            await self._initialize_components()
            
            # Start crawling
            self._status = CrawlerStatus.RUNNING
            self._current_session.status = CrawlerStatus.RUNNING
            self._start_time = time.time()
            self._action_count = 0
            
            logger.info(f"Started crawling session {session_id} for app {target_app}")
            
            # Main crawling loop
            await self._crawling_loop()
            
            return self._current_session
            
        except Exception as e:
            logger.error(f"Failed to start crawling: {e}")
            self._status = CrawlerStatus.ERROR
            if self._current_session:
                self._current_session.status = CrawlerStatus.ERROR
            raise RuntimeError(f"Failed to start crawling: {e}")
    
    async def stop_crawling(self) -> None:
        """Stop the current crawling session."""
        try:
            if self._status not in [CrawlerStatus.RUNNING, CrawlerStatus.PAUSED]:
                logger.warning(f"Cannot stop crawler in status: {self._status}")
                return
            
            self._status = CrawlerStatus.STOPPING
            
            # Finalize session
            if self._current_session:
                self._current_session.end_time = time.time()
                self._current_session.status = CrawlerStatus.STOPPED
                
                # Calculate final metrics
                self._current_session.metrics = self._get_session_metrics()
            
            # Cleanup components
            await self._cleanup_components()
            
            self._status = CrawlerStatus.IDLE
            logger.info("Crawling session stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop crawling: {e}")
            self._status = CrawlerStatus.ERROR
            raise RuntimeError(f"Failed to stop crawling: {e}")
    
    async def pause_crawling(self) -> None:
        """Pause the current crawling session."""
        if self._status == CrawlerStatus.RUNNING:
            self._status = CrawlerStatus.PAUSED
            if self._current_session:
                self._current_session.status = CrawlerStatus.PAUSED
            logger.info("Crawling session paused")
    
    async def resume_crawling(self) -> None:
        """Resume the paused crawling session."""
        if self._status == CrawlerStatus.PAUSED:
            self._status = CrawlerStatus.RUNNING
            if self._current_session:
                self._current_session.status = CrawlerStatus.RUNNING
            logger.info("Crawling session resumed")
    
    def get_status(self) -> CrawlerStatus:
        """Get current crawler status."""
        return self._status
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get current session information."""
        if not self._current_session:
            return None
        
        return {
            "session_id": self._current_session.session_id,
            "status": self._current_session.status,
            "mode": self._current_session.mode,
            "target_app": self._current_session.target_app,
            "start_time": self._current_session.start_time,
            "end_time": self._current_session.end_time,
            "duration": (self._current_session.end_time or time.time()) - self._current_session.start_time,
            "action_count": self._action_count,
            "current_state_id": self._current_state_id,
            "metrics": self._current_session.metrics
        }
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Get comprehensive coverage report."""
        return {
            "coverage_metrics": self.coverage_tracker.get_coverage_metrics(),
            "exploration_progress": self.coverage_tracker.get_exploration_progress(),
            "state_statistics": self.coverage_tracker.get_state_statistics(),
            "navigation_metrics": self.navigation_engine.get_navigation_metrics(),
            "analysis_metrics": self.state_analyzer.get_analysis_metrics(),
            "planning_stats": self.action_planner.get_planning_stats()
        }
    
    def set_callback(self, event_type: str, callback: callable) -> None:
        """
        Set callback for crawler events.
        
        Args:
            event_type: Event type ('state_change', 'action', 'error')
            callback: Callback function
        """
        if event_type == "state_change":
            self._on_state_change_callback = callback
        elif event_type == "action":
            self._on_action_callback = callback
        elif event_type == "error":
            self._on_error_callback = callback
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def _crawling_loop(self) -> None:
        """Main crawling loop."""
        try:
            while self._status == CrawlerStatus.RUNNING:
                # Check stopping conditions
                if self._should_stop_crawling():
                    break
                
                # Get current screen
                current_image = await self._capture_screen()
                if not current_image:
                    logger.warning("Failed to capture screen, retrying...")
                    await asyncio.sleep(1)
                    continue
                
                # Analyze current state
                state_analysis = await self._analyze_current_state(current_image)
                
                # Plan next actions
                actions = await self._plan_next_actions(state_analysis)
                
                if not actions:
                    logger.info("No actions available, stopping crawling")
                    break
                
                # Execute best action
                action_result = await self._execute_action(actions[0])
                
                # Update state based on result
                await self._update_crawler_state(action_result)
                
                # Check for errors
                if self._consecutive_failures >= self._max_consecutive_failures:
                    logger.error("Too many consecutive failures, stopping crawling")
                    break
                
                # Wait between actions
                await asyncio.sleep(self.crawler_config.get_action_delay_seconds())
                
        except Exception as e:
            logger.error(f"Crawling loop error: {e}")
            self._status = CrawlerStatus.ERROR
            if self._on_error_callback:
                self._on_error_callback(e)
            raise
    
    async def _initialize_components(self) -> None:
        """Initialize all crawler components."""
        try:
            # Initialize action planner
            self.action_planner.reset_planning_state()
            
            # Initialize coverage tracker
            self.coverage_tracker.clear_data()
            
            # Initialize navigation engine
            self.navigation_engine.reset_navigation_state()
            
            # Initialize state analyzer
            self.state_analyzer.clear_analysis_data()
            
            logger.info("All components initialized")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def _cleanup_components(self) -> None:
        """Cleanup all crawler components."""
        try:
            # Cleanup components
            self.action_planner.cleanup()
            self.coverage_tracker.cleanup()
            self.navigation_engine.cleanup()
            self.state_analyzer.cleanup()
            
            logger.info("All components cleaned up")
            
        except Exception as e:
            logger.error(f"Component cleanup failed: {e}")
    
    async def _capture_screen(self) -> Optional[Image.Image]:
        """Capture current screen (placeholder)."""
        # This would integrate with the device controller
        # For now, return a placeholder image
        try:
            # Create a placeholder image
            image = Image.new('RGB', (1080, 1920), color='white')
            return image
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    async def _analyze_current_state(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze the current app state."""
        try:
            # Add state to coverage tracker
            state_id = self.coverage_tracker.add_state(image, StateType.SCREEN)
            
            # Analyze state
            state_analysis = self.state_analyzer.analyze_state(
                image, state_id, self._current_state_id
            )
            
            # Update current state
            self._current_state_id = state_id
            self._current_image = image
            
            # Notify callback
            if self._on_state_change_callback:
                self._on_state_change_callback(state_id, state_analysis)
            
            return {
                "state_id": state_id,
                "analysis": state_analysis,
                "image": image
            }
            
        except Exception as e:
            logger.error(f"State analysis failed: {e}")
            return {}
    
    async def _plan_next_actions(self, state_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan next actions based on current state."""
        try:
            if not state_info:
                return []
            
            state_id = state_info["state_id"]
            image = state_info["image"]
            
            # Get current state info
            current_state = {
                "id": state_id,
                "type": state_info["analysis"].state_type,
                "exploration_status": ExplorationStatus.UNEXPLORED
            }
            
            # Get exploration context
            exploration_context = {
                "visited_states": list(self.coverage_tracker._states.keys()),
                "current_goal": self._mode,
                "action_count": self._action_count
            }
            
            # Plan actions based on mode
            if self._mode == CrawlerMode.EXPLORATION:
                actions = self.action_planner.plan_exploration_actions(
                    image, current_state, exploration_context["visited_states"]
                )
            else:
                actions = self.action_planner.plan_actions(
                    image, current_state, exploration_context
                )
            
            return actions
            
        except Exception as e:
            logger.error(f"Action planning failed: {e}")
            return []
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action (placeholder)."""
        try:
            start_time = time.time()
            
            # This would integrate with the action executor
            # For now, simulate action execution
            success = np.random.random() > 0.1  # 90% success rate
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Record action
            action_id = self.coverage_tracker.record_action(
                action, self._current_state_id, None, success, duration_ms
            )
            
            # Update action planner
            self.action_planner.update_action_result(action, success)
            
            # Update navigation engine
            self.navigation_engine.update_navigation_result(
                self._current_state_id, None, success, 1
            )
            
            # Notify callback
            if self._on_action_callback:
                self._on_action_callback(action, success)
            
            self._action_count += 1
            self._last_action_time = time.time()
            
            return {
                "action": action,
                "success": success,
                "duration_ms": duration_ms,
                "action_id": action_id
            }
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "action": action,
                "success": False,
                "duration_ms": 0,
                "action_id": None,
                "error": str(e)
            }
    
    async def _update_crawler_state(self, action_result: Dict[str, Any]) -> None:
        """Update crawler state based on action result."""
        try:
            success = action_result.get("success", False)
            
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
            
            # Update coverage tracker
            if self._current_state_id:
                if success:
                    self.coverage_tracker.update_state_exploration(
                        self._current_state_id, ExplorationStatus.PARTIALLY_EXPLORED
                    )
                else:
                    self.coverage_tracker.update_state_exploration(
                        self._current_state_id, ExplorationStatus.BLOCKED
                    )
            
        except Exception as e:
            logger.error(f"State update failed: {e}")
    
    def _should_stop_crawling(self) -> bool:
        """Check if crawling should stop."""
        if not self._current_session:
            return True
        
        # Check action limit
        max_actions = self._current_session.config.get("max_actions", self.crawler_config.max_actions)
        if self._action_count >= max_actions:
            logger.info(f"Reached action limit: {max_actions}")
            return True
        
        # Check time limit
        max_time_minutes = self._current_session.config.get("max_time_minutes", self.crawler_config.max_time_minutes)
        if self._start_time:
            elapsed_minutes = (time.time() - self._start_time) / 60
            if elapsed_minutes >= max_time_minutes:
                logger.info(f"Reached time limit: {max_time_minutes} minutes")
                return True
        
        # Check coverage threshold
        coverage_metrics = self.coverage_tracker.get_coverage_metrics()
        coverage_percentage = coverage_metrics.get("coverage_percentage", 0)
        if coverage_percentage >= self.crawler_config.coverage_threshold * 100:
            logger.info(f"Reached coverage threshold: {self.crawler_config.coverage_threshold}")
            return True
        
        return False
    
    def _get_session_metrics(self) -> Dict[str, Any]:
        """Get session metrics."""
        if not self._current_session:
            return {}
        
        duration = (self._current_session.end_time or time.time()) - self._current_session.start_time
        
        return {
            "duration_seconds": duration,
            "action_count": self._action_count,
            "actions_per_minute": self._action_count / (duration / 60) if duration > 0 else 0,
            "consecutive_failures": self._consecutive_failures,
            "coverage_metrics": self.coverage_tracker.get_coverage_metrics(),
            "navigation_metrics": self.navigation_engine.get_navigation_metrics(),
            "analysis_metrics": self.state_analyzer.get_analysis_metrics()
        }
    
    def cleanup(self) -> None:
        """Clean up crawler engine."""
        try:
            # Stop current session if running
            if self._status in [CrawlerStatus.RUNNING, CrawlerStatus.PAUSED]:
                asyncio.create_task(self.stop_crawling())
            
            # Cleanup components
            self.action_planner.cleanup()
            self.coverage_tracker.cleanup()
            self.navigation_engine.cleanup()
            self.state_analyzer.cleanup()
            
            # Reset state
            self._status = CrawlerStatus.IDLE
            self._current_session = None
            self._current_state_id = None
            self._current_image = None
            
            logger.info("Crawler engine cleaned up")
            
        except Exception as e:
            logger.error(f"Crawler engine cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
