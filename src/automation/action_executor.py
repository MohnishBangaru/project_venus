"""
Action Executor

This module provides UI action execution capabilities for Android devices,
including touch, swipe, input, and gesture operations.
"""

import logging
import time
import math
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

from .device_controller import DeviceController
from config.device_config import DeviceConfig


logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of UI actions."""
    CLICK = "click"
    LONG_PRESS = "long_press"
    SWIPE = "swipe"
    INPUT = "input"
    BACK = "back"
    HOME = "home"
    RECENT_APPS = "recent_apps"
    SCROLL = "scroll"
    PINCH = "pinch"
    ROTATE = "rotate"


class ActionResult:
    """Result of an action execution."""
    
    def __init__(self, success: bool, action_type: str, duration_ms: int, 
                 error_message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.action_type = action_type
        self.duration_ms = duration_ms
        self.error_message = error_message
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "action_type": self.action_type,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class ActionExecutor:
    """
    UI action executor for Android devices.
    
    Provides comprehensive UI interaction capabilities including touch gestures,
    input operations, and system actions with proper error handling and validation.
    """
    
    def __init__(self, device_controller: DeviceController, config: DeviceConfig):
        """
        Initialize the action executor.
        
        Args:
            device_controller: Device controller instance
            config: Device configuration
        """
        self.device_controller = device_controller
        self.config = config
        self._action_history: List[ActionResult] = []
        self._max_history = 100
        
        logger.info("Action executor initialized")
    
    def execute_action(self, action: Dict[str, Any]) -> ActionResult:
        """
        Execute a UI action.
        
        Args:
            action: Action dictionary with type and parameters
            
        Returns:
            Action execution result
        """
        try:
            action_type = action.get("type", "").lower()
            start_time = time.time()
            
            # Execute based on action type
            if action_type == ActionType.CLICK:
                result = self._execute_click(action)
            elif action_type == ActionType.LONG_PRESS:
                result = self._execute_long_press(action)
            elif action_type == ActionType.SWIPE:
                result = self._execute_swipe(action)
            elif action_type == ActionType.INPUT:
                result = self._execute_input(action)
            elif action_type == ActionType.BACK:
                result = self._execute_back(action)
            elif action_type == ActionType.HOME:
                result = self._execute_home(action)
            elif action_type == ActionType.RECENT_APPS:
                result = self._execute_recent_apps(action)
            elif action_type == ActionType.SCROLL:
                result = self._execute_scroll(action)
            elif action_type == ActionType.PINCH:
                result = self._execute_pinch(action)
            elif action_type == ActionType.ROTATE:
                result = self._execute_rotate(action)
            else:
                result = ActionResult(
                    False, action_type, 0, 
                    f"Unknown action type: {action_type}"
                )
            
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            result.duration_ms = duration_ms
            
            # Add to history
            self._add_to_history(result)
            
            # Log result
            if result.success:
                logger.debug(f"Action executed successfully: {action_type}")
            else:
                logger.warning(f"Action failed: {action_type} - {result.error_message}")
            
            return result
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            result = ActionResult(
                False, action.get("type", "unknown"), duration_ms,
                f"Action execution error: {str(e)}"
            )
            self._add_to_history(result)
            logger.error(f"Action execution failed: {e}")
            return result
    
    def click(self, x: int, y: int) -> ActionResult:
        """
        Perform a click at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Action execution result
        """
        action = {
            "type": ActionType.CLICK,
            "coordinates": [x, y]
        }
        return self.execute_action(action)
    
    def long_press(self, x: int, y: int, duration_ms: Optional[int] = None) -> ActionResult:
        """
        Perform a long press at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration_ms: Press duration in milliseconds
            
        Returns:
            Action execution result
        """
        action = {
            "type": ActionType.LONG_PRESS,
            "coordinates": [x, y],
            "duration_ms": duration_ms or self.config.long_press_duration_ms
        }
        return self.execute_action(action)
    
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, 
              duration_ms: Optional[int] = None) -> ActionResult:
        """
        Perform a swipe gesture.
        
        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration_ms: Swipe duration in milliseconds
            
        Returns:
            Action execution result
        """
        action = {
            "type": ActionType.SWIPE,
            "start_coordinates": [start_x, start_y],
            "end_coordinates": [end_x, end_y],
            "duration_ms": duration_ms or self.config.swipe_duration_ms
        }
        return self.execute_action(action)
    
    def input_text(self, text: str, clear_first: bool = True) -> ActionResult:
        """
        Input text into the current input field.
        
        Args:
            text: Text to input
            clear_first: Whether to clear existing text first
            
        Returns:
            Action execution result
        """
        action = {
            "type": ActionType.INPUT,
            "text": text,
            "clear_first": clear_first
        }
        return self.execute_action(action)
    
    def scroll(self, direction: str, distance: Optional[int] = None) -> ActionResult:
        """
        Perform a scroll gesture.
        
        Args:
            direction: Scroll direction ('up', 'down', 'left', 'right')
            distance: Scroll distance in pixels
            
        Returns:
            Action execution result
        """
        action = {
            "type": ActionType.SCROLL,
            "direction": direction,
            "distance": distance
        }
        return self.execute_action(action)
    
    def _execute_click(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a click action."""
        try:
            coordinates = action.get("coordinates", [])
            if len(coordinates) < 2:
                return ActionResult(False, ActionType.CLICK, 0, "Invalid coordinates")
            
            x, y = coordinates[0], coordinates[1]
            
            # Validate coordinates
            if not self._validate_coordinates(x, y):
                return ActionResult(False, ActionType.CLICK, 0, "Coordinates out of bounds")
            
            # Execute click
            success = self._perform_tap(x, y)
            
            if success:
                return ActionResult(True, ActionType.CLICK, 0, metadata={
                    "coordinates": [x, y],
                    "target": action.get("target", "unknown")
                })
            else:
                return ActionResult(False, ActionType.CLICK, 0, "Click execution failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.CLICK, 0, f"Click error: {str(e)}")
    
    def _execute_long_press(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a long press action."""
        try:
            coordinates = action.get("coordinates", [])
            if len(coordinates) < 2:
                return ActionResult(False, ActionType.LONG_PRESS, 0, "Invalid coordinates")
            
            x, y = coordinates[0], coordinates[1]
            duration_ms = action.get("duration_ms", self.config.long_press_duration_ms)
            
            # Validate coordinates
            if not self._validate_coordinates(x, y):
                return ActionResult(False, ActionType.LONG_PRESS, 0, "Coordinates out of bounds")
            
            # Execute long press
            success = self._perform_long_press(x, y, duration_ms)
            
            if success:
                return ActionResult(True, ActionType.LONG_PRESS, 0, metadata={
                    "coordinates": [x, y],
                    "duration_ms": duration_ms,
                    "target": action.get("target", "unknown")
                })
            else:
                return ActionResult(False, ActionType.LONG_PRESS, 0, "Long press execution failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.LONG_PRESS, 0, f"Long press error: {str(e)}")
    
    def _execute_swipe(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a swipe action."""
        try:
            start_coords = action.get("start_coordinates", [])
            end_coords = action.get("end_coordinates", [])
            
            if len(start_coords) < 2 or len(end_coords) < 2:
                return ActionResult(False, ActionType.SWIPE, 0, "Invalid coordinates")
            
            start_x, start_y = start_coords[0], start_coords[1]
            end_x, end_y = end_coords[0], end_coords[1]
            duration_ms = action.get("duration_ms", self.config.swipe_duration_ms)
            
            # Validate coordinates
            if not self._validate_coordinates(start_x, start_y) or not self._validate_coordinates(end_x, end_y):
                return ActionResult(False, ActionType.SWIPE, 0, "Coordinates out of bounds")
            
            # Execute swipe
            success = self._perform_swipe(start_x, start_y, end_x, end_y, duration_ms)
            
            if success:
                return ActionResult(True, ActionType.SWIPE, 0, metadata={
                    "start_coordinates": [start_x, start_y],
                    "end_coordinates": [end_x, end_y],
                    "duration_ms": duration_ms,
                    "target": action.get("target", "unknown")
                })
            else:
                return ActionResult(False, ActionType.SWIPE, 0, "Swipe execution failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.SWIPE, 0, f"Swipe error: {str(e)}")
    
    def _execute_input(self, action: Dict[str, Any]) -> ActionResult:
        """Execute an input action."""
        try:
            text = action.get("text", "")
            clear_first = action.get("clear_first", True)
            
            if not text:
                return ActionResult(False, ActionType.INPUT, 0, "No text provided")
            
            # Clear existing text if requested
            if clear_first:
                self._clear_input_field()
            
            # Input text
            success = self._input_text(text)
            
            if success:
                return ActionResult(True, ActionType.INPUT, 0, metadata={
                    "text": text,
                    "clear_first": clear_first,
                    "target": action.get("target", "unknown")
                })
            else:
                return ActionResult(False, ActionType.INPUT, 0, "Input execution failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.INPUT, 0, f"Input error: {str(e)}")
    
    def _execute_back(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a back action."""
        try:
            success = self.device_controller.press_back()
            
            if success:
                return ActionResult(True, ActionType.BACK, 0, metadata={
                    "target": action.get("target", "back_button")
                })
            else:
                return ActionResult(False, ActionType.BACK, 0, "Back button press failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.BACK, 0, f"Back action error: {str(e)}")
    
    def _execute_home(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a home action."""
        try:
            success = self.device_controller.press_home()
            
            if success:
                return ActionResult(True, ActionType.HOME, 0, metadata={
                    "target": action.get("target", "home_button")
                })
            else:
                return ActionResult(False, ActionType.HOME, 0, "Home button press failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.HOME, 0, f"Home action error: {str(e)}")
    
    def _execute_recent_apps(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a recent apps action."""
        try:
            success = self.device_controller.press_recent_apps()
            
            if success:
                return ActionResult(True, ActionType.RECENT_APPS, 0, metadata={
                    "target": action.get("target", "recent_apps_button")
                })
            else:
                return ActionResult(False, ActionType.RECENT_APPS, 0, "Recent apps button press failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.RECENT_APPS, 0, f"Recent apps action error: {str(e)}")
    
    def _execute_scroll(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a scroll action."""
        try:
            direction = action.get("direction", "down").lower()
            distance = action.get("distance", 300)
            
            # Get screen dimensions
            screen_width, screen_height = self.config.screen_resolution
            center_x, center_y = screen_width // 2, screen_height // 2
            
            # Calculate scroll coordinates based on direction
            if direction == "down":
                start_x, start_y = center_x, center_y + distance // 2
                end_x, end_y = center_x, center_y - distance // 2
            elif direction == "up":
                start_x, start_y = center_x, center_y - distance // 2
                end_x, end_y = center_x, center_y + distance // 2
            elif direction == "left":
                start_x, start_y = center_x + distance // 2, center_y
                end_x, end_y = center_x - distance // 2, center_y
            elif direction == "right":
                start_x, start_y = center_x - distance // 2, center_y
                end_x, end_y = center_x + distance // 2, center_y
            else:
                return ActionResult(False, ActionType.SCROLL, 0, f"Invalid scroll direction: {direction}")
            
            # Execute swipe for scroll
            success = self._perform_swipe(start_x, start_y, end_x, end_y, self.config.swipe_duration_ms)
            
            if success:
                return ActionResult(True, ActionType.SCROLL, 0, metadata={
                    "direction": direction,
                    "distance": distance,
                    "target": action.get("target", "screen")
                })
            else:
                return ActionResult(False, ActionType.SCROLL, 0, "Scroll execution failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.SCROLL, 0, f"Scroll error: {str(e)}")
    
    def _execute_pinch(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a pinch gesture."""
        try:
            # Pinch gestures require multi-touch, which is complex with ADB
            # For now, implement as a simple zoom gesture
            center_x = action.get("center_x", self.config.screen_resolution[0] // 2)
            center_y = action.get("center_y", self.config.screen_resolution[1] // 2)
            scale = action.get("scale", 1.5)
            
            # Simulate pinch with multiple swipes
            distance = 100
            success = True
            
            # First swipe (outward)
            success &= self._perform_swipe(
                center_x - distance, center_y, 
                center_x + distance, center_y, 
                self.config.swipe_duration_ms
            )
            
            time.sleep(0.1)
            
            # Second swipe (outward)
            success &= self._perform_swipe(
                center_x, center_y - distance,
                center_x, center_y + distance,
                self.config.swipe_duration_ms
            )
            
            if success:
                return ActionResult(True, ActionType.PINCH, 0, metadata={
                    "center": [center_x, center_y],
                    "scale": scale,
                    "target": action.get("target", "screen")
                })
            else:
                return ActionResult(False, ActionType.PINCH, 0, "Pinch execution failed")
                
        except Exception as e:
            return ActionResult(False, ActionType.PINCH, 0, f"Pinch error: {str(e)}")
    
    def _execute_rotate(self, action: Dict[str, Any]) -> ActionResult:
        """Execute a rotation action."""
        try:
            # Rotation is typically handled by the device controller
            # This is a placeholder for device rotation
            return ActionResult(True, ActionType.ROTATE, 0, metadata={
                "target": action.get("target", "device")
            })
            
        except Exception as e:
            return ActionResult(False, ActionType.ROTATE, 0, f"Rotate error: {str(e)}")
    
    def _perform_tap(self, x: int, y: int) -> bool:
        """Perform a tap at the specified coordinates."""
        try:
            result = self.device_controller._device.shell(
                "input", "tap", str(x), str(y)
            )
            time.sleep(self.config.get_touch_duration_seconds())
            return result is not None
        except Exception as e:
            logger.error(f"Tap failed: {e}")
            return False
    
    def _perform_long_press(self, x: int, y: int, duration_ms: int) -> bool:
        """Perform a long press at the specified coordinates."""
        try:
            result = self.device_controller._device.shell(
                "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms)
            )
            return result is not None
        except Exception as e:
            logger.error(f"Long press failed: {e}")
            return False
    
    def _perform_swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int) -> bool:
        """Perform a swipe gesture."""
        try:
            result = self.device_controller._device.shell(
                "input", "swipe", 
                str(start_x), str(start_y), str(end_x), str(end_y), str(duration_ms)
            )
            return result is not None
        except Exception as e:
            logger.error(f"Swipe failed: {e}")
            return False
    
    def _input_text(self, text: str) -> bool:
        """Input text into the current input field."""
        try:
            # Escape special characters
            escaped_text = text.replace(' ', '%s').replace('&', '\\&')
            result = self.device_controller._device.shell(
                "input", "text", escaped_text
            )
            return result is not None
        except Exception as e:
            logger.error(f"Text input failed: {e}")
            return False
    
    def _clear_input_field(self) -> None:
        """Clear the current input field."""
        try:
            # Select all text and delete it
            self.device_controller._device.shell("input", "keyevent", "KEYCODE_CTRL_A")
            time.sleep(0.1)
            self.device_controller._device.shell("input", "keyevent", "KEYCODE_DEL")
        except Exception as e:
            logger.warning(f"Failed to clear input field: {e}")
    
    def _validate_coordinates(self, x: int, y: int) -> bool:
        """Validate that coordinates are within screen bounds."""
        screen_width, screen_height = self.config.screen_resolution
        return 0 <= x <= screen_width and 0 <= y <= screen_height
    
    def _add_to_history(self, result: ActionResult) -> None:
        """Add action result to history."""
        self._action_history.append(result)
        
        # Limit history size
        if len(self._action_history) > self._max_history:
            self._action_history = self._action_history[-self._max_history:]
    
    def get_action_history(self) -> List[ActionResult]:
        """Get action execution history."""
        return self._action_history.copy()
    
    def get_success_rate(self) -> float:
        """Get action success rate."""
        if not self._action_history:
            return 0.0
        
        successful_actions = sum(1 for result in self._action_history if result.success)
        return successful_actions / len(self._action_history)
    
    def get_recent_actions(self, count: int = 10) -> List[ActionResult]:
        """Get recent action results."""
        return self._action_history[-count:] if self._action_history else []
    
    def clear_history(self) -> None:
        """Clear action history."""
        self._action_history.clear()
        logger.info("Action history cleared")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_history()
        logger.info("Action executor cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
