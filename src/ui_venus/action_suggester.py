"""
UI-Venus Action Suggester

This module provides intelligent action suggestion capabilities using UI-Venus,
focusing on navigation tasks and action planning for maximum app coverage.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
from enum import Enum

from .model_client import UIVenusModelClient
from .element_detector import UIVenusElementDetector
from config.ui_venus_config import UIVenusConfig


logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of actions that can be suggested."""
    CLICK = "click"
    SWIPE = "swipe"
    INPUT = "input"
    LONG_PRESS = "long_press"
    BACK = "back"
    HOME = "home"
    RECENT_APPS = "recent_apps"
    SCROLL = "scroll"
    PINCH = "pinch"
    ROTATE = "rotate"


class ActionPriority(str, Enum):
    """Priority levels for actions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UIVenusActionSuggester:
    """
    Intelligent action suggester using UI-Venus navigation capabilities.
    
    Analyzes current screen state and suggests optimal actions for
    maximum app coverage and exploration.
    """
    
    def __init__(self, config: UIVenusConfig):
        """
        Initialize the action suggester.
        
        Args:
            config: UI-Venus configuration
        """
        self.config = config
        self.model_client = UIVenusModelClient(config)
        self.element_detector = UIVenusElementDetector(config)
        self._action_history = []
        self._max_history = 50
    
    def suggest_actions(self, image: Union[str, Path, Image.Image, np.ndarray], 
                       context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Suggest optimal actions for the current screen.
        
        Args:
            image: Current screen image
            context: Optional context about current state and goals
            
        Returns:
            List of suggested actions with priorities and metadata
        """
        try:
            # Get UI-Venus suggestions
            ui_venus_actions = self.model_client.suggest_actions(image, self._format_context(context))
            
            # Get detected elements
            elements = self.element_detector.detect_all_elements(image)
            
            # Generate element-based actions
            element_actions = self._generate_element_actions(elements)
            
            # Combine and prioritize actions
            all_actions = ui_venus_actions + element_actions
            prioritized_actions = self._prioritize_actions(all_actions, context)
            
            # Filter and validate actions
            valid_actions = self._validate_actions(prioritized_actions, image)
            
            # Add to history
            self._add_to_history(valid_actions)
            
            logger.info(f"Suggested {len(valid_actions)} actions")
            return valid_actions
            
        except Exception as e:
            logger.error(f"Action suggestion failed: {e}")
            raise RuntimeError(f"Action suggestion failed: {e}")
    
    def suggest_exploration_actions(self, image: Union[str, Path, Image.Image, np.ndarray], 
                                   visited_screens: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest actions focused on exploring unexplored areas.
        
        Args:
            image: Current screen image
            visited_screens: List of previously visited screen identifiers
            
        Returns:
            List of exploration-focused actions
        """
        try:
            # Get all suggested actions
            all_actions = self.suggest_actions(image)
            
            # Filter for exploration actions
            exploration_actions = []
            for action in all_actions:
                if self._is_exploration_action(action, visited_screens):
                    exploration_actions.append(action)
            
            # Prioritize by exploration potential
            exploration_actions.sort(key=lambda x: x.get("exploration_score", 0), reverse=True)
            
            logger.info(f"Suggested {len(exploration_actions)} exploration actions")
            return exploration_actions
            
        except Exception as e:
            logger.error(f"Exploration action suggestion failed: {e}")
            raise RuntimeError(f"Exploration action suggestion failed: {e}")
    
    def suggest_navigation_actions(self, image: Union[str, Path, Image.Image, np.ndarray], 
                                  target_screen: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Suggest actions for navigation to a specific screen or general navigation.
        
        Args:
            image: Current screen image
            target_screen: Optional target screen identifier
            
        Returns:
            List of navigation actions
        """
        try:
            # Get detected elements
            elements = self.element_detector.detect_all_elements(image)
            
            # Focus on navigation elements
            navigation_elements = self.element_detector.detect_navigation_elements(image)
            
            # Generate navigation actions
            navigation_actions = []
            
            # Add back button action
            navigation_actions.append({
                "type": ActionType.BACK,
                "target": "back_button",
                "priority": ActionPriority.MEDIUM,
                "description": "Navigate back to previous screen",
                "confidence": 0.8,
                "navigation_score": 0.9
            })
            
            # Add home button action
            navigation_actions.append({
                "type": ActionType.HOME,
                "target": "home_button",
                "priority": ActionPriority.LOW,
                "description": "Navigate to home screen",
                "confidence": 0.8,
                "navigation_score": 0.7
            })
            
            # Add actions for navigation elements
            for element in navigation_elements:
                action = self._create_element_action(element, ActionType.CLICK)
                action["navigation_score"] = 0.8
                navigation_actions.append(action)
            
            # Prioritize actions
            navigation_actions.sort(key=lambda x: x.get("navigation_score", 0), reverse=True)
            
            logger.info(f"Suggested {len(navigation_actions)} navigation actions")
            return navigation_actions
            
        except Exception as e:
            logger.error(f"Navigation action suggestion failed: {e}")
            raise RuntimeError(f"Navigation action suggestion failed: {e}")
    
    def suggest_recovery_actions(self, image: Union[str, Path, Image.Image, np.ndarray], 
                                error_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Suggest actions for recovering from errors or stuck states.
        
        Args:
            image: Current screen image
            error_context: Optional context about the error
            
        Returns:
            List of recovery actions
        """
        try:
            recovery_actions = []
            
            # Standard recovery actions
            recovery_actions.extend([
                {
                    "type": ActionType.BACK,
                    "target": "back_button",
                    "priority": ActionPriority.HIGH,
                    "description": "Try back button to recover",
                    "confidence": 0.9,
                    "recovery_score": 0.9
                },
                {
                    "type": ActionType.HOME,
                    "target": "home_button",
                    "priority": ActionPriority.HIGH,
                    "description": "Navigate to home screen",
                    "confidence": 0.9,
                    "recovery_score": 0.8
                },
                {
                    "type": ActionType.RECENT_APPS,
                    "target": "recent_apps",
                    "priority": ActionPriority.MEDIUM,
                    "description": "Open recent apps",
                    "confidence": 0.8,
                    "recovery_score": 0.7
                }
            ])
            
            # Get clickable elements for additional recovery options
            clickable_elements = self.element_detector.detect_clickable_elements(image)
            for element in clickable_elements[:3]:  # Limit to top 3
                action = self._create_element_action(element, ActionType.CLICK)
                action["recovery_score"] = 0.6
                action["priority"] = ActionPriority.MEDIUM
                recovery_actions.append(action)
            
            # Prioritize by recovery score
            recovery_actions.sort(key=lambda x: x.get("recovery_score", 0), reverse=True)
            
            logger.info(f"Suggested {len(recovery_actions)} recovery actions")
            return recovery_actions
            
        except Exception as e:
            logger.error(f"Recovery action suggestion failed: {e}")
            raise RuntimeError(f"Recovery action suggestion failed: {e}")
    
    def _generate_element_actions(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate actions based on detected elements.
        
        Args:
            elements: List of detected elements
            
        Returns:
            List of element-based actions
        """
        actions = []
        
        for element in elements:
            element_type = element.get("type", "").lower()
            
            # Generate appropriate action for element type
            if element_type in ["button", "link", "clickable", "menu_item", "tab"]:
                action = self._create_element_action(element, ActionType.CLICK)
                actions.append(action)
            
            elif element_type in ["input", "text_input", "password", "email", "search"]:
                action = self._create_element_action(element, ActionType.INPUT)
                actions.append(action)
            
            elif element_type in ["image", "icon"]:
                # Check if image is clickable
                if element.get("clickability_score", 0) > 0.5:
                    action = self._create_element_action(element, ActionType.CLICK)
                    actions.append(action)
        
        return actions
    
    def _create_element_action(self, element: Dict[str, Any], action_type: ActionType) -> Dict[str, Any]:
        """
        Create an action for a specific element.
        
        Args:
            element: Element dictionary
            action_type: Type of action to create
            
        Returns:
            Action dictionary
        """
        bounds = element.get("bounds", [])
        center = element.get("center", [0, 0])
        
        action = {
            "type": action_type,
            "target": element.get("id", "unknown"),
            "element": element,
            "bounds": bounds,
            "coordinates": center,
            "priority": self._determine_action_priority(element, action_type),
            "description": f"{action_type.value} on {element.get('type', 'element')}",
            "confidence": element.get("confidence", 0.5),
            "clickability_score": element.get("clickability_score", 0.0)
        }
        
        # Add action-specific parameters
        if action_type == ActionType.INPUT:
            action["input_text"] = self._suggest_input_text(element)
        elif action_type == ActionType.SWIPE:
            action["direction"] = self._suggest_swipe_direction(element)
        elif action_type == ActionType.LONG_PRESS:
            action["duration"] = 1000  # 1 second
        
        return action
    
    def _prioritize_actions(self, actions: List[Dict[str, Any]], 
                           context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize actions based on various factors.
        
        Args:
            actions: List of actions to prioritize
            context: Optional context for prioritization
            
        Returns:
            Prioritized list of actions
        """
        for action in actions:
            # Calculate priority score
            priority_score = 0.0
            
            # Base priority by type
            action_type = action.get("type", "")
            type_priorities = {
                ActionType.CLICK: 0.8,
                ActionType.INPUT: 0.6,
                ActionType.SWIPE: 0.7,
                ActionType.BACK: 0.5,
                ActionType.HOME: 0.3,
                ActionType.RECENT_APPS: 0.4
            }
            priority_score += type_priorities.get(action_type, 0.5)
            
            # Confidence factor
            confidence = action.get("confidence", 0.5)
            priority_score += confidence * 0.3
            
            # Clickability factor
            clickability = action.get("clickability_score", 0.0)
            priority_score += clickability * 0.2
            
            # Context factor
            if context:
                if context.get("exploration_mode", False):
                    exploration_score = action.get("exploration_score", 0.0)
                    priority_score += exploration_score * 0.3
                
                if context.get("recovery_mode", False):
                    recovery_score = action.get("recovery_score", 0.0)
                    priority_score += recovery_score * 0.4
            
            # History factor (avoid recently used actions)
            if action in self._action_history[-10:]:  # Last 10 actions
                priority_score *= 0.7
            
            action["priority_score"] = priority_score
        
        # Sort by priority score
        actions.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        return actions
    
    def _validate_actions(self, actions: List[Dict[str, Any]], 
                         image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Validate and filter actions.
        
        Args:
            actions: List of actions to validate
            image: Current screen image
            
        Returns:
            List of valid actions
        """
        valid_actions = []
        
        for action in actions:
            if self._is_valid_action(action, image):
                valid_actions.append(action)
        
        return valid_actions
    
    def _is_valid_action(self, action: Dict[str, Any], 
                        image: Union[str, Path, Image.Image, np.ndarray]) -> bool:
        """
        Check if an action is valid for the current screen.
        
        Args:
            action: Action to validate
            image: Current screen image
            
        Returns:
            True if action is valid
        """
        try:
            # Check required fields
            if not action.get("type") or not action.get("target"):
                return False
            
            # Check bounds for element-based actions
            if "bounds" in action:
                bounds = action["bounds"]
                if len(bounds) < 4:
                    return False
                
                # Check if bounds are within screen
                if isinstance(image, (str, Path)):
                    pil_image = Image.open(image)
                elif isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image
                
                img_width, img_height = pil_image.size
                left, top, right, bottom = bounds[:4]
                
                if (left < 0 or top < 0 or right > img_width or bottom > img_height or
                    left >= right or top >= bottom):
                    return False
            
            # Check confidence threshold
            confidence = action.get("confidence", 0.0)
            if confidence < 0.3:  # Minimum confidence threshold
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Action validation failed: {e}")
            return False
    
    def _is_exploration_action(self, action: Dict[str, Any], 
                              visited_screens: List[str]) -> bool:
        """
        Check if an action is likely to lead to exploration.
        
        Args:
            action: Action to check
            visited_screens: List of visited screens
            
        Returns:
            True if action is exploration-focused
        """
        action_type = action.get("type", "")
        
        # High exploration potential actions
        if action_type in [ActionType.CLICK, ActionType.SWIPE]:
            return True
        
        # Low exploration potential actions
        if action_type in [ActionType.BACK, ActionType.HOME]:
            return False
        
        # Check element type for click actions
        if action_type == ActionType.CLICK:
            element = action.get("element", {})
            element_type = element.get("type", "").lower()
            
            # High exploration potential elements
            if element_type in ["button", "link", "menu_item", "tab"]:
                return True
        
        return False
    
    def _determine_action_priority(self, element: Dict[str, Any], 
                                  action_type: ActionType) -> ActionPriority:
        """
        Determine priority for an action based on element and action type.
        
        Args:
            element: Element dictionary
            action_type: Type of action
            
        Returns:
            Action priority
        """
        element_type = element.get("type", "").lower()
        clickability = element.get("clickability_score", 0.0)
        
        # High priority conditions
        if (element_type in ["button", "link", "menu_item"] and 
            clickability > 0.7 and action_type == ActionType.CLICK):
            return ActionPriority.HIGH
        
        # Medium priority conditions
        if (element_type in ["input", "text_input"] and 
            action_type == ActionType.INPUT):
            return ActionPriority.MEDIUM
        
        if (element_type in ["tab", "checkbox", "radio"] and 
            action_type == ActionType.CLICK):
            return ActionPriority.MEDIUM
        
        # Default to low priority
        return ActionPriority.LOW
    
    def _suggest_input_text(self, element: Dict[str, Any]) -> str:
        """
        Suggest input text for input elements.
        
        Args:
            element: Input element
            
        Returns:
            Suggested input text
        """
        element_type = element.get("type", "").lower()
        
        # Suggest appropriate text based on input type
        if "email" in element_type:
            return "test@example.com"
        elif "password" in element_type:
            return "password123"
        elif "search" in element_type:
            return "test search"
        elif "phone" in element_type:
            return "1234567890"
        else:
            return "test input"
    
    def _suggest_swipe_direction(self, element: Dict[str, Any]) -> str:
        """
        Suggest swipe direction for swipe actions.
        
        Args:
            element: Element to swipe on
            
        Returns:
            Suggested swipe direction
        """
        # Default to vertical swipe
        return "vertical"
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Format context for UI-Venus model.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        if not context:
            return None
        
        context_parts = []
        
        if context.get("exploration_mode"):
            context_parts.append("Focus on exploring new areas")
        
        if context.get("recovery_mode"):
            context_parts.append("Recovery mode - suggest safe actions")
        
        if context.get("target_screen"):
            context_parts.append(f"Target screen: {context['target_screen']}")
        
        if context.get("visited_screens"):
            context_parts.append(f"Visited {len(context['visited_screens'])} screens")
        
        return "; ".join(context_parts) if context_parts else None
    
    def _add_to_history(self, actions: List[Dict[str, Any]]) -> None:
        """
        Add actions to history for learning and avoiding repetition.
        
        Args:
            actions: Actions to add to history
        """
        for action in actions:
            self._action_history.append(action)
        
        # Limit history size
        if len(self._action_history) > self._max_history:
            self._action_history = self._action_history[-self._max_history:]
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history."""
        return self._action_history.copy()
    
    def clear_history(self) -> None:
        """Clear action history."""
        self._action_history.clear()
        logger.info("Action history cleared")
    
    def get_suggestion_stats(self) -> Dict[str, Any]:
        """Get suggestion statistics."""
        return {
            "history_size": len(self._action_history),
            "max_history": self._max_history,
            "recent_actions": self._action_history[-10:] if self._action_history else []
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_history()
        if hasattr(self, 'model_client'):
            self.model_client.cleanup()
        if hasattr(self, 'element_detector'):
            self.element_detector.cleanup()
        logger.info("Action suggester cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
