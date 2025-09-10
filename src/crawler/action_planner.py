"""
Action Planner

This module provides intelligent action planning capabilities that bridge UI-Venus
suggestions with crawler strategy and exploration goals.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
from enum import Enum
import time

from src.ui_venus.action_suggester import UIVenusActionSuggester, ActionType, ActionPriority
from src.ui_venus.element_detector import UIVenusElementDetector
from config.crawler_config import CrawlerConfig, CrawlingStrategy
from config.ui_venus_config import UIVenusConfig


logger = logging.getLogger(__name__)


class ExplorationGoal(str, Enum):
    """Exploration goals for action planning."""
    MAXIMIZE_COVERAGE = "maximize_coverage"
    DEEP_EXPLORATION = "deep_exploration"
    FEATURE_DISCOVERY = "feature_discovery"
    ERROR_RECOVERY = "error_recovery"
    NAVIGATION = "navigation"


class ActionPlanner:
    """
    Intelligent action planner that combines UI-Venus suggestions with crawler strategy.
    
    This component serves as the bridge between the UI-Venus model's action suggestions
    and the crawler's exploration strategy, ensuring optimal action selection for
    maximum app coverage and systematic exploration.
    """
    
    def __init__(self, crawler_config: CrawlerConfig, ui_venus_config: UIVenusConfig):
        """
        Initialize the action planner.
        
        Args:
            crawler_config: Crawler configuration
            ui_venus_config: UI-Venus model configuration
        """
        self.crawler_config = crawler_config
        self.ui_venus_config = ui_venus_config
        
        # Initialize UI-Venus components
        self.action_suggester = UIVenusActionSuggester(ui_venus_config)
        self.element_detector = UIVenusElementDetector(ui_venus_config)
        
        # Planning state
        self._current_goal = ExplorationGoal.MAXIMIZE_COVERAGE
        self._action_history = []
        self._failed_actions = []
        self._successful_actions = []
        self._exploration_context = {}
        
        # Strategy-specific state
        self._breadth_first_queue = []
        self._depth_first_stack = []
        self._priority_weights = self._initialize_priority_weights()
        
        logger.info("Action planner initialized")
    
    def plan_actions(self, 
                    image: Image.Image,
                    current_state: Dict[str, Any],
                    exploration_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Plan optimal actions for the current screen state.
        
        Args:
            image: Current screen image
            current_state: Current app state information
            exploration_context: Additional context for exploration
            
        Returns:
            List of planned actions with priorities and metadata
        """
        try:
            # Update exploration context
            if exploration_context:
                self._exploration_context.update(exploration_context)
            
            # Get UI-Venus suggestions
            ui_venus_actions = self.action_suggester.suggest_actions(image, self._exploration_context)
            
            # Apply crawler strategy
            strategy_actions = self._apply_crawling_strategy(ui_venus_actions, current_state)
            
            # Filter and prioritize actions
            filtered_actions = self._filter_actions(strategy_actions, current_state)
            prioritized_actions = self._prioritize_actions(filtered_actions, current_state)
            
            # Add planning metadata
            planned_actions = self._add_planning_metadata(prioritized_actions, current_state)
            
            # Update planning state
            self._update_planning_state(planned_actions, current_state)
            
            logger.info(f"Planned {len(planned_actions)} actions using {self.crawler_config.strategy} strategy")
            return planned_actions
            
        except Exception as e:
            logger.error(f"Action planning failed: {e}")
            raise RuntimeError(f"Action planning failed: {e}")
    
    def plan_exploration_actions(self, 
                                image: Image.Image,
                                current_state: Dict[str, Any],
                                visited_states: List[str]) -> List[Dict[str, Any]]:
        """
        Plan actions specifically for exploration of unexplored areas.
        
        Args:
            image: Current screen image
            current_state: Current app state information
            visited_states: List of previously visited state identifiers
            
        Returns:
            List of exploration-focused actions
        """
        try:
            # Set exploration goal
            self._current_goal = ExplorationGoal.MAXIMIZE_COVERAGE
            
            # Get exploration-focused suggestions
            exploration_context = {
                "exploration_mode": True,
                "visited_states": visited_states,
                "goal": "maximize_coverage"
            }
            
            exploration_actions = self.action_suggester.suggest_exploration_actions(
                image, visited_states
            )
            
            # Apply exploration strategy
            strategy_actions = self._apply_exploration_strategy(exploration_actions, visited_states)
            
            # Prioritize by exploration potential
            prioritized_actions = self._prioritize_exploration_actions(strategy_actions, visited_states)
            
            logger.info(f"Planned {len(prioritized_actions)} exploration actions")
            return prioritized_actions
            
        except Exception as e:
            logger.error(f"Exploration action planning failed: {e}")
            raise RuntimeError(f"Exploration action planning failed: {e}")
    
    def plan_recovery_actions(self, 
                             image: Image.Image,
                             current_state: Dict[str, Any],
                             error_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Plan actions for recovering from errors or stuck states.
        
        Args:
            image: Current screen image
            current_state: Current app state information
            error_context: Context about the error or stuck state
            
        Returns:
            List of recovery actions
        """
        try:
            # Set recovery goal
            self._current_goal = ExplorationGoal.ERROR_RECOVERY
            
            # Get recovery suggestions
            recovery_actions = self.action_suggester.suggest_recovery_actions(image, error_context)
            
            # Apply recovery strategy
            strategy_actions = self._apply_recovery_strategy(recovery_actions, error_context)
            
            # Prioritize by recovery potential
            prioritized_actions = self._prioritize_recovery_actions(strategy_actions, error_context)
            
            logger.info(f"Planned {len(prioritized_actions)} recovery actions")
            return prioritized_actions
            
        except Exception as e:
            logger.error(f"Recovery action planning failed: {e}")
            raise RuntimeError(f"Recovery action planning failed: {e}")
    
    def plan_navigation_actions(self, 
                               image: Image.Image,
                               current_state: Dict[str, Any],
                               target_state: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Plan actions for navigation to specific states or general navigation.
        
        Args:
            image: Current screen image
            current_state: Current app state information
            target_state: Optional target state identifier
            
        Returns:
            List of navigation actions
        """
        try:
            # Set navigation goal
            self._current_goal = ExplorationGoal.NAVIGATION
            
            # Get navigation suggestions
            navigation_actions = self.action_suggester.suggest_navigation_actions(image, target_state)
            
            # Apply navigation strategy
            strategy_actions = self._apply_navigation_strategy(navigation_actions, target_state)
            
            # Prioritize by navigation potential
            prioritized_actions = self._prioritize_navigation_actions(strategy_actions, target_state)
            
            logger.info(f"Planned {len(prioritized_actions)} navigation actions")
            return prioritized_actions
            
        except Exception as e:
            logger.error(f"Navigation action planning failed: {e}")
            raise RuntimeError(f"Navigation action planning failed: {e}")
    
    def update_action_result(self, action: Dict[str, Any], success: bool, 
                           new_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the planner with the result of an executed action.
        
        Args:
            action: The executed action
            success: Whether the action was successful
            new_state: New state after action execution
        """
        try:
            # Add to appropriate history
            if success:
                self._successful_actions.append({
                    "action": action,
                    "timestamp": time.time(),
                    "new_state": new_state
                })
            else:
                self._failed_actions.append({
                    "action": action,
                    "timestamp": time.time(),
                    "error": "Action execution failed"
                })
            
            # Update strategy-specific state
            self._update_strategy_state(action, success, new_state)
            
            # Update exploration context
            if new_state:
                self._exploration_context.update({
                    "last_action": action,
                    "last_success": success,
                    "current_state": new_state
                })
            
            logger.debug(f"Updated action result: success={success}")
            
        except Exception as e:
            logger.error(f"Failed to update action result: {e}")
    
    def _apply_crawling_strategy(self, actions: List[Dict[str, Any]], 
                                current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply the configured crawling strategy to actions."""
        strategy = self.crawler_config.strategy
        
        if strategy == CrawlingStrategy.BREADTH_FIRST:
            return self._apply_breadth_first_strategy(actions, current_state)
        elif strategy == CrawlingStrategy.DEPTH_FIRST:
            return self._apply_depth_first_strategy(actions, current_state)
        elif strategy == CrawlingStrategy.PRIORITY_BASED:
            return self._apply_priority_based_strategy(actions, current_state)
        elif strategy == CrawlingStrategy.RANDOM:
            return self._apply_random_strategy(actions, current_state)
        else:
            logger.warning(f"Unknown strategy: {strategy}, using priority-based")
            return self._apply_priority_based_strategy(actions, current_state)
    
    def _apply_breadth_first_strategy(self, actions: List[Dict[str, Any]], 
                                     current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply breadth-first exploration strategy."""
        # Add new actions to queue
        for action in actions:
            if action not in self._breadth_first_queue:
                self._breadth_first_queue.append(action)
        
        # Return actions from queue (FIFO)
        return self._breadth_first_queue[:self.crawler_config.max_actions]
    
    def _apply_depth_first_strategy(self, actions: List[Dict[str, Any]], 
                                   current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply depth-first exploration strategy."""
        # Add new actions to stack
        for action in actions:
            if action not in self._depth_first_stack:
                self._depth_first_stack.append(action)
        
        # Return actions from stack (LIFO)
        return self._depth_first_stack[:self.crawler_config.max_actions]
    
    def _apply_priority_based_strategy(self, actions: List[Dict[str, Any]], 
                                      current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply priority-based strategy using element priorities."""
        prioritized_actions = []
        
        for action in actions:
            # Calculate priority score
            priority_score = self._calculate_priority_score(action, current_state)
            action["priority_score"] = priority_score
            prioritized_actions.append(action)
        
        # Sort by priority score
        prioritized_actions.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        return prioritized_actions
    
    def _apply_random_strategy(self, actions: List[Dict[str, Any]], 
                              current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply random exploration strategy."""
        import random
        random.shuffle(actions)
        return actions
    
    def _apply_exploration_strategy(self, actions: List[Dict[str, Any]], 
                                   visited_states: List[str]) -> List[Dict[str, Any]]:
        """Apply exploration-specific strategy."""
        exploration_actions = []
        
        for action in actions:
            # Calculate exploration score
            exploration_score = self._calculate_exploration_score(action, visited_states)
            action["exploration_score"] = exploration_score
            exploration_actions.append(action)
        
        # Sort by exploration score
        exploration_actions.sort(key=lambda x: x.get("exploration_score", 0), reverse=True)
        
        return exploration_actions
    
    def _apply_recovery_strategy(self, actions: List[Dict[str, Any]], 
                                error_context: Optional[str]) -> List[Dict[str, Any]]:
        """Apply recovery-specific strategy."""
        recovery_actions = []
        
        for action in actions:
            # Calculate recovery score
            recovery_score = self._calculate_recovery_score(action, error_context)
            action["recovery_score"] = recovery_score
            recovery_actions.append(action)
        
        # Sort by recovery score
        recovery_actions.sort(key=lambda x: x.get("recovery_score", 0), reverse=True)
        
        return recovery_actions
    
    def _apply_navigation_strategy(self, actions: List[Dict[str, Any]], 
                                  target_state: Optional[str]) -> List[Dict[str, Any]]:
        """Apply navigation-specific strategy."""
        navigation_actions = []
        
        for action in actions:
            # Calculate navigation score
            navigation_score = self._calculate_navigation_score(action, target_state)
            action["navigation_score"] = navigation_score
            navigation_actions.append(action)
        
        # Sort by navigation score
        navigation_actions.sort(key=lambda x: x.get("navigation_score", 0), reverse=True)
        
        return navigation_actions
    
    def _filter_actions(self, actions: List[Dict[str, Any]], 
                       current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter actions based on current state and constraints."""
        filtered_actions = []
        
        for action in actions:
            if self._is_action_valid(action, current_state):
                filtered_actions.append(action)
        
        return filtered_actions
    
    def _prioritize_actions(self, actions: List[Dict[str, Any]], 
                           current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize actions based on multiple factors."""
        for action in actions:
            # Calculate comprehensive priority score
            priority_score = 0.0
            
            # Base priority from UI-Venus
            priority_score += action.get("priority_score", 0.0) * 0.4
            
            # Element priority from config
            element_type = action.get("element", {}).get("type", "").lower()
            element_priority = self.crawler_config.get_element_priority(element_type)
            priority_score += self._priority_weights.get(element_priority, 0.0) * 0.3
            
            # Confidence factor
            confidence = action.get("confidence", 0.5)
            priority_score += confidence * 0.2
            
            # Exploration bonus
            if self._current_goal == ExplorationGoal.MAXIMIZE_COVERAGE:
                exploration_score = action.get("exploration_score", 0.0)
                priority_score += exploration_score * self.crawler_config.exploration_bonus * 0.1
            
            action["final_priority_score"] = priority_score
        
        # Sort by final priority score
        actions.sort(key=lambda x: x.get("final_priority_score", 0), reverse=True)
        
        return actions
    
    def _prioritize_exploration_actions(self, actions: List[Dict[str, Any]], 
                                       visited_states: List[str]) -> List[Dict[str, Any]]:
        """Prioritize actions for exploration."""
        for action in actions:
            exploration_score = action.get("exploration_score", 0.0)
            action["final_priority_score"] = exploration_score
        
        actions.sort(key=lambda x: x.get("final_priority_score", 0), reverse=True)
        return actions
    
    def _prioritize_recovery_actions(self, actions: List[Dict[str, Any]], 
                                    error_context: Optional[str]) -> List[Dict[str, Any]]:
        """Prioritize actions for recovery."""
        for action in actions:
            recovery_score = action.get("recovery_score", 0.0)
            action["final_priority_score"] = recovery_score
        
        actions.sort(key=lambda x: x.get("final_priority_score", 0), reverse=True)
        return actions
    
    def _prioritize_navigation_actions(self, actions: List[Dict[str, Any]], 
                                      target_state: Optional[str]) -> List[Dict[str, Any]]:
        """Prioritize actions for navigation."""
        for action in actions:
            navigation_score = action.get("navigation_score", 0.0)
            action["final_priority_score"] = navigation_score
        
        actions.sort(key=lambda x: x.get("final_priority_score", 0), reverse=True)
        return actions
    
    def _add_planning_metadata(self, actions: List[Dict[str, Any]], 
                              current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add planning-specific metadata to actions."""
        for i, action in enumerate(actions):
            action["planning_metadata"] = {
                "planned_at": time.time(),
                "strategy": self.crawler_config.strategy,
                "goal": self._current_goal,
                "order": i,
                "state_id": current_state.get("id", "unknown")
            }
        
        return actions
    
    def _update_planning_state(self, actions: List[Dict[str, Any]], 
                              current_state: Dict[str, Any]) -> None:
        """Update internal planning state."""
        self._action_history.extend(actions)
        
        # Limit history size
        max_history = self.crawler_config.max_state_history
        if len(self._action_history) > max_history:
            self._action_history = self._action_history[-max_history:]
    
    def _update_strategy_state(self, action: Dict[str, Any], success: bool, 
                              new_state: Optional[Dict[str, Any]]) -> None:
        """Update strategy-specific state."""
        strategy = self.crawler_config.strategy
        
        if strategy == CrawlingStrategy.BREADTH_FIRST:
            if action in self._breadth_first_queue:
                self._breadth_first_queue.remove(action)
        elif strategy == CrawlingStrategy.DEPTH_FIRST:
            if action in self._depth_first_stack:
                self._depth_first_stack.remove(action)
    
    def _calculate_priority_score(self, action: Dict[str, Any], 
                                 current_state: Dict[str, Any]) -> float:
        """Calculate priority score for an action."""
        score = 0.0
        
        # Base score from UI-Venus
        score += action.get("priority_score", 0.0)
        
        # Element type bonus
        element_type = action.get("element", {}).get("type", "").lower()
        if self.crawler_config.is_high_priority_element(element_type):
            score += 0.5
        
        # Confidence bonus
        confidence = action.get("confidence", 0.5)
        score += confidence * 0.3
        
        return score
    
    def _calculate_exploration_score(self, action: Dict[str, Any], 
                                    visited_states: List[str]) -> float:
        """Calculate exploration score for an action."""
        score = 0.0
        
        # Base exploration score from UI-Venus
        score += action.get("exploration_score", 0.0)
        
        # Action type bonus
        action_type = action.get("type", "")
        if action_type in [ActionType.CLICK, ActionType.SWIPE]:
            score += 0.3
        
        # Element type bonus
        element_type = action.get("element", {}).get("type", "").lower()
        if element_type in ["button", "link", "menu_item", "tab"]:
            score += 0.2
        
        return score
    
    def _calculate_recovery_score(self, action: Dict[str, Any], 
                                 error_context: Optional[str]) -> float:
        """Calculate recovery score for an action."""
        score = 0.0
        
        # Base recovery score from UI-Venus
        score += action.get("recovery_score", 0.0)
        
        # Action type bonus for recovery
        action_type = action.get("type", "")
        if action_type in [ActionType.BACK, ActionType.HOME]:
            score += 0.5
        
        return score
    
    def _calculate_navigation_score(self, action: Dict[str, Any], 
                                   target_state: Optional[str]) -> float:
        """Calculate navigation score for an action."""
        score = 0.0
        
        # Base navigation score from UI-Venus
        score += action.get("navigation_score", 0.0)
        
        # Action type bonus for navigation
        action_type = action.get("type", "")
        if action_type in [ActionType.CLICK, ActionType.BACK, ActionType.HOME]:
            score += 0.3
        
        return score
    
    def _is_action_valid(self, action: Dict[str, Any], 
                        current_state: Dict[str, Any]) -> bool:
        """Check if an action is valid for the current state."""
        # Check if action was recently failed
        for failed_action in self._failed_actions[-5:]:  # Last 5 failed actions
            if self._actions_are_similar(action, failed_action["action"]):
                return False
        
        # Check confidence threshold
        confidence = action.get("confidence", 0.0)
        if confidence < 0.3:
            return False
        
        # Check if action is within bounds
        bounds = action.get("bounds", [])
        if bounds and len(bounds) >= 4:
            left, top, right, bottom = bounds[:4]
            if left >= right or top >= bottom:
                return False
        
        return True
    
    def _actions_are_similar(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Check if two actions are similar."""
        # Compare action type
        if action1.get("type") != action2.get("type"):
            return False
        
        # Compare target
        if action1.get("target") != action2.get("target"):
            return False
        
        # Compare coordinates (if present)
        coords1 = action1.get("coordinates", [])
        coords2 = action2.get("coordinates", [])
        if coords1 and coords2 and len(coords1) >= 2 and len(coords2) >= 2:
            # Check if coordinates are within 50 pixels
            distance = ((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2) ** 0.5
            if distance < 50:
                return True
        
        return False
    
    def _initialize_priority_weights(self) -> Dict[ActionPriority, float]:
        """Initialize priority weights for different action priorities."""
        return {
            ActionPriority.HIGH: 1.0,
            ActionPriority.MEDIUM: 0.7,
            ActionPriority.LOW: 0.4
        }
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return {
            "current_goal": self._current_goal,
            "action_history_size": len(self._action_history),
            "successful_actions": len(self._successful_actions),
            "failed_actions": len(self._failed_actions),
            "strategy": self.crawler_config.strategy,
            "breadth_first_queue_size": len(self._breadth_first_queue),
            "depth_first_stack_size": len(self._depth_first_stack)
        }
    
    def reset_planning_state(self) -> None:
        """Reset planning state."""
        self._action_history.clear()
        self._failed_actions.clear()
        self._successful_actions.clear()
        self._breadth_first_queue.clear()
        self._depth_first_stack.clear()
        self._exploration_context.clear()
        self._current_goal = ExplorationGoal.MAXIMIZE_COVERAGE
        
        logger.info("Planning state reset")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.reset_planning_state()
        if hasattr(self, 'action_suggester'):
            self.action_suggester.cleanup()
        if hasattr(self, 'element_detector'):
            self.element_detector.cleanup()
        logger.info("Action planner cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
