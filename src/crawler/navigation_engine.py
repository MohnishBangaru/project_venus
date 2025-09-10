"""
Navigation Engine

This module provides systematic exploration strategies and navigation logic
for the crawler, implementing different exploration algorithms and pathfinding.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import numpy as np
from PIL import Image
import time
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

from config.crawler_config import CrawlerConfig, CrawlingStrategy
from src.crawler.coverage_tracker import CoverageTracker, AppState, ExplorationStatus


logger = logging.getLogger(__name__)


class NavigationStrategy(str, Enum):
    """Navigation strategies for exploration."""
    SYSTEMATIC = "systematic"
    ADAPTIVE = "adaptive"
    RANDOM_WALK = "random_walk"
    PRIORITY_DRIVEN = "priority_driven"
    COVERAGE_OPTIMIZED = "coverage_optimized"


class NavigationGoal(str, Enum):
    """Navigation goals."""
    EXPLORE_UNEXPLORED = "explore_unexplored"
    REACH_TARGET = "reach_target"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    MINIMIZE_BACKTRACKING = "minimize_backtracking"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class NavigationPlan:
    """Represents a navigation plan."""
    goal: NavigationGoal
    strategy: NavigationStrategy
    target_states: List[str]
    current_path: List[str]
    estimated_actions: int
    confidence: float
    metadata: Dict[str, Any]


class NavigationEngine:
    """
    Systematic navigation engine for intelligent app exploration.
    
    Implements various navigation strategies and algorithms to optimize
    exploration coverage while minimizing redundant actions and backtracking.
    """
    
    def __init__(self, config: CrawlerConfig, coverage_tracker: CoverageTracker):
        """
        Initialize the navigation engine.
        
        Args:
            config: Crawler configuration
            coverage_tracker: Coverage tracker instance
        """
        self.config = config
        self.coverage_tracker = coverage_tracker
        
        # Navigation state
        self._current_strategy = NavigationStrategy.SYSTEMATIC
        self._current_goal = NavigationGoal.MAXIMIZE_COVERAGE
        self._navigation_history: List[str] = []
        self._backtrack_stack: deque = deque()
        self._exploration_frontier: Set[str] = set()
        
        # Strategy-specific state
        self._systematic_queue: deque = deque()
        self._adaptive_weights: Dict[str, float] = {}
        self._priority_queue: List[Tuple[str, float]] = []
        
        # Navigation metrics
        self._navigation_metrics = {
            "total_navigations": 0,
            "successful_navigations": 0,
            "failed_navigations": 0,
            "backtrack_count": 0,
            "exploration_efficiency": 0.0,
            "coverage_gain": 0.0
        }
        
        logger.info("Navigation engine initialized")
    
    def plan_navigation(self, 
                       current_state_id: str,
                       goal: Optional[NavigationGoal] = None,
                       target_states: Optional[List[str]] = None) -> NavigationPlan:
        """
        Plan navigation from current state to achieve the specified goal.
        
        Args:
            current_state_id: Current state ID
            goal: Navigation goal (uses current goal if None)
            target_states: Specific target states (if applicable)
            
        Returns:
            Navigation plan
        """
        try:
            # Use provided goal or current goal
            navigation_goal = goal or self._current_goal
            
            # Plan based on strategy
            if self._current_strategy == NavigationStrategy.SYSTEMATIC:
                plan = self._plan_systematic_navigation(current_state_id, navigation_goal, target_states)
            elif self._current_strategy == NavigationStrategy.ADAPTIVE:
                plan = self._plan_adaptive_navigation(current_state_id, navigation_goal, target_states)
            elif self._current_strategy == NavigationStrategy.RANDOM_WALK:
                plan = self._plan_random_walk_navigation(current_state_id, navigation_goal, target_states)
            elif self._current_strategy == NavigationStrategy.PRIORITY_DRIVEN:
                plan = self._plan_priority_driven_navigation(current_state_id, navigation_goal, target_states)
            elif self._current_strategy == NavigationStrategy.COVERAGE_OPTIMIZED:
                plan = self._plan_coverage_optimized_navigation(current_state_id, navigation_goal, target_states)
            else:
                logger.warning(f"Unknown strategy: {self._current_strategy}, using systematic")
                plan = self._plan_systematic_navigation(current_state_id, navigation_goal, target_states)
            
            # Update navigation state
            self._update_navigation_state(plan)
            
            logger.info(f"Planned navigation: {navigation_goal} using {self._current_strategy}")
            return plan
            
        except Exception as e:
            logger.error(f"Navigation planning failed: {e}")
            raise RuntimeError(f"Navigation planning failed: {e}")
    
    def get_next_exploration_target(self, current_state_id: str) -> Optional[str]:
        """
        Get the next state to explore from the current state.
        
        Args:
            current_state_id: Current state ID
            
        Returns:
            Next state ID to explore, or None if no targets available
        """
        try:
            # Get exploration candidates
            candidates = self.coverage_tracker.get_exploration_candidates(limit=20)
            
            if not candidates:
                logger.debug("No exploration candidates available")
                return None
            
            # Filter out current state
            candidates = [c for c in candidates if c != current_state_id]
            
            if not candidates:
                logger.debug("No exploration candidates after filtering current state")
                return None
            
            # Select best candidate based on strategy
            if self._current_strategy == NavigationStrategy.SYSTEMATIC:
                target = self._select_systematic_target(candidates, current_state_id)
            elif self._current_strategy == NavigationStrategy.ADAPTIVE:
                target = self._select_adaptive_target(candidates, current_state_id)
            elif self._current_strategy == NavigationStrategy.RANDOM_WALK:
                target = self._select_random_target(candidates)
            elif self._current_strategy == NavigationStrategy.PRIORITY_DRIVEN:
                target = self._select_priority_target(candidates, current_state_id)
            elif self._current_strategy == NavigationStrategy.COVERAGE_OPTIMIZED:
                target = self._select_coverage_optimized_target(candidates, current_state_id)
            else:
                target = candidates[0]  # Default to first candidate
            
            logger.debug(f"Selected exploration target: {target}")
            return target
            
        except Exception as e:
            logger.error(f"Failed to get next exploration target: {e}")
            return None
    
    def plan_path_to_state(self, from_state_id: str, to_state_id: str) -> Optional[List[str]]:
        """
        Plan a path from one state to another.
        
        Args:
            from_state_id: Source state ID
            to_state_id: Target state ID
            
        Returns:
            List of state IDs representing the path, or None if no path exists
        """
        try:
            # Use coverage tracker's pathfinding
            path = self.coverage_tracker.get_state_path(from_state_id, to_state_id)
            
            if path:
                logger.debug(f"Found path from {from_state_id} to {to_state_id}: {len(path)} states")
            else:
                logger.debug(f"No path found from {from_state_id} to {to_state_id}")
            
            return path
            
        except Exception as e:
            logger.error(f"Path planning failed: {e}")
            return None
    
    def should_backtrack(self, current_state_id: str) -> bool:
        """
        Determine if backtracking is needed from the current state.
        
        Args:
            current_state_id: Current state ID
            
        Returns:
            True if backtracking is recommended
        """
        try:
            # Check if current state is blocked or has no exploration options
            state = self.coverage_tracker._states.get(current_state_id)
            if not state:
                return True
            
            # Check if state is blocked or in error
            if state.exploration_status in [ExplorationStatus.BLOCKED, ExplorationStatus.ERROR]:
                return True
            
            # Check if we've been in this state too many times
            if state.visit_count > self.config.revisit_threshold:
                return True
            
            # Check if no new exploration options are available
            exploration_candidates = self.coverage_tracker.get_exploration_candidates(limit=5)
            if not exploration_candidates or current_state_id not in exploration_candidates:
                return True
            
            # Check if we're stuck in a loop
            if self._is_in_navigation_loop(current_state_id):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Backtrack decision failed: {e}")
            return True  # Default to backtracking on error
    
    def get_backtrack_target(self, current_state_id: str) -> Optional[str]:
        """
        Get the target state for backtracking.
        
        Args:
            current_state_id: Current state ID
            
        Returns:
            Target state ID for backtracking, or None if no target available
        """
        try:
            # Check backtrack stack first
            if self._backtrack_stack:
                target = self._backtrack_stack.pop()
                logger.debug(f"Backtracking to {target} from stack")
                return target
            
            # Find a state with unexplored options
            for state_id in reversed(self._navigation_history[-10:]):  # Check last 10 states
                if state_id != current_state_id:
                    state = self.coverage_tracker._states.get(state_id)
                    if state and state.exploration_status in [ExplorationStatus.UNEXPLORED, 
                                                             ExplorationStatus.PARTIALLY_EXPLORED]:
                        logger.debug(f"Backtracking to {state_id} with exploration potential")
                        return state_id
            
            # Find any state with exploration potential
            exploration_candidates = self.coverage_tracker.get_exploration_candidates(limit=10)
            if exploration_candidates:
                target = exploration_candidates[0]
                logger.debug(f"Backtracking to exploration candidate {target}")
                return target
            
            logger.debug("No backtrack target available")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get backtrack target: {e}")
            return None
    
    def update_navigation_result(self, 
                                from_state_id: str,
                                to_state_id: Optional[str],
                                success: bool,
                                action_count: int = 1) -> None:
        """
        Update navigation engine with the result of navigation.
        
        Args:
            from_state_id: Source state ID
            to_state_id: Target state ID (if successful)
            success: Whether navigation was successful
            action_count: Number of actions taken
        """
        try:
            # Update navigation history
            self._navigation_history.append(from_state_id)
            if to_state_id:
                self._navigation_history.append(to_state_id)
            
            # Update metrics
            self._navigation_metrics["total_navigations"] += 1
            if success:
                self._navigation_metrics["successful_navigations"] += 1
            else:
                self._navigation_metrics["failed_navigations"] += 1
            
            # Update strategy-specific state
            self._update_strategy_state(from_state_id, to_state_id, success, action_count)
            
            # Update exploration frontier
            if success and to_state_id:
                self._exploration_frontier.add(to_state_id)
            
            # Calculate exploration efficiency
            self._calculate_exploration_efficiency()
            
            logger.debug(f"Updated navigation result: {from_state_id} -> {to_state_id}, success={success}")
            
        except Exception as e:
            logger.error(f"Failed to update navigation result: {e}")
    
    def set_navigation_strategy(self, strategy: NavigationStrategy) -> None:
        """Set the navigation strategy."""
        self._current_strategy = strategy
        logger.info(f"Navigation strategy set to: {strategy}")
    
    def set_navigation_goal(self, goal: NavigationGoal) -> None:
        """Set the navigation goal."""
        self._current_goal = goal
        logger.info(f"Navigation goal set to: {goal}")
    
    def _plan_systematic_navigation(self, 
                                   current_state_id: str,
                                   goal: NavigationGoal,
                                   target_states: Optional[List[str]]) -> NavigationPlan:
        """Plan systematic navigation."""
        if goal == NavigationGoal.EXPLORE_UNEXPLORED:
            # Find unexplored states
            unexplored = self.coverage_tracker.get_unexplored_states()
            target_states = unexplored[:5]  # Limit to 5 targets
        elif goal == NavigationGoal.REACH_TARGET and target_states:
            # Use provided target states
            pass
        else:
            # Default to exploration candidates
            target_states = self.coverage_tracker.get_exploration_candidates(limit=5)
        
        # Plan path to first target
        current_path = [current_state_id]
        if target_states:
            path = self.plan_path_to_state(current_state_id, target_states[0])
            if path:
                current_path = path
        
        return NavigationPlan(
            goal=goal,
            strategy=NavigationStrategy.SYSTEMATIC,
            target_states=target_states or [],
            current_path=current_path,
            estimated_actions=len(current_path) - 1,
            confidence=0.8,
            metadata={"systematic_queue_size": len(self._systematic_queue)}
        )
    
    def _plan_adaptive_navigation(self, 
                                 current_state_id: str,
                                 goal: NavigationGoal,
                                 target_states: Optional[List[str]]) -> NavigationPlan:
        """Plan adaptive navigation based on exploration history."""
        # Get exploration candidates
        candidates = self.coverage_tracker.get_exploration_candidates(limit=10)
        
        # Apply adaptive weights
        weighted_candidates = []
        for candidate in candidates:
            weight = self._adaptive_weights.get(candidate, 1.0)
            weighted_candidates.append((candidate, weight))
        
        # Sort by weight
        weighted_candidates.sort(key=lambda x: x[1], reverse=True)
        target_states = [c[0] for c in weighted_candidates[:5]]
        
        # Plan path to best target
        current_path = [current_state_id]
        if target_states:
            path = self.plan_path_to_state(current_state_id, target_states[0])
            if path:
                current_path = path
        
        return NavigationPlan(
            goal=goal,
            strategy=NavigationStrategy.ADAPTIVE,
            target_states=target_states,
            current_path=current_path,
            estimated_actions=len(current_path) - 1,
            confidence=0.7,
            metadata={"adaptive_weights_count": len(self._adaptive_weights)}
        )
    
    def _plan_random_walk_navigation(self, 
                                    current_state_id: str,
                                    goal: NavigationGoal,
                                    target_states: Optional[List[str]]) -> NavigationPlan:
        """Plan random walk navigation."""
        # Get random exploration candidates
        candidates = self.coverage_tracker.get_exploration_candidates(limit=10)
        if candidates:
            target_states = random.sample(candidates, min(3, len(candidates)))
        
        # Random path (just current state for now)
        current_path = [current_state_id]
        
        return NavigationPlan(
            goal=goal,
            strategy=NavigationStrategy.RANDOM_WALK,
            target_states=target_states or [],
            current_path=current_path,
            estimated_actions=random.randint(1, 5),
            confidence=0.5,
            metadata={"random_seed": random.randint(0, 1000)}
        )
    
    def _plan_priority_driven_navigation(self, 
                                        current_state_id: str,
                                        goal: NavigationGoal,
                                        target_states: Optional[List[str]]) -> NavigationPlan:
        """Plan priority-driven navigation."""
        # Get priority-ordered candidates
        candidates = self.coverage_tracker.get_exploration_candidates(limit=10)
        
        # Calculate priorities
        priority_candidates = []
        for candidate in candidates:
            priority = self._calculate_state_priority(candidate, current_state_id)
            priority_candidates.append((candidate, priority))
        
        # Sort by priority
        priority_candidates.sort(key=lambda x: x[1], reverse=True)
        target_states = [c[0] for c in priority_candidates[:5]]
        
        # Plan path to highest priority target
        current_path = [current_state_id]
        if target_states:
            path = self.plan_path_to_state(current_state_id, target_states[0])
            if path:
                current_path = path
        
        return NavigationPlan(
            goal=goal,
            strategy=NavigationStrategy.PRIORITY_DRIVEN,
            target_states=target_states,
            current_path=current_path,
            estimated_actions=len(current_path) - 1,
            confidence=0.8,
            metadata={"priority_queue_size": len(self._priority_queue)}
        )
    
    def _plan_coverage_optimized_navigation(self, 
                                           current_state_id: str,
                                           goal: NavigationGoal,
                                           target_states: Optional[List[str]]) -> NavigationPlan:
        """Plan coverage-optimized navigation."""
        # Get coverage metrics
        coverage_metrics = self.coverage_tracker.get_coverage_metrics()
        
        # Find states that maximize coverage gain
        candidates = self.coverage_tracker.get_exploration_candidates(limit=10)
        coverage_candidates = []
        
        for candidate in candidates:
            coverage_gain = self._estimate_coverage_gain(candidate)
            coverage_candidates.append((candidate, coverage_gain))
        
        # Sort by coverage gain
        coverage_candidates.sort(key=lambda x: x[1], reverse=True)
        target_states = [c[0] for c in coverage_candidates[:5]]
        
        # Plan path to best coverage target
        current_path = [current_state_id]
        if target_states:
            path = self.plan_path_to_state(current_state_id, target_states[0])
            if path:
                current_path = path
        
        return NavigationPlan(
            goal=goal,
            strategy=NavigationStrategy.COVERAGE_OPTIMIZED,
            target_states=target_states,
            current_path=current_path,
            estimated_actions=len(current_path) - 1,
            confidence=0.9,
            metadata={"coverage_percentage": coverage_metrics.get("coverage_percentage", 0)}
        )
    
    def _select_systematic_target(self, candidates: List[str], current_state_id: str) -> str:
        """Select target using systematic approach."""
        # Add candidates to systematic queue if not already present
        for candidate in candidates:
            if candidate not in self._systematic_queue:
                self._systematic_queue.append(candidate)
        
        # Return first candidate from queue
        if self._systematic_queue:
            return self._systematic_queue.popleft()
        else:
            return candidates[0] if candidates else current_state_id
    
    def _select_adaptive_target(self, candidates: List[str], current_state_id: str) -> str:
        """Select target using adaptive approach."""
        # Apply adaptive weights
        weighted_candidates = []
        for candidate in candidates:
            weight = self._adaptive_weights.get(candidate, 1.0)
            weighted_candidates.append((candidate, weight))
        
        # Sort by weight and return best
        weighted_candidates.sort(key=lambda x: x[1], reverse=True)
        return weighted_candidates[0][0] if weighted_candidates else candidates[0]
    
    def _select_random_target(self, candidates: List[str]) -> str:
        """Select target randomly."""
        return random.choice(candidates) if candidates else ""
    
    def _select_priority_target(self, candidates: List[str], current_state_id: str) -> str:
        """Select target based on priority."""
        # Calculate priorities
        priority_candidates = []
        for candidate in candidates:
            priority = self._calculate_state_priority(candidate, current_state_id)
            priority_candidates.append((candidate, priority))
        
        # Sort by priority and return best
        priority_candidates.sort(key=lambda x: x[1], reverse=True)
        return priority_candidates[0][0] if priority_candidates else candidates[0]
    
    def _select_coverage_optimized_target(self, candidates: List[str], current_state_id: str) -> str:
        """Select target that maximizes coverage gain."""
        # Calculate coverage gains
        coverage_candidates = []
        for candidate in candidates:
            coverage_gain = self._estimate_coverage_gain(candidate)
            coverage_candidates.append((candidate, coverage_gain))
        
        # Sort by coverage gain and return best
        coverage_candidates.sort(key=lambda x: x[1], reverse=True)
        return coverage_candidates[0][0] if coverage_candidates else candidates[0]
    
    def _calculate_state_priority(self, state_id: str, current_state_id: str) -> float:
        """Calculate priority for a state."""
        priority = 0.0
        
        # Base priority from state type
        state = self.coverage_tracker._states.get(state_id)
        if state:
            if state.state_type == "screen":
                priority += 1.0
            elif state.state_type == "menu":
                priority += 0.8
            elif state.state_type == "dialog":
                priority += 0.6
        
        # Distance factor (closer states have higher priority)
        path = self.plan_path_to_state(current_state_id, state_id)
        if path:
            distance = len(path) - 1
            priority += max(0, 1.0 - distance * 0.1)
        
        # Exploration status factor
        if state:
            if state.exploration_status == ExplorationStatus.UNEXPLORED:
                priority += 0.5
            elif state.exploration_status == ExplorationStatus.PARTIALLY_EXPLORED:
                priority += 0.3
        
        return priority
    
    def _estimate_coverage_gain(self, state_id: str) -> float:
        """Estimate coverage gain from exploring a state."""
        state = self.coverage_tracker._states.get(state_id)
        if not state:
            return 0.0
        
        # Base coverage gain
        coverage_gain = 0.0
        
        # Unexplored states have higher coverage gain
        if state.exploration_status == ExplorationStatus.UNEXPLORED:
            coverage_gain += 1.0
        elif state.exploration_status == ExplorationStatus.PARTIALLY_EXPLORED:
            coverage_gain += 0.5
        
        # State type factor
        if state.state_type == "screen":
            coverage_gain += 0.3
        elif state.state_type == "menu":
            coverage_gain += 0.2
        
        return coverage_gain
    
    def _is_in_navigation_loop(self, current_state_id: str) -> bool:
        """Check if we're stuck in a navigation loop."""
        # Check if we've visited this state recently
        recent_history = self._navigation_history[-10:]
        if current_state_id in recent_history:
            # Count occurrences
            count = recent_history.count(current_state_id)
            if count >= 3:  # Visited 3 times in last 10 navigations
                return True
        
        return False
    
    def _update_navigation_state(self, plan: NavigationPlan) -> None:
        """Update navigation state based on plan."""
        # Add targets to appropriate queues
        if plan.strategy == NavigationStrategy.SYSTEMATIC:
            for target in plan.target_states:
                if target not in self._systematic_queue:
                    self._systematic_queue.append(target)
    
    def _update_strategy_state(self, 
                              from_state_id: str,
                              to_state_id: Optional[str],
                              success: bool,
                              action_count: int) -> None:
        """Update strategy-specific state."""
        if self._current_strategy == NavigationStrategy.ADAPTIVE:
            # Update adaptive weights
            if success and to_state_id:
                # Increase weight for successful navigation
                self._adaptive_weights[to_state_id] = self._adaptive_weights.get(to_state_id, 1.0) + 0.1
            else:
                # Decrease weight for failed navigation
                self._adaptive_weights[from_state_id] = max(0.1, self._adaptive_weights.get(from_state_id, 1.0) - 0.1)
        
        elif self._current_strategy == NavigationStrategy.PRIORITY_DRIVEN:
            # Update priority queue
            if success and to_state_id:
                priority = self._calculate_state_priority(to_state_id, from_state_id)
                self._priority_queue.append((to_state_id, priority))
                # Keep queue size manageable
                if len(self._priority_queue) > 20:
                    self._priority_queue = self._priority_queue[-20:]
    
    def _calculate_exploration_efficiency(self) -> None:
        """Calculate exploration efficiency metrics."""
        total_navigations = self._navigation_metrics["total_navigations"]
        successful_navigations = self._navigation_metrics["successful_navigations"]
        
        if total_navigations > 0:
            success_rate = successful_navigations / total_navigations
            coverage_metrics = self.coverage_tracker.get_coverage_metrics()
            coverage_percentage = coverage_metrics.get("coverage_percentage", 0)
            
            self._navigation_metrics["exploration_efficiency"] = success_rate * (coverage_percentage / 100)
    
    def get_navigation_metrics(self) -> Dict[str, Any]:
        """Get navigation metrics."""
        return self._navigation_metrics.copy()
    
    def get_navigation_stats(self) -> Dict[str, Any]:
        """Get detailed navigation statistics."""
        return {
            "current_strategy": self._current_strategy,
            "current_goal": self._current_goal,
            "navigation_history_size": len(self._navigation_history),
            "backtrack_stack_size": len(self._backtrack_stack),
            "exploration_frontier_size": len(self._exploration_frontier),
            "systematic_queue_size": len(self._systematic_queue),
            "adaptive_weights_count": len(self._adaptive_weights),
            "priority_queue_size": len(self._priority_queue),
            "metrics": self._navigation_metrics
        }
    
    def reset_navigation_state(self) -> None:
        """Reset navigation state."""
        self._navigation_history.clear()
        self._backtrack_stack.clear()
        self._exploration_frontier.clear()
        self._systematic_queue.clear()
        self._adaptive_weights.clear()
        self._priority_queue.clear()
        
        # Reset metrics
        for key in self._navigation_metrics:
            self._navigation_metrics[key] = 0
        
        logger.info("Navigation state reset")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.reset_navigation_state()
        logger.info("Navigation engine cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
