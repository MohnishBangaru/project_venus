"""
Coverage Tracker

This module provides comprehensive coverage tracking capabilities for the crawler,
including state management, exploration tracking, and coverage optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

from config.crawler_config import CrawlerConfig


logger = logging.getLogger(__name__)


class StateType(str, Enum):
    """Types of app states."""
    SCREEN = "screen"
    DIALOG = "dialog"
    MENU = "menu"
    FORM = "form"
    ERROR = "error"
    LOADING = "loading"
    UNKNOWN = "unknown"


class ExplorationStatus(str, Enum):
    """Exploration status of a state."""
    UNEXPLORED = "unexplored"
    PARTIALLY_EXPLORED = "partially_explored"
    FULLY_EXPLORED = "fully_explored"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class AppState:
    """Represents an app state with metadata."""
    id: str
    state_type: StateType
    screenshot_path: Optional[str]
    screenshot_hash: str
    timestamp: float
    exploration_status: ExplorationStatus
    visit_count: int
    last_visited: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ActionRecord:
    """Record of an action taken."""
    action_id: str
    action_type: str
    target: str
    coordinates: Tuple[int, int]
    timestamp: float
    success: bool
    from_state_id: str
    to_state_id: Optional[str]
    duration_ms: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        """Create from dictionary."""
        return cls(**data)


class CoverageTracker:
    """
    Comprehensive coverage tracker for app exploration.
    
    Tracks visited states, actions taken, exploration progress, and provides
    insights for optimizing coverage and identifying unexplored areas.
    """
    
    def __init__(self, config: CrawlerConfig):
        """
        Initialize the coverage tracker.
        
        Args:
            config: Crawler configuration
        """
        self.config = config
        
        # State tracking
        self._states: Dict[str, AppState] = {}
        self._state_transitions: Dict[str, List[str]] = defaultdict(list)
        self._action_records: List[ActionRecord] = []
        
        # Exploration tracking
        self._exploration_queue: deque = deque()
        self._blocked_states: Set[str] = set()
        self._error_states: Set[str] = set()
        
        # Coverage metrics
        self._coverage_metrics = {
            "total_states": 0,
            "explored_states": 0,
            "partially_explored_states": 0,
            "blocked_states": 0,
            "error_states": 0,
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "coverage_percentage": 0.0,
            "exploration_efficiency": 0.0
        }
        
        # State similarity tracking
        self._state_similarity_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("Coverage tracker initialized")
    
    def add_state(self, 
                  image: Image.Image,
                  state_type: StateType = StateType.SCREEN,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new state to the tracker.
        
        Args:
            image: Screenshot of the state
            state_type: Type of the state
            metadata: Additional metadata about the state
            
        Returns:
            State ID
        """
        try:
            # Generate state ID from image hash
            state_id = self._generate_state_id(image)
            
            # Check if state already exists
            if state_id in self._states:
                existing_state = self._states[state_id]
                existing_state.visit_count += 1
                existing_state.last_visited = time.time()
                logger.debug(f"State {state_id} already exists, updated visit count")
                return state_id
            
            # Create new state
            screenshot_hash = self._calculate_image_hash(image)
            current_time = time.time()
            
            new_state = AppState(
                id=state_id,
                state_type=state_type,
                screenshot_path=None,  # Will be set if saving screenshots
                screenshot_hash=screenshot_hash,
                timestamp=current_time,
                exploration_status=ExplorationStatus.UNEXPLORED,
                visit_count=1,
                last_visited=current_time,
                metadata=metadata or {}
            )
            
            # Add to tracker
            self._states[state_id] = new_state
            self._exploration_queue.append(state_id)
            
            # Update metrics
            self._update_coverage_metrics()
            
            logger.info(f"Added new state: {state_id} ({state_type})")
            return state_id
            
        except Exception as e:
            logger.error(f"Failed to add state: {e}")
            raise RuntimeError(f"Failed to add state: {e}")
    
    def record_action(self, 
                     action: Dict[str, Any],
                     from_state_id: str,
                     to_state_id: Optional[str] = None,
                     success: bool = True,
                     duration_ms: int = 0) -> str:
        """
        Record an action taken between states.
        
        Args:
            action: Action dictionary
            from_state_id: Source state ID
            to_state_id: Target state ID (if known)
            success: Whether the action was successful
            duration_ms: Action duration in milliseconds
            
        Returns:
            Action record ID
        """
        try:
            # Generate action ID
            action_id = self._generate_action_id(action, from_state_id)
            
            # Create action record
            action_record = ActionRecord(
                action_id=action_id,
                action_type=action.get("type", "unknown"),
                target=action.get("target", "unknown"),
                coordinates=tuple(action.get("coordinates", [0, 0])),
                timestamp=time.time(),
                success=success,
                from_state_id=from_state_id,
                to_state_id=to_state_id,
                duration_ms=duration_ms,
                metadata=action.get("metadata", {})
            )
            
            # Add to records
            self._action_records.append(action_record)
            
            # Update state transitions
            if to_state_id and success:
                self._state_transitions[from_state_id].append(to_state_id)
            
            # Update metrics
            self._update_coverage_metrics()
            
            logger.debug(f"Recorded action: {action_id} ({action_record.action_type})")
            return action_id
            
        except Exception as e:
            logger.error(f"Failed to record action: {e}")
            raise RuntimeError(f"Failed to record action: {e}")
    
    def update_state_exploration(self, 
                                state_id: str,
                                status: ExplorationStatus,
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the exploration status of a state.
        
        Args:
            state_id: State ID
            status: New exploration status
            metadata: Additional metadata
        """
        try:
            if state_id not in self._states:
                logger.warning(f"State {state_id} not found for exploration update")
                return
            
            state = self._states[state_id]
            state.exploration_status = status
            
            # Update metadata
            if metadata:
                state.metadata.update(metadata)
            
            # Update tracking sets
            if status == ExplorationStatus.BLOCKED:
                self._blocked_states.add(state_id)
            elif status == ExplorationStatus.ERROR:
                self._error_states.add(state_id)
            elif status == ExplorationStatus.FULLY_EXPLORED:
                # Remove from exploration queue if fully explored
                if state_id in self._exploration_queue:
                    self._exploration_queue.remove(state_id)
            
            # Update metrics
            self._update_coverage_metrics()
            
            logger.debug(f"Updated state {state_id} exploration status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update state exploration: {e}")
    
    def get_unexplored_states(self) -> List[str]:
        """Get list of unexplored state IDs."""
        unexplored = [
            state_id for state_id, state in self._states.items()
            if state.exploration_status == ExplorationStatus.UNEXPLORED
        ]
        return unexplored
    
    def get_partially_explored_states(self) -> List[str]:
        """Get list of partially explored state IDs."""
        partially_explored = [
            state_id for state_id, state in self._states.items()
            if state.exploration_status == ExplorationStatus.PARTIALLY_EXPLORED
        ]
        return partially_explored
    
    def get_exploration_candidates(self, limit: int = 10) -> List[str]:
        """
        Get states that are good candidates for exploration.
        
        Args:
            limit: Maximum number of candidates to return
            
        Returns:
            List of state IDs prioritized for exploration
        """
        candidates = []
        
        # Prioritize unexplored states
        unexplored = self.get_unexplored_states()
        candidates.extend(unexplored[:limit])
        
        # Add partially explored states if we need more
        if len(candidates) < limit:
            partially_explored = self.get_partially_explored_states()
            remaining = limit - len(candidates)
            candidates.extend(partially_explored[:remaining])
        
        # Remove blocked and error states
        candidates = [s for s in candidates if s not in self._blocked_states and s not in self._error_states]
        
        return candidates[:limit]
    
    def get_state_transitions(self, state_id: str) -> List[str]:
        """Get states that can be reached from the given state."""
        return self._state_transitions.get(state_id, [])
    
    def get_state_path(self, from_state_id: str, to_state_id: str) -> Optional[List[str]]:
        """
        Find a path between two states using BFS.
        
        Args:
            from_state_id: Source state ID
            to_state_id: Target state ID
            
        Returns:
            List of state IDs representing the path, or None if no path exists
        """
        try:
            if from_state_id == to_state_id:
                return [from_state_id]
            
            # BFS to find shortest path
            queue = deque([(from_state_id, [from_state_id])])
            visited = {from_state_id}
            
            while queue:
                current_state, path = queue.popleft()
                
                # Get transitions from current state
                transitions = self._state_transitions.get(current_state, [])
                
                for next_state in transitions:
                    if next_state == to_state_id:
                        return path + [next_state]
                    
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [next_state]))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find state path: {e}")
            return None
    
    def calculate_state_similarity(self, state_id1: str, state_id2: str) -> float:
        """
        Calculate similarity between two states.
        
        Args:
            state_id1: First state ID
            state_id2: Second state ID
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Check cache first
            cache_key = f"{state_id1}_{state_id2}"
            if cache_key in self._state_similarity_cache:
                return self._state_similarity_cache[cache_key]
            
            if state_id1 not in self._states or state_id2 not in self._states:
                return 0.0
            
            state1 = self._states[state_id1]
            state2 = self._states[state_id2]
            
            # If hashes are identical, states are identical
            if state1.screenshot_hash == state2.screenshot_hash:
                similarity = 1.0
            else:
                # For now, use a simple similarity based on state type and metadata
                # In a real implementation, you would use image similarity algorithms
                similarity = self._calculate_metadata_similarity(state1, state2)
            
            # Cache result
            self._state_similarity_cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            logger.error(f"Failed to calculate state similarity: {e}")
            return 0.0
    
    def find_similar_states(self, state_id: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find states similar to the given state.
        
        Args:
            state_id: State ID to find similar states for
            threshold: Similarity threshold
            
        Returns:
            List of (state_id, similarity_score) tuples
        """
        similar_states = []
        
        for other_state_id in self._states:
            if other_state_id != state_id:
                similarity = self.calculate_state_similarity(state_id, other_state_id)
                if similarity >= threshold:
                    similar_states.append((other_state_id, similarity))
        
        # Sort by similarity score
        similar_states.sort(key=lambda x: x[1], reverse=True)
        
        return similar_states
    
    def get_coverage_metrics(self) -> Dict[str, Any]:
        """Get current coverage metrics."""
        return self._coverage_metrics.copy()
    
    def get_exploration_progress(self) -> Dict[str, Any]:
        """Get detailed exploration progress information."""
        total_states = len(self._states)
        explored_states = len([s for s in self._states.values() 
                              if s.exploration_status in [ExplorationStatus.PARTIALLY_EXPLORED, 
                                                         ExplorationStatus.FULLY_EXPLORED]])
        
        progress = {
            "total_states": total_states,
            "explored_states": explored_states,
            "unexplored_states": len(self.get_unexplored_states()),
            "partially_explored_states": len(self.get_partially_explored_states()),
            "fully_explored_states": len([s for s in self._states.values() 
                                        if s.exploration_status == ExplorationStatus.FULLY_EXPLORED]),
            "blocked_states": len(self._blocked_states),
            "error_states": len(self._error_states),
            "coverage_percentage": (explored_states / total_states * 100) if total_states > 0 else 0,
            "exploration_queue_size": len(self._exploration_queue),
            "total_actions": len(self._action_records),
            "successful_actions": len([a for a in self._action_records if a.success]),
            "failed_actions": len([a for a in self._action_records if not a.success])
        }
        
        return progress
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about states."""
        if not self._states:
            return {}
        
        state_types = defaultdict(int)
        exploration_statuses = defaultdict(int)
        visit_counts = []
        
        for state in self._states.values():
            state_types[state.state_type] += 1
            exploration_statuses[state.exploration_status] += 1
            visit_counts.append(state.visit_count)
        
        return {
            "state_types": dict(state_types),
            "exploration_statuses": dict(exploration_statuses),
            "visit_count_stats": {
                "min": min(visit_counts) if visit_counts else 0,
                "max": max(visit_counts) if visit_counts else 0,
                "avg": sum(visit_counts) / len(visit_counts) if visit_counts else 0
            }
        }
    
    def export_coverage_data(self, file_path: str) -> None:
        """
        Export coverage data to a file.
        
        Args:
            file_path: Path to export file
        """
        try:
            export_data = {
                "states": {state_id: state.to_dict() for state_id, state in self._states.items()},
                "action_records": [record.to_dict() for record in self._action_records],
                "state_transitions": dict(self._state_transitions),
                "coverage_metrics": self._coverage_metrics,
                "exploration_progress": self.get_exploration_progress(),
                "export_timestamp": time.time()
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Coverage data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export coverage data: {e}")
            raise RuntimeError(f"Failed to export coverage data: {e}")
    
    def import_coverage_data(self, file_path: str) -> None:
        """
        Import coverage data from a file.
        
        Args:
            file_path: Path to import file
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            # Import states
            self._states = {
                state_id: AppState.from_dict(state_data)
                for state_id, state_data in import_data.get("states", {}).items()
            }
            
            # Import action records
            self._action_records = [
                ActionRecord.from_dict(record_data)
                for record_data in import_data.get("action_records", [])
            ]
            
            # Import state transitions
            self._state_transitions = defaultdict(list, import_data.get("state_transitions", {}))
            
            # Rebuild tracking sets
            self._blocked_states = {
                state_id for state_id, state in self._states.items()
                if state.exploration_status == ExplorationStatus.BLOCKED
            }
            
            self._error_states = {
                state_id for state_id, state in self._states.items()
                if state.exploration_status == ExplorationStatus.ERROR
            }
            
            # Rebuild exploration queue
            self._exploration_queue = deque([
                state_id for state_id, state in self._states.items()
                if state.exploration_status in [ExplorationStatus.UNEXPLORED, 
                                               ExplorationStatus.PARTIALLY_EXPLORED]
            ])
            
            # Update metrics
            self._update_coverage_metrics()
            
            logger.info(f"Coverage data imported from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to import coverage data: {e}")
            raise RuntimeError(f"Failed to import coverage data: {e}")
    
    def _generate_state_id(self, image: Image.Image) -> str:
        """Generate a unique state ID from image."""
        # Use image hash as base for state ID
        image_hash = self._calculate_image_hash(image)
        return f"state_{image_hash[:12]}"
    
    def _generate_action_id(self, action: Dict[str, Any], from_state_id: str) -> str:
        """Generate a unique action ID."""
        action_str = f"{action.get('type', 'unknown')}_{action.get('target', 'unknown')}_{from_state_id}_{time.time()}"
        return hashlib.md5(action_str.encode()).hexdigest()[:12]
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate hash of an image."""
        # Convert to bytes and hash
        image_bytes = image.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _calculate_metadata_similarity(self, state1: AppState, state2: AppState) -> float:
        """Calculate similarity based on metadata."""
        similarity = 0.0
        
        # State type similarity
        if state1.state_type == state2.state_type:
            similarity += 0.5
        
        # Metadata similarity (simple key overlap)
        keys1 = set(state1.metadata.keys())
        keys2 = set(state2.metadata.keys())
        if keys1 or keys2:
            key_overlap = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
            similarity += key_overlap * 0.5
        
        return min(similarity, 1.0)
    
    def _update_coverage_metrics(self) -> None:
        """Update coverage metrics."""
        total_states = len(self._states)
        explored_states = len([s for s in self._states.values() 
                              if s.exploration_status in [ExplorationStatus.PARTIALLY_EXPLORED, 
                                                         ExplorationStatus.FULLY_EXPLORED]])
        
        self._coverage_metrics.update({
            "total_states": total_states,
            "explored_states": explored_states,
            "partially_explored_states": len(self.get_partially_explored_states()),
            "blocked_states": len(self._blocked_states),
            "error_states": len(self._error_states),
            "total_actions": len(self._action_records),
            "successful_actions": len([a for a in self._action_records if a.success]),
            "failed_actions": len([a for a in self._action_records if not a.success]),
            "coverage_percentage": (explored_states / total_states * 100) if total_states > 0 else 0
        })
        
        # Calculate exploration efficiency
        if self._coverage_metrics["total_actions"] > 0:
            success_rate = self._coverage_metrics["successful_actions"] / self._coverage_metrics["total_actions"]
            self._coverage_metrics["exploration_efficiency"] = success_rate * self._coverage_metrics["coverage_percentage"] / 100
    
    def clear_data(self) -> None:
        """Clear all tracking data."""
        self._states.clear()
        self._state_transitions.clear()
        self._action_records.clear()
        self._exploration_queue.clear()
        self._blocked_states.clear()
        self._error_states.clear()
        self._state_similarity_cache.clear()
        
        # Reset metrics
        for key in self._coverage_metrics:
            self._coverage_metrics[key] = 0
        
        logger.info("Coverage tracker data cleared")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_data()
        logger.info("Coverage tracker cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
