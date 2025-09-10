"""
State Analyzer

This module provides comprehensive app state analysis capabilities, including
state change detection, similarity analysis, and state classification.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

from config.crawler_config import CrawlerConfig
from src.crawler.coverage_tracker import AppState, StateType, ExplorationStatus


logger = logging.getLogger(__name__)


class StateChangeType(str, Enum):
    """Types of state changes."""
    NEW_SCREEN = "new_screen"
    DIALOG_OPENED = "dialog_opened"
    DIALOG_CLOSED = "dialog_closed"
    MENU_OPENED = "menu_opened"
    MENU_CLOSED = "menu_closed"
    FORM_CHANGED = "form_changed"
    CONTENT_LOADED = "content_loaded"
    ERROR_OCCURRED = "error_occurred"
    LOADING_STARTED = "loading_started"
    LOADING_FINISHED = "loading_finished"
    NAVIGATION = "navigation"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class StateSimilarityLevel(str, Enum):
    """Levels of state similarity."""
    IDENTICAL = "identical"
    VERY_SIMILAR = "very_similar"
    SIMILAR = "similar"
    DIFFERENT = "different"
    COMPLETELY_DIFFERENT = "completely_different"
    UNKNOWN = "unknown"


@dataclass
class StateChange:
    """Represents a state change."""
    change_id: str
    change_type: StateChangeType
    from_state_id: str
    to_state_id: str
    timestamp: float
    confidence: float
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class StateAnalysis:
    """Comprehensive state analysis result."""
    state_id: str
    state_type: StateType
    similarity_level: StateSimilarityLevel
    similarity_scores: Dict[str, float]
    change_history: List[StateChange]
    ui_elements: List[Dict[str, Any]]
    text_content: str
    visual_features: Dict[str, Any]
    metadata: Dict[str, Any]


class StateAnalyzer:
    """
    Comprehensive state analyzer for app state detection and analysis.
    
    Provides capabilities for detecting state changes, analyzing state similarity,
    classifying states, and extracting meaningful information from app states.
    """
    
    def __init__(self, config: CrawlerConfig):
        """
        Initialize the state analyzer.
        
        Args:
            config: Crawler configuration
        """
        self.config = config
        
        # Analysis state
        self._state_history: deque = deque(maxlen=config.max_state_history)
        self._change_history: List[StateChange] = []
        self._similarity_cache: Dict[str, Dict[str, float]] = {}
        
        # Analysis metrics
        self._analysis_metrics = {
            "total_analyses": 0,
            "state_changes_detected": 0,
            "similarity_calculations": 0,
            "cache_hits": 0,
            "analysis_time_avg_ms": 0.0
        }
        
        # Visual analysis settings
        self._similarity_threshold = config.state_similarity_threshold
        self._feature_extraction_enabled = True
        
        logger.info("State analyzer initialized")
    
    def analyze_state(self, 
                     image: Image.Image,
                     state_id: Optional[str] = None,
                     previous_state_id: Optional[str] = None) -> StateAnalysis:
        """
        Perform comprehensive analysis of a state.
        
        Args:
            image: State screenshot
            state_id: State ID (generated if None)
            previous_state_id: Previous state ID for change detection
            
        Returns:
            Comprehensive state analysis
        """
        try:
            start_time = time.time()
            
            # Generate state ID if not provided
            if not state_id:
                state_id = self._generate_state_id(image)
            
            # Detect state type
            state_type = self.classify_state_type(image)
            
            # Calculate similarity with previous state
            similarity_scores = {}
            if previous_state_id:
                similarity_scores[previous_state_id] = self._calculate_state_similarity(
                    image, previous_state_id
                )
            
            # Detect state changes
            change_history = []
            if previous_state_id:
                state_change = self.detect_state_change(image, previous_state_id, state_id)
                if state_change:
                    change_history.append(state_change)
                    self._change_history.append(state_change)
            
            # Extract UI elements (placeholder - would integrate with UI-Venus)
            ui_elements = self._extract_ui_elements(image)
            
            # Extract text content (placeholder - would use OCR)
            text_content = self._extract_text_content(image)
            
            # Extract visual features
            visual_features = self._extract_visual_features(image)
            
            # Create analysis result
            analysis = StateAnalysis(
                state_id=state_id,
                state_type=state_type,
                similarity_level=self._determine_similarity_level(similarity_scores),
                similarity_scores=similarity_scores,
                change_history=change_history,
                ui_elements=ui_elements,
                text_content=text_content,
                visual_features=visual_features,
                metadata={
                    "analysis_timestamp": time.time(),
                    "image_size": image.size,
                    "image_mode": image.mode
                }
            )
            
            # Update analysis state
            self._state_history.append({
                "state_id": state_id,
                "timestamp": time.time(),
                "analysis": analysis
            })
            
            # Update metrics
            self._update_analysis_metrics(time.time() - start_time)
            
            logger.debug(f"Analyzed state {state_id}: {state_type}")
            return analysis
            
        except Exception as e:
            logger.error(f"State analysis failed: {e}")
            raise RuntimeError(f"State analysis failed: {e}")
    
    def detect_state_change(self, 
                           current_image: Image.Image,
                           previous_state_id: str,
                           current_state_id: str) -> Optional[StateChange]:
        """
        Detect state changes between two states.
        
        Args:
            current_image: Current state image
            previous_state_id: Previous state ID
            current_state_id: Current state ID
            
        Returns:
            State change object if change detected, None otherwise
        """
        try:
            # Calculate similarity
            similarity_score = self._calculate_state_similarity(current_image, previous_state_id)
            
            # Determine change type based on similarity
            if similarity_score >= self._similarity_threshold:
                change_type = StateChangeType.NO_CHANGE
            else:
                change_type = self._classify_change_type(current_image, previous_state_id, similarity_score)
            
            # Create change record
            change = StateChange(
                change_id=self._generate_change_id(previous_state_id, current_state_id),
                change_type=change_type,
                from_state_id=previous_state_id,
                to_state_id=current_state_id,
                timestamp=time.time(),
                confidence=self._calculate_change_confidence(similarity_score, change_type),
                similarity_score=similarity_score,
                metadata={
                    "similarity_threshold": self._similarity_threshold,
                    "change_detection_method": "visual_similarity"
                }
            )
            
            logger.debug(f"Detected state change: {change_type} (similarity: {similarity_score:.3f})")
            return change
            
        except Exception as e:
            logger.error(f"State change detection failed: {e}")
            return None
    
    def calculate_state_similarity(self, 
                                  image1: Image.Image,
                                  image2: Image.Image) -> float:
        """
        Calculate similarity between two state images.
        
        Args:
            image1: First state image
            image2: Second state image
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Generate state IDs for caching
            state_id1 = self._generate_state_id(image1)
            state_id2 = self._generate_state_id(image2)
            
            # Check cache
            cache_key = f"{state_id1}_{state_id2}"
            if cache_key in self._similarity_cache:
                self._analysis_metrics["cache_hits"] += 1
                return self._similarity_cache[cache_key]
            
            # Calculate similarity
            similarity = self._calculate_visual_similarity(image1, image2)
            
            # Cache result
            self._similarity_cache[cache_key] = similarity
            self._analysis_metrics["similarity_calculations"] += 1
            
            return similarity
            
        except Exception as e:
            logger.error(f"State similarity calculation failed: {e}")
            return 0.0
    
    def find_similar_states(self, 
                           image: Image.Image,
                           threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find states similar to the given image.
        
        Args:
            image: State image
            threshold: Similarity threshold
            
        Returns:
            List of (state_id, similarity_score) tuples
        """
        try:
            similar_states = []
            
            # Check against state history
            for state_record in self._state_history:
                state_id = state_record["state_id"]
                state_analysis = state_record["analysis"]
                
                # Get cached similarity if available
                current_state_id = self._generate_state_id(image)
                cache_key = f"{current_state_id}_{state_id}"
                
                if cache_key in self._similarity_cache:
                    similarity = self._similarity_cache[cache_key]
                else:
                    # Calculate similarity (would need the original image)
                    similarity = 0.5  # Placeholder
                
                if similarity >= threshold:
                    similar_states.append((state_id, similarity))
            
            # Sort by similarity
            similar_states.sort(key=lambda x: x[1], reverse=True)
            
            return similar_states
            
        except Exception as e:
            logger.error(f"Similar state search failed: {e}")
            return []
    
    def classify_state_type(self, image: Image.Image) -> StateType:
        """
        Classify the type of a state.
        
        Args:
            image: State image
            
        Returns:
            State type classification
        """
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Simple classification based on visual features
            # In a real implementation, this would use more sophisticated ML models
            
            # Check for dialog-like features (smaller, centered content)
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Check for loading indicators
            if self._has_loading_indicators(img_array):
                return StateType.LOADING
            
            # Check for error indicators
            if self._has_error_indicators(img_array):
                return StateType.ERROR
            
            # Check for form-like features
            if self._has_form_elements(img_array):
                return StateType.FORM
            
            # Check for menu-like features
            if self._has_menu_elements(img_array):
                return StateType.MENU
            
            # Check for dialog-like features
            if self._is_dialog_like(img_array):
                return StateType.DIALOG
            
            # Default to screen
            return StateType.SCREEN
            
        except Exception as e:
            logger.error(f"State type classification failed: {e}")
            return StateType.UNKNOWN
    
    def get_state_change_history(self, limit: int = 50) -> List[StateChange]:
        """Get recent state change history."""
        return self._change_history[-limit:]
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get analysis metrics."""
        return self._analysis_metrics.copy()
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about analyzed states."""
        if not self._state_history:
            return {}
        
        # Count state types
        state_types = defaultdict(int)
        similarity_levels = defaultdict(int)
        
        for state_record in self._state_history:
            analysis = state_record["analysis"]
            state_types[analysis.state_type] += 1
            similarity_levels[analysis.similarity_level] += 1
        
        return {
            "total_states_analyzed": len(self._state_history),
            "state_types": dict(state_types),
            "similarity_levels": dict(similarity_levels),
            "total_changes_detected": len(self._change_history),
            "cache_size": len(self._similarity_cache)
        }
    
    def _generate_state_id(self, image: Image.Image) -> str:
        """Generate a unique state ID from image."""
        # Use image hash as base for state ID
        image_bytes = image.tobytes()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        return f"state_{image_hash[:12]}"
    
    def _generate_change_id(self, from_state_id: str, to_state_id: str) -> str:
        """Generate a unique change ID."""
        change_str = f"{from_state_id}_{to_state_id}_{time.time()}"
        return hashlib.md5(change_str.encode()).hexdigest()[:12]
    
    def _calculate_state_similarity(self, image: Image.Image, state_id: str) -> float:
        """Calculate similarity with a specific state."""
        # This is a placeholder - in a real implementation, you would:
        # 1. Retrieve the original image for the state_id
        # 2. Calculate visual similarity between the images
        # 3. Return the similarity score
        
        # For now, return a random similarity score
        return np.random.uniform(0.3, 0.9)
    
    def _calculate_visual_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """Calculate visual similarity between two images."""
        try:
            # Convert to numpy arrays
            img1_array = np.array(image1)
            img2_array = np.array(image2)
            
            # Resize images to same size for comparison
            target_size = (224, 224)  # Standard size for feature extraction
            img1_resized = cv2.resize(img1_array, target_size)
            img2_resized = cv2.resize(img2_array, target_size)
            
            # Convert to grayscale for simpler comparison
            if len(img1_resized.shape) == 3:
                img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1_resized
            
            if len(img2_resized.shape) == 3:
                img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
            else:
                img2_gray = img2_resized
            
            # Calculate structural similarity
            similarity = self._calculate_structural_similarity(img1_gray, img2_gray)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Visual similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate structural similarity between two grayscale images."""
        try:
            # Simple pixel-wise similarity
            diff = np.abs(img1.astype(float) - img2.astype(float))
            similarity = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Structural similarity calculation failed: {e}")
            return 0.0
    
    def _classify_change_type(self, 
                             current_image: Image.Image,
                             previous_state_id: str,
                             similarity_score: float) -> StateChangeType:
        """Classify the type of state change."""
        try:
            # Classify based on similarity score and visual features
            if similarity_score < 0.1:
                return StateChangeType.NEW_SCREEN
            elif similarity_score < 0.3:
                return StateChangeType.NAVIGATION
            elif similarity_score < 0.6:
                # Check for specific change types
                if self._is_dialog_like(np.array(current_image)):
                    return StateChangeType.DIALOG_OPENED
                elif self._has_menu_elements(np.array(current_image)):
                    return StateChangeType.MENU_OPENED
                elif self._has_form_elements(np.array(current_image)):
                    return StateChangeType.FORM_CHANGED
                else:
                    return StateChangeType.CONTENT_LOADED
            else:
                return StateChangeType.UNKNOWN
                
        except Exception as e:
            logger.error(f"Change type classification failed: {e}")
            return StateChangeType.UNKNOWN
    
    def _calculate_change_confidence(self, similarity_score: float, change_type: StateChangeType) -> float:
        """Calculate confidence in the detected change."""
        # Base confidence on similarity score and change type
        if change_type == StateChangeType.NO_CHANGE:
            return similarity_score
        else:
            return 1.0 - similarity_score
    
    def _determine_similarity_level(self, similarity_scores: Dict[str, float]) -> StateSimilarityLevel:
        """Determine similarity level based on scores."""
        if not similarity_scores:
            return StateSimilarityLevel.UNKNOWN
        
        max_similarity = max(similarity_scores.values())
        
        if max_similarity >= 0.95:
            return StateSimilarityLevel.IDENTICAL
        elif max_similarity >= 0.8:
            return StateSimilarityLevel.VERY_SIMILAR
        elif max_similarity >= 0.6:
            return StateSimilarityLevel.SIMILAR
        elif max_similarity >= 0.3:
            return StateSimilarityLevel.DIFFERENT
        else:
            return StateSimilarityLevel.COMPLETELY_DIFFERENT
    
    def _extract_ui_elements(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract UI elements from the image (placeholder)."""
        # This would integrate with UI-Venus element detection
        # For now, return empty list
        return []
    
    def _extract_text_content(self, image: Image.Image) -> str:
        """Extract text content from the image (placeholder)."""
        # This would use OCR to extract text
        # For now, return empty string
        return ""
    
    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features from the image."""
        try:
            img_array = np.array(image)
            
            features = {
                "size": image.size,
                "mode": image.mode,
                "aspect_ratio": image.size[0] / image.size[1],
                "brightness": np.mean(img_array),
                "contrast": np.std(img_array),
                "dominant_colors": self._extract_dominant_colors(img_array),
                "edge_density": self._calculate_edge_density(img_array)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            return {}
    
    def _extract_dominant_colors(self, img_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the image."""
        try:
            # Simple dominant color extraction
            # In a real implementation, you would use more sophisticated methods
            colors = []
            
            # Sample colors from different regions
            height, width = img_array.shape[:2]
            regions = [
                (0, 0, width//2, height//2),  # Top-left
                (width//2, 0, width, height//2),  # Top-right
                (0, height//2, width//2, height),  # Bottom-left
                (width//2, height//2, width, height)  # Bottom-right
            ]
            
            for region in regions:
                x1, y1, x2, y2 = region
                region_pixels = img_array[y1:y2, x1:x2]
                if len(region_pixels) > 0:
                    avg_color = np.mean(region_pixels, axis=(0, 1))
                    colors.append(tuple(avg_color.astype(int)))
            
            return colors
            
        except Exception as e:
            logger.error(f"Dominant color extraction failed: {e}")
            return []
    
    def _calculate_edge_density(self, img_array: np.ndarray) -> float:
        """Calculate edge density in the image."""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Calculate edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_density = edge_pixels / total_pixels
            
            return edge_density
            
        except Exception as e:
            logger.error(f"Edge density calculation failed: {e}")
            return 0.0
    
    def _has_loading_indicators(self, img_array: np.ndarray) -> bool:
        """Check if image has loading indicators."""
        # Simple heuristic - look for circular patterns
        # In a real implementation, you would use more sophisticated detection
        return False
    
    def _has_error_indicators(self, img_array: np.ndarray) -> bool:
        """Check if image has error indicators."""
        # Simple heuristic - look for red colors or error text
        # In a real implementation, you would use more sophisticated detection
        return False
    
    def _has_form_elements(self, img_array: np.ndarray) -> bool:
        """Check if image has form elements."""
        # Simple heuristic - look for rectangular input-like areas
        # In a real implementation, you would use more sophisticated detection
        return False
    
    def _has_menu_elements(self, img_array: np.ndarray) -> bool:
        """Check if image has menu elements."""
        # Simple heuristic - look for list-like structures
        # In a real implementation, you would use more sophisticated detection
        return False
    
    def _is_dialog_like(self, img_array: np.ndarray) -> bool:
        """Check if image is dialog-like."""
        # Simple heuristic - check aspect ratio and content distribution
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        
        # Dialogs are often more square-like
        return 0.7 <= aspect_ratio <= 1.3
    
    def _update_analysis_metrics(self, analysis_time: float) -> None:
        """Update analysis metrics."""
        self._analysis_metrics["total_analyses"] += 1
        
        # Update average analysis time
        total_analyses = self._analysis_metrics["total_analyses"]
        current_avg = self._analysis_metrics["analysis_time_avg_ms"]
        new_avg = ((current_avg * (total_analyses - 1)) + (analysis_time * 1000)) / total_analyses
        self._analysis_metrics["analysis_time_avg_ms"] = new_avg
    
    def clear_analysis_data(self) -> None:
        """Clear analysis data."""
        self._state_history.clear()
        self._change_history.clear()
        self._similarity_cache.clear()
        
        # Reset metrics
        for key in self._analysis_metrics:
            self._analysis_metrics[key] = 0
        
        logger.info("State analyzer data cleared")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_analysis_data()
        logger.info("State analyzer cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
