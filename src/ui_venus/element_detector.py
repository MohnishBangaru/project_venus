"""
UI-Venus Element Detector

This module provides specialized element detection capabilities using UI-Venus,
focusing on grounding tasks and element identification.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from .model_client import UIVenusModelClient
from config.ui_venus_config import UIVenusConfig


logger = logging.getLogger(__name__)


class UIVenusElementDetector:
    """
    Specialized element detector using UI-Venus grounding capabilities.
    
    Focuses on identifying and localizing UI elements in screenshots
    with high accuracy and confidence scores.
    """
    
    def __init__(self, config: UIVenusConfig):
        """
        Initialize the element detector.
        
        Args:
            config: UI-Venus configuration
        """
        self.config = config
        self.model_client = UIVenusModelClient(config)
        self._element_cache = {}
        self._cache_size = 100  # Maximum cached elements
    
    def detect_all_elements(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect all UI elements in the given image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of detected elements with bounding boxes and metadata
        """
        try:
            # Get cache key
            cache_key = self._get_cache_key(image)
            if cache_key in self._element_cache:
                logger.debug("Returning cached elements")
                return self._element_cache[cache_key]
            
            # Detect elements using UI-Venus
            elements = self.model_client.detect_elements(image)
            
            # Post-process elements
            processed_elements = self._post_process_elements(elements, image)
            
            # Cache results
            self._cache_elements(cache_key, processed_elements)
            
            logger.info(f"Detected {len(processed_elements)} elements")
            return processed_elements
            
        except Exception as e:
            logger.error(f"Element detection failed: {e}")
            raise RuntimeError(f"Element detection failed: {e}")
    
    def detect_clickable_elements(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect only clickable UI elements (buttons, links, etc.).
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of clickable elements
        """
        all_elements = self.detect_all_elements(image)
        
        clickable_types = {
            "button", "link", "clickable", "menu_item", "tab", 
            "checkbox", "radio", "toggle", "switch"
        }
        
        clickable_elements = [
            element for element in all_elements
            if element.get("type", "").lower() in clickable_types
        ]
        
        logger.info(f"Detected {len(clickable_elements)} clickable elements")
        return clickable_elements
    
    def detect_input_elements(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect input elements (text fields, forms, etc.).
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of input elements
        """
        all_elements = self.detect_all_elements(image)
        
        input_types = {
            "input", "text_input", "password", "email", "search",
            "textarea", "form", "field"
        }
        
        input_elements = [
            element for element in all_elements
            if element.get("type", "").lower() in input_types
        ]
        
        logger.info(f"Detected {len(input_elements)} input elements")
        return input_elements
    
    def detect_navigation_elements(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect navigation elements (menus, tabs, back buttons, etc.).
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of navigation elements
        """
        all_elements = self.detect_all_elements(image)
        
        navigation_types = {
            "menu", "tab", "navigation", "back", "forward", "home",
            "breadcrumb", "pagination", "sidebar", "drawer"
        }
        
        navigation_elements = [
            element for element in all_elements
            if element.get("type", "").lower() in navigation_types
        ]
        
        logger.info(f"Detected {len(navigation_elements)} navigation elements")
        return navigation_elements
    
    def detect_elements_by_type(self, image: Union[str, Path, Image.Image, np.ndarray], 
                               element_type: str) -> List[Dict[str, Any]]:
        """
        Detect elements of a specific type.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            element_type: Type of elements to detect
            
        Returns:
            List of elements of the specified type
        """
        all_elements = self.detect_all_elements(image)
        
        filtered_elements = [
            element for element in all_elements
            if element.get("type", "").lower() == element_type.lower()
        ]
        
        logger.info(f"Detected {len(filtered_elements)} {element_type} elements")
        return filtered_elements
    
    def get_element_at_position(self, image: Union[str, Path, Image.Image, np.ndarray], 
                               x: int, y: int) -> Optional[Dict[str, Any]]:
        """
        Get the element at a specific position.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Element at the specified position, or None if not found
        """
        elements = self.detect_all_elements(image)
        
        for element in elements:
            bounds = element.get("bounds", [])
            if len(bounds) >= 4:
                left, top, right, bottom = bounds[:4]
                if left <= x <= right and top <= y <= bottom:
                    return element
        
        return None
    
    def get_elements_in_region(self, image: Union[str, Path, Image.Image, np.ndarray], 
                              region: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Get all elements within a specific region.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            region: Region as (left, top, right, bottom)
            
        Returns:
            List of elements within the region
        """
        elements = self.detect_all_elements(image)
        left, top, right, bottom = region
        
        region_elements = []
        for element in elements:
            bounds = element.get("bounds", [])
            if len(bounds) >= 4:
                elem_left, elem_top, elem_right, elem_bottom = bounds[:4]
                
                # Check if element overlaps with region
                if (elem_left < right and elem_right > left and 
                    elem_top < bottom and elem_bottom > top):
                    region_elements.append(element)
        
        logger.info(f"Found {len(region_elements)} elements in region")
        return region_elements
    
    def _post_process_elements(self, elements: List[Dict[str, Any]], 
                              image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Post-process detected elements to improve quality and add metadata.
        
        Args:
            elements: Raw detected elements
            image: Original image
            
        Returns:
            Post-processed elements
        """
        processed_elements = []
        
        # Get image dimensions
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        img_width, img_height = pil_image.size
        
        for element in elements:
            try:
                # Validate and normalize bounds
                bounds = element.get("bounds", [])
                if len(bounds) >= 4:
                    left, top, right, bottom = bounds[:4]
                    
                    # Ensure bounds are within image
                    left = max(0, min(left, img_width))
                    top = max(0, min(top, img_height))
                    right = max(left, min(right, img_width))
                    bottom = max(top, min(bottom, img_height))
                    
                    # Update bounds
                    element["bounds"] = [left, top, right, bottom]
                    
                    # Add additional metadata
                    element["width"] = right - left
                    element["height"] = bottom - top
                    element["area"] = element["width"] * element["height"]
                    element["center"] = [(left + right) // 2, (top + bottom) // 2]
                    
                    # Add confidence if not present
                    if "confidence" not in element:
                        element["confidence"] = 0.8  # Default confidence
                    
                    # Add element ID if not present
                    if "id" not in element:
                        element["id"] = f"{element.get('type', 'unknown')}_{len(processed_elements)}"
                    
                    # Add clickability score
                    element["clickability_score"] = self._calculate_clickability_score(element)
                    
                    # Add text content if available
                    if "text" not in element:
                        element["text"] = self._extract_text_from_element(element, pil_image)
                    
                    processed_elements.append(element)
                
            except Exception as e:
                logger.warning(f"Failed to process element: {e}")
                continue
        
        # Sort elements by confidence and area
        processed_elements.sort(key=lambda x: (x.get("confidence", 0), x.get("area", 0)), reverse=True)
        
        return processed_elements
    
    def _calculate_clickability_score(self, element: Dict[str, Any]) -> float:
        """
        Calculate a clickability score for an element.
        
        Args:
            element: Element dictionary
            
        Returns:
            Clickability score between 0 and 1
        """
        score = 0.0
        
        # Base score by type
        element_type = element.get("type", "").lower()
        type_scores = {
            "button": 0.9,
            "link": 0.8,
            "clickable": 0.7,
            "menu_item": 0.8,
            "tab": 0.7,
            "checkbox": 0.6,
            "radio": 0.6,
            "toggle": 0.7,
            "switch": 0.7,
            "input": 0.5,
            "text": 0.1,
            "image": 0.3
        }
        
        score += type_scores.get(element_type, 0.2)
        
        # Size factor (larger elements are more clickable)
        area = element.get("area", 0)
        if area > 0:
            # Normalize area score (assuming max reasonable area is 10000 pixels)
            area_score = min(area / 10000.0, 1.0) * 0.2
            score += area_score
        
        # Confidence factor
        confidence = element.get("confidence", 0.5)
        score += confidence * 0.3
        
        # Text content factor (elements with text are often more clickable)
        text = element.get("text", "")
        if text and len(text.strip()) > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_text_from_element(self, element: Dict[str, Any], image: Image.Image) -> str:
        """
        Extract text content from an element using OCR.
        
        Args:
            element: Element dictionary
            image: Original image
            
        Returns:
            Extracted text content
        """
        try:
            bounds = element.get("bounds", [])
            if len(bounds) < 4:
                return ""
            
            left, top, right, bottom = bounds[:4]
            
            # Crop element from image
            cropped = image.crop((left, top, right, bottom))
            
            # Convert to numpy array for OpenCV
            img_array = np.array(cropped)
            
            # Use OpenCV for text detection (simple approach)
            # This is a placeholder - in practice, you might want to use
            # more sophisticated OCR like Tesseract or EasyOCR
            
            # For now, return empty string
            # In a real implementation, you would:
            # 1. Preprocess the cropped image
            # 2. Use OCR to extract text
            # 3. Return the extracted text
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to extract text from element: {e}")
            return ""
    
    def _get_cache_key(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        """Generate cache key for image."""
        if isinstance(image, (str, Path)):
            return str(image)
        else:
            # For PIL Image or numpy array, use hash
            if isinstance(image, Image.Image):
                return str(hash(image.tobytes()))
            else:
                return str(hash(image.tobytes()))
    
    def _cache_elements(self, cache_key: str, elements: List[Dict[str, Any]]) -> None:
        """Cache elements with size limit."""
        if len(self._element_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._element_cache))
            del self._element_cache[oldest_key]
        
        self._element_cache[cache_key] = elements
    
    def clear_cache(self) -> None:
        """Clear the element cache."""
        self._element_cache.clear()
        logger.info("Element cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._element_cache),
            "max_cache_size": self._cache_size,
            "cache_keys": list(self._element_cache.keys())
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()
        if hasattr(self, 'model_client'):
            self.model_client.cleanup()
        logger.info("Element detector cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
