"""
Screenshot Manager

This module provides screenshot capture and management capabilities for Android devices,
including image processing, storage, and optimization features.
"""

import logging
import time
import hashlib
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from PIL import Image, ImageOps
import io
import base64

from .device_controller import DeviceController
from config.device_config import DeviceConfig


logger = logging.getLogger(__name__)


class ScreenshotManager:
    """
    Screenshot manager for Android devices.
    
    Provides comprehensive screenshot capture, processing, and management
    capabilities with optimization and storage features.
    """
    
    def __init__(self, device_controller: DeviceController, config: DeviceConfig):
        """
        Initialize the screenshot manager.
        
        Args:
            device_controller: Device controller instance
            config: Device configuration
        """
        self.device_controller = device_controller
        self.config = config
        self._screenshot_cache: Dict[str, Image.Image] = {}
        self._cache_size = 50
        self._last_screenshot: Optional[Image.Image] = None
        self._screenshot_count = 0
        
        logger.info("Screenshot manager initialized")
    
    def capture_screenshot(self, save_path: Optional[str] = None, 
                          compress: Optional[bool] = None) -> Optional[Image.Image]:
        """
        Capture a screenshot from the device.
        
        Args:
            save_path: Optional path to save the screenshot
            compress: Whether to compress the image (overrides config)
            
        Returns:
            PIL Image object or None if capture failed
        """
        try:
            start_time = time.time()
            
            # Capture screenshot using ADB
            screenshot_data = self._capture_raw_screenshot()
            if not screenshot_data:
                logger.error("Failed to capture raw screenshot")
                return None
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot_data))
            
            # Apply scaling if configured
            if self.config.screenshot_scale != 1.0:
                new_size = (
                    int(image.width * self.config.screenshot_scale),
                    int(image.height * self.config.screenshot_scale)
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply orientation correction if needed
            image = self._correct_orientation(image)
            
            # Save if path provided
            if save_path:
                self._save_screenshot(image, save_path, compress)
            
            # Update state
            self._last_screenshot = image
            self._screenshot_count += 1
            
            # Cache the screenshot
            self._cache_screenshot(image)
            
            capture_time = (time.time() - start_time) * 1000
            logger.debug(f"Screenshot captured in {capture_time:.1f}ms")
            
            return image
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def capture_and_save(self, output_dir: str, filename_prefix: str = "screenshot") -> Optional[str]:
        """
        Capture a screenshot and save it to the specified directory.
        
        Args:
            output_dir: Directory to save the screenshot
            filename_prefix: Prefix for the filename
            
        Returns:
            Path to saved screenshot or None if failed
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = int(time.time() * 1000)
            filename = f"{filename_prefix}_{timestamp}.{self.config.screenshot_format.lower()}"
            file_path = output_path / filename
            
            # Capture and save
            image = self.capture_screenshot(str(file_path))
            
            if image:
                logger.info(f"Screenshot saved to: {file_path}")
                return str(file_path)
            else:
                logger.error("Failed to capture screenshot for saving")
                return None
                
        except Exception as e:
            logger.error(f"Screenshot save failed: {e}")
            return None
    
    def get_screenshot_info(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get information about a screenshot.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with screenshot information
        """
        try:
            # Calculate image hash for comparison
            image_hash = self._calculate_image_hash(image)
            
            info = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(image.tobytes()),
                "hash": image_hash,
                "timestamp": time.time(),
                "aspect_ratio": image.width / image.height,
                "pixel_count": image.width * image.height
            }
            
            # Add color information
            if image.mode == "RGB":
                # Calculate average color
                pixels = list(image.getdata())
                if pixels:
                    avg_r = sum(p[0] for p in pixels) / len(pixels)
                    avg_g = sum(p[1] for p in pixels) / len(pixels)
                    avg_b = sum(p[2] for p in pixels) / len(pixels)
                    info["average_color"] = (int(avg_r), int(avg_g), int(avg_b))
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get screenshot info: {e}")
            return {}
    
    def compare_screenshots(self, image1: Image.Image, image2: Image.Image) -> Dict[str, Any]:
        """
        Compare two screenshots and return similarity metrics.
        
        Args:
            image1: First screenshot
            image2: Second screenshot
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Resize images to same size for comparison
            if image1.size != image2.size:
                min_width = min(image1.width, image2.width)
                min_height = min(image1.height, image2.height)
                image1 = image1.resize((min_width, min_height), Image.Resampling.LANCZOS)
                image2 = image2.resize((min_width, min_height), Image.Resampling.LANCZOS)
            
            # Convert to same mode
            if image1.mode != image2.mode:
                image1 = image1.convert("RGB")
                image2 = image2.convert("RGB")
            
            # Calculate pixel-wise difference
            pixels1 = list(image1.getdata())
            pixels2 = list(image2.getdata())
            
            if len(pixels1) != len(pixels2):
                return {"error": "Images have different pixel counts"}
            
            # Calculate differences
            total_diff = 0
            max_diff = 0
            different_pixels = 0
            
            for p1, p2 in zip(pixels1, pixels2):
                if isinstance(p1, tuple) and isinstance(p2, tuple):
                    # RGB comparison
                    diff = sum(abs(a - b) for a, b in zip(p1, p2))
                else:
                    # Grayscale comparison
                    diff = abs(p1 - p2)
                
                total_diff += diff
                max_diff = max(max_diff, diff)
                
                if diff > 10:  # Threshold for "different" pixel
                    different_pixels += 1
            
            # Calculate similarity metrics
            total_pixels = len(pixels1)
            avg_diff = total_diff / total_pixels
            similarity = 1.0 - (avg_diff / 255.0)  # Normalize to 0-1
            difference_percentage = (different_pixels / total_pixels) * 100
            
            return {
                "similarity": max(0.0, min(1.0, similarity)),
                "difference_percentage": difference_percentage,
                "average_difference": avg_diff,
                "max_difference": max_diff,
                "different_pixels": different_pixels,
                "total_pixels": total_pixels,
                "is_identical": similarity > 0.99
            }
            
        except Exception as e:
            logger.error(f"Screenshot comparison failed: {e}")
            return {"error": str(e)}
    
    def get_last_screenshot(self) -> Optional[Image.Image]:
        """Get the last captured screenshot."""
        return self._last_screenshot
    
    def get_screenshot_count(self) -> int:
        """Get the total number of screenshots captured."""
        return self._screenshot_count
    
    def clear_cache(self) -> None:
        """Clear the screenshot cache."""
        self._screenshot_cache.clear()
        logger.info("Screenshot cache cleared")
    
    def _capture_raw_screenshot(self) -> Optional[bytes]:
        """Capture raw screenshot data from device."""
        try:
            # Use ADB to capture screenshot
            result = self.device_controller._run_adb_command("exec-out", "screencap", "-p")
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Screenshot capture failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Raw screenshot capture failed: {e}")
            return None
    
    def _correct_orientation(self, image: Image.Image) -> Image.Image:
        """Correct image orientation based on device orientation."""
        try:
            # Check if image needs rotation based on device orientation
            if self.config.orientation == "landscape" and image.height > image.width:
                # Rotate 90 degrees clockwise
                image = image.rotate(-90, expand=True)
            elif self.config.orientation == "portrait" and image.width > image.height:
                # Rotate 90 degrees counter-clockwise
                image = image.rotate(90, expand=True)
            
            return image
            
        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")
            return image
    
    def _save_screenshot(self, image: Image.Image, file_path: str, compress: Optional[bool] = None) -> bool:
        """Save screenshot to file."""
        try:
            save_path = Path(file_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine compression settings
            should_compress = compress if compress is not None else self.config.compress_screenshots
            quality = self.config.get_screenshot_quality_for_format()
            
            # Save with appropriate settings
            if should_compress and quality:
                image.save(save_path, format=self.config.screenshot_format, 
                          quality=quality, optimize=True)
            else:
                image.save(save_path, format=self.config.screenshot_format)
            
            return True
            
        except Exception as e:
            logger.error(f"Screenshot save failed: {e}")
            return False
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate hash of an image for comparison."""
        try:
            # Convert to bytes and hash
            image_bytes = image.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"Image hash calculation failed: {e}")
            return ""
    
    def _cache_screenshot(self, image: Image.Image) -> None:
        """Cache a screenshot with size limit."""
        try:
            # Calculate cache key
            image_hash = self._calculate_image_hash(image)
            if not image_hash:
                return
            
            # Add to cache
            self._screenshot_cache[image_hash] = image.copy()
            
            # Limit cache size
            if len(self._screenshot_cache) > self._cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._screenshot_cache))
                del self._screenshot_cache[oldest_key]
                
        except Exception as e:
            logger.warning(f"Screenshot caching failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get screenshot cache statistics."""
        return {
            "cache_size": len(self._screenshot_cache),
            "max_cache_size": self._cache_size,
            "screenshot_count": self._screenshot_count,
            "has_last_screenshot": self._last_screenshot is not None
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()
        self._last_screenshot = None
        logger.info("Screenshot manager cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
