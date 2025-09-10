"""
UI-Venus Model Client

This module provides the main interface for interacting with the UI-Venus model,
handling both local model inference and remote API calls.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import requests
import json

from config.ui_venus_config import UIVenusConfig


logger = logging.getLogger(__name__)


class UIVenusModelClient:
    """
    Client for interacting with UI-Venus model.
    
    Supports both local model inference and remote API calls.
    Handles model loading, image preprocessing, and inference.
    """
    
    def __init__(self, config: UIVenusConfig):
        """
        Initialize the UI-Venus model client.
        
        Args:
            config: UI-Venus configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None
        self._is_initialized = False
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the UI-Venus model and tokenizer."""
        try:
            if self.config.is_remote_api():
                logger.info("Using remote UI-Venus API")
                self._is_initialized = True
                return
            
            logger.info(f"Initializing UI-Venus model: {self.config.model_name}")
            
            # Set device
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif self.config.device == "mps" and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS device")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU device")
            
            # Load model and tokenizer
            model_path = self.config.get_model_path()
            cache_dir = self.config.get_cache_dir()
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Cache directory: {cache_dir}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load processor (for multimodal models)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    trust_remote_code=self.config.trust_remote_code
                )
            except Exception as e:
                logger.warning(f"Could not load processor: {e}")
                self.processor = None
            
            # Load model - UI-Venus uses AutoModelForVision2Seq
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    trust_remote_code=self.config.trust_remote_code,
                    torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logger.warning(f"Failed to load as Vision2Seq model, trying CausalLM: {e}")
                # Fallback to CausalLM if Vision2Seq fails
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    cache_dir=cache_dir,
                    trust_remote_code=self.config.trust_remote_code,
                    torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
            
            # Move to device if not using device_map
            if self.config.device != "cuda" or not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Configure memory usage
            if self.config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.config.max_memory_usage)
            
            self._is_initialized = True
            logger.info("UI-Venus model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize UI-Venus model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def detect_elements(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect UI elements in an image using UI-Venus grounding capabilities.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            List of detected elements with bounding boxes and metadata
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            if self.config.is_remote_api():
                return self._detect_elements_remote(processed_image)
            else:
                return self._detect_elements_local(processed_image)
                
        except Exception as e:
            logger.error(f"Element detection failed: {e}")
            raise RuntimeError(f"Element detection failed: {e}")
    
    def suggest_actions(self, image: Union[str, Path, Image.Image, np.ndarray], 
                       context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Suggest actions to take on the current screen using UI-Venus navigation capabilities.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            context: Optional context about the current state
            
        Returns:
            List of suggested actions with priorities and metadata
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized")
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            if self.config.is_remote_api():
                return self._suggest_actions_remote(processed_image, context)
            else:
                return self._suggest_actions_local(processed_image, context)
                
        except Exception as e:
            logger.error(f"Action suggestion failed: {e}")
            raise RuntimeError(f"Action suggestion failed: {e}")
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Preprocess image for UI-Venus model input.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize to model input size
        target_size = self.config.image_size
        if pil_image.size != target_size:
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Normalize if required
        if self.config.normalize_images:
            # Convert to numpy for normalization
            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        return pil_image
    
    def _detect_elements_local(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect elements using local model inference."""
        try:
            # Prepare messages in UI-Venus format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Detect all UI elements in this image and provide their coordinates and types."}
                    ]
                }
            ]
            
            # Use processor to apply chat template
            if self.processor:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Fallback to simple text processing
                text = "User: Detect all UI elements in this image\nAssistant:"
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
                )
            
            # Decode response
            if self.processor:
                response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response to extract elements
            elements = self._parse_element_response(response)
            
            return elements
            
        except Exception as e:
            logger.error(f"Local element detection failed: {e}")
            raise
    
    def _detect_elements_remote(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect elements using remote API."""
        try:
            # Convert image to base64
            import base64
            import io
            
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare request payload
            payload = {
                "image": image_b64,
                "task": "element_detection",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
            
            # Make API request
            headers = self.config.get_api_headers()
            response = requests.post(
                self.config.api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            elements = self._parse_element_response(result.get("response", ""))
            
            return elements
            
        except Exception as e:
            logger.error(f"Remote element detection failed: {e}")
            raise
    
    def _suggest_actions_local(self, image: Image.Image, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest actions using local model inference."""
        try:
            # Prepare prompt for UI-Venus
            prompt = "Suggest the best actions to take on this screen for maximum app coverage."
            if context:
                prompt += f" Context: {context}"
            
            # Prepare messages in UI-Venus format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Use processor to apply chat template
            if self.processor:
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Fallback to simple text processing
                text = f"User: {prompt}\nAssistant:"
                inputs = self.tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
                )
            
            # Decode response
            if self.processor:
                response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            else:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response to extract actions
            actions = self._parse_action_response(response)
            
            return actions
            
        except Exception as e:
            logger.error(f"Local action suggestion failed: {e}")
            raise
    
    def _suggest_actions_remote(self, image: Image.Image, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest actions using remote API."""
        try:
            # Convert image to base64
            import base64
            import io
            
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare request payload
            payload = {
                "image": image_b64,
                "task": "action_suggestion",
                "context": context,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }
            
            # Make API request
            headers = self.config.get_api_headers()
            response = requests.post(
                self.config.api_endpoint,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            actions = self._parse_action_response(result.get("response", ""))
            
            return actions
            
        except Exception as e:
            logger.error(f"Remote action suggestion failed: {e}")
            raise
    
    def _parse_element_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse model response to extract UI elements.
        
        Args:
            response: Raw model response
            
        Returns:
            List of parsed elements
        """
        elements = []
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith("{"):
                data = json.loads(response)
                if isinstance(data, list):
                    elements = data
                elif isinstance(data, dict) and "elements" in data:
                    elements = data["elements"]
            
            # Fallback to text parsing
            else:
                # Look for element patterns in the response
                lines = response.split("\n")
                for line in lines:
                    if "element" in line.lower() or "button" in line.lower() or "click" in line.lower():
                        # Extract element information
                        element = self._extract_element_from_text(line)
                        if element:
                            elements.append(element)
            
            # Validate and clean elements
            validated_elements = []
            for element in elements:
                if self._validate_element(element):
                    validated_elements.append(element)
            
            return validated_elements
            
        except Exception as e:
            logger.warning(f"Failed to parse element response: {e}")
            return []
    
    def _parse_action_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse model response to extract suggested actions.
        
        Args:
            response: Raw model response
            
        Returns:
            List of parsed actions
        """
        actions = []
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith("{"):
                data = json.loads(response)
                if isinstance(data, list):
                    actions = data
                elif isinstance(data, dict) and "actions" in data:
                    actions = data["actions"]
            
            # Fallback to text parsing
            else:
                # Look for action patterns in the response
                lines = response.split("\n")
                for line in lines:
                    if "action" in line.lower() or "click" in line.lower() or "tap" in line.lower():
                        # Extract action information
                        action = self._extract_action_from_text(line)
                        if action:
                            actions.append(action)
            
            # Validate and clean actions
            validated_actions = []
            for action in actions:
                if self._validate_action(action):
                    validated_actions.append(action)
            
            return validated_actions
            
        except Exception as e:
            logger.warning(f"Failed to parse action response: {e}")
            return []
    
    def _extract_element_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract element information from text line."""
        try:
            # Simple text parsing - can be enhanced based on actual model output
            element = {
                "type": "unknown",
                "bounds": [0, 0, 100, 100],  # Default bounds
                "confidence": 0.5,
                "text": text.strip()
            }
            
            # Try to extract type
            if "button" in text.lower():
                element["type"] = "button"
            elif "text" in text.lower():
                element["type"] = "text"
            elif "image" in text.lower():
                element["type"] = "image"
            elif "input" in text.lower():
                element["type"] = "input"
            
            return element
            
        except Exception:
            return None
    
    def _extract_action_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract action information from text line."""
        try:
            # Simple text parsing - can be enhanced based on actual model output
            action = {
                "type": "click",
                "target": "unknown",
                "priority": "medium",
                "description": text.strip()
            }
            
            # Try to extract action type
            if "click" in text.lower() or "tap" in text.lower():
                action["type"] = "click"
            elif "swipe" in text.lower():
                action["type"] = "swipe"
            elif "input" in text.lower() or "type" in text.lower():
                action["type"] = "input"
            
            return action
            
        except Exception:
            return None
    
    def _validate_element(self, element: Dict[str, Any]) -> bool:
        """Validate element structure."""
        required_fields = ["type", "bounds"]
        return all(field in element for field in required_fields)
    
    def _validate_action(self, action: Dict[str, Any]) -> bool:
        """Validate action structure."""
        required_fields = ["type", "target"]
        return all(field in action for field in required_fields)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "model_name": self.config.model_name,
            "device": str(self.device) if self.device else "unknown",
            "is_remote": self.config.is_remote_api(),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        if self.config.device == "cuda" and torch.cuda.is_available():
            info["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        
        return info
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_initialized = False
        logger.info("UI-Venus model client cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
