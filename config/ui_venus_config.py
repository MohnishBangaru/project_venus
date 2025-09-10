"""
UI-Venus Model Configuration

This module defines the configuration for UI-Venus model integration,
including model settings, inference parameters, and performance tuning.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Tuple
import os


class UIVenusConfig(BaseModel):
    """Configuration for UI-Venus model integration."""
    
    # Model Settings
    model_name: str = Field(
        default="ui-venus-7b", 
        description="UI-Venus model variant to use"
    )
    model_path: Optional[str] = Field(
        default=None, 
        description="Local path to model files (if None, will download from Hugging Face)"
    )
    device: str = Field(
        default="cuda", 
        description="Device for model inference (cuda/cpu)"
    )
    
    # Inference Settings
    max_tokens: int = Field(
        default=512, 
        description="Maximum tokens for generation"
    )
    temperature: float = Field(
        default=0.1, 
        description="Sampling temperature (lower = more deterministic)"
    )
    top_p: float = Field(
        default=0.9, 
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=50, 
        description="Top-k sampling parameter"
    )
    
    # Performance Settings
    batch_size: int = Field(
        default=1, 
        description="Batch size for inference"
    )
    max_memory_usage: float = Field(
        default=0.8, 
        description="Max GPU memory usage (0.0-1.0)"
    )
    use_half_precision: bool = Field(
        default=True, 
        description="Use half precision (FP16) for faster inference"
    )
    
    # API Settings (if using remote UI-Venus)
    api_endpoint: Optional[str] = Field(
        default=None, 
        description="Remote API endpoint for UI-Venus"
    )
    api_key: Optional[str] = Field(
        default=None, 
        description="API key for remote access"
    )
    timeout: int = Field(
        default=30, 
        description="API timeout in seconds"
    )
    
    # Image Processing
    image_size: Tuple[int, int] = Field(
        default=(1024, 1024), 
        description="Input image size for UI-Venus"
    )
    image_format: str = Field(
        default="RGB", 
        description="Image format for processing"
    )
    normalize_images: bool = Field(
        default=True, 
        description="Normalize images before processing"
    )
    
    # Model Loading Settings
    trust_remote_code: bool = Field(
        default=True, 
        description="Trust remote code when loading model"
    )
    cache_dir: Optional[str] = Field(
        default=None, 
        description="Directory to cache model files"
    )
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('top_p must be between 0.0 and 1.0')
        return v
    
    @validator('max_memory_usage')
    def validate_memory_usage(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('max_memory_usage must be between 0.0 and 1.0')
        return v
    
    @validator('image_size')
    def validate_image_size(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError('image_size must be a tuple of two positive integers')
        return v
    
    @validator('device')
    def validate_device(cls, v):
        valid_devices = ['cuda', 'cpu', 'mps']
        if v not in valid_devices:
            raise ValueError(f'device must be one of {valid_devices}')
        return v
    
    class Config:
        env_prefix = "UI_VENUS_"
        case_sensitive = False
        validate_assignment = True
        
    def get_model_path(self) -> str:
        """Get the actual model path, handling environment variables and defaults."""
        if self.model_path:
            return self.model_path
        
        # Default to Hugging Face model name
        return f"inclusionAI/{self.model_name}"
    
    def get_cache_dir(self) -> str:
        """Get the cache directory, with fallback to default."""
        if self.cache_dir:
            return self.cache_dir
        
        # Default cache directory
        default_cache = os.path.expanduser("~/.cache/ui-venus")
        os.makedirs(default_cache, exist_ok=True)
        return default_cache
    
    def is_remote_api(self) -> bool:
        """Check if using remote API instead of local model."""
        return self.api_endpoint is not None
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
