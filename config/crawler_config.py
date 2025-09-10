"""
Crawler Configuration

This module defines the configuration for the intelligent crawling engine,
including crawling strategies, coverage settings, and action priorities.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from enum import Enum


class CrawlingStrategy(str, Enum):
    """Available crawling strategies."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    PRIORITY_BASED = "priority_based"
    RANDOM = "random"


class ActionPriority(str, Enum):
    """Action priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CrawlerConfig(BaseModel):
    """Configuration for crawler behavior and coverage."""
    
    # Crawling Strategy
    strategy: CrawlingStrategy = Field(
        default=CrawlingStrategy.PRIORITY_BASED,
        description="Crawling strategy to use"
    )
    max_depth: int = Field(
        default=10, 
        description="Maximum navigation depth"
    )
    max_actions: int = Field(
        default=1000, 
        description="Maximum actions per session"
    )
    max_time_minutes: int = Field(
        default=60, 
        description="Maximum crawling time in minutes"
    )
    
    # Coverage Settings
    coverage_threshold: float = Field(
        default=0.8, 
        description="Target coverage percentage (0.0-1.0)"
    )
    revisit_threshold: int = Field(
        default=3, 
        description="Maximum revisits to same screen"
    )
    exploration_bonus: float = Field(
        default=2.0, 
        description="Bonus multiplier for exploring new areas"
    )
    
    # Action Settings
    action_delay_ms: int = Field(
        default=1000, 
        description="Delay between actions in milliseconds"
    )
    screenshot_delay_ms: int = Field(
        default=500, 
        description="Delay before taking screenshot in milliseconds"
    )
    retry_attempts: int = Field(
        default=3, 
        description="Retry attempts for failed actions"
    )
    retry_delay_ms: int = Field(
        default=2000, 
        description="Delay between retry attempts in milliseconds"
    )
    
    # Element Priority Configuration
    element_priorities: Dict[str, ActionPriority] = Field(
        default={
            "button": ActionPriority.HIGH,
            "link": ActionPriority.HIGH,
            "clickable": ActionPriority.HIGH,
            "input": ActionPriority.MEDIUM,
            "text_input": ActionPriority.MEDIUM,
            "image": ActionPriority.LOW,
            "text": ActionPriority.LOW,
            "icon": ActionPriority.MEDIUM,
            "menu_item": ActionPriority.HIGH,
            "tab": ActionPriority.HIGH,
        },
        description="Priority mapping for different UI element types"
    )
    
    # Navigation Settings
    enable_backtracking: bool = Field(
        default=True, 
        description="Enable backtracking to unexplored areas"
    )
    enable_swipe_navigation: bool = Field(
        default=True, 
        description="Enable swipe gestures for navigation"
    )
    enable_form_filling: bool = Field(
        default=False, 
        description="Enable automatic form filling"
    )
    enable_long_press: bool = Field(
        default=True, 
        description="Enable long press actions"
    )
    
    # Swipe Configuration
    swipe_distance: int = Field(
        default=300, 
        description="Default swipe distance in pixels"
    )
    swipe_duration_ms: int = Field(
        default=300, 
        description="Swipe duration in milliseconds"
    )
    
    # Form Filling Settings (when enabled)
    form_filling_strategy: str = Field(
        default="random", 
        description="Form filling strategy (random/realistic/empty)"
    )
    test_data_sources: List[str] = Field(
        default=["faker", "predefined"], 
        description="Sources for test data generation"
    )
    
    # Logging and Output
    save_screenshots: bool = Field(
        default=True, 
        description="Save screenshots for each action"
    )
    save_action_logs: bool = Field(
        default=True, 
        description="Save detailed action logs"
    )
    save_ui_venus_responses: bool = Field(
        default=True, 
        description="Save UI-Venus model responses"
    )
    output_directory: str = Field(
        default="./crawl_results", 
        description="Base directory for output files"
    )
    
    # State Management
    state_similarity_threshold: float = Field(
        default=0.95, 
        description="Threshold for considering screens as similar"
    )
    max_state_history: int = Field(
        default=100, 
        description="Maximum number of states to keep in history"
    )
    
    # Performance Settings
    parallel_screenshot_processing: bool = Field(
        default=False, 
        description="Process screenshots in parallel"
    )
    max_concurrent_actions: int = Field(
        default=1, 
        description="Maximum concurrent actions (usually 1 for mobile)"
    )
    
    @validator('coverage_threshold')
    def validate_coverage_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('coverage_threshold must be between 0.0 and 1.0')
        return v
    
    @validator('max_depth')
    def validate_max_depth(cls, v):
        if v <= 0:
            raise ValueError('max_depth must be positive')
        return v
    
    @validator('max_actions')
    def validate_max_actions(cls, v):
        if v <= 0:
            raise ValueError('max_actions must be positive')
        return v
    
    @validator('max_time_minutes')
    def validate_max_time(cls, v):
        if v <= 0:
            raise ValueError('max_time_minutes must be positive')
        return v
    
    @validator('action_delay_ms')
    def validate_action_delay(cls, v):
        if v < 0:
            raise ValueError('action_delay_ms must be non-negative')
        return v
    
    @validator('screenshot_delay_ms')
    def validate_screenshot_delay(cls, v):
        if v < 0:
            raise ValueError('screenshot_delay_ms must be non-negative')
        return v
    
    @validator('retry_attempts')
    def validate_retry_attempts(cls, v):
        if v < 0:
            raise ValueError('retry_attempts must be non-negative')
        return v
    
    @validator('state_similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('state_similarity_threshold must be between 0.0 and 1.0')
        return v
    
    class Config:
        env_prefix = "CRAWLER_"
        case_sensitive = False
        validate_assignment = True
    
    def get_element_priority(self, element_type: str) -> ActionPriority:
        """Get priority for a specific element type."""
        return self.element_priorities.get(element_type.lower(), ActionPriority.LOW)
    
    def is_high_priority_element(self, element_type: str) -> bool:
        """Check if an element type is high priority."""
        return self.get_element_priority(element_type) == ActionPriority.HIGH
    
    def get_action_delay_seconds(self) -> float:
        """Get action delay in seconds."""
        return self.action_delay_ms / 1000.0
    
    def get_screenshot_delay_seconds(self) -> float:
        """Get screenshot delay in seconds."""
        return self.screenshot_delay_ms / 1000.0
    
    def get_retry_delay_seconds(self) -> float:
        """Get retry delay in seconds."""
        return self.retry_delay_ms / 1000.0
    
    def get_swipe_duration_seconds(self) -> float:
        """Get swipe duration in seconds."""
        return self.swipe_duration_ms / 1000.0
