"""
UI-Venus Integration Module

This module provides integration with the UI-Venus model for intelligent
UI understanding and action generation in mobile app crawling.
"""

from .model_client import UIVenusModelClient
from .element_detector import UIVenusElementDetector
from .action_suggester import UIVenusActionSuggester, ActionType, ActionPriority

__all__ = [
    "UIVenusModelClient",
    "UIVenusElementDetector", 
    "UIVenusActionSuggester",
    "ActionType",
    "ActionPriority"
]

__version__ = "1.0.0"
__author__ = "UI-Venus Mobile Crawler Team"
