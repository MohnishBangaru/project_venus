#!/usr/bin/env python3
"""
UI-Venus Integration Demo

This script demonstrates the UI-Venus integration capabilities including
element detection and action suggestion.
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import UIVenusConfig, ProjectConfig
from ui_venus import UIVenusModelClient, UIVenusElementDetector, UIVenusActionSuggester
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def demo_model_client():
    """Demonstrate UI-Venus model client functionality."""
    console.print("\n🤖 [bold blue]UI-Venus Model Client Demo[/bold blue]")
    console.print("=" * 50)
    
    # Create configuration
    config = UIVenusConfig(
        model_name="ui-venus-7b",
        device="cpu",  # Use CPU for demo
        max_tokens=256,
        temperature=0.1
    )
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Initializing UI-Venus model...", total=None)
            
            # Initialize model client
            client = UIVenusModelClient(config)
            progress.update(task, completed=100)
        
        # Get model info
        info = client.get_model_info()
        
        console.print("\n✅ Model initialized successfully!")
        console.print(f"📊 Model: {info['model_name']}")
        console.print(f"🖥️ Device: {info['device']}")
        console.print(f"🔧 Max Tokens: {info['max_tokens']}")
        console.print(f"🌡️ Temperature: {info['temperature']}")
        
        # Cleanup
        client.cleanup()
        
    except Exception as e:
        console.print(f"❌ Model initialization failed: {e}")
        console.print("💡 This is expected if UI-Venus model is not available locally")
        console.print("💡 The demo will continue with mock functionality")


def demo_element_detector():
    """Demonstrate element detection functionality."""
    console.print("\n🎯 [bold green]Element Detection Demo[/bold green]")
    console.print("=" * 50)
    
    # Create configuration
    config = UIVenusConfig(
        model_name="ui-venus-7b",
        device="cpu",
        max_tokens=256
    )
    
    try:
        # Initialize element detector
        detector = UIVenusElementDetector(config)
        
        # Create a sample image for testing
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.new('RGB', (200, 300), color='white')
        
        console.print("📱 Created test image (200x300 pixels)")
        
        # Mock element detection (since we don't have the actual model)
        mock_elements = [
            {
                "type": "button",
                "bounds": [50, 50, 150, 80],
                "confidence": 0.9,
                "text": "Login",
                "clickability_score": 0.95
            },
            {
                "type": "input",
                "bounds": [50, 100, 150, 130],
                "confidence": 0.8,
                "text": "Username",
                "clickability_score": 0.7
            },
            {
                "type": "text",
                "bounds": [50, 150, 150, 170],
                "confidence": 0.7,
                "text": "Welcome to the app",
                "clickability_score": 0.1
            }
        ]
        
        console.print(f"🔍 Detected {len(mock_elements)} elements:")
        
        # Display elements in a table
        table = Table(title="Detected Elements")
        table.add_column("Type", style="cyan")
        table.add_column("Bounds", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("Text", style="yellow")
        table.add_column("Clickability", style="red")
        
        for element in mock_elements:
            bounds_str = f"{element['bounds'][0]},{element['bounds'][1]},{element['bounds'][2]},{element['bounds'][3]}"
            table.add_row(
                element["type"],
                bounds_str,
                f"{element['confidence']:.2f}",
                element["text"],
                f"{element['clickability_score']:.2f}"
            )
        
        console.print(table)
        
        # Demonstrate element filtering
        clickable_elements = [e for e in mock_elements if e["clickability_score"] > 0.5]
        console.print(f"\n🎯 Found {len(clickable_elements)} clickable elements")
        
        # Demonstrate element at position
        element_at_pos = next((e for e in mock_elements if 75 >= e["bounds"][0] and 75 <= e["bounds"][2] and 65 >= e["bounds"][1] and 65 <= e["bounds"][3]), None)
        if element_at_pos:
            console.print(f"📍 Element at position (75, 65): {element_at_pos['type']}")
        
        # Cleanup
        detector.cleanup()
        
    except Exception as e:
        console.print(f"❌ Element detection demo failed: {e}")


def demo_action_suggester():
    """Demonstrate action suggestion functionality."""
    console.print("\n🎬 [bold yellow]Action Suggestion Demo[/bold yellow]")
    console.print("=" * 50)
    
    # Create configuration
    config = UIVenusConfig(
        model_name="ui-venus-7b",
        device="cpu",
        max_tokens=256
    )
    
    try:
        # Initialize action suggester
        suggester = UIVenusActionSuggester(config)
        
        # Create a sample image for testing
        from PIL import Image
        test_image = Image.new('RGB', (200, 300), color='white')
        
        # Mock action suggestions
        mock_actions = [
            {
                "type": "click",
                "target": "login_button",
                "priority": "high",
                "description": "Click on login button",
                "confidence": 0.9,
                "coordinates": [100, 65],
                "priority_score": 0.95
            },
            {
                "type": "input",
                "target": "username_field",
                "priority": "medium",
                "description": "Enter username",
                "confidence": 0.8,
                "coordinates": [100, 115],
                "priority_score": 0.75
            },
            {
                "type": "swipe",
                "target": "scroll_area",
                "priority": "low",
                "description": "Swipe to scroll",
                "confidence": 0.6,
                "coordinates": [100, 150],
                "priority_score": 0.45
            }
        ]
        
        console.print(f"💡 Suggested {len(mock_actions)} actions:")
        
        # Display actions in a table
        table = Table(title="Suggested Actions")
        table.add_column("Type", style="cyan")
        table.add_column("Target", style="magenta")
        table.add_column("Priority", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Confidence", style="red")
        table.add_column("Score", style="blue")
        
        for action in mock_actions:
            table.add_row(
                action["type"],
                action["target"],
                action["priority"],
                action["description"],
                f"{action['confidence']:.2f}",
                f"{action['priority_score']:.2f}"
            )
        
        console.print(table)
        
        # Demonstrate action filtering
        high_priority_actions = [a for a in mock_actions if a["priority"] == "high"]
        console.print(f"\n⭐ {len(high_priority_actions)} high-priority actions")
        
        # Demonstrate exploration actions
        exploration_actions = [a for a in mock_actions if a["type"] in ["click", "swipe"]]
        console.print(f"🔍 {len(exploration_actions)} exploration actions")
        
        # Demonstrate recovery actions
        recovery_actions = [
            {
                "type": "back",
                "target": "back_button",
                "priority": "high",
                "description": "Navigate back",
                "confidence": 0.9,
                "recovery_score": 0.9
            },
            {
                "type": "home",
                "target": "home_button",
                "priority": "high",
                "description": "Go to home screen",
                "confidence": 0.9,
                "recovery_score": 0.8
            }
        ]
        
        console.print(f"🔄 {len(recovery_actions)} recovery actions available")
        
        # Cleanup
        suggester.cleanup()
        
    except Exception as e:
        console.print(f"❌ Action suggestion demo failed: {e}")


def demo_integration_workflow():
    """Demonstrate complete integration workflow."""
    console.print("\n🔄 [bold magenta]Complete Integration Workflow Demo[/bold magenta]")
    console.print("=" * 60)
    
    # Create configuration
    config = UIVenusConfig(
        model_name="ui-venus-7b",
        device="cpu",
        max_tokens=256
    )
    
    try:
        # Initialize all components
        console.print("🚀 Initializing UI-Venus components...")
        
        client = UIVenusModelClient(config)
        detector = UIVenusElementDetector(config)
        suggester = UIVenusActionSuggester(config)
        
        console.print("✅ All components initialized successfully!")
        
        # Simulate workflow
        console.print("\n📱 Simulating mobile app crawling workflow...")
        
        # Step 1: Take screenshot
        console.print("1️⃣ Taking screenshot...")
        from PIL import Image
        screenshot = Image.new('RGB', (400, 600), color='lightblue')
        console.print("   ✅ Screenshot captured")
        
        # Step 2: Detect elements
        console.print("2️⃣ Detecting UI elements...")
        # Mock element detection
        elements = [
            {"type": "button", "bounds": [100, 100, 200, 130], "confidence": 0.9},
            {"type": "input", "bounds": [100, 150, 200, 180], "confidence": 0.8},
            {"type": "text", "bounds": [100, 200, 200, 220], "confidence": 0.7}
        ]
        console.print(f"   ✅ Detected {len(elements)} elements")
        
        # Step 3: Suggest actions
        console.print("3️⃣ Suggesting actions...")
        # Mock action suggestions
        actions = [
            {"type": "click", "target": "button1", "priority": "high", "confidence": 0.9},
            {"type": "input", "target": "input1", "priority": "medium", "confidence": 0.8}
        ]
        console.print(f"   ✅ Suggested {len(actions)} actions")
        
        # Step 4: Execute action
        console.print("4️⃣ Executing action...")
        selected_action = actions[0]  # Select highest priority action
        console.print(f"   ✅ Executed: {selected_action['type']} on {selected_action['target']}")
        
        # Step 5: Update state
        console.print("5️⃣ Updating crawling state...")
        console.print("   ✅ State updated, ready for next iteration")
        
        # Display workflow summary
        workflow_summary = Panel(
            f"[bold]Workflow Summary:[/bold]\n"
            f"• Screenshot: 400x600 pixels\n"
            f"• Elements detected: {len(elements)}\n"
            f"• Actions suggested: {len(actions)}\n"
            f"• Action executed: {selected_action['type']}\n"
            f"• Status: Ready for next iteration",
            title="🔄 Crawling Workflow",
            border_style="green"
        )
        console.print(workflow_summary)
        
        # Cleanup
        client.cleanup()
        detector.cleanup()
        suggester.cleanup()
        
    except Exception as e:
        console.print(f"❌ Integration workflow demo failed: {e}")


def main():
    """Run all UI-Venus integration demos."""
    console.print("🚀 [bold]UI-Venus Mobile Crawler - Integration Demo[/bold]")
    console.print("=" * 60)
    console.print("This demo showcases the UI-Venus integration capabilities")
    console.print("for intelligent mobile app crawling.\n")
    
    # Run demos
    demo_model_client()
    demo_element_detector()
    demo_action_suggester()
    demo_integration_workflow()
    
    console.print("\n🎉 [bold green]UI-Venus Integration Demo Completed![/bold green]")
    console.print("\n💡 [bold]Next Steps:[/bold]")
    console.print("1. Set up UI-Venus model on your RunPod instance")
    console.print("2. Configure the model path in your configuration")
    console.print("3. Test with real Android device screenshots")
    console.print("4. Integrate with the crawler engine")
    
    console.print("\n📚 [bold]Documentation:[/bold]")
    console.print("• UI-Venus Model: https://github.com/inclusionAI/UI-Venus")
    console.print("• Configuration: config/ui_venus_config.py")
    console.print("• API Reference: docs/api_reference.md")


if __name__ == "__main__":
    main()
