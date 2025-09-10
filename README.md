# UI-Venus Mobile Crawler

An intelligent Android app crawler that leverages the UI-Venus model for maximum app coverage and systematic exploration.

## 🎯 Project Overview

The UI-Venus Mobile Crawler is an advanced automation framework that combines state-of-the-art computer vision with intelligent navigation strategies to systematically explore Android applications. Built on the UI-Venus 7B model, it provides superior UI understanding and action prediction for comprehensive app testing and analysis.

### Key Features

- **🤖 UI-Venus Integration**: Leverages the UI-Venus 7B model for superior UI understanding
- **📱 Android Automation**: Complete Android device control via ADB and UIAutomator2
- **🎯 Intelligent Coverage**: Systematic exploration for maximum app coverage
- **🔄 Smart Navigation**: Context-aware action planning and execution
- **📊 Comprehensive Reporting**: Detailed PDF reports with screenshots and analysis
- **⚙️ Flexible Configuration**: Type-safe configuration system with environment variable support
- **🧪 Extensive Testing**: Comprehensive test suite with mock and integration tests

## 🏗️ Architecture

```
project_venus/
├── src/
│   ├── ui_venus/              # UI-Venus model integration
│   │   ├── model_client.py    # Main model interface
│   │   ├── element_detector.py # Element detection and grounding
│   │   └── action_suggester.py # Action suggestion and planning
│   ├── crawler/               # Intelligent crawling engine
│   │   ├── coverage_tracker.py # Track explored screens/states
│   │   ├── action_planner.py   # Rule-based action planning
│   │   ├── navigation_engine.py # Systematic exploration logic
│   │   └── state_analyzer.py   # Analyze app state changes
│   ├── automation/            # Android device control
│   │   ├── device_controller.py # ADB/device control
│   │   ├── action_executor.py   # Execute UI actions
│   │   └── screenshot_manager.py # Screenshot capture
│   ├── core/                  # Core orchestration
│   │   ├── orchestrator.py     # Main crawling orchestration
│   │   ├── recovery.py         # Error recovery and retry logic
│   │   └── logger.py           # Logging and debugging
│   ├── utils/                 # Utilities
│   │   ├── image_utils.py      # Image processing utilities
│   │   ├── config_utils.py     # Configuration management
│   │   └── file_utils.py       # File operations
│   └── vision/                # Image processing
│       ├── image_processor.py  # Image preprocessing for UI-Venus
│       └── element_detector.py # Fallback element detection
├── config/                    # Configuration system
│   ├── ui_venus_config.py     # UI-Venus model configuration
│   ├── crawler_config.py      # Crawler behavior settings
│   └── device_config.py       # Device-specific settings
├── scripts/                   # CLI entry points
│   ├── venus_crawler.py       # Main entry point
│   ├── setup_device.py        # Device setup and testing
│   └── analyze_results.py     # Result analysis tools
├── tests/                     # Test suite
│   ├── test_ui_venus/         # UI-Venus tests
│   ├── test_crawler/          # Crawler tests
│   └── test_automation/       # Automation tests
└── docs/                      # Documentation
    ├── setup_guide.md         # Setup instructions
    ├── usage_guide.md         # Usage guide
    └── api_reference.md       # API documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Android device or emulator with ADB enabled
- RunPod instance with GPU (for UI-Venus model)
- UI-Venus 7B model files

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MohnishBangaru/project_venus.git
   cd project_venus
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Basic Usage

1. **Setup Android device:**
   ```bash
   python scripts/venus_crawler.py setup-device
   ```

2. **Test UI-Venus integration:**
   ```bash
   python scripts/venus_crawler.py test-ui-venus
   ```

3. **Crawl an Android app:**
   ```bash
   python scripts/venus_crawler.py crawl --apk /path/to/app.apk
   ```

## ⚙️ Configuration

The project uses a comprehensive configuration system with Pydantic validation:

### UI-Venus Configuration

```python
from config import UIVenusConfig

config = UIVenusConfig(
    model_name="ui-venus-7b",
    device="cuda",
    max_tokens=512,
    temperature=0.1,
    max_memory_usage=0.8
)
```

### Crawler Configuration

```python
from config import CrawlerConfig

config = CrawlerConfig(
    strategy="priority_based",
    max_depth=10,
    max_actions=1000,
    coverage_threshold=0.8,
    enable_backtracking=True
)
```

### Environment Variables

```bash
# UI-Venus settings
export UI_VENUS_MODEL_NAME="ui-venus-7b"
export UI_VENUS_DEVICE="cuda"
export UI_VENUS_TEMPERATURE="0.1"

# Crawler settings
export CRAWLER_STRATEGY="priority_based"
export CRAWLER_MAX_ACTIONS="1000"
export CRAWLER_COVERAGE_THRESHOLD="0.8"

# Device settings
export DEVICE_ADB_HOST="localhost"
export DEVICE_ADB_PORT="5037"
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# UI-Venus integration tests
pytest tests/test_ui_venus/ -v

# Crawler engine tests
pytest tests/test_crawler/ -v

# Automation tests
pytest tests/test_automation/ -v
```

### Mac Testing (Mock Mode)

For testing on Mac without GPU:

```bash
python scripts/test_ui_venus_mac.py
```

## 📊 Current Implementation Status

### ✅ Completed Components

- **Project Structure**: Complete modular architecture
- **Configuration System**: Pydantic-based with validation
- **Dependencies**: All required packages and tools
- **UI-Venus Integration**: Complete model client, element detection, and action suggestion
- **Testing Framework**: Comprehensive test suite with mocks
- **Documentation**: API documentation and usage guides

### 🚧 In Progress

- **Crawler Engine**: Intelligent crawling logic and coverage tracking
- **Device Automation**: Android device control and action execution
- **CLI Implementation**: Command-line interface with progress reporting
- **PDF Reporting**: Comprehensive result analysis and reporting

### 📋 Next Steps

1. **Crawler Engine Implementation**
   - Coverage tracking and state management
   - Action planning and prioritization
   - Navigation strategies and backtracking

2. **Device Automation**
   - ADB integration and device control
   - Screenshot capture and processing
   - Action execution and error recovery

3. **CLI Implementation**
   - Command-line interface with rich progress reporting
   - Device setup and testing utilities
   - Result analysis and reporting tools

4. **Integration Testing**
   - End-to-end testing with real devices
   - Performance optimization and tuning
   - Error handling and recovery testing

## 🔧 Development

### Code Quality

The project uses several tools for code quality:

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/

# Type checking
mypy src/ config/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📚 Documentation

- **Setup Guide**: `docs/setup_guide.md`
- **Usage Guide**: `docs/usage_guide.md`
- **API Reference**: `docs/api_reference.md`
- **Configuration**: `config/` directory
- **Examples**: `scripts/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **UI-Venus Team**: For the state-of-the-art UI understanding model
- **AA_VA-Phi Project**: For the Android automation foundation
- **Open Source Community**: For the various libraries and tools used

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/MohnishBangaru/project_venus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MohnishBangaru/project_venus/discussions)
- **Email**: mohnish@example.com

## 🔗 Related Projects

- [UI-Venus](https://github.com/inclusionAI/UI-Venus) - The core UI understanding model
- [AA_VA-Phi](https://github.com/MohnishBangaru/AA_VA-Phi) - Android automation framework
- [UIAutomator2](https://github.com/openatx/uiautomator2) - Android UI automation

---

**Status**: 🚧 In Development | **Version**: 1.0.0 | **Python**: 3.8+ | **License**: MIT
