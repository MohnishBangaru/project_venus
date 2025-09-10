# RunPod + Local ADB Server Setup Guide

This guide will help you set up the UI-Venus Mobile Crawler on RunPod to connect to your local emulator via ADB server. The RunPod instance will only install the ADB client to connect to your local machine where the emulator runs.

## 🚀 RunPod Setup

### 1. Create RunPod Instance
- **Template**: PyTorch 2.0+ with CUDA
- **GPU**: RTX 4090 or A100 (recommended for UI-Venus 7B)
- **RAM**: 32GB+ (UI-Venus model requires significant memory)
- **Storage**: 100GB+ (for model cache and results)

### 2. Connect to RunPod
```bash
# SSH into your RunPod instance
ssh root@<your-runpod-ip>
```

### 3. Install Dependencies
```bash
# Update system and install essential tools
apt update && apt upgrade -y
apt install -y unzip wget curl

# Install Python dependencies (includes adbutils for ADB functionality)
pip install -r requirements.txt
```

## 📱 Local ADB Server Setup

### 1. On Your Local Machine (where emulator runs)

**Start ADB Server:**
```bash
# Start ADB server
adb start-server

# Check devices
adb devices
# Should show: emulator-5554    device
```

### 2. On RunPod Instance

**Connect to Local ADB Server:**
```bash
# Connect to your local machine's ADB server
adb connect <your-local-ip>:5037

# Verify connection
adb devices
# Should show your emulator
```

**Test connection (optional)**
```bash
python -c "import adbutils; adb = adbutils.AdbClient(); print(adb.device_list())"
```

## 🔧 Configuration

### 1. Environment Variables
```bash
# Set Hugging Face token
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

### 2. RunPod-Specific Settings
The `runpod_crawler.py` script is optimized for RunPod with:
- **GPU acceleration** for UI-Venus model
- **Workspace paths** (`/workspace/`) for persistent storage
- **Remote ADB connection** support
- **Extended timeouts** for network operations

## 🚀 Running the Crawler

### 1. Basic Usage
```bash
# Run with default settings
python scripts/runpod_crawler.py

# Custom app and parameters
python scripts/runpod_crawler.py --app com.android.settings --max-actions 200 --max-time 60
```

### 2. Advanced Configuration
```bash
# Custom ADB server connection
python scripts/runpod_crawler.py \
  --adb-host your-local-ip \
  --adb-port 5037 \
  --app com.example.myapp \
  --max-actions 500 \
  --max-time 120
```

## 📊 Expected Performance

### RunPod RTX 4090:
- **Model Loading**: ~2-3 minutes (first time)
- **Inference Speed**: ~2-3 seconds per screenshot
- **Memory Usage**: ~12-15GB GPU memory
- **Actions per minute**: ~15-20 actions

### RunPod A100:
- **Model Loading**: ~1-2 minutes (first time)
- **Inference Speed**: ~1-2 seconds per screenshot
- **Memory Usage**: ~12-15GB GPU memory
- **Actions per minute**: ~25-30 actions

## 🔍 Monitoring and Debugging

### 1. Check GPU Usage
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Check ADB Connection
```bash
# Test adbutils connection
python -c "import adbutils; adb = adbutils.AdbClient(); print('Devices:', adb.device_list())"

# Check if devices are accessible
python -c "import adbutils; adb = adbutils.AdbClient(); devices = adb.device_list(); print('Connected devices:', len(devices))"
```

### 3. View Logs
```bash
# Real-time log monitoring
tail -f /workspace/crawler.log

# Check results
ls -la /workspace/crawl_results/
```

## 🛠️ Troubleshooting

### Installation Issues
```bash
# If you get "unzip: command not found" error
apt install -y unzip wget curl

# If you get "wget: command not found" error
apt install -y wget curl

# If you get permission errors
sudo apt update && sudo apt install -y unzip wget curl
```

### ADB Connection Issues
```bash
# Restart ADB server
adb kill-server
adb start-server

# Check network connectivity
ping your-local-ip

# Test port connectivity
telnet your-local-ip 5037
```

### GPU Memory Issues
```bash
# Reduce batch size in config
# Set max_memory_usage to 0.6 instead of 0.8
```

## 🧪 Testing Setup

### 1. Run Setup Tests
```bash
# Test RunPod environment
python scripts/test_runpod_setup.py

# Test adbutils integration
python scripts/test_adbutils.py
```

### 2. Test Configuration
```bash
# Test configuration loading
python scripts/config_demo.py
```

## 📁 File Structure

```
/workspace/
├── crawl_results/          # Crawling results and reports
├── ui_venus_cache/         # Model cache and temporary files
├── scripts/
│   ├── runpod_crawler.py   # Main RunPod crawler
│   ├── test_runpod_setup.py # Setup testing
│   └── test_adbutils.py    # ADB testing
└── config/                 # Configuration files
```

## 🔄 Workflow

1. **Local Setup**: Start emulator and ADB server on local machine
2. **RunPod Setup**: Deploy and configure RunPod instance
3. **Connection**: Establish ADB connection between RunPod and local machine
4. **Crawling**: Run UI-Venus crawler on RunPod with remote device control
5. **Results**: Collect results from RunPod workspace

## 📋 Checklist

- [ ] RunPod instance created with GPU
- [ ] Local emulator running
- [ ] ADB server started on local machine
- [ ] RunPod connected to local ADB server
- [ ] Hugging Face token set
- [ ] Dependencies installed on RunPod
- [ ] Setup tests passing
- [ ] Configuration validated
- [ ] First crawling session successful

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run the setup tests to identify problems
3. Check logs for detailed error messages
4. Verify network connectivity between RunPod and local machine
5. Ensure all dependencies are properly installed

For additional help, refer to the main project documentation or create an issue in the repository.