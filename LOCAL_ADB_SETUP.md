# Local ADB Server Setup Guide

This guide helps you set up your local machine to work with the RunPod deployment.

## 🏠 Local Machine Setup (Where your emulator runs)

### 1. Start Your Emulator
```bash
# Start your Android emulator (however you normally do it)
# This could be through Android Studio, command line, etc.
```

### 2. Start ADB Server
```bash
# Start the ADB server
adb start-server

# Verify your emulator is connected
adb devices
# Should show: emulator-5554    device
```

### 3. Enable TCP/IP Mode
```bash
# Enable TCP/IP mode on your emulator
adb tcpip 5555

# Or use the default ADB port
adb tcpip 5037
```

### 4. Find Your Local IP
```bash
# On macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# On Windows
ipconfig | findstr "IPv4"
```

## 🌐 RunPod Connection

### 1. Configure Connection
The `adbutils` package will automatically handle ADB connections. You just need to ensure your local ADB server is accessible.

### 2. Test Connection
```bash
# Test adbutils connection from RunPod
python -c "import adbutils; adb = adbutils.AdbClient(); print('Devices:', adb.device_list())"

# Test basic device operations
python -c "import adbutils; adb = adbutils.AdbClient(); device = adb.device_list()[0] if adb.device_list() else None; print('Device info:', device.shell('getprop ro.build.version.release') if device else 'No devices')"
```

## 🔧 Troubleshooting

### Connection Issues
```bash
# If connection fails, try:
# 1. Check firewall settings on local machine
# 2. Ensure ADB server is running: adb start-server
# 3. Try different port: adb tcpip 5037
# 4. Restart ADB: adb kill-server && adb start-server
```

### Port Forwarding (Alternative)
```bash
# On local machine, set up port forwarding
adb -s emulator-5554 forward tcp:5037 tcp:5037

# Then connect from RunPod using port 5037
adb connect <your-local-ip>:5037
```

### Firewall Configuration
```bash
# On macOS, allow incoming connections
sudo pfctl -f /etc/pf.conf

# On Linux, open port in firewall
sudo ufw allow 5555
sudo ufw allow 5037
```

## 📱 Expected Setup

```
Local Machine                    RunPod Instance
┌─────────────────┐             ┌─────────────────┐
│ Android Emulator│             │ UI-Venus Crawler│
│ (emulator-5554) │◄────────────┤                 │
│                 │             │                 │
│ ADB Server      │◄────────────┤ ADB Client      │
│ (port 5555)     │   Network   │                 │
└─────────────────┘             └─────────────────┘
```

## 🚀 Quick Start Commands

### Local Machine:
```bash
# 1. Start emulator
# 2. Start ADB server
adb start-server

# 3. Enable TCP/IP
adb tcpip 5555

# 4. Note your IP address
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### RunPod Instance:
```bash
# 1. Test adbutils connection
python -c "import adbutils; adb = adbutils.AdbClient(); print('Devices:', adb.device_list())"

# 2. Run comprehensive adbutils test
python scripts/test_adbutils.py

# 3. Run crawler (adbutils handles connection automatically)
python scripts/runpod_crawler.py
```

## 💡 Tips

1. **Keep ADB server running** on your local machine while using RunPod
2. **Use a static IP** or note your IP address for consistent connections
3. **Test connection** before running the crawler
4. **Check firewall settings** if connection fails
5. **Use port forwarding** as an alternative to TCP/IP mode
