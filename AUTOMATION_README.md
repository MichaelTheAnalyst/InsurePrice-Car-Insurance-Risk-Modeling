# 🚗 InsurePrice Dashboard Automation Guide

## Automated Dashboard Startup Solutions

This guide provides multiple methods to automatically start the InsurePrice dashboard without manual intervention.

---

## 📋 Quick Start Options

### 1. **One-Click Windows Launch** (Simplest)
```cmd
# Double-click this file
start_dashboard.bat
```
**Features:** Automatic dependency checks, virtual environment detection, error handling

### 2. **PowerShell Automation** (Advanced)
```powershell
# Interactive launch
.\start_dashboard.ps1

# Background mode
.\start_dashboard.ps1 -Background

# Custom port
.\start_dashboard.ps1 -Port 8502
```
**Features:** Port checking, logging, background operation, custom configuration

### 3. **Auto-Service Mode** (Enterprise)
```bash
# Start as service
python auto_start_dashboard.py --daemon

# Check status
python auto_start_dashboard.py --status

# Stop service
python auto_start_dashboard.py --stop
```
**Features:** Auto-restart, health monitoring, system integration, comprehensive logging

---

## 🎯 Automation Methods by Use Case

| Use Case | Method | Setup Time | Maintenance |
|----------|--------|------------|-------------|
| **Personal Use** | Batch file (.bat) | 1 minute | None |
| **Development** | PowerShell script | 2 minutes | Low |
| **Production Server** | Auto-service | 5 minutes | Medium |
| **System Integration** | Windows Task Scheduler | 10 minutes | Low |

---

## 🪟 Windows Automation (Recommended)

### Method 1: Simple Batch File
1. **Double-click** `start_dashboard.bat`
2. Dashboard starts automatically on `http://localhost:8501`

**Features:**
- ✅ Automatic dependency verification
- ✅ Virtual environment detection
- ✅ Error messages and troubleshooting
- ✅ Clean shutdown with Ctrl+C

### Method 2: PowerShell Script (Advanced)
```powershell
# Install execution policy (one-time)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run the script
.\start_dashboard.ps1
```

**Features:**
- ✅ Port availability checking
- ✅ Process management
- ✅ Comprehensive logging
- ✅ Background operation support
- ✅ Custom port configuration

### Method 3: Windows Task Scheduler (Automatic)
```powershell
# Run as Administrator
.\setup_windows_task.ps1
```

**Options:**
1. **Boot startup** - Starts when Windows boots
2. **Daily schedule** - Starts at specific time (e.g., 9 AM)
3. **Both** - Boot + scheduled startup

**Features:**
- ✅ Fully automatic operation
- ✅ System integration
- ✅ Failure recovery
- ✅ No user interaction required

---

## 🔧 Configuration Options

### Environment Variables
```cmd
# Set custom port
set DASHBOARD_PORT=8502

# Set custom host
set DASHBOARD_HOST=0.0.0.0

# Enable debug logging
set DASHBOARD_DEBUG=1
```

### Virtual Environment
The automation scripts automatically detect and use virtual environments:
- `venv/` - Standard virtual environment
- `.venv/` - Alternative location

### Logging
All automated solutions create log files:
- `dashboard_service.log` - Service operation logs
- `dashboard_task.log` - Task scheduler logs

---

## 🐧 Linux/Mac Automation

### Systemd Service (Linux)
```bash
# Create service file
sudo nano /etc/systemd/system/insureprice-dashboard.service
```

**Service file content:**
```ini
[Unit]
Description=InsurePrice Dashboard Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/insureprice
ExecStart=/usr/bin/python3 auto_start_dashboard.py --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Control commands:**
```bash
# Enable and start
sudo systemctl enable insureprice-dashboard
sudo systemctl start insureprice-dashboard

# Status and logs
sudo systemctl status insureprice-dashboard
sudo journalctl -u insureprice-dashboard -f
```

### LaunchDaemon (macOS)
```bash
# Create plist file
sudo nano /Library/LaunchDaemons/com.insureprice.dashboard.plist
```

**Plist content:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.insureprice.dashboard</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/path/to/auto_start_dashboard.py</string>
        <string>--daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/insureprice-dashboard.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/insureprice-dashboard-error.log</string>
</dict>
</plist>
```

### Cron Job (Simple Scheduling)
```bash
# Edit crontab
crontab -e

# Add daily startup at 9 AM
0 9 * * * cd /path/to/insureprice && python3 auto_start_dashboard.py --daemon
```

---

## 🐳 Docker Automation

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/healthz || exit 1

# Start command
CMD ["python", "auto_start_dashboard.py", "--daemon"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  insureprice-dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    environment:
      - DASHBOARD_PORT=8501
      - DASHBOARD_HOST=0.0.0.0
```

**Run commands:**
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f insureprice-dashboard

# Stop
docker-compose down
```

---

## 📊 Monitoring & Management

### Service Status
```bash
# Check if running
python auto_start_dashboard.py --status

# View logs
tail -f dashboard_service.log

# Check port usage
netstat -an | grep 8501
```

### Windows Task Scheduler
```powershell
# View task status
.\setup_windows_task.ps1 -Status

# Remove tasks
.\setup_windows_task.ps1 -Uninstall
```

### Process Management
```bash
# Find dashboard processes
ps aux | grep streamlit

# Kill specific process
kill -9 <PID>

# Windows
taskkill /F /IM python.exe /FI "WINDOWTITLE eq streamlit*"
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Find what's using the port
netstat -an | grep 8501
# Kill the process
kill -9 <PID>
```

**2. Python Not Found**
```bash
# Check Python installation
python --version
# Add to PATH or use full path
```

**3. Virtual Environment Issues**
```bash
# Activate manually
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**4. Permission Errors**
```bash
# Windows: Run as Administrator
# Linux/Mac: Use sudo or change permissions
chmod +x auto_start_dashboard.py
```

### Log Files
- `dashboard_service.log` - Main service logs
- `dashboard_service.status` - Service status JSON
- `dashboard_service.pid` - Process ID file

---

## 🎯 Best Practices

### Security
- Run with minimal required permissions
- Use virtual environments
- Regularly update dependencies
- Monitor access logs

### Performance
- Configure appropriate memory limits
- Use SSD storage for data files
- Monitor system resources
- Implement log rotation

### Backup & Recovery
- Backup configuration files
- Regular data backups
- Test recovery procedures
- Document customizations

---

## 📞 Support

### Quick Diagnostics
```bash
# Run diagnostic script
python -c "
import sys
print('Python version:', sys.version)
import streamlit
print('Streamlit OK')
import pandas as pd
print('Pandas OK')
import os
print('Files exist:', os.path.exists('insureprice_dashboard.py'))
"
```

### Common Commands
```bash
# Manual start
python run_dashboard.py

# Service start
python auto_start_dashboard.py --daemon

# Check health
curl http://localhost:8501/healthz
```

---

*This automation guide ensures the InsurePrice dashboard can run reliably in any environment with minimal manual intervention.*

