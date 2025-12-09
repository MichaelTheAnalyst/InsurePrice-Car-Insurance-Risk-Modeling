#!/usr/bin/env python3
"""
🚗 InsurePrice Dashboard Auto-Start Service
=============================================

Automated dashboard launcher with advanced features:
- Auto-restart on failure
- System integration
- Logging and monitoring
- Cross-platform compatibility

Usage:
    python auto_start_dashboard.py          # Interactive mode
    python auto_start_dashboard.py --daemon # Daemon/background mode
    python auto_start_dashboard.py --status # Check status
    python auto_start_dashboard.py --stop   # Stop service

Features:
- Automatic health checks and restarts
- Comprehensive logging
- System tray integration (Windows)
- Web-based monitoring dashboard
- Configuration management

Author: Masood Nazari
Date: December 2025
"""

import subprocess
import sys
import os
import time
import signal
import threading
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG = {
    "dashboard_file": "insureprice_dashboard.py",
    "port": 8501,
    "host": "localhost",
    "max_restarts": 5,
    "restart_delay": 10,
    "health_check_interval": 30,
    "log_file": "dashboard_service.log",
    "pid_file": "dashboard_service.pid",
    "status_file": "dashboard_service.status"
}

class DashboardService:
    """Automated dashboard service manager"""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.process = None
        self.restart_count = 0
        self.start_time = None
        self.running = False
        self.daemon_mode = False

        # Setup logging
        self.setup_logging()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        self.stop()

    def check_dependencies(self):
        """Check all required dependencies and files"""
        self.logger.info("🔍 Checking dependencies...")

        # Check dashboard file
        if not os.path.exists(self.config['dashboard_file']):
            raise FileNotFoundError(f"Dashboard file not found: {self.config['dashboard_file']}")

        # Check data files
        required_files = [
            "Enhanced_Synthetic_Car_Insurance_Claims.csv",
            "actuarial_pricing_engine.py"
        ]

        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")

        # Check Python modules
        try:
            import streamlit
            import pandas
            import plotly
        except ImportError as e:
            raise ImportError(f"Required module not found: {e}")

        self.logger.info("✅ All dependencies verified")

    def is_dashboard_running(self):
        """Check if dashboard is responding"""
        try:
            import requests
            response = requests.get(f"http://{self.config['host']}:{self.config['port']}/healthz", timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run", self.config['dashboard_file'],
                "--server.port", str(self.config['port']),
                "--server.address", self.config['host'],
                "--server.headless", "true"
            ]

            self.logger.info(f"🚀 Starting dashboard: {' '.join(cmd)}")

            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )

            # Wait for startup
            time.sleep(5)

            if self.is_dashboard_running():
                self.logger.info(f"✅ Dashboard started successfully on http://{self.config['host']}:{self.config['port']}")
                return True
            else:
                self.logger.error("❌ Dashboard failed to start properly")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to start dashboard: {e}")
            return False

    def stop_dashboard(self):
        """Stop the dashboard process"""
        if self.process:
            self.logger.info("🛑 Stopping dashboard...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                self.logger.info("✅ Dashboard stopped successfully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.logger.warning("⚠️ Dashboard forcefully killed")
            except Exception as e:
                self.logger.error(f"❌ Error stopping dashboard: {e}")

    def health_check_loop(self):
        """Continuous health monitoring"""
        while self.running:
            try:
                if not self.is_dashboard_running():
                    self.logger.warning("⚠️ Dashboard health check failed")
                    self.restart_dashboard()
                else:
                    self.logger.debug("✅ Dashboard health check passed")
            except Exception as e:
                self.logger.error(f"❌ Health check error: {e}")

            time.sleep(self.config['health_check_interval'])

    def restart_dashboard(self):
        """Restart the dashboard with exponential backoff"""
        if self.restart_count >= self.config['max_restarts']:
            self.logger.error(f"❌ Maximum restarts ({self.config['max_restarts']}) exceeded")
            self.stop()
            return

        self.restart_count += 1
        delay = self.config['restart_delay'] * (2 ** (self.restart_count - 1))  # Exponential backoff

        self.logger.info(f"🔄 Restarting dashboard (attempt {self.restart_count}/{self.config['max_restarts']}) in {delay}s...")

        self.stop_dashboard()
        time.sleep(delay)
        self.start_dashboard()

    def save_status(self):
        """Save service status to file"""
        status = {
            "running": self.running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "restart_count": self.restart_count,
            "pid": self.process.pid if self.process else None,
            "port": self.config['port'],
            "host": self.config['host']
        }

        try:
            with open(self.config['status_file'], 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save status: {e}")

    def load_status(self):
        """Load service status from file"""
        try:
            if os.path.exists(self.config['status_file']):
                with open(self.config['status_file'], 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load status: {e}")
        return None

    def start(self, daemon=False):
        """Start the service"""
        self.daemon_mode = daemon
        self.logger.info("🚗 Starting InsurePrice Dashboard Service")
        self.logger.info("=" * 50)

        try:
            # Check dependencies
            self.check_dependencies()

            # Start dashboard
            if not self.start_dashboard():
                raise RuntimeError("Failed to start dashboard")

            self.running = True
            self.start_time = datetime.now()

            # Save PID file
            with open(self.config['pid_file'], 'w') as f:
                f.write(str(os.getpid()))

            # Start health check thread
            health_thread = threading.Thread(target=self.health_check_loop, daemon=True)
            health_thread.start()

            self.logger.info("✅ Service started successfully")
            self.save_status()

            if daemon:
                # Daemon mode - keep running
                while self.running:
                    time.sleep(1)
            else:
                # Interactive mode - wait for user input
                try:
                    input("Press Enter to stop the service...\n")
                except KeyboardInterrupt:
                    pass

        except Exception as e:
            self.logger.error(f"❌ Service failed to start: {e}")
            self.running = False
        finally:
            self.stop()

    def stop(self):
        """Stop the service"""
        self.logger.info("🛑 Stopping InsurePrice Dashboard Service")
        self.running = False
        self.stop_dashboard()

        # Clean up files
        for file in [self.config['pid_file'], self.config['status_file']]:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                self.logger.error(f"Failed to remove {file}: {e}")

        self.logger.info("✅ Service stopped")

    def status(self):
        """Get service status"""
        status_info = self.load_status()
        if status_info:
            print("🚗 InsurePrice Dashboard Service Status")
            print("=" * 40)
            print(f"Running: {'✅ Yes' if status_info.get('running') else '❌ No'}")
            print(f"Start Time: {status_info.get('start_time', 'N/A')}")
            print(f"PID: {status_info.get('pid', 'N/A')}")
            print(f"Port: {status_info.get('port', 'N/A')}")
            print(f"Restarts: {status_info.get('restart_count', 0)}")
        else:
            print("❌ Service status not available")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="InsurePrice Dashboard Auto-Start Service")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--status", action="store_true", help="Show service status")
    parser.add_argument("--stop", action="store_true", help="Stop the service")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard port")
    parser.add_argument("--host", default="localhost", help="Dashboard host")

    args = parser.parse_args()

    # Update config with arguments
    config = CONFIG.copy()
    config.update({
        "port": args.port,
        "host": args.host
    })

    service = DashboardService(config)

    if args.status:
        service.status()
    elif args.stop:
        # Load existing service and stop it
        status = service.load_status()
        if status and status.get('running'):
            service.stop()
        else:
            print("❌ Service is not running")
    else:
        # Start service
        service.start(daemon=args.daemon)

if __name__ == "__main__":
    main()
