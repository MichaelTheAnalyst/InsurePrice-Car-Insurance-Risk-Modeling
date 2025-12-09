# ========================================================
# 🚗 InsurePrice Dashboard Auto-Launcher (PowerShell)
# ========================================================
#
# Advanced PowerShell script for automated dashboard startup
# with enhanced error handling, logging, and system integration
#
# Usage:
#   .\start_dashboard.ps1
#   .\start_dashboard.ps1 -Port 8502
#   .\start_dashboard.ps1 -Background
#
# Parameters:
#   -Port: Custom port number (default: 8501)
#   -Background: Run in background without blocking console
#   -LogFile: Path to log file for output
#
# Author: Masood Nazari
# Date: December 2025
# ========================================================

param(
    [int]$Port = 8501,
    [switch]$Background,
    [string]$LogFile = "",
    [switch]$Headless
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    $ColorMap = @{
        "Red" = [ConsoleColor]::Red
        "Green" = [ConsoleColor]::Green
        "Yellow" = [ConsoleColor]::Yellow
        "Blue" = [ConsoleColor]::Blue
        "Cyan" = [ConsoleColor]::Cyan
        "Magenta" = [ConsoleColor]::Magenta
    }
    if ($ColorMap.ContainsKey($Color)) {
        Write-Host $Message -ForegroundColor $ColorMap[$Color]
    } else {
        Write-Host $Message
    }
}

# Function to log messages
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] $Message"

    if ($LogFile) {
        Add-Content -Path $LogFile -Value $LogEntry
    }

    switch ($Level) {
        "ERROR" { Write-ColorOutput $LogEntry "Red" }
        "WARN"  { Write-ColorOutput $LogEntry "Yellow" }
        "INFO"  { Write-ColorOutput $LogEntry "Green" }
        "DEBUG" { Write-ColorOutput $LogEntry "Cyan" }
        default { Write-Host $LogEntry }
    }
}

# Function to check if port is available
function Test-PortAvailable {
    param([int]$Port)
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect("localhost", $Port)
        $tcpClient.Close()
        return $false
    } catch {
        return $true
    }
}

# Function to kill process on port
function Stop-ProcessOnPort {
    param([int]$Port)
    try {
        $processes = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
                     Select-Object -ExpandProperty OwningProcess
        if ($processes) {
            foreach ($process in $processes) {
                Stop-Process -Id $process -Force
                Write-Log "Killed process $process on port $Port" "WARN"
            }
        }
    } catch {
        Write-Log "Could not kill processes on port $Port" "WARN"
    }
}

# Main execution block
try {
    Write-Log "🚗 Starting InsurePrice Dashboard..." "INFO"
    Write-Log "========================================" "INFO"
    Write-Log "🎨 AI-Powered Car Insurance Risk Modeling" "INFO"
    Write-Log "🎯 Interactive Dashboard with Color Psychology" "INFO"
    Write-Log "========================================" "INFO"

    # Get script directory and change to it
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $ScriptDir
    Write-Log "Working directory: $ScriptDir" "DEBUG"

    # Check if port is available
    if (-not (Test-PortAvailable $Port)) {
        Write-Log "Port $Port is already in use. Attempting to free it..." "WARN"
        Stop-ProcessOnPort $Port
        Start-Sleep -Seconds 2
        if (-not (Test-PortAvailable $Port)) {
            Write-Log "Could not free port $Port. Please choose a different port." "ERROR"
            exit 1
        }
    }

    # Check Python installation
    try {
        $pythonVersion = python --version 2>&1
        Write-Log "Python version: $pythonVersion" "INFO"
    } catch {
        Write-Log "Python is not installed or not in PATH!" "ERROR"
        Write-Log "Please install Python 3.8+ from https://python.org" "ERROR"
        exit 1
    }

    # Check required files
    $requiredFiles = @(
        "insureprice_dashboard.py",
        "Enhanced_Synthetic_Car_Insurance_Claims.csv",
        "actuarial_pricing_engine.py"
    )

    $missingFiles = @()
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            $missingFiles += $file
        }
    }

    if ($missingFiles.Count -gt 0) {
        Write-Log "Missing required files:" "ERROR"
        foreach ($file in $missingFiles) {
            Write-Log "  - $file" "ERROR"
        }
        exit 1
    }

    Write-Log "✅ All required files verified" "INFO"

    # Check for virtual environment
    $venvActivated = $false
    if (Test-Path "venv\Scripts\activate.ps1") {
        Write-Log "🔧 Activating virtual environment..." "INFO"
        & "venv\Scripts\activate.ps1"
        $venvActivated = $true
    } elseif (Test-Path ".venv\Scripts\activate.ps1") {
        Write-Log "🔧 Activating virtual environment (.venv)..." "INFO"
        & ".venv\Scripts\activate.ps1"
        $venvActivated = $true
    } else {
        Write-Log "ℹ️ No virtual environment found, using system Python" "INFO"
    }

    # Test Streamlit import
    try {
        python -c "import streamlit; print('Streamlit OK')" | Out-Null
        Write-Log "✅ Streamlit verified" "INFO"
    } catch {
        Write-Log "❌ Streamlit not installed or not accessible!" "ERROR"
        Write-Log "Please run: pip install -r requirements.txt" "ERROR"
        exit 1
    }

    # Prepare Streamlit command
    $streamlitArgs = @(
        "-m", "streamlit", "run", "insureprice_dashboard.py",
        "--server.port", $Port.ToString(),
        "--server.address", "localhost"
    )

    if ($Headless) {
        $streamlitArgs += "--server.headless", "true"
    }

    Write-Log "🚀 Launching dashboard on http://localhost:$Port" "INFO"
    Write-Log "Press Ctrl+C to stop the dashboard" "INFO"
    Write-Log "" "INFO"

    # Launch dashboard
    if ($Background) {
        # Run in background
        Write-Log "Running in background mode..." "INFO"
        $process = Start-Process -FilePath "python" -ArgumentList $streamlitArgs -NoNewWindow -PassThru
        Write-Log "Dashboard started with PID: $($process.Id)" "INFO"
        Write-Log "Dashboard is running in background on http://localhost:$Port" "INFO"

        # Optional: Wait for user input to keep script alive
        Read-Host "Press Enter to exit (dashboard will continue running)"
    } else {
        # Run in foreground (blocking)
        & python $streamlitArgs
    }

} catch {
    Write-Log "❌ An error occurred: $($_.Exception.Message)" "ERROR"
    exit 1
} finally {
    Write-Log "👋 Dashboard launcher finished" "INFO"
}
