# ========================================================
# 🚗 InsurePrice Dashboard - Windows Task Scheduler Setup
# ========================================================
#
# This script creates a Windows Scheduled Task to automatically
# start the InsurePrice dashboard on system boot or at scheduled times.
#
# Features:
# - Auto-start on system boot
# - Scheduled daily startup
# - Automatic restart on failure
# - User-friendly task management
#
# Usage:
#   .\setup_windows_task.ps1                 # Interactive setup
#   .\setup_windows_task.ps1 -Uninstall      # Remove task
#   .\setup_windows_task.ps1 -Status         # Check task status
#
# Author: Masood Nazari
# Date: December 2025
# ========================================================

param(
    [switch]$Uninstall,
    [switch]$Status,
    [switch]$Boot,
    [switch]$Daily,
    [string]$Time = "09:00",
    [string]$TaskName = "InsurePrice Dashboard"
)

# Requires administrator privileges for task creation
#Requires -RunAsAdministrator

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonPath = (Get-Command python).Source
$ScriptPath = Join-Path $ScriptDir "auto_start_dashboard.py"
$WorkingDir = $ScriptDir
$LogFile = Join-Path $ScriptDir "dashboard_task.log"

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

# Function to check if task exists
function Test-TaskExists {
    param([string]$TaskName)
    try {
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        return $null -ne $task
    } catch {
        return $false
    }
}

# Function to create boot task
function New-BootTask {
    param([string]$TaskName)

    Write-ColorOutput "🚀 Creating boot startup task..." "Green"

    try {
        $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`" --daemon" -WorkingDirectory $WorkingDir
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

        # Set to run as current user
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

        $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Automatically start InsurePrice Dashboard on system boot"

        Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

        Write-ColorOutput "✅ Boot task created successfully!" "Green"
        Write-ColorOutput "   Dashboard will start automatically when Windows boots" "Cyan"

    } catch {
        Write-ColorOutput "❌ Failed to create boot task: $($_.Exception.Message)" "Red"
        return $false
    }

    return $true
}

# Function to create daily task
function New-DailyTask {
    param([string]$TaskName, [string]$Time)

    Write-ColorOutput "📅 Creating daily scheduled task..." "Green"

    try {
        $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$ScriptPath`" --daemon" -WorkingDirectory $WorkingDir
        $trigger = New-ScheduledTaskTrigger -Daily -At $Time
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

        $task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Start InsurePrice Dashboard daily at $Time"

        Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null

        Write-ColorOutput "✅ Daily task created successfully!" "Green"
        Write-ColorOutput "   Dashboard will start daily at $Time" "Cyan"

    } catch {
        Write-ColorOutput "❌ Failed to create daily task: $($_.Exception.Message)" "Red"
        return $false
    }

    return $true
}

# Function to remove task
function Remove-DashboardTask {
    param([string]$TaskName)

    Write-ColorOutput "🗑️ Removing dashboard task..." "Yellow"

    try {
        if (Test-TaskExists $TaskName) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-ColorOutput "✅ Task removed successfully!" "Green"
        } else {
            Write-ColorOutput "ℹ️ Task '$TaskName' not found" "Yellow"
        }
    } catch {
        Write-ColorOutput "❌ Failed to remove task: $($_.Exception.Message)" "Red"
        return $false
    }

    return $true
}

# Function to show task status
function Show-TaskStatus {
    param([string]$TaskName)

    Write-ColorOutput "📊 Task Status for '$TaskName'" "Cyan"
    Write-ColorOutput "=" * 40 "Cyan"

    try {
        if (Test-TaskExists $TaskName) {
            $task = Get-ScheduledTask -TaskName $TaskName
            $info = Get-ScheduledTaskInfo -TaskName $TaskName

            Write-ColorOutput "Status: ✅ Task exists" "Green"
            Write-ColorOutput "State: $($task.State)" "White"
            Write-ColorOutput "Last Run: $($info.LastRunTime)" "White"
            Write-ColorOutput "Last Result: $($info.LastTaskResult)" "White"
            Write-ColorOutput "Next Run: $($info.NextRunTime)" "White"
            Write-ColorOutput "Author: $($task.Author)" "White"

            # Show triggers
            Write-ColorOutput "Triggers:" "Yellow"
            foreach ($trigger in $task.Triggers) {
                Write-ColorOutput "  - $($trigger)" "White"
            }

        } else {
            Write-ColorOutput "Status: ❌ Task not found" "Red"
        }
    } catch {
        Write-ColorOutput "❌ Error getting task status: $($_.Exception.Message)" "Red"
    }
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-ColorOutput "🔍 Checking prerequisites..." "Cyan"

    $checks = @(
        @{ Name = "Python"; Check = { Get-Command python -ErrorAction SilentlyContinue } },
        @{ Name = "Auto-start script"; Check = { Test-Path $ScriptPath } },
        @{ Name = "Dashboard file"; Check = { Test-Path (Join-Path $ScriptDir "insureprice_dashboard.py") } },
        @{ Name = "Data file"; Check = { Test-Path (Join-Path $ScriptDir "Enhanced_Synthetic_Car_Insurance_Claims.csv") } }
    )

    $allPassed = $true

    foreach ($check in $checks) {
        try {
            $result = & $check.Check
            if ($result) {
                Write-ColorOutput "✅ $($check.Name)" "Green"
            } else {
                Write-ColorOutput "❌ $($check.Name)" "Red"
                $allPassed = $false
            }
        } catch {
            Write-ColorOutput "❌ $($check.Name): $($_.Exception.Message)" "Red"
            $allPassed = $false
        }
    }

    return $allPassed
}

# Main execution
Write-ColorOutput "🚗 InsurePrice Dashboard - Task Scheduler Setup" "Magenta"
Write-ColorOutput "=" * 50 "Magenta"
Write-ColorOutput "Automate dashboard startup with Windows Task Scheduler" "Cyan"
Write-ColorOutput "" "White"

# Check if running as administrator
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
$isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-ColorOutput "⚠️ This script requires administrator privileges!" "Yellow"
    Write-ColorOutput "Please run PowerShell as Administrator and try again." "Yellow"
    exit 1
}

# Handle command line arguments
if ($Uninstall) {
    Remove-DashboardTask $TaskName
    exit
}

if ($Status) {
    Show-TaskStatus $TaskName
    exit
}

# Check prerequisites
if (-not (Test-Prerequisites)) {
    Write-ColorOutput "❌ Prerequisites not met. Please fix the issues above and try again." "Red"
    exit 1
}

# Interactive setup if no specific mode requested
if (-not $Boot -and -not $Daily) {
    Write-ColorOutput "Choose automation type:" "Cyan"
    Write-ColorOutput "1. Boot startup (starts when Windows boots)" "White"
    Write-ColorOutput "2. Daily schedule (starts at specific time)" "White"
    Write-ColorOutput "3. Both boot and daily" "White"
    Write-ColorOutput "" "White"

    $choice = Read-Host "Enter your choice (1-3)"

    switch ($choice) {
        "1" { $Boot = $true }
        "2" {
            $Daily = $true
            $Time = Read-Host "Enter daily start time (HH:mm, default: $Time)"
            if (-not $Time) { $Time = "09:00" }
        }
        "3" {
            $Boot = $true
            $Daily = $true
            $Time = Read-Host "Enter daily start time (HH:mm, default: $Time)"
            if (-not $Time) { $Time = "09:00" }
        }
        default {
            Write-ColorOutput "❌ Invalid choice. Exiting." "Red"
            exit 1
        }
    }
}

# Remove existing task if it exists
if (Test-TaskExists $TaskName) {
    Write-ColorOutput "📝 Updating existing task..." "Yellow"
    Remove-DashboardTask $TaskName
}

# Create tasks
$success = $true

if ($Boot) {
    $bootTaskName = "$TaskName (Boot)"
    if (-not (New-BootTask $bootTaskName)) {
        $success = $false
    }
}

if ($Daily) {
    $dailyTaskName = "$TaskName (Daily)"
    if (-not (New-DailyTask $dailyTaskName $Time)) {
        $success = $false
    }
}

if ($success) {
    Write-ColorOutput "" "White"
    Write-ColorOutput "🎉 Setup completed successfully!" "Green"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Next steps:" "Cyan"
    Write-ColorOutput "1. Reboot your system (for boot task)" "White"
    Write-ColorOutput "2. Or wait for the scheduled time (for daily task)" "White"
    Write-ColorOutput "3. Dashboard will start automatically" "White"
    Write-ColorOutput "4. Check Task Scheduler for task status" "White"
    Write-ColorOutput "" "White"
    Write-ColorOutput "Manual control:" "Cyan"
    Write-ColorOutput "• python auto_start_dashboard.py --status  # Check status" "White"
    Write-ColorOutput "• python auto_start_dashboard.py --stop    # Stop service" "White"
    Write-ColorOutput "" "White"
} else {
    Write-ColorOutput "❌ Setup failed. Please check the error messages above." "Red"
}
