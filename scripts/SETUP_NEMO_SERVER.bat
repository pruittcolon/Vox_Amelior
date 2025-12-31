@echo off
REM ============================================================================
REM Nemo Server - One-Click Windows Setup
REM ============================================================================
REM INSTRUCTIONS:
REM   1. Right-click this file
REM   2. Select "Run as administrator"
REM   3. Follow the prompts
REM ============================================================================

echo.
echo ============================================================================
echo                    NEMO SERVER - WINDOWS SETUP
echo ============================================================================
echo.
echo This will set up your computer as a remote Nemo Server node.
echo.
echo REQUIREMENTS:
echo   - Windows 10/11 (64-bit)
echo   - 8GB RAM minimum (16GB recommended)
echo   - 50GB free disk space
echo   - Internet connection
echo.
echo Press any key to continue or close this window to cancel...
pause > nul

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo.
    echo ERROR: This script requires Administrator privileges!
    echo.
    echo Please right-click this file and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Run the PowerShell setup script
echo.
echo Starting setup...
echo.
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%windows_remote_setup.ps1"

echo.
echo ============================================================================
echo Setup process completed. Check above for any errors.
echo ============================================================================
echo.
pause
