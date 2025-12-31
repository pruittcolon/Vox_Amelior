# ============================================================================
# Nemo Server - Windows Remote Setup Script
# ============================================================================
# Run this on your Windows computer in PowerShell as Administrator:
#   Right-click PowerShell -> Run as Administrator
#   Then: Set-ExecutionPolicy Bypass -Scope Process -Force; .\windows_remote_setup.ps1
#
# This script will:
#   1. Install Tailscale (secure VPN)
#   2. Install Docker Desktop (requires WSL2)
#   3. Clone and configure Nemo Server
#   4. Start all services
# ============================================================================

param(
    [string]$TailscaleAuthKey = "",
    [string]$GitRepoUrl = "https://github.com/YOUR_USERNAME/Nemo_Server.git",
    [string]$InstallPath = "$env:USERPROFILE\Desktop\Nemo_Server",
    [switch]$SkipTailscale,
    [switch]$SkipDocker,
    [switch]$GPUMode
)

$ErrorActionPreference = "Stop"

# --- Color Output Functions ---
function Write-Step { param($msg) Write-Host "`n[STEP] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red }

# --- Check Admin ---
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Fail "This script must be run as Administrator!"
    Write-Host "Right-click PowerShell and select 'Run as Administrator'"
    exit 1
}

Write-Host @"
============================================================================
                    NEMO SERVER - WINDOWS REMOTE SETUP
============================================================================
This script will set up your Windows PC as a remote Nemo Server node.
Your Linux machine will be able to access this server via Tailscale.

Requirements:
  - Windows 10/11 (64-bit)
  - At least 8GB RAM (16GB recommended)
  - 50GB free disk space
  - Internet connection

Press Enter to continue or Ctrl+C to cancel...
"@
Read-Host

# ============================================================================
# STEP 1: Install Tailscale
# ============================================================================
if (-not $SkipTailscale) {
    Write-Step "Installing Tailscale..."
    
    $tailscaleInstalled = Get-Command tailscale -ErrorAction SilentlyContinue
    
    if ($tailscaleInstalled) {
        Write-Success "Tailscale already installed"
    } else {
        Write-Host "Downloading Tailscale installer..."
        $tailscaleUrl = "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe"
        $tailscaleInstaller = "$env:TEMP\tailscale-setup.exe"
        
        Invoke-WebRequest -Uri $tailscaleUrl -OutFile $tailscaleInstaller
        
        Write-Host "Running Tailscale installer (follow the prompts)..."
        Start-Process -FilePath $tailscaleInstaller -Wait
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Success "Tailscale installed"
    }
    
    # Connect to Tailscale
    Write-Host "`nConnecting to Tailscale network..."
    if ($TailscaleAuthKey) {
        & tailscale up --authkey $TailscaleAuthKey
    } else {
        Write-Host "A browser window will open. Log in with the same account as your Linux machine."
        & tailscale up
    }
    
    # Get Tailscale IP
    $tailscaleIP = & tailscale ip -4 2>$null
    if ($tailscaleIP) {
        Write-Success "Tailscale connected! Your IP: $tailscaleIP"
        Write-Host "Share this IP with your Linux machine to connect." -ForegroundColor Yellow
    }
}

# ============================================================================
# STEP 2: Enable WSL2 and Install Docker Desktop
# ============================================================================
if (-not $SkipDocker) {
    Write-Step "Setting up WSL2 and Docker..."
    
    # Check if WSL is installed
    $wslInstalled = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
    
    if ($wslInstalled.State -ne "Enabled") {
        Write-Host "Enabling WSL..."
        Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
        Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
        
        Write-Warn "WSL features enabled. A REBOOT is required."
        Write-Host "After reboot, run this script again with: -SkipTailscale"
        
        $reboot = Read-Host "Reboot now? (y/n)"
        if ($reboot -eq "y") {
            Restart-Computer
        }
        exit 0
    }
    
    # Set WSL2 as default
    wsl --set-default-version 2 2>$null
    
    # Check if Docker is installed
    $dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
    
    if ($dockerInstalled) {
        Write-Success "Docker already installed"
    } else {
        Write-Host "Downloading Docker Desktop..."
        $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
        $dockerInstaller = "$env:TEMP\DockerDesktopInstaller.exe"
        
        Invoke-WebRequest -Uri $dockerUrl -OutFile $dockerInstaller
        
        Write-Host "Running Docker Desktop installer..."
        Write-Host "(This may take several minutes)"
        Start-Process -FilePath $dockerInstaller -ArgumentList "install", "--quiet" -Wait
        
        Write-Success "Docker Desktop installed"
        Write-Warn "Docker Desktop needs to start. Please:"
        Write-Host "  1. Open Docker Desktop from Start Menu"
        Write-Host "  2. Accept the license agreement"
        Write-Host "  3. Wait for Docker to finish starting"
        Write-Host "  4. Run this script again with: -SkipTailscale -SkipDocker"
        
        Read-Host "Press Enter after Docker is running..."
    }
    
    # Verify Docker is running
    try {
        docker version | Out-Null
        Write-Success "Docker is running"
    } catch {
        Write-Fail "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    }
}

# ============================================================================
# STEP 3: Clone/Update Nemo Server Repository
# ============================================================================
Write-Step "Setting up Nemo Server repository..."

if (Test-Path $InstallPath) {
    Write-Host "Nemo Server directory exists. Updating..."
    Push-Location $InstallPath
    git pull origin main
    Pop-Location
} else {
    Write-Host "Cloning Nemo Server repository..."
    Write-Host "Repository: $GitRepoUrl"
    
    # Check if git is installed
    $gitInstalled = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitInstalled) {
        Write-Host "Installing Git..."
        winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    
    git clone $GitRepoUrl $InstallPath
}

Set-Location $InstallPath

# ============================================================================
# STEP 4: Configure Secrets
# ============================================================================
Write-Step "Configuring secrets..."

$secretsPath = Join-Path $InstallPath "docker\secrets"

if (-not (Test-Path $secretsPath)) {
    New-Item -ItemType Directory -Path $secretsPath -Force | Out-Null
}

# Check if secrets exist
$requiredSecrets = @(
    "jwt_secret_primary",
    "jwt_secret_previous", 
    "jwt_secret",
    "session_key",
    "postgres_user",
    "postgres_password",
    "users_db_key",
    "rag_db_key",
    "redis_password"
)

$missingSecrets = @()
foreach ($secret in $requiredSecrets) {
    $secretFile = Join-Path $secretsPath $secret
    if (-not (Test-Path $secretFile)) {
        $missingSecrets += $secret
    }
}

if ($missingSecrets.Count -gt 0) {
    Write-Warn "Missing secrets files:"
    $missingSecrets | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
    Write-Host "You need to copy the secrets from your Linux machine." -ForegroundColor Yellow
    Write-Host "On your Linux machine, run:"
    Write-Host "  scp -r ~/Desktop/Nemo_Server/docker/secrets/* <this-pc-tailscale-ip>:$secretsPath\"
    Write-Host ""
    Read-Host "Press Enter after copying secrets..."
}

Write-Success "Secrets configured"

# ============================================================================
# STEP 5: Start Nemo Server
# ============================================================================
Write-Step "Starting Nemo Server..."

Set-Location (Join-Path $InstallPath "docker")

if ($GPUMode) {
    Write-Host "Starting in GPU mode (full services)..."
    docker compose -f docker-compose.yml up -d --build
} else {
    Write-Host "Starting in CPU-only mode (no Gemma/Transcription)..."
    docker compose -f docker-compose.remote.yml up -d --build
}

# Wait for services to be healthy
Write-Host "Waiting for services to start (this may take 2-5 minutes)..."
Start-Sleep -Seconds 30

# Check health
$attempts = 0
$maxAttempts = 20
while ($attempts -lt $maxAttempts) {
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 5
        if ($health.status -eq "healthy") {
            Write-Success "Nemo Server is healthy!"
            break
        }
    } catch {
        $attempts++
        if ($attempts -lt $maxAttempts) {
            Write-Host "  Waiting... ($attempts/$maxAttempts)"
            Start-Sleep -Seconds 15
        }
    }
}

if ($attempts -ge $maxAttempts) {
    Write-Warn "Services may still be starting. Check with: docker ps"
}

# ============================================================================
# FINAL: Display Connection Info
# ============================================================================
$tailscaleIP = & tailscale ip -4 2>$null

Write-Host @"

============================================================================
                    NEMO SERVER - SETUP COMPLETE
============================================================================

Your Windows PC is now running Nemo Server!

TAILSCALE IP: $tailscaleIP
LOCAL URL:    http://localhost:8000
REMOTE URL:   http://${tailscaleIP}:8000

FROM YOUR LINUX MACHINE:
  Open browser: http://${tailscaleIP}:8000/ui/login.html

USEFUL COMMANDS (run in PowerShell from docker folder):
  View logs:    docker compose logs -f
  Stop:         docker compose down
  Restart:      docker compose restart
  Status:       docker ps

SERVICES RUNNING:
"@

docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host ""
Write-Success "Setup complete! Share your Tailscale IP with your Linux machine."
