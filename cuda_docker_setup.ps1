# Run as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "Please run this script as Administrator!"
    Break
}

Write-Host "Setting up NVIDIA Container Toolkit for Windows..." -ForegroundColor Green

# Check if Docker Desktop is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Warning "Docker Desktop is not installed. Please install Docker Desktop first."
    Break
}

# Enable Windows containers feature
Write-Host "Enabling Windows containers feature..." -ForegroundColor Yellow
Enable-WindowsOptionalFeature -Online -FeatureName containers -All

# Install NVIDIA Driver if not present
if (!(Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*NVIDIA*"})) {
    Write-Warning "NVIDIA GPU not detected or drivers not installed."
    Write-Host "Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx"
    Break
}

# Configure Docker to use NVIDIA Runtime
$dockerConfig = "$env:USERPROFILE\.docker\daemon.json"
$configContent = @{
    "runtimes" = @{
        "nvidia" = @{
            "path" = "nvidia-container-runtime"
            "runtimeArgs" = @()
        }
    }
    "default-runtime" = "nvidia"
} | ConvertTo-Json -Depth 10

# Create or update Docker config
if (!(Test-Path $dockerConfig)) {
    New-Item -Path $dockerConfig -Force
}
Set-Content -Path $dockerConfig -Value $configContent

Write-Host "Configuration complete. Please:" -ForegroundColor Green
Write-Host "1. Restart Docker Desktop" -ForegroundColor Yellow
Write-Host "2. Enable WSL 2 based engine in Docker Desktop settings" -ForegroundColor Yellow
Write-Host "3. Enable NVIDIA GPU integration in Docker Desktop settings" -ForegroundColor Yellow 