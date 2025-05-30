@echo off
title Cloudflare Tunnel for Virtual Patient

echo ========================================================================================================================
echo WARNING: Cloudflare Tunnel for "Virtual Patient"!
echo ========================================================================================================================
echo This script downloads (if necessary) and runs cloudflared.exe from Cloudflare
echo to create a temporary HTTPS tunnel to your locally running "Virtual Patient" application.
echo.
echo The "Virtual Patient" application MUST BE RUNNING before creating the tunnel
echo (usually with the start_virtual_patient.bat file).
echo.
echo Using the generated temporary tunnel URL, anyone can access
echo your "Virtual Patient" application over the Internet while this script and the tunnel are active.
echo.
echo IMPORTANT:
echo 1. Keep the tunnel URL secret if you do not want public access.
echo 2. Ensure your "Virtual Patient" application does not expose
echo    sensitive data without proper authentication (if applicable).
echo 3. This script only creates the tunnel. The security of the application itself is your responsibility.
echo.
echo To stop the tunnel, press Ctrl+C in this window or simply close it!
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause
echo.

echo Checking for cloudflared.exe...
if not exist cloudflared.exe (
    echo cloudflared.exe not found. Attempting to download...
    REM Using PowerShell for a more reliable download, as curl might not be universally installed
    powershell -Command "Invoke-WebRequest -Uri https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -OutFile cloudflared.exe"
    
    IF NOT EXIST "cloudflared.exe" (
        echo ERROR: Failed to download cloudflared.exe.
        echo Please check your internet connection or download it manually from the Cloudflare website
        echo and place cloudflared.exe in the same folder as this script.
        echo.
        pause
        exit /b 1
    )
    echo Download of cloudflared.exe complete.
) else (
    echo cloudflared.exe found.
)
echo.

echo Starting Cloudflare Tunnel for localhost:8501 ...
echo (Port 8501 is the default for Streamlit applications)
echo.
cloudflared.exe tunnel --url localhost:8501

echo.
echo The Cloudflare tunnel has been closed or encountered an error.
pause