@echo off
setlocal
echo Checking for the presence of Docker
docker --version
if %errorlevel% neq 0 (
     echo Docker is not installed. Installing Docker...
     winget.exe install --id Docker.DockerDesktop -e
 ) else (
     echo Docker is already installed.
)
if %errorlevel% neq 0 (
     echo Docker installation failed. Exiting...
     exit 1)
REM Determine the directory the script is located in
set scriptDir=%~dp0

REM Path to the certificate file
set certPath=%scriptDir%unnamed.crt

REM Add the certificate to the Trusted People store
certutil -addstore TrustedPeople "%certPath%"

if %errorlevel% equ 0 (
    echo Certificate added to Trusted People store successfully.
) else (
    echo Failed to add certificate.
    echo Error Code: %errorlevel%
)

endlocal
.\Unnamed_installer.msi
pause
