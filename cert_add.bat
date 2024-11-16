@echo off
setlocal

REM Determine the directory the script is located in
set scriptDir=%~dp0

REM Path to the certificate file
set certPath=%scriptDir%YourCert.crt

REM Add the certificate to the Trusted People store
certutil -addstore TrustedPeople "%certPath%"

if %errorlevel% equ 0 (
    echo Certificate added to Trusted People store successfully.
) else (
    echo Failed to add certificate.
    echo Error Code: %errorlevel%
)

endlocal
pause
