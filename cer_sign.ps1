# Define the parameters
$certName = "Anantha Venkata Subramanyam G"
$exeFile = "Main.exe"
$signedExeFile = "Main-signed.exe"
$expiryYears = 10
$pfxPassword = "P@ssw0rd" # Replace with a strong password of your choice

# Create a self-signed certificate
$cert = New-SelfSignedCertificate -Type CodeSigningCert -CertStoreLocation Cert:\CurrentUser\My -Subject $certName -NotAfter (Get-Date).AddYears($expiryYears) -KeyExportPolicy Exportable -KeySpec Signature

# Export the certificate to a PFX file in the current working directory
$pfxPath = "$PWD\MySelfSignedCert.pfx"
Export-PfxCertificate -Cert $cert -FilePath $pfxPath -Password (ConvertTo-SecureString -String $pfxPassword -Force -AsPlainText)
# Export the certificate to a .cer file in the current working directory
$cerPath = "$PWD\MySelfSignedCert.cer"
Export-Certificate -Cert $cert -FilePath $cerPath

# Sign the executable file using signtool
#Start-Process "signtool.exe" -ArgumentList "sign /fd SHA256 /f $pfxPath /p $pfxPassword /t http://timestamp.digicert.com $exeFile /out $signedExeFile" -NoNewWindow -Wait

#Write-Host "Executable signed successfully and saved as $signedExeFile"
