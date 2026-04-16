$WshShell = New-Object -comObject WScript.Shell
$Desktop = [System.Environment]::GetFolderPath("Desktop")

# Intentar escritorio de OneDrive si existe
$OneDriveDesktop = "$env:USERPROFILE\OneDrive\Desktop"
if (Test-Path $OneDriveDesktop) {
    $Desktop = $OneDriveDesktop
}

$Shortcut = $WshShell.CreateShortcut("$Desktop\Claude Code Gratis.lnk")
$Shortcut.TargetPath = "C:\Users\titan\free-claude-code\INICIAR_CLAUDE_GRATIS.bat"
$Shortcut.WorkingDirectory = "C:\Users\titan\free-claude-code"
$Shortcut.IconLocation = "C:\Windows\System32\cmd.exe,0"
$Shortcut.Description = "Claude Code gratis via NVIDIA NIM Proxy"
$Shortcut.WindowStyle = 1
$Shortcut.Save()

Write-Host "Acceso directo creado en: $Desktop\Claude Code Gratis.lnk" -ForegroundColor Green
