@echo off
title Claude Code Gratis (NIM Proxy)
color 0A

echo.
echo  ==========================================
echo   CLAUDE CODE GRATIS - NVIDIA NIM
echo  ==========================================
echo.

REM Verificar si el proxy ya esta corriendo
curl -s http://localhost:3000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo  [OK] Proxy ya esta corriendo
    goto :iniciar_claude
)

echo  Iniciando servidor proxy...
start "NIM Proxy" /min cmd /c "cd /d C:\Users\titan\free-claude-code && uv run python server.py"

echo  Esperando que el proxy arranque...
:esperar
timeout /t 2 /nobreak >nul
curl -s http://localhost:3000/health >nul 2>&1
if %errorlevel% neq 0 goto :esperar

:iniciar_claude
echo  [OK] Proxy activo en localhost:3000
echo  Modelo: Llama 3.1 8B (NVIDIA NIM - GRATIS)
echo.
echo  Iniciando Claude Code...
echo.

set ANTHROPIC_BASE_URL=http://localhost:3000
set ANTHROPIC_AUTH_TOKEN=freecc

claude
