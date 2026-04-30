# fcc.ps1 — Free Claude Code one-click launcher (Windows PowerShell)
# Usage:
#   ./fcc.ps1           Start proxy in background
#   ./fcc.ps1 claude    Start proxy + launch Claude Code
#   ./fcc.ps1 stop      Stop the proxy
#
# Requires: uv, Claude Code CLI

param(
    [string]$Action = "proxy"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Port = if ($env:FCC_PORT) { [int]$env:FCC_PORT } else { 8082 }
$BaseUrl = "http://localhost:$Port"
$AuthToken = if ($env:ANTHROPIC_AUTH_TOKEN) { $env:ANTHROPIC_AUTH_TOKEN } else { "freecc" }
$LogFile = "$env:TEMP\fcc-proxy.log"

function Get-ProcessIdByPort {
    param([int]$Port)
    # Get-NetTCPConnection 需要管理员权限，失败时回退到 netstat
    $procId = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess
    if (-not $procId) {
        $line = netstat -ano | Select-String ":$Port\s"
        if ($line) { $procId = ($line.Line -split '\s+')[-1] }
    }
    return $procId
}

function Start-Proxy {
    Write-Host "🔧 清理端口 $Port ..."
    $existingPids = Get-ProcessIdByPort -Port $Port
    if ($existingPids) {
        $existingPids | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 1
    }
    Start-Sleep -Seconds 1

    Write-Host "🚀 启动代理服务器 → $BaseUrl"
    Set-Location $ScriptDir
    Remove-Job -Name "FCCProxy" -ErrorAction SilentlyContinue
    $job = Start-Job -Name "FCCProxy" -ScriptBlock {
        param($dir, $port, $log)
        Set-Location $dir
        uv run uvicorn server:app --host 0.0.0.0 --port $port *>"$log"
    } -ArgumentList $ScriptDir, $Port, $LogFile

    # 等待代理就绪
    for ($i = 0; $i -lt 10; $i++) {
        $procId = Get-ProcessIdByPort -Port $Port
        if ($procId) {
            Write-Host "✅ 代理已就绪"
            return
        }
        Start-Sleep -Seconds 1
    }

    Write-Host "❌ 代理启动失败，查看日志: $LogFile"
    exit 1
}

function Stop-Proxy {
    $procIds = Get-ProcessIdByPort -Port $Port
    if ($procIds) {
        Write-Host "🛑 停止代理 (PID: $($procIds -join ','))..."
        $procIds | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
        Write-Host "✅ 已停止"
    } else {
        Write-Host "ℹ️  代理未运行"
    }
}

switch ($Action) {
    "proxy" {
        Start-Proxy
    }
    "claude" {
        if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
            Write-Host "❌ 未找到 claude 命令，请先安装 Claude Code"
            exit 1
        }
        Start-Proxy
        Write-Host ""
        Write-Host "=============================================="
        Write-Host "  启动 Claude Code..."
        Write-Host "=============================================="
        Write-Host ""
        $env:ANTHROPIC_AUTH_TOKEN = $AuthToken
        $env:ANTHROPIC_BASE_URL = $BaseUrl
        claude
    }
    "stop" {
        Stop-Proxy
    }
    default {
        Write-Host "用法: .\fcc.ps1 [proxy|claude|stop]"
        Write-Host ""
        Write-Host "  .\fcc.ps1         后台启动代理"
        Write-Host "  .\fcc.ps1 claude  启动代理 + 启动 Claude Code"
        Write-Host "  .\fcc.ps1 stop    停止代理"
    }
}
