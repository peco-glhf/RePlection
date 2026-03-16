﻿# RePlection — 起動スクリプト（Windows PowerShell）
#
# 使い方:
#   1. .env.example を .env にコピーして GEMINI_API_KEY を設定
#   2. PowerShell で: .\start.ps1
#   3. 停止: Ctrl+C（sidecar） + companion window を閉じる
#
# 前提:
#   - Python 3.12+ と uv がインストール済み
#   - Rust と Node.js がインストール済み（Tauri companion 用）
#   - companion/node_modules が存在しない場合は初回 npm install が走る

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot

Write-Host "=== RePlection 起動 ===" -ForegroundColor Cyan
Write-Host "プロジェクト: $ProjectRoot"

# .env チェック
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "[ERROR] .env が見つかりません。.env.example をコピーして GEMINI_API_KEY を設定してください。" -ForegroundColor Red
    exit 1
}

# --- sidecar（Python FastAPI）を別ウィンドウで起動 ---
Write-Host ""
Write-Host "[1/2] sidecar を起動中（port 8765）..." -ForegroundColor Yellow

$sidecarProc = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$ProjectRoot'; uv run python -m sidecar.main"
) -PassThru

Write-Host "      sidecar PID: $($sidecarProc.Id)"

# sidecar の起動待ち（最大 10 秒）
Write-Host "      起動確認中..."
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8765/health" -UseBasicParsing -TimeoutSec 1
        if ($resp.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # まだ起動中
    }
}

if ($ready) {
    Write-Host "      sidecar 起動完了" -ForegroundColor Green
} else {
    Write-Host "      [WARN] sidecar の応答確認タイムアウト（起動中の可能性あり）" -ForegroundColor Yellow
}

# --- Tauri companion window を起動 ---
Write-Host ""
Write-Host "[2/2] companion window を起動中..." -ForegroundColor Yellow

$companionDir = Join-Path $ProjectRoot "companion"

# 初回: npm install
$nodeModules = Join-Path $companionDir "node_modules"
if (-not (Test-Path $nodeModules)) {
    Write-Host "      npm install を実行中（初回のみ）..."
    Push-Location $companionDir
    npm install
    Pop-Location
}

Push-Location $companionDir
$companionProc = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$companionDir'; npm run tauri dev"
) -PassThru
Pop-Location

Write-Host "      companion PID: $($companionProc.Id)"

# --- 起動完了 ---
Write-Host ""
Write-Host "=== 起動完了 ===" -ForegroundColor Green
Write-Host "sidecar:   http://127.0.0.1:8765"
Write-Host "companion: Tauri ウィンドウが開きます"
Write-Host ""
Write-Host "停止方法:"
Write-Host "  - sidecar:   PowerShell ウィンドウを閉じる or Ctrl+C"
Write-Host "  - companion: ウィンドウの ✕ ボタン or PowerShell ウィンドウを閉じる"
Write-Host ""
Write-Host "このウィンドウは閉じて構いません。"
