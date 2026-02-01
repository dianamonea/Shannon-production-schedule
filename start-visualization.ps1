#!/usr/bin/env pwsh
<#
.SYNOPSIS
    多智能体交互可视化 - 快速启动脚本 (PowerShell 版本)
    Agent Interaction Visualization - Quick Start Script (PowerShell)

.DESCRIPTION
    提供菜单选择来启动生产调度演示和可视化服务的组合

.AUTHOR
    Shannon 多智能体系统团队
#>

# 设置颜色输出
$colors = @{
    Success = 'Green'
    Warning = 'Yellow'
    Error   = 'Red'
    Info    = 'Cyan'
    Header  = 'Magenta'
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = 'White'
    )
    Write-Host $Message -ForegroundColor $Color
}

function Show-Header {
    Write-Host ""
    Write-ColorOutput "╔══════════════════════════════════════════════════════════════╗" $colors.Header
    Write-ColorOutput "║  多智能体交互可视化 快速启动脚本                                ║" $colors.Header
    Write-ColorOutput "║  Agent Interaction Visualization - Quick Start                ║" $colors.Header
    Write-ColorOutput "╚══════════════════════════════════════════════════════════════╝" $colors.Header
    Write-Host ""
}

function Show-Menu {
    Write-Host ""
    Write-Host "请选择启动方式 (Choose startup method):" -ForegroundColor White
    Write-Host ""
    Write-Host "  1 - 🌐 启动网页版可视化 (Web Visualization)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2 - 🖥️  启动完整演示 (Full Demo)" -ForegroundColor Cyan
    Write-Host "       • 运行生产调度演示" -ForegroundColor Gray
    Write-Host "       • 自动启动可视化服务" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3 - 📊 仅启动可视化服务 (Visualization Server Only)" -ForegroundColor Cyan
    Write-Host "  4 - 🚀 运行演示脚本 (Run Demo Script Only)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  5 - ❌ 退出 (Exit)" -ForegroundColor Cyan
    Write-Host ""
}

function Start-WebVisualization {
    Write-Host ""
    Write-ColorOutput "🌐 启动网页版可视化..." $colors.Info
    Write-Host ""
    Write-Host "📂 当前目录: $(Get-Location)"
    Write-Host ""
    
    & python visualization-server.py localhost 8888
}

function Start-FullDemo {
    Write-Host ""
    Write-ColorOutput "🚀 启动完整演示流程..." $colors.Info
    Write-Host ""
    
    Write-ColorOutput "第一步：运行生产调度演示..." $colors.Warning
    Write-Host ""
    
    & python production_scheduler_demo.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "❌ 演示脚本执行失败" $colors.Error
        Read-Host "按 Enter 继续..."
        return
    }
    
    Write-Host ""
    Write-ColorOutput "✓ 演示完成！现在启动可视化服务..." $colors.Success
    Write-Host ""
    Write-Host "⏳ 3 秒后启动可视化服务..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    Write-Host ""
    
    & python visualization-server.py localhost 8888
}

function Start-VizOnly {
    Write-Host ""
    Write-ColorOutput "📊 启动可视化服务..." $colors.Info
    Write-Host ""
    Write-Host "💡 提示：确保已运行过 production_scheduler_demo.py" -ForegroundColor Yellow
    Write-Host "💡 Tip: Make sure you have run production_scheduler_demo.py" -ForegroundColor Yellow
    Write-Host ""
    
    & python visualization-server.py localhost 8888
}

function Start-DemoOnly {
    Write-Host ""
    Write-ColorOutput "🚀 运行生产调度演示..." $colors.Info
    Write-Host ""
    
    & python production_scheduler_demo.py
    
    Write-Host ""
    Write-ColorOutput "✓ 演示完成！" $colors.Success
    Write-Host ""
    Write-Host "💡 如要查看交互流程可视化，请运行：" -ForegroundColor Yellow
    Write-Host "   .\start-visualization.ps1" -ForegroundColor Cyan
    Write-Host "   或选择菜单选项 3" -ForegroundColor Cyan
    Write-Host ""
}

function Check-Prerequisites {
    Write-Host ""
    Write-ColorOutput "检查前置条件..." $colors.Info
    Write-Host ""
    
    # 检查 Python
    try {
        $pythonVersion = python --version 2>&1
        Write-ColorOutput "✓ Python: $pythonVersion" $colors.Success
    } catch {
        Write-ColorOutput "❌ Python 未安装或未在 PATH 中" $colors.Error
        return $false
    }
    
    # 检查必要的文件
    $files = @(
        'production_scheduler_demo.py',
        'visualization-server.py',
        'agent-interaction-visualization.html'
    )
    
    foreach ($file in $files) {
        if (Test-Path $file) {
            Write-ColorOutput "✓ 文件: $file" $colors.Success
        } else {
            Write-ColorOutput "❌ 文件缺失: $file" $colors.Error
            return $false
        }
    }
    
    Write-Host ""
    return $true
}

# 主程序
function Main {
    Show-Header
    
    # 检查前置条件
    if (-not (Check-Prerequisites)) {
        Write-ColorOutput "❌ 前置条件检查失败" $colors.Error
        Read-Host "按 Enter 退出..."
        exit 1
    }
    
    Write-Host ""
    Write-ColorOutput "✓ 前置条件检查通过！" $colors.Success
    Write-Host ""
    
    # 显示菜单循环
    do {
        Show-Menu
        
        $choice = Read-Host "请输入选择 (Enter your choice) [1-5]"
        
        switch ($choice) {
            "1" {
                Start-WebVisualization
                break
            }
            "2" {
                Start-FullDemo
                break
            }
            "3" {
                Start-VizOnly
                break
            }
            "4" {
                Start-DemoOnly
                break
            }
            "5" {
                Write-Host ""
                Write-ColorOutput "👋 再见！(Goodbye!)" $colors.Info
                Write-Host ""
                exit 0
            }
            default {
                Write-ColorOutput "❌ 无效的选择 (Invalid choice)" $colors.Error
            }
        }
    } while ($true)
}

# 运行主程序
Main
