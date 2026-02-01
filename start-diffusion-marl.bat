@echo off
REM 扩散式 MARL 集成 - 快速启动脚本
REM Quick Start Script for Diffusion MARL Integration

setlocal enabledelayedexpansion

REM 获取项目目录
set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%

REM 检查虚拟环境
if not exist ".venv" (
    echo ❌ 虚拟环境不存在
    echo 请先运行: python -m venv .venv
    pause
    exit /b 1
)

REM 激活虚拟环境
call .venv\Scripts\activate.bat

echo.
echo ============================================================
echo    扩散式多智能体强化学习集成 - 快速启动
echo    Diffusion MARL Integration - Quick Start
echo ============================================================
echo.

:menu
echo 请选择要运行的程序:
echo.
echo 1. 运行扩散模型示例 (diffusion_marl.py)
echo    - 演示基本的扩散模型和多智能体协调
echo    - 运行时间: ~30 秒
echo.
echo 2. 运行混合调度器演示 (hybrid_diffusion_scheduler.py) [推荐]
echo    - 展示推荐的集成方案
echo    - 包括传统方法对比
echo    - 运行时间: ~30 秒
echo.
echo 3. 显示快速参考卡片
echo    - 常见问题、参数调优、故障排除
echo    - 运行时间: ~5 秒
echo.
echo 4. 打开文档目录
echo    - 用资源管理器打开项目目录
echo.
echo 5. 查看扩散模型源代码
echo    - 在默认编辑器中打开 diffusion_marl.py
echo.
echo 6. 查看混合调度器源代码
echo    - 在默认编辑器中打开 hybrid_diffusion_scheduler.py
echo.
echo 7. 查看完整指南
echo    - 在默认编辑器中打开 DIFFUSION_MARL_GUIDE.md
echo.
echo 8. 退出
echo.

set /p choice="请输入选项 (1-8): "

if "%choice%"=="1" (
    echo.
    echo 运行扩散模型示例...
    echo.
    python diffusion_marl.py
    pause
    goto menu
) else if "%choice%"=="2" (
    echo.
    echo 运行混合调度器演示...
    echo.
    python hybrid_diffusion_scheduler.py
    pause
    goto menu
) else if "%choice%"=="3" (
    echo.
    echo 显示快速参考卡片...
    echo.
    python DIFFUSION_MARL_QUICK_REFERENCE.py
    pause
    goto menu
) else if "%choice%"=="4" (
    echo.
    echo 打开文档目录...
    start explorer .
    goto menu
) else if "%choice%"=="5" (
    echo.
    echo 打开源代码...
    start notepad diffusion_marl.py
    goto menu
) else if "%choice%"=="6" (
    echo.
    echo 打开源代码...
    start notepad hybrid_diffusion_scheduler.py
    goto menu
) else if "%choice%"=="7" (
    echo.
    echo 打开详细指南...
    start notepad DIFFUSION_MARL_GUIDE.md
    goto menu
) else if "%choice%"=="8" (
    echo.
    echo 再见!
    exit /b 0
) else (
    echo.
    echo ❌ 无效的选项，请重试
    echo.
    goto menu
)
