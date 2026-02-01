@echo off
REM 多智能体交互可视化 - 一键启动脚本
REM Agent Interaction Visualization - Quick Start Script

setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║  多智能体交互可视化 快速启动脚本                                ║
echo ║  Agent Interaction Visualization - Quick Start                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM 检查是否在 Shannon 目录中
if not exist "production_scheduler_demo.py" (
    echo ❌ 错误：请在 Shannon 根目录中运行此脚本
    echo ❌ Error: Please run this script in Shannon root directory
    pause
    exit /b 1
)

echo ✓ 检测到 Shannon 项目目录
echo.

REM 设置菜单
:menu
echo 请选择启动方式 (Choose startup method):
echo.
echo   1 - 🌐 启动网页版可视化 (Web Visualization)
echo   2 - 🖥️  启动完整演示 (Full Demo)
echo.        - 运行生产调度演示
echo.        - 自动启动可视化服务
echo.
echo   3 - 📊 仅启动可视化服务 (Visualization Server Only)
echo   4 - 🚀 运行演示脚本 (Run Demo Script Only)
echo.
echo   5 - ❌ 退出 (Exit)
echo.

set /p choice="请输入选择 (Enter your choice) [1-5]: "

if "%choice%"=="1" goto web_only
if "%choice%"=="2" goto full_demo
if "%choice%"=="3" goto viz_only
if "%choice%"=="4" goto demo_only
if "%choice%"=="5" goto exit_script
echo ❌ 无效的选择 (Invalid choice)
goto menu

REM ========== 1. 网页版可视化 ==========
:web_only
echo.
echo 🌐 启动网页版可视化...
echo.
call python visualization-server.py localhost 8888
pause
goto end

REM ========== 2. 完整演示 ==========
:full_demo
echo.
echo 🚀 启动完整演示流程...
echo.
echo 第一步：运行生产调度演示...
echo.
python production_scheduler_demo.py
if errorlevel 1 (
    echo ❌ 演示脚本执行失败
    pause
    goto end
)
echo.
echo ✓ 演示完成！现在启动可视化服务...
echo.
timeout /t 3 /nobreak
echo.
call python visualization-server.py localhost 8888
pause
goto end

REM ========== 3. 仅启动可视化服务 ==========
:viz_only
echo.
echo 📊 启动可视化服务...
echo.
echo 💡 提示：确保已运行过 production_scheduler_demo.py
echo 💡 Tip: Make sure you have run production_scheduler_demo.py
echo.
call python visualization-server.py localhost 8888
pause
goto end

REM ========== 4. 运行演示脚本 ==========
:demo_only
echo.
echo 🚀 运行生产调度演示...
echo.
python production_scheduler_demo.py
echo.
echo ✓ 演示完成！
echo.
echo 💡 如要查看交互流程可视化，请运行：
echo 💡 python visualization-server.py localhost 8888
echo.
pause
goto end

REM ========== 退出 ==========
:exit_script
echo.
echo 👋 再见！(Goodbye!)
echo.
goto end

REM ========== 结束 ==========
:end
endlocal
