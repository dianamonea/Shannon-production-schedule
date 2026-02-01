@echo off
chcp 65001 >nul
echo ================================================================================
echo 🎯 Shannon 生产调度系统 - 快速启动脚本
echo ================================================================================
echo.

set SESSION_ID=production-scheduler-%RANDOM%%RANDOM%

echo 📌 新建 Session ID: %SESSION_ID%
echo.
echo ✨ 正在启动生产调度系统...
echo 💡 同时可以通过以下方式查看实时进度：
echo.
echo    1. Temporal UI: http://localhost:8088
echo    2. Shannon Desktop: 搜索 Session ID
echo    3. 终端实时监控（见下方）
echo.
echo ================================================================================
echo.

:: 设置UTF-8编码
set PYTHONIOENCODING=utf-8

:: 在后台启动监控
start /B python visualize_progress.py %SESSION_ID%

:: 运行主程序
python main.py

pause
