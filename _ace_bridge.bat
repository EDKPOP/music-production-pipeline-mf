@echo off
chcp 65001 >nul
rem 송캠프 ACE 브리지 래퍼 - 크래시해도 창이 닫히지 않는다
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo [ace-bridge] python ace_bridge.py 시작 - 업스트림 http://127.0.0.1:8001
python ace_bridge.py
echo.
echo [ace-bridge] 브리지 종료 - 종료 코드 %errorlevel% - 위 오류를 확인하세요
pause
