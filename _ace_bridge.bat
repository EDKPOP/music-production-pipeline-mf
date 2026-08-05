@echo off
chcp 65001 >nul
rem 송캠프 ACE 브리지 래퍼 - 크래시/업데이트(/update) 후 자동 재기동 루프
cd /d "%~dp0"
set PYTHONUTF8=1
call .venv\Scripts\activate.bat
:loop
echo [ace-bridge] python ace_bridge.py 시작 - 업스트림 http://127.0.0.1:8001
python ace_bridge.py
echo.
echo [ace-bridge] 종료 코드 %errorlevel% - 5초 후 재시작 (창을 닫으면 완전 종료)
timeout /t 5 /nobreak >nul
goto loop
