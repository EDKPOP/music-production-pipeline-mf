@echo off
chcp 65001 >nul
rem 송캠프 SAO-Instruct 서버 래퍼 - 크래시/업데이트 후 자동 재기동 루프
cd /d "%~dp0"
set PYTHONUTF8=1
if not defined SAO_DIR set "SAO_DIR=%USERPROFILE%\sao-instruct"
:loop
echo [sao-server] python sao_server.py 시작 - 레포 %SAO_DIR%
"%SAO_DIR%\.venv\Scripts\python.exe" sao_server.py
echo.
echo [sao-server] 종료 코드 %errorlevel% - 5초 후 재시작 (창을 닫으면 완전 종료)
timeout /t 5 /nobreak >nul
goto loop
