@echo off
chcp 65001 >nul
rem 송캠프 SAO-Instruct 서버 래퍼 - 크래시해도 창이 닫히지 않는다
cd /d "%~dp0"
if not defined SAO_DIR set "SAO_DIR=%USERPROFILE%\sao-instruct"
echo [sao-server] python sao_server.py 시작 - 레포 %SAO_DIR%
"%SAO_DIR%\.venv\Scripts\python.exe" sao_server.py
echo.
echo [sao-server] 서버 종료 - 종료 코드 %errorlevel% - 위 오류를 확인하세요
pause
