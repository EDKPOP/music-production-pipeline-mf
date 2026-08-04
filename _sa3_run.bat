@echo off
chcp 65001 >nul
rem 송캠프 SA3 서버 래퍼 - 크래시/업데이트 후 자동 재기동 루프
cd /d "%~dp0"
set PYTHONUTF8=1
set "SA3DIR="
if exist "..\stable-audio-3\.venv\Scripts\python.exe" set "SA3DIR=%~dp0..\stable-audio-3"
if exist "stable-audio-3\.venv\Scripts\python.exe" set "SA3DIR=%~dp0stable-audio-3"
if exist "%USERPROFILE%\stable-audio-3\.venv\Scripts\python.exe" set "SA3DIR=%USERPROFILE%\stable-audio-3"
if not defined SA3DIR (
  echo [sa3-server] X stable-audio-3 venv 를 찾지 못했습니다 - README 참조
  pause
  exit /b 1
)
:loop
echo [sa3-server] python sa3_server.py 시작 - 환경 %SA3DIR%\.venv
"%SA3DIR%\.venv\Scripts\python.exe" sa3_server.py
echo.
echo [sa3-server] 종료 코드 %errorlevel% - 5초 후 재시작 (창을 닫으면 완전 종료)
timeout /t 5 /nobreak >nul
goto loop
