@echo off
chcp 65001 >nul
rem SA3 리터치 서버 단독 실행 — 환경 탐색은 run.bat 과 동일 (레포 옆/안/사용자 홈)
cd /d "%~dp0"
set "SA3DIR="
if exist "..\stable-audio-3\.venv\Scripts\activate.bat" set "SA3DIR=%~dp0..\stable-audio-3"
if exist "stable-audio-3\.venv\Scripts\activate.bat" set "SA3DIR=%~dp0stable-audio-3"
if exist "%USERPROFILE%\stable-audio-3\.venv\Scripts\activate.bat" set "SA3DIR=%USERPROFILE%\stable-audio-3"
if not defined SA3DIR (
  echo X stable-audio-3 환경을 찾지 못했습니다 - README "Stable Audio 3 리터치 서버" 절 참고
  echo   탐색한 경로: ..\stable-audio-3 / .\stable-audio-3 / %USERPROFILE%\stable-audio-3
  pause
  exit /b 1
)
call "%SA3DIR%\.venv\Scripts\activate.bat"
python sa3_server.py
pause
