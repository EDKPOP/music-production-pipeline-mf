@echo off
chcp 65001 >nul
rem 자율 송캠프 GPU 서버 일괄 기동 — MF(8400) + SA3 리터치(8500)
rem 두 서버 프로세스는 각자의 창에서 상주하고, VRAM의 '모델'은 맥의 GPU
rem 중재자가 /load /unload 로 교대시킨다 (야간 심사=MF, 작업실 리터치=SA3).
cd /d "%~dp0"

echo [1/2] MF 심사 서버 시작 - 새 창 "mf-server 8400"
start "mf-server 8400" cmd /k "call .venv\Scripts\activate.bat && python mf_server.py"

rem SA3 환경 탐색 — 레포 옆 / 레포 안 / 사용자 홈 순서
set "SA3DIR="
if exist "..\stable-audio-3\.venv\Scripts\activate.bat" set "SA3DIR=%~dp0..\stable-audio-3"
if exist "stable-audio-3\.venv\Scripts\activate.bat" set "SA3DIR=%~dp0stable-audio-3"
if exist "%USERPROFILE%\stable-audio-3\.venv\Scripts\activate.bat" set "SA3DIR=%USERPROFILE%\stable-audio-3"

if defined SA3DIR (
  echo [2/2] SA3 리터치 서버 시작 - 새 창 "sa3-server 8500"  ^(환경: %SA3DIR%\.venv^)
  start "sa3-server 8500" cmd /k call "%SA3DIR%\.venv\Scripts\activate.bat" ^&^& python sa3_server.py
) else (
  echo.
  echo [2/2] X SA3 리터치 서버^(8500^)는 켜지 못했습니다 - stable-audio-3 미설치.
  echo      최초 1회 설치가 필요합니다. README.md 의 "Stable Audio 3 리터치 서버"
  echo      절을 따라 stable-audio-3 를 클론하고 uv sync + requirements-sa3.txt 를
  echo      설치하세요. 설치가 끝나면 run.bat 재실행으로 두 서버가 함께 켜집니다.
  echo      탐색한 경로: ..\stable-audio-3 / .\stable-audio-3 / %USERPROFILE%\stable-audio-3
)
echo.
echo 상태 확인:  http://localhost:8400/health    http://localhost:8500/health
echo 이 창은 닫아도 됩니다 - 서버는 각자의 창에서 돕니다.
pause
