@echo off
rem 자율 송캠프 GPU 서버 일괄 기동 — MF(8400) + SA3 리터치(8500)
rem 두 서버는 항상 함께 떠 있고, 어떤 '모델'이 VRAM을 쓸지는 맥의 GPU
rem 중재자가 /load /unload 로 조율한다 (야간 심사=MF, 작업실 리터치=SA3).
cd /d %~dp0

start "mf-server 8400" cmd /k "call .venv\Scripts\activate.bat && python mf_server.py"

if exist ..\stable-audio-3\.venv\Scripts\activate.bat (
  start "sa3-server 8500" cmd /k "call ..\stable-audio-3\.venv\Scripts\activate.bat && python sa3_server.py"
) else (
  echo [안내] SA3 리터치 서버 미설치 — README의 "Stable Audio 3 리터치 서버" 절 참고
  echo        (설치 후 run.bat 재실행 또는 run_sa3.bat 단독 실행)
)
