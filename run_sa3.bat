@echo off
rem SA3 리터치 서버 — stable-audio-3 의 uv 가상환경으로 실행
rem 최초 1회 설치는 README의 "Stable Audio 3 리터치 서버" 절 참고
cd /d %~dp0
call ..\stable-audio-3\.venv\Scripts\activate.bat
python sa3_server.py
pause
