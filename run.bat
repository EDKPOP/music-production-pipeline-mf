@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python mf_server.py
pause
