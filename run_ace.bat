@echo off
chcp 65001 >nul
rem ACE-Step 1.5 단독 원클릭 - 로직은 _ace_start.bat (run.bat 에도 통합됨)
cd /d "%~dp0"
call "%~dp0_ace_start.bat"
if errorlevel 1 echo X ACE 시동 실패 - 위 오류를 확인하세요
echo.
echo 확인:  curl http://localhost:8600/health
pause
