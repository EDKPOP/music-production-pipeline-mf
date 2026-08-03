@echo off
chcp 65001 >nul
rem ACE 공식 API 서버 래퍼 - ACEDIR 는 부모(_ace_start.bat)의 환경변수를 상속
if not defined ACEDIR (
  echo X ACEDIR 미설정 - run.bat 또는 run_ace.bat 으로 실행하세요
  pause
  exit /b 1
)
cd /d "%ACEDIR%"
echo [ace-api] 작업 폴더: %CD%
echo [ace-api] uv run acestep-api 시작 - 최초엔 모델 다운로드 수 GB
uv run acestep-api
echo.
echo [ace-api] 서버 종료 - 종료 코드 %errorlevel%
echo [ace-api] 정상 종료가 아니라면 위 마지막 오류 줄을 맥 쪽에 전달해 주세요
pause
