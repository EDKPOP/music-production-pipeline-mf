@echo off
chcp 65001 >nul
rem ACE 공식 API 서버 래퍼 — 크래시해도 창이 닫히지 않고 오류가 남는다
rem 사용: _ace_api.bat <ACE-Step-1.5 경로>
cd /d "%~1"
echo [ace-api] 작업 폴더: %CD%
echo [ace-api] uv run acestep-api 시작… (최초엔 모델 다운로드 수 GB)
uv run acestep-api
echo.
echo ══════════════════════════════════════════════════════
echo [ace-api] 서버 프로세스가 종료되었습니다 (정상 종료가 아니라면
echo           위쪽의 마지막 오류 줄을 맥 쪽에 그대로 전달해 주세요)
echo ══════════════════════════════════════════════════════
pause
