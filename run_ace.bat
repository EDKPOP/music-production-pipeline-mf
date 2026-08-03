@echo off
chcp 65001 >nul
rem ═══════════════════════════════════════════════════════════════
rem ACE-Step 1.5 원클릭 — 환경 확인 → 설치 → 모델(자동 다운로드) → 기동
rem   창 2개가 뜹니다: "ace-api 8001"(공식 서버) + "ace-bridge 8600"(송캠프 통역)
rem   맥 쪽 설정: 리터치 엔진 ace, 주소 http://윈도우IP:8600
rem ═══════════════════════════════════════════════════════════════
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo X uv 가 없습니다 - 먼저 설치하세요:  winget install astral-sh.uv
  pause
  exit /b 1
)
if not exist ".venv\Scripts\activate.bat" (
  echo X 이 레포의 .venv 가 없습니다 - README 의 MF 서버 설치 절차를 먼저 진행하세요
  pause
  exit /b 1
)

set "ACEDIR="
if exist "..\ACE-Step-1.5\pyproject.toml" set "ACEDIR=%~dp0..\ACE-Step-1.5"
if exist "%USERPROFILE%\ACE-Step-1.5\pyproject.toml" set "ACEDIR=%USERPROFILE%\ACE-Step-1.5"
if not defined ACEDIR (
  echo [1/4] ACE-Step 1.5 저장소 클론 - %USERPROFILE%\ACE-Step-1.5
  git clone https://github.com/ace-step/ACE-Step-1.5.git "%USERPROFILE%\ACE-Step-1.5"
  if errorlevel 1 ( pause & exit /b 1 )
  set "ACEDIR=%USERPROFILE%\ACE-Step-1.5"
) else (
  echo [1/4] ACE-Step 1.5 발견: %ACEDIR%
)

echo [2/4] 의존성 동기화 (uv sync - 최초엔 수 분)…
pushd "%ACEDIR%"
uv sync
if errorlevel 1 ( popd & pause & exit /b 1 )
rem CUDA torch 확인 - 윈도우 PyPI torch 는 CPU 빌드 함정 (MF README 실사고)
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo   CPU torch 감지 - CUDA 빌드로 재설치합니다 (수 GB, 몇 분)…
  uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 ( popd & pause & exit /b 1 )
)
popd

echo [3/4] ACE 공식 API 서버 시작 - 새 창 "ace-api 8001" (최초 실행 시 모델 자동 다운로드 - 수 GB)
rem 래퍼 bat 사용 - 서버가 죽어도 창이 닫히지 않고 오류가 남는다
start "ace-api 8001" /D "%~dp0" _ace_api.bat "%ACEDIR%"

echo [4/4] 송캠프 브리지 시작 - 새 창 "ace-bridge 8600"
start "ace-bridge 8600" /D "%~dp0" _ace_bridge.bat

echo.
echo 확인:  curl http://localhost:8600/health   → upstream 연결까지 최초 수 분
echo 방화벽 (관리자 PowerShell, 최초 1회):
echo   New-NetFirewallRule -DisplayName "songcamp-ace 8600" -Direction Inbound -Protocol TCP -LocalPort 8600 -Action Allow -Profile Private,Domain
echo 이 창은 닫아도 됩니다 - 서버는 각자의 창에서 돕니다.
pause
