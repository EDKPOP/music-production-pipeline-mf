@echo off
chcp 65001 >nul
rem ACE-Step 1.5 공통 시동 로직 - run.bat / run_ace.bat 이 call 한다 (pause 없음)
rem 환경 확인 → (없으면) 클론 → uv sync → CUDA 확인 → 서버 2창 기동

where uv >nul 2>nul
if errorlevel 1 (
  echo X uv 가 없습니다 - winget install astral-sh.uv 후 다시 실행하세요
  exit /b 1
)

set "ACEDIR="
if exist "..\ACE-Step-1.5\pyproject.toml" set "ACEDIR=%~dp0..\ACE-Step-1.5"
if exist "%USERPROFILE%\ACE-Step-1.5\pyproject.toml" set "ACEDIR=%USERPROFILE%\ACE-Step-1.5"
if not defined ACEDIR (
  echo   ACE-Step 1.5 저장소 클론 - %USERPROFILE%\ACE-Step-1.5
  git clone https://github.com/ace-step/ACE-Step-1.5.git "%USERPROFILE%\ACE-Step-1.5"
  if errorlevel 1 exit /b 1
  set "ACEDIR=%USERPROFILE%\ACE-Step-1.5"
)
echo   ACE 폴더: %ACEDIR%

echo   의존성 동기화 uv sync - 최초엔 수 분…
pushd "%ACEDIR%"
uv sync
if errorlevel 1 ( popd & exit /b 1 )
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo   CPU torch 감지 - CUDA 빌드로 재설치 - 수 GB…
  uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 ( popd & exit /b 1 )
)
popd

rem 자식 창은 ACEDIR 환경변수를 상속한다 - 인자 전달 없음
rem (start /D "%~dp0" 는 끝 백슬래시가 따옴표를 이스케이프해 구문 오류를 냈다)
echo   ACE 공식 서버 시작 - 새 창 "ace-api 8001" - 최초엔 모델 자동 다운로드 수 GB
start "ace-api 8001" cmd /k "%~dp0_ace_api.bat"
echo   송캠프 브리지 시작 - 새 창 "ace-bridge 8600"
start "ace-bridge 8600" cmd /k "%~dp0_ace_bridge.bat"
exit /b 0
