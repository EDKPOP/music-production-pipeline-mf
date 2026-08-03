@echo off
chcp 65001 >nul
rem ACE-Step 1.5 시동 (run.bat [3/3] 또는 run_ace.bat 이 호출)
rem v2: 브리지가 upstream(acestep-api)을 스스로 스폰/종료 - MF 와 GPU 교대.
rem 창은 브리지 하나만 뜬다. 모델: 3090 티어 = acestep-v15-xl-base.

where uv >nul 2>nul
if errorlevel 1 (
  echo      X uv 가 없습니다 - winget install astral-sh.uv 후 다시 실행
  exit /b 1
)

set "ACEDIR="
if exist "%~dp0..\ACE-Step-1.5\pyproject.toml" set "ACEDIR=%~dp0..\ACE-Step-1.5"
if exist "%USERPROFILE%\ACE-Step-1.5\pyproject.toml" set "ACEDIR=%USERPROFILE%\ACE-Step-1.5"
if not defined ACEDIR (
  echo      ACE-Step 1.5 클론 중 - %USERPROFILE%\ACE-Step-1.5
  git clone https://github.com/ace-step/ACE-Step-1.5.git "%USERPROFILE%\ACE-Step-1.5"
  if errorlevel 1 exit /b 1
  set "ACEDIR=%USERPROFILE%\ACE-Step-1.5"
)

echo      의존성 동기화 - uv sync
pushd "%ACEDIR%"
uv sync
if errorlevel 1 ( popd & exit /b 1 )
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo      CPU torch 감지 - CUDA 재설치
  uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 ( popd & exit /b 1 )
)
popd

rem 브리지의 자동 보컬 분리용 demucs - MF venv 에 1회 설치
"%~dp0.venv\Scripts\python.exe" -c "import demucs" >nul 2>nul
if errorlevel 1 (
  echo      demucs 설치 - 브리지 자동 보컬 분리용
  "%~dp0.venv\Scripts\python.exe" -m pip install demucs
)

set "ACE_DIR=%ACEDIR%"
set "ACE_MODEL=acestep-v15-xl-base"
echo      브리지 시작 - 새 창 "ace-bridge 8600" (upstream 은 리터치 시작 때 자동 기동)
start "ace-bridge 8600" cmd /k "%~dp0_ace_bridge.bat"
exit /b 0
