@echo off
chcp 65001 >nul
rem SAO-Instruct 편집 서버(8700) 시동 - run.bat [4/4] 이 호출
rem 유형(3) 악기/음색 교체 전용 - demucs 스템 격리 + 자연어 지시 편집.
rem 환경 확인 -> 클론 -> venv -> 의존성 -> CUDA 복구 -> 모델 다운로드 -> 기동.

where uv >nul 2>nul
if errorlevel 1 (
  echo      X uv 가 없습니다 - winget install astral-sh.uv 후 다시 실행
  exit /b 1
)

set "SAODIR="
if exist "%~dp0..\sao-instruct\model\requirements.txt" set "SAODIR=%~dp0..\sao-instruct"
if exist "%USERPROFILE%\sao-instruct\model\requirements.txt" set "SAODIR=%USERPROFILE%\sao-instruct"
if not defined SAODIR (
  echo      sao-instruct 클론 중 - %USERPROFILE%\sao-instruct
  git clone https://github.com/ETH-DISCO/sao-instruct.git "%USERPROFILE%\sao-instruct"
  if errorlevel 1 exit /b 1
  set "SAODIR=%USERPROFILE%\sao-instruct"
)

pushd "%SAODIR%"
if not exist ".venv\Scripts\python.exe" (
  echo      가상환경 생성 - uv venv --python 3.10
  uv venv --python 3.10
  if errorlevel 1 ( popd & exit /b 1 )
)
echo      의존성 설치/확인 - SAO-Instruct + 서버 (최초 수 분)
uv pip install -r model\requirements.txt
if errorlevel 1 ( popd & exit /b 1 )
uv pip install .\model\stable-audio-tools
if errorlevel 1 ( popd & exit /b 1 )
uv pip install fastapi uvicorn soundfile demucs requests pydantic
if errorlevel 1 ( popd & exit /b 1 )
rem CUDA torch 확인 - 의존성 설치가 CPU torch 로 되돌렸으면 복구
rem (주의: 이 venv 에서 uv sync 금지 - CUDA torch 가 다시 밀려난다)
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo      CPU torch 감지 - CUDA 재설치
  uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 ( popd & exit /b 1 )
)
rem 모델 웨이트 선다운로드(HF 캐시) - 실패해도 서버는 뜬다(첫 /load 때 재시도)
echo      모델 웨이트 확인/다운로드 - disco-eth/sao-instruct
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('disco-eth/sao-instruct')"
if errorlevel 1 echo      ! 선다운로드 실패 - 첫 리터치 때 자동 재시도 ^(네트워크 확인^)
popd

set "SAO_DIR=%SAODIR%"
echo      서버 시작 - 새 창 "sao-server 8700" (모델은 리터치 시작 때 VRAM 로드)
start "sao-server 8700" cmd /k "%~dp0_sao_run.bat"
exit /b 0
