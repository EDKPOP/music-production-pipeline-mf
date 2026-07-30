# 🔥 songcamp-mf — Music Flamingo 추론 서버 (Windows 11 + NVIDIA GPU)

[자율 송캠프](https://github.com/EDKPOP/music-production-pipeline)의 **게이트②
A&R 심사(Music Flamingo 8B)** 를 NVIDIA GPU가 달린 윈도우 PC에서 대신 돌려주는
서버입니다. 맥(본체)이 곡을 HTTP로 보내면, 이 서버가 GPU로 채점해 돌려줍니다.

> 왜 분리? 맥의 GPU(MPS)는 현재 PyTorch 버그로 이 모델을 돌릴 수 없고
> CPU는 곡당 십수 분이 걸립니다. NVIDIA GPU면 곡당 수 초~수십 초입니다.

## 요구 사항

- Windows 11 + **NVIDIA 그래픽카드 (VRAM 12GB 이상 권장, 16GB이면 여유)**
- 최신 NVIDIA 드라이버 ([nvidia.com/drivers](https://www.nvidia.com/drivers))
- Python 3.11 ([python.org](https://python.org) — 설치 시 "Add python.exe to PATH" 체크!)
- 디스크 여유 ~20GB (모델 16GB + 라이브러리)
- 맥과 같은 공유기(같은 네트워크)에 연결

## 설치 (PowerShell — 최초 1회, 20~40분)

`Win + X` → "터미널" 을 열고, 아래 상자를 순서대로 붙여넣으세요.

```powershell
# 1) 코드 받기
cd $HOME
git clone https://github.com/EDKPOP/music-production-pipeline-mf.git songcamp-mf
cd songcamp-mf

# 2) 실행 환경
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# 3) CUDA용 PyTorch (제일 중요 — 이걸 빼먹으면 CPU로 돌아 매우 느립니다)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4) 나머지 의존성
pip install -r requirements.txt

# 5) GPU 인식 확인 — True 가 나와야 합니다
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> `git` 이 없다고 나오면: `winget install Git.Git` 후 터미널을 껐다 다시 여세요.

## 실행

```powershell
cd $HOME\songcamp-mf
.\.venv\Scripts\Activate.ps1
python mf_server.py
```

첫 실행은 모델(~16GB)을 자동으로 내려받아 오래 걸립니다. 이후엔 1분 내 기동.
`mf-server 대기 중 — http://0.0.0.0:8400` 이 보이면 준비 완료입니다.
(또는 `run.bat` 더블클릭)

**방화벽 창이 뜨면 "액세스 허용"** 을 누르세요 (사설 네트워크만 체크해도 됩니다).

## 맥(본체)과 연결

1. 윈도우 PC의 IP 확인 (PowerShell): `ipconfig` → "IPv4 주소" (예: `192.168.0.23`)
2. 맥의 `~/music-production-pipeline/config.yaml` 에서:

```yaml
gate2:
  backend: http
  http_url: "http://192.168.0.23:8400"   # ← 윈도우 PC의 IP로
```

3. 맥에서 연결 확인:

```bash
curl http://192.168.0.23:8400/health
# {"status":"ok","model_loaded":true,"cuda":true,"device":"cuda:0"} 이면 성공
```

이후 맥에서 밤 파이프라인/inbox 처리를 돌리면 게이트② 채점이 자동으로
이 서버를 경유합니다. 서버가 꺼져 있으면 곡은 탈락되지 않고 "심사 대기"로
보류됐다가, 서버를 켠 뒤 다시 처리하면 이어집니다.

## 자주 묻는 것

**Q. `CUDA: False` 가 나온다** → 3번(CUDA용 PyTorch)을 건너뛰었거나 드라이버가
낡은 것. `pip uninstall torch` 후 3번을 다시 실행하고, 드라이버를 업데이트하세요.

**Q. VRAM이 부족하다(Out of memory)** → 다른 GPU 사용 앱(게임·브라우저 하드웨어
가속)을 끄세요. 8GB VRAM이면 `mf_server.py`의 `torch_dtype`은 이미 bf16(≈16GB→
GPU가 절반을 시스템 램에 오프로딩)이라 느려질 수 있습니다 — 12GB+ 권장.

**Q. 맥에서 /health 가 안 열린다** → 순서대로 (관리자 PowerShell):

```powershell
# 1) 서버 자체 확인 (윈도우에서 — JSON 나오면 서버는 정상, 방화벽 문제)
curl.exe http://localhost:8400/health

# 2) 네트워크를 '개인(Private)'으로 — 공용(Public)이면 인바운드 전체 차단됨
Set-NetConnectionProfile -NetworkCategory Private

# 3) 방화벽 인바운드 허용 규칙
New-NetFirewallRule -DisplayName "songcamp-mf 8400" -Direction Inbound -Protocol TCP -LocalPort 8400 -Action Allow -Profile Private,Domain

# 4) IP 확인 후 맥에서 curl http://IP:8400/health
ipconfig
```

그래도 안 되면: 맥과 같은 공유기인지(게스트 Wi-Fi는 기기 간 통신 차단 — AP 격리),
공유기의 "AP 격리" 옵션 여부 확인. IP 고정(DHCP 예약)을 걸어두면 재부팅 후에도
주소가 안 바뀝니다.

## 프로토콜 (참고)

- `GET /health` → 상태 확인
- `POST /` body: `{"mode":"rubric"|"compare","prompt":"...","audio_b64":"..."[,"audio_b64_b":"..."]}`
  → 채점 JSON (rubric: hook/production/structure/vocal 점수+근거+heard)

오디오는 본체가 훅 근처 20초 발췌를 base64로 보냅니다. 서버는 파일을 임시
저장 후 즉시 삭제하며, 아무것도 디스크에 남기지 않습니다.

## Stable Audio 3 리터치 서버 (선택 — 트랙 작업실 구간 리터치)

트랙 작업실의 "이 구간만 프롬프트대로 고치기"를 담당하는 두 번째 서버입니다
(포트 8500). 곡 전체를 받아 demucs로 보컬/반주를 분리하고, 반주만 Stable
Audio 3 인페인팅으로 선택 구간을 재생성한 뒤, 마스크 밖 원본과 보컬을 그대로
합쳐 돌려줍니다.

### 설치 (PowerShell — 최초 1회)

```powershell
# 1) uv (파이썬 패키지 매니저) — 이미 있으면 생략
winget install astral-sh.uv

# 2) Stable Audio 3 코드 + 환경 (모델은 첫 실행 때 자동 다운로드)
cd $HOME
git clone https://github.com/Stability-AI/stable-audio-3.git
cd stable-audio-3
uv sync

# 3) 서버 추가 의존성 (같은 환경에 얹기)
#    ⚠ uv가 만든 .venv에는 pip이 없습니다 — 반드시 uv pip에 대상 파이썬을
#      명시하세요 (activate+pip은 다른 환경의 pip을 잡아 조용히 어긋납니다)
uv pip install --python .venv\Scripts\python.exe -r $HOME\songcamp-mf\requirements-sa3.txt

# 4) CUDA용 PyTorch 강제 (윈도우 함정 — MF 설치 3번과 같은 이유)
#    ⚠ 윈도우에서 PyPI 기본 torch는 CPU 빌드입니다. uv sync/의존성 설치가
#      CPU 빌드를 깔거나 '교체'할 수 있으므로 마지막에 CUDA 빌드로 덮습니다.
uv pip install --python .venv\Scripts\python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# 5) 확인 — 두 줄 다 통과해야 합니다 (CUDA: True / deps OK)
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.__version__)"
.venv\Scripts\python.exe -c "import uvicorn, fastapi, soundfile, demucs; print('deps OK')"
```

> `CUDA: False` 가 나오면: torch 버전 끝에 `+cpu` 가 붙어 있는지 확인하고
> 4번을 다시 실행하세요. 그래도 False면 `cu126` 대신 `cu124` 로 시도
> (드라이버가 오래된 경우). requirements-sa3 를 재설치한 뒤에는 항상
> 4번을 마지막에 한 번 더 — 의존성 해석이 torch를 CPU 빌드로 되돌릴 수 있습니다.

### 모델 접근 승인 (최초 1회 — 게이트 모델)

`stable-audio-3-medium` 은 Hugging Face **게이트 모델**입니다 — 라이선스에
동의하고 계정 인증을 해야 다운로드됩니다 (안 하면 첫 리터치가 401
GatedRepoError 로 실패):

1. 브라우저에서 https://huggingface.co/stabilityai/stable-audio-3-medium 접속
   → HF 계정 로그인 → 라이선스 동의(Agree and access) — 보통 즉시 승인
2. 토큰 발급: https://huggingface.co/settings/tokens → "Read" 권한 토큰 생성
3. PowerShell에서 로그인 (토큰은 로컬에 저장 — 서버 재시작 불필요):

```powershell
cd $HOME\stable-audio-3
# 숨김 입력 프롬프트는 윈도우 콘솔에서 붙여넣기가 새는 경우가 있어(빈 값
# 전송 → 400 Bad Request), 토큰을 명령에 직접 넣는 방식이 확실합니다.
# hf_로 시작하는 본인 토큰으로 바꿔 실행:
.venv\Scripts\python.exe -c "from huggingface_hub import login; login(token='hf_여기에_토큰')"

# 확인 — 본인 HF 계정 이름이 출력되면 성공:
.venv\Scripts\python.exe -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

> 토큰은 로컬 캐시에 저장되므로 위 명령은 한 번만 실행하면 됩니다.
> 명령 기록에 토큰이 남는 게 싫다면 실행 후 `Clear-History` 를 한 번.

### 실행

**`run.bat` 더블클릭 한 번이면 MF(8400)와 SA3(8500)가 함께 켜집니다**
(SA3 미설치면 안내만 뜨고 MF만 켜짐). 단독 실행:

```powershell
cd $HOME\songcamp-mf
..\stable-audio-3\.venv\Scripts\Activate.ps1
python sa3_server.py          # 기본 0.0.0.0:8500  (또는 run_sa3.bat)
```

방화벽 허용 (관리자 PowerShell, 최초 1회):

```powershell
New-NetFirewallRule -DisplayName "songcamp-sa3 8500" -Direction Inbound -Protocol TCP -LocalPort 8500 -Action Allow -Profile Private,Domain
```

맥 쪽 설정은 songcamp 웹 UI **⚙️ 설정/상태 탭의 SA3 주소** 칸에
`http://윈도우IP:8500` 을 넣으면 됩니다. 확인: `curl http://윈도우IP:8500/health`

> **VRAM/GPU 중재**: MF(≈16GB)와 SA3(≈6.5GB)는 같은 GPU를 나눠 씁니다.
> 두 서버 프로세스는 항상 함께 떠 있고, **어느 모델이 VRAM에 올라갈지는
> 맥이 자동 조율**합니다 — 야간 심사·inbox 처리 전엔 SA3 모델을 내리고 MF를
> 예열하며, 작업실에서 리터치를 시작하면 MF를 내리고 SA3로 전환합니다.
> 진행 중인 작업이 있으면 언로드를 거절(409)하므로 잡이 깨질 일은 없습니다.

### ⚠ 출력이 '지지직'/초저음질 해시로 들릴 때 (medium 전용 — flash-attn)

공식 README Troubleshooting: **"Output audio is a static glitch sound" =
flash-attention 설치 문제**입니다 (medium 의 SAME-L 오토인코더가 flash-attn 필수).
실측 증상: 생성 구간만 8kHz 이상이 백색잡음처럼 채워짐 (16kbps mp3 같은 인상).

**자동 복구가 기본입니다** — `run.bat`(또는 `run_sa3.bat`)로 서버가 뜰 때
flash-attn 이 깨져 있거나 **torch 가 CPU 빌드로 바뀌어 있으면**(`uv sync`
사고의 전형 — 윈도우 PyPI torch 는 CPU 전용이라 cuda=False 가 된다)
`ensure_flash_attn.py` 가 알아서 ⓪ nvidia-smi 로 드라이버 CUDA 를 읽어
PyTorch 공식 CUDA 인덱스에서 같은 버전 torch/torchaudio 재설치 후,
① 환경 태그(파이썬 cpXY · torch x.y · CUDA cuXYZ) 감지 →
② 휠 저장소들([mjun0812](https://github.com/mjun0812/flash-attention-prebuild-wheels)
→ [kingbri1](https://github.com/kingbri1/flash-attention) 등)에서 정확히 맞는
win_amd64 휠 선택 → ③ `pip --no-deps` 설치 → ④ 임포트 재검증까지 합니다.
결과는 서버 시작 로그와 `/health` 의 `flash_attn` 필드에 보입니다.
단독 실행: `python ensure_flash_attn.py` (SA3 venv 활성 상태에서).

자동 복구가 "맞는 휠 없음"으로 실패할 때만 수동으로: 위 저장소에서 로그가
알려준 태그(예: `cp310`·`torch2.7`·`cu128`·`win_amd64`)와 맞는 휠을 받아
`uv pip install <whl>`. flash-attn 은 pyproject 에 없으므로 이후 의존성
갱신은 `uv sync --inexact` (그냥 sync 하면 도로 지워짐).

### 프로토콜 (본체 retouch 클라이언트·GPU 중재자와 계약)

- `GET /health` → `{"status","version":"sa3-v4-stems","cuda","model_loaded","busy","max_audio_s","queue"}`
- `POST /edit` `{"audio_b64": 44.1kHz wav, "edits":[{start_s,end_s,prompt,…}]
  [,"keep_vocals","vocals_b64"(이전 리터치의 보컬 스템 — 있으면 demucs 재분리 생략)]}` → `{"job_id"}`
- `GET /jobs/{id}` → `{"status","phase","progress","elapsed_s"[,"error"]}` (OOM 시 error="cuda_oom")
- `GET /jobs/{id}/result` → `{"audio_b64","sr"[,"vocals_b64" — 클라이언트가 사이드카 저장, 체이닝 재전송용]}`
- `POST /unload` → 모델 언로드 (잡 진행/대기 중이면 409 busy) · MF 쪽도 동일하게
  `POST /load`·`POST /unload` 지원

## 라이선스

코드는 MIT. **Music Flamingo 모델은 NVIDIA OneWay Noncommercial 라이선스**
(비상업 연구 용도 전용), **Stable Audio 3 는 Stability AI Community License**
(비상업 OK)입니다 — 이 프로젝트는 비상업 개인 프로듀싱 용도로만 쓰세요.
