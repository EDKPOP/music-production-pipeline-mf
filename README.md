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

## 라이선스

코드는 MIT. **Music Flamingo 모델은 NVIDIA OneWay Noncommercial 라이선스**
(비상업 연구 용도 전용)입니다 — 이 프로젝트는 비상업 개인 프로듀싱 용도로만 쓰세요.
