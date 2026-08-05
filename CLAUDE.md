# CLAUDE.md — songcamp-mf (Music Flamingo 추론 서버)

이 레포에서 작업하는 Claude를 위한 기준 문서. 본체 레포는
github.com/EDKPOP/music-production-pipeline — **본체의 `docs/HANDOFF.md`와
`songcamp/critic.py`(클라이언트)를 먼저 읽어야 이 서버의 계약이 보인다.**

## 역할

자율 송캠프의 GPU 추론 서버 2종을 담는 레포 (윈도우 11 + NVIDIA GPU 24GB):

1. **`mf_server.py` (포트 8400)** — 게이트② A&R 청취 심사. Music Flamingo 8B,
   맥(본체)이 보낸 오디오의 평문 심사평/구조 분석 반환.
   버전 표식: `/health` → `"version": "v7-arbiter"`.
2. **`sa3_server.py` (포트 8500)** — 트랙 작업실 구간 리터치. 곡 전체 수신 →
   demucs 보컬/반주 분리(또는 클라이언트가 보낸 보컬 스템 재사용 — 체이닝 열화
   차단) → 반주만 Stable Audio 3 인페인팅(구간 ± 문맥 창, 마스크 구간만 재생성)
   → 마스크 밖 원본 스플라이스 + 보컬 재합성(보컬 없는 구간은 스템 미합성 게이트).
   잡 기반(POST /edit → GET /jobs/{id} 폴링 → /jobs/{id}/result).
   버전 표식: `"sa3-v5-audedit"`. 본체 클라이언트는 `songcamp/postprod/retouch.py`.
   설치는 README "Stable Audio 3 리터치 서버" 절 (stable-audio-3 uv 환경 공유).
   `POST /audedit`(AudEdit 속도장 차분 편집 — 실측 보류)과 **`POST /update`
   (git pull 자가 재기동 — `_sa3_run.bat` 루프 전제)** 포함.
3. **`ace_bridge.py` (포트 8600)** — ACE-Step 1.5 통역 브리지
   (`ace-v4-stemroute`): 풀믹스 repaint + **드럼 스템 경로**(`stem_repaint`
   — 유일하게 귀 합격한 드럼 리듬·필인 경로) + upstream 스폰/종료로 GPU 교대.
4. **`sao_server.py` (포트 8700)** — SAO-Instruct 편집 서버. **실측 불합격
   판정(2026-08-05)으로 기본 미사용** — 본체 `retouch.sao_enable=1` 때만
   라우팅됨. `/update` 자가 배포 있음. 판정 근거는 본체
   `docs/HANDOFF.md` §0 (재도전 전에 반드시 읽을 것).

### GPU 중재 (24GB를 MF 16GB + SA3 6.5GB가 나눠 쓰는 방법)

- `run.bat` 은 **두 서버 프로세스를 함께** 띄운다. VRAM에 어느 '모델'이
  올라갈지는 **맥의 중재자**(`songcamp/gpu.py`)가 조율한다:
  야간 심사/inbox 처리 전 → SA3 `POST /unload` 후 MF `POST /load`(예열),
  작업실 리터치 시작 → MF `POST /unload` (SA3는 잡에서 자동 로드).
- 양쪽 모두 **작업 진행 중이면 /unload 를 409(busy)로 거절**한다 — 진행 중
  잡을 깨뜨리지 않는 것이 계약. 언로드된 서버에 요청이 오면 자동 재로드되므로
  중재 실패는 성능 문제일 뿐 정합성 문제가 아니다.
- SA3는 기본 상주(연속 리터치 가속), `SA3_UNLOAD_EACH=1` 이면 잡마다 언로드.
- OOM 시 잡/응답 error="cuda_oom" (본체가 이 문자열로 안내 분기 — 의미 변경 금지).

## 절대 계약 (본체 critic.py가 의존 — 임의 변경 금지)

1. **평문 원칙**: MF에게 JSON을 강요하지 않는다. `mode:"text"`는 모델 출력
   전문을 `{"text": ...}`로 **무절단** 반환. 구조화·번역은 맥 쪽 책임.
   (과거 JSON 강제 시절 스키마 불일치가 끝없이 났던 것이 분리 이유)
2. **요청 형식**: POST `/` `{mode, prompt, audio_b64[, gen]}` — 오디오는
   16kHz 모노 wav의 base64. 텍스트 1 : 오디오 1 제약(프로세서 한계) —
   여러 발췌는 맥이 무음 간격으로 이어붙여 보낸다.
3. **OOM 신호**: GPU 메모리 부족 시 캐시를 비우고 **HTTP 507
   `{"error":"cuda_oom"}`** 반환. 맥의 강등 사다리(전곡 300s→180s→발췌)가
   이 신호로 동작하므로 상태 코드·의미를 바꾸면 안 된다.
4. **길이 가드**: `MF_MAX_AUDIO_S`(기본 420s) 초과 오디오는 RIFF 청크 워커로
   앞부분만 절단(헤더 보정). 본체는 기본 300s 캡으로 보내므로 이건 자기방어선.
5. `/health` 필드(`version`, `cuda`, `enforcer`, `max_audio_s`)는 본체
   설정 화면·헬스체크가 읽는다 — 제거 금지, 추가는 자유.

## 구동 방식 핵심 (버그 재발 방지 — 각각 실측으로 확정된 것)

- **SDPA 어텐션**으로 로드(`attn_implementation="sdpa"`, 미지원 폴백).
  eager는 전곡 오디오에서 어텐션 메모리가 시퀀스 제곱으로 폭발
  (24GB에서 16GB+ 단일 할당 실측)해 OOM의 주범이었다.
- **autocast 필수**: 모델에 bf16 모듈(conv)과 fp32 고정 모듈(layer_norm)이
  섞여 있어 입력을 한쪽에 맞추면 반대쪽이 죽는다. 입력 dtype은 float64→
  float32 강등만 하고 나머지는 autocast에 맡긴다.
- **greedy 금지**: do_sample=False는 이 모델에서 템플릿 복사·무한 반복에
  빠진다(실측). 약한 샘플링(temp 0.4, top_p 0.9, repetition_penalty 1.15)이
  정답이고, 결정성은 맥 쪽 3회 채점 중앙값이 확보한다.
- 요청마다 종료 시 `torch.cuda.empty_cache()` — 단편화 누적 방지.
- 모델 로딩은 기동 시 1회(약 16GB 다운로드는 최초 1회). `run.bat` 실행.

## 운영

- 실행: PowerShell에서 `run.bat` (venv 활성화 + 서버 기동 포함).
- 코드 갱신: `git pull` 후 **서버 재시작 필수** (상주 프로세스).
- 정상 확인: 브라우저에서 `http://localhost:8400/health` — `version`이
  최신 표식인지, `cuda: true`인지.
- 본체 쪽 MF 주소는 songcamp 웹 UI의 시스템 화면에서 설정한다.

## 작업 규율 (본체와 동일)

- 검증 없이 커밋 없다. 이 레포는 GPU 없는 환경에서 전체 실행이 불가하므로
  최소한 구문 검사(ast.parse)와, 순수 함수(_clamp_wav 등)는 단위 검증을
  하고 커밋한다. GPU 필요 변경은 사용자에게 실기 확인 항목을 명시해 전달.
- 조용한 실패 금지: 생성 원문 미리보기, OOM, 절단 등 모든 특이 상황은
  콘솔 로그로 드러낸다 (사용자가 창을 보고 있는 유일한 관측 수단).
- 커밋 메시지는 한국어로 상세하게 (원인→수정→검증).
