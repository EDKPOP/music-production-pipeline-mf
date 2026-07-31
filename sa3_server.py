"""Stable Audio 3 구간 리터치 서버 — Windows 11 + NVIDIA GPU 전용.

자율 송캠프 본체(맥)의 트랙 작업실에서 "이 구간만 프롬프트대로 고쳐줘"를
HTTP로 위임받아 처리한다. 파이프라인:

  전체 곡(wav) 수신 → demucs로 보컬/반주 분리 → 반주만 Stable Audio 3
  인페인팅(전체 반주를 컨텍스트로, 마스크 구간만 재생성) → 마스크 밖은
  원본 그대로 스플라이스(크로스페이드) → 보컬 재합성 → 반환

왜 이 구조인가 — SA3는 보컬 품질 평가가 낮아 반주만 맡기고, 보컬은 원본을
그대로 얹는다. 마스크 밖 구간은 모델 출력조차 쓰지 않고 원본 믹스를 유지해
"전체 톤이 틀어지지 않는" 것을 구조적으로 보장한다.

실행:  run_sa3.bat   (stable-audio-3 의 uv 가상환경에서 python sa3_server.py)
       python sa3_server.py --port 8500

프로토콜 (본체 songcamp/postprod/retouch.py 와 계약 — 임의 변경 금지):
  GET  /health          → {"status","version","cuda","model_loaded","max_audio_s"}
  POST /edit            → {"audio_b64": 44.1kHz 스테레오 wav의 b64,
                           "edits": [{"start_s","end_s","prompt"
                                      [,"mode","strength","cfg_scale","fill"]}]
                           [, "keep_vocals"=true, "seed","steps",
                            "vocals_b64": 이전 리터치의 보컬 스템(v4 — 있으면
                            demucs 재분리 생략, 체이닝 열화 차단)]} → {"job_id"}
                          (구형 단건 start_s/end_s/prompt 도 계속 허용)
  GET  /jobs/{id}       → {"status":"queued|running|done|failed","phase",
                           "progress":0~1,"elapsed_s"[,"error"]}
  GET  /jobs/{id}/result→ {"audio_b64": 결과 전체 곡 wav b64, "sr":44100
                           [,"vocals_b64": 보컬 스템 — 클라이언트가 사이드카로
                            저장해 다음 체이닝에 재전송]}
  OOM 시 잡 error = "cuda_oom" (본체가 이 문자열로 안내 분기)

배치가 기본: 예약 여러 건을 한 잡으로 받아 디코드·보컬 분리를 1회만 하고
반주 위에 구간별 생성(inpaint=전곡 컨텍스트 재생성 / a2a=구간±문맥 창을
원본 초기값으로 변형)을 누적한 뒤, 마지막에 원본과 1회 합성한다.
생성 구간은 원본 구간과 RMS 매칭(음량 꺼짐 방지). cfg 는 기본 2.0·상한
4.5 — cfg 7 + negative 조합이 마스크 구간을 백색잡음으로 만든 실사고의
재발 방지 (negative 는 명시 요청 시에만 전달).

GPU 메모리: 이 PC는 MF 8B(~16GB)가 상주하므로 SA3(~6.5GB)+demucs는
기본적으로 잡이 끝나면 언로드한다 (SA3_KEEP_LOADED=1 로 상주 전환).

라이선스: Stable Audio 3 는 Stability AI Community License — 비상업 용도 OK.
"""
import argparse
import base64
import io
import os
import sys
import threading
import time
import traceback
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VERSION = "sa3-v4.6-eqmatch"
SR = 44100
A2A_CTX_S = 10.0         # a2a(변형) 창: 구간 ± 문맥 초 — 짧을수록 충실도·속도↑
# inpaint(재생성)도 전곡이 아니라 구간 ± 문맥 창만 모델에 보낸다 — SA3 논문의
# 인페인팅 평가는 클립의 2~20%만 마스크했고, 장시간 duration 조건은 품질을
# 떨어뜨린다. 0 이면 과거(v3)처럼 전곡 컨텍스트.
INPAINT_CTX_S = float(os.environ.get("SA3_INPAINT_CTX_S", 60))
# 보컬 게이트: 수정 구간의 보컬 에너지가 믹스 대비 이 비율 미만이면(간주 등)
# 분리 스템(악기 블리드 포함)을 얹지 않는다 — 반주 구간 음질 열화 방지
VOCAL_MIN_RATIO = 0.05
NEG_PROMPT = ("low quality, muffled, lo-fi, noisy, distorted, degraded audio, "
              "artifacts, harsh, tinny, dark, dull")
MAX_AUDIO_S = float(os.environ.get("SA3_MAX_AUDIO_S", 370))  # 모델 상한 380s 아래 안전선
# 기본은 '상주' — 리터치를 연달아 할 때 매번 로드하지 않는다. GPU가 MF에
# 필요해지면 맥의 중재자가 POST /unload 로 내린다 (SA3_UNLOAD_EACH=1 이면
# 과거처럼 잡마다 언로드).
UNLOAD_EACH = os.environ.get("SA3_UNLOAD_EACH", "") == "1"
XFADE_S = 0.10           # 마스크 경계 크로스페이드
RESULT_KEEP = 5          # 메모리에 보관할 완료 잡 수 (결과 wav가 수십 MB)

app = FastAPI(title="sa3-server", docs_url=None)
_sa3 = None
_demucs = None
_jobs = {}               # id → dict(status, phase, progress, ...)
_queue = []
_lock = threading.Lock()


def _log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_FA = None


def _flash_attn_status() -> dict:
    """flash-attn 동작 여부 — medium 모델은 SAME-L 오토인코더가 flash-attn 을
    요구하며, 깨져 있으면 출력 전체가 '지지직' 글리치가 된다 (공식 README
    Troubleshooting: 'static glitch sound = flash-attention 설치 문제').
    실측 증상: 생성 구간 8kHz+ 스펙트럼 평탄도 0.003→0.2 (백색잡음성 해시)."""
    global _FA
    if _FA is None:
        try:
            import flash_attn
            from flash_attn import flash_attn_func  # noqa: F401 — 실기능 임포트 검증
            _FA = {"ok": True, "version": getattr(flash_attn, "__version__", "?")}
        except Exception as e:
            _FA = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return _FA


def _load_demucs():
    global _demucs
    if _demucs is None:
        import torch
        from demucs.pretrained import get_model
        _log("demucs(htdemucs) 로딩…")
        _demucs = get_model("htdemucs")
        _demucs.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return _demucs


def _load_sa3():
    global _sa3
    if _sa3 is None:
        from stable_audio_3 import StableAudioModel
        _log("Stable Audio 3 medium 로딩… (최초 실행 시 모델 다운로드)")
        _sa3 = StableAudioModel.from_pretrained("medium")
    return _sa3


def _drop_models():
    global _sa3, _demucs
    _sa3 = _demucs = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _unload():
    """잡 종료 후 처리 — 기본은 상주(빠른 연속 리터치), SA3_UNLOAD_EACH=1 이면
    과거처럼 매 잡 언로드. GPU 양보는 맥 중재자의 POST /unload 가 담당한다."""
    if not UNLOAD_EACH:
        return
    _drop_models()
    _log("모델 언로드 (SA3_UNLOAD_EACH=1)")


def _decode_wav(b64: str):
    """b64 wav → float32 (2, T) @44.1k. 모노는 스테레오 복제, 타 SR은 리샘플."""
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32",
                       always_2d=True)          # (T, C)
    wav = data.T                                 # (C, T)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != SR:
        import torch
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, SR).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def _encode_wav(wav: np.ndarray) -> str:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav.T, -1.0, 1.0), SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _separate(wav: np.ndarray, phase):
    """demucs 2-스템: (vocals, inst). inst = 원본 − vocals (스템 합 보존)."""
    import torch
    from demucs.apply import apply_model
    model = _load_demucs()
    phase("보컬/반주 분리 (demucs)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(wav)[None]              # (1, C, T)
    ref = t.mean(0)
    t = (t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(model, t.to(dev), device=dev,
                              split=True, overlap=0.25, progress=False)[0]
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    vocals = sources[model.sources.index("vocals")].cpu().numpy()
    inst = wav - vocals
    return vocals.astype(np.float32), inst.astype(np.float32)


def _inpaint(inst: np.ndarray, start_s: float, end_s: float, prompt: str,
             extra: dict, phase):
    """반주 전체를 컨텍스트로 SA3 인페인팅 → 재생성된 전체 반주 (2, T)."""
    import torch
    model = _load_sa3()
    dur = inst.shape[1] / SR
    mode = (extra.get("mode") or "inpaint").lower()
    common = dict(
        prompt=prompt,
        duration=dur,
        # 기본 sample_size 는 ~120s 상한 — 컨텍스트가 조용히 잘리지 않게 명시.
        # 실제 길이와 정확히 일치시킨다 — 여유 패딩(+8s)은 duration 조건과
        # 어긋나 후반부가 무너지는 원인이 된다 (ComfyUI #14825 동계열)
        sample_size=int(min(dur, 380.0) * SR),
        # ⚠ 실측 확정(2026-07-30, /diag): 이 post-trained 체크포인트(8스텝
        # 핑퐁, 기본 cfg 1.0)에 cfg 7 + negative 를 얹으면 마스크 구간이
        # 백색잡음이 된다 (>10kHz 비중 79%, '16kbps 저음질' 증상의 진범).
        # cfg 1~4 는 실측 정상(0.5~3%) — 기본 2.0, 상한 4.5 로 강제.
        cfg_scale=min(max(float(extra.get("cfg_scale") or 2.0), 1.0), 4.5),
    )
    # negative 는 cfg>1 과 결합하면 백색잡음 붕괴(실사고), 반대로 APG 체제
    # (cfg=1)에서는 inpaint 의 '먹먹함'을 걷어내는 실측 효과가 있다
    # (센트로이드 777→1202Hz) — cfg가 1일 때만 기본 적용한다.
    neg = str(extra.get("negative") or "").strip()
    if not neg and common["cfg_scale"] <= 1.01:
        neg = NEG_PROMPT
    if neg:
        common["negative_prompt"] = neg
    if extra.get("apg_scale") is not None:   # APG — 노이즈 없는 준수 강화 (실측)
        common["apg_scale"] = min(max(float(extra["apg_scale"]), 1.0), 10.0)
    if mode == "t2a":
        # 순수 생성 — overlay 모드용: 마스크 길이만큼 '요청한 요소만' 만든다
        # (init/inpaint 조건 없음 — 프롬프트가 곧 결과)
        phase(f"요소 생성 (t2a, {dur:.1f}s)")
        kwargs = dict(**common)
    elif mode == "a2a":
        # audio-to-audio: 원본 반주를 초기값으로 깔고 노이즈를 부분만 섞어
        # 뼈대(멜로디·리듬)를 유지한 채 변형. 창 전체가 다시 그려지지만
        # 마스크 밖은 이후 _splice 가 진짜 원본으로 되돌린다.
        # (문서: 0.1=밀접한 변형, 0.5=중간 혼합, 1.0=원본 무시)
        phase(f"구간 변형 (SA3 a2a, {start_s:.1f}~{end_s:.1f}s / {dur:.0f}s, "
              f"노이즈 {float(extra.get('strength') or 0.35):.2f})")
        kwargs = dict(
            init_audio=(SR, torch.from_numpy(inst)),
            init_noise_level=min(max(float(extra.get("strength") or 0.35), 0.1), 1.0),
            **common,
        )
    else:
        phase(f"구간 재생성 (Stable Audio 3, {start_s:.1f}~{end_s:.1f}s / {dur:.0f}s)")
        kwargs = dict(
            # 공식 시그니처는 (sample_rate, audio) 순서 — README 예제(torchaudio.load
            # 반환 순서)와 반대다. 뒤집으면 'int has no attribute to' 로 즉사 (실기 재현)
            inpaint_audio=(SR, torch.from_numpy(inst)),
            inpaint_mask_start_seconds=float(start_s),
            inpaint_mask_end_seconds=float(end_s),
            **common,
        )
    for k in ("seed", "steps"):                  # 선택 파라미터 — API가 모르면 제거
        if extra.get(k) is not None:
            kwargs[k] = extra[k]
    try:
        out = model.generate(**kwargs)
    except TypeError as e:
        _log(f"⚠ 선택 파라미터 미지원({e}) — 기본 파라미터로 재시도")
        for k in ("seed", "steps", "cfg_scale", "negative_prompt"):
            kwargs.pop(k, None)
        out = model.generate(**kwargs)
    if isinstance(out, tuple):                   # (tensor, sr) 형태 방어
        out, out_sr = out
    else:
        out_sr = SR
    if hasattr(out, "cpu"):
        out = out.float().cpu().numpy()
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:                            # (B, C, T) 방어
        out = out[0]
    if out.ndim == 1:
        out = np.stack([out, out])
    if out_sr != SR:
        import torchaudio
        out = torchaudio.functional.resample(
            torch.from_numpy(out), out_sr, SR).numpy()
    return out


def _align_len(gen: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """생성 길이를 기준 오디오에 정렬 — 초과는 자르고 부족분은 기준으로 채움."""
    d = gen.shape[1] - ref.shape[1]
    if d == 0:
        return gen
    _log(f"  생성 길이 보정: {d:+d} 샘플")
    if d > 0:
        return gen[:, :ref.shape[1]]
    return np.concatenate([gen, ref[:, gen.shape[1]:]], axis=1)


PHASE_MAX_LAG_S = 0.045   # 위상 정렬 탐색 폭 — 지각적 페이징은 수십 ms 대


def _phase_align(gen: np.ndarray, ref: np.ndarray, a: int, b: int):
    """접합 경계의 위상 정렬 — 페이징(빗질 소리) 근본 대책.

    생성물이 원본과 수십 ms 어긋난 채 크로스페이드되면 두 파형이 간섭해
    페이징이 난다. 마스크 양 경계 주변 파형의 상호상관으로 지연을 실측해
    생성물 전체를 그만큼 밀어 원본 그리드에 맞춘다 (최대 ±45ms).
    반환 (정렬된 gen, 적용 지연 샘플)."""
    n = gen.shape[1]
    max_lag = int(PHASE_MAX_LAG_S * SR)
    win = int(0.25 * SR)
    corr = None
    for edge in (a, b):
        lo = max(edge - win, 0)
        hi = min(edge + win, n, ref.shape[1])
        if hi - lo < max_lag * 3:
            continue
        r = ref[:, lo:hi].mean(0)
        g = gen[:, lo:hi].mean(0)
        if float(np.sqrt((r ** 2).mean())) < 1e-3:
            continue                     # 원본이 사실상 무음 — 정렬 무의미
        L = hi - lo
        c = np.correlate(r, g, "full")[L - 1 - max_lag: L + max_lag]
        corr = c if corr is None else corr + c
    if corr is None or not len(corr):
        return gen, 0
    lag = int(np.argmax(corr)) - max_lag   # out[j] = gen[j - lag] 이 최적 정합
    if lag == 0:
        return gen, 0
    out = np.empty_like(gen)
    if lag > 0:
        out[:, lag:] = gen[:, :n - lag]
        out[:, :lag] = gen[:, :1]
    else:
        out[:, :n + lag] = gen[:, -lag:]
        out[:, n + lag:] = gen[:, -1:]
    return out, lag


def _spectral_new(gen: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """생성물에서 '원본에 없던 성분'만 추출 — overlay 의 핵심.

    |G|−|O| 를 정류(음수=0)한 크기에 생성물의 위상을 입혀 복원한다.
    원본이 이미 갖고 있던 에너지 대역은 차감되어 사라지고, 모델이 새로
    추가한 소리(필인·레이어)만 남는다 — 문맥 조건(inpaint) 생성이라
    곡의 팔레트·그루브를 물려받은 상태의 '추가분'이다."""
    import torch
    n_fft, hop = 2048, 512
    win = torch.hann_window(n_fft)
    out = []
    for ch in range(gen.shape[0]):
        G = torch.stft(torch.from_numpy(np.ascontiguousarray(gen[ch])),
                       n_fft, hop, window=win, return_complex=True)
        O = torch.stft(torch.from_numpy(np.ascontiguousarray(orig[ch])),
                       n_fft, hop, window=win, return_complex=True)
        m = min(G.shape[1], O.shape[1])
        G, O = G[:, :m], O[:, :m]
        mag = (G.abs() - O.abs()).clamp(min=0.0)
        y = torch.istft(mag * torch.exp(1j * G.angle()), n_fft, hop,
                        window=win, length=gen.shape[1])
        out.append(y.numpy())
    return np.stack(out).astype(np.float32)


def _match_spectrum(gen: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """생성 창의 장기 스펙트럼 기울기를 원본 창에 매칭 — 강한 cfg 유도의
    '어두워짐' 비용을 사후 EQ 로 상쇄한다 (실측: 평탄도 0.120→0.019,
    센트로이드 690→1316Hz). 강한 스무딩 + 클램프(-6dB~+12dB)로 미세 구조와
    프롬프트가 의도한 추가 성분은 보존한다. SA3_EQ_MATCH=0 으로 끌 수 있다."""
    if os.environ.get("SA3_EQ_MATCH", "1") == "0":
        return gen
    n = min(gen.shape[1], ref.shape[1])
    if n < SR // 2:
        return gen
    if float(np.sqrt(np.mean(ref[:, :n] ** 2))) < 1e-3:
        return gen                                  # 원본이 무음(빈 블록) — 기준 없음
    G = np.abs(np.fft.rfft(gen[:, :n].mean(0)))
    R = np.abs(np.fft.rfft(ref[:, :n].mean(0)))
    # 비율을 직접 스무딩하면 G≈0 지점의 폭주 비율이 평균을 지배한다 —
    # 분자/분모를 각각 스무딩한 뒤 나눠야 안정적인 EQ 커브가 나온다
    w = max(64, len(G) // 40)
    Gs = np.convolve(G, np.ones(w) / w, mode="same")
    Rs = np.convolve(R, np.ones(w) / w, mode="same")
    fr = np.arange(len(G), dtype=np.float32)
    c_g = float((Gs ** 2 * fr).sum() / max((Gs ** 2).sum(), 1e-9))
    c_r = float((Rs ** 2 * fr).sum() / max((Rs ** 2).sum(), 1e-9))
    if c_g >= 0.8 * c_r:
        return gen        # 이미 원본급 밝기 — EQ 는 어두워진 결과 전용 (실측)
    _log(f"  EQ 매칭: 센트로이드 {c_g / c_r * 100:.0f}% → 원본 기울기로 보정")
    k = np.clip(Rs / np.maximum(Gs, 1e-9), 0.5, 4.0)
    spec = np.fft.rfft(gen, axis=1)
    k2 = np.interp(np.linspace(0, 1, spec.shape[1]), np.linspace(0, 1, len(k)), k)
    out = np.fft.irfft(spec * k2, n=gen.shape[1], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _match_rms(gen: np.ndarray, ref: np.ndarray, a: int, b: int) -> np.ndarray:
    """생성 구간의 음량을 원본 구간에 맞춘다 — 구간만 조용해지는
    '음질 다운' 인상 방지. 원본이 무음(빈 블록)이면 건드리지 않는다."""
    ref_rms = float(np.sqrt(np.mean(ref[:, a:b] ** 2)))
    gen_rms = float(np.sqrt(np.mean(gen[:, a:b] ** 2)))
    if ref_rms < 1e-4 or gen_rms < 1e-4:
        return gen
    g = min(max(ref_rms / gen_rms, 0.5), 2.0)
    if abs(g - 1.0) > 0.05:
        _log(f"  RMS 매칭: ×{g:.2f}")
    return gen * g


def _splice(original: np.ndarray, generated: np.ndarray,
            start_s: float, end_s: float) -> np.ndarray:
    """마스크 구간만 generated, 밖은 original — 경계 크로스페이드.

    모델이 마스크 밖도 통째로 다시 그려 주지만 그 부분은 신뢰하지 않는다.
    원본 유지가 '톤이 틀어지지 않는다'의 구조적 보증이다."""
    n = min(original.shape[1], generated.shape[1])
    out = original[:, :n].copy()
    gen = generated[:, :n]
    a, b = int(start_s * SR), min(int(end_s * SR), n)
    if b <= a:
        return out
    out[:, a:b] = gen[:, a:b]
    xf = int(XFADE_S * SR)
    # 등전력(equal-power) 크로스페이드 — 두 소스의 위상이 어긋난 구간에서
    # 선형 페이드는 중앙이 움푹 꺼지며 '밀리는' 인상을 만든다 (sin/cos 게인)
    theta = np.linspace(0.0, np.pi / 2, xf, dtype=np.float32)
    g_in, g_out = np.sin(theta), np.cos(theta)
    lo = max(a - xf, 0)
    if a - lo > 0:                               # 들어가는 경계
        gi, go = g_in[-(a - lo):], g_out[-(a - lo):]
        out[:, lo:a] = original[:, lo:a] * go + gen[:, lo:a] * gi
    hi = min(b + xf, n)
    if hi - b > 0:                               # 나오는 경계
        gi, go = g_in[: hi - b], g_out[: hi - b]
        out[:, b:hi] = gen[:, b:hi] * go + original[:, b:hi] * gi
    return out


def _run_job(job: dict):
    def phase(p, prog=None):
        job["phase"] = p
        if prog is not None:
            job["progress"] = prog
        _log(f"잡 {job['id'][:8]}: {p}")

    t0 = time.time()
    try:
        phase("오디오 디코드", 0.05)
        wav = _decode_wav(job.pop("audio_b64"))
        dur = wav.shape[1] / SR
        edits = job["edits"]
        for e in edits:
            s, t = float(e["start_s"]), float(e["end_s"])
            if not (0 <= s < t <= dur + 0.5):
                raise ValueError(f"구간이 곡 길이를 벗어남: {s}~{t}s (곡 {dur:.1f}s)")
            e["start_s"], e["end_s"] = s, min(t, dur)
        lo = min(e["start_s"] for e in edits)
        hi = max(e["end_s"] for e in edits)

        # 모델 상한 초과 곡 → 수정 구간 묶음을 중심에 둔 윈도우만 모델에 보낸다
        off = 0.0
        full = wav
        if dur > MAX_AUDIO_S:
            if hi - lo > MAX_AUDIO_S:
                raise ValueError(f"수정 구간들이 한 번에 처리 가능한 폭"
                                 f"({MAX_AUDIO_S:.0f}s)을 넘습니다 — 예약을 나눠 실행하세요")
            off = max(0.0, min(lo - (MAX_AUDIO_S - (hi - lo)) / 2, dur - MAX_AUDIO_S))
            a = int(off * SR)
            wav = wav[:, a:a + int(MAX_AUDIO_S * SR)]
            _log(f"  곡 {dur:.0f}s > {MAX_AUDIO_S:.0f}s — 윈도우 {off:.1f}s~ 적용")

        # 배치의 핵심: 디코드·분리를 잡당 1회만 하고 구간 생성만 누적한다.
        # v4: 클라이언트가 이전 리터치의 보컬 스템(vocals_b64)을 보내면 demucs
        # 재분리를 생략한다 — 리터치 위에 리터치를 이어갈 때 분리 아티팩트가
        # 회차마다 누적되는 것(체이닝 열화)이 v3 최대 품질 문제였다.
        vb64 = job.pop("vocals_b64", None)
        if not job.get("keep_vocals", True):
            vocals, inst = np.zeros_like(wav), wav
        elif vb64:
            phase("보컬 스템 재사용 (이전 리터치의 분리 결과 — 재분리 생략)", 0.2)
            vocals = _decode_wav(vb64)
            if off:
                a0 = int(off * SR)
                vocals = vocals[:, a0:a0 + wav.shape[1]]
            if vocals.shape[1] < wav.shape[1]:
                vocals = np.concatenate(
                    [vocals, np.zeros((2, wav.shape[1] - vocals.shape[1]),
                                      np.float32)], axis=1)
            vocals = np.ascontiguousarray(vocals[:, :wav.shape[1]])
            inst = wav - vocals
        else:
            phase("보컬/반주 분리 (demucs) — 배치당 1회", 0.15)
            vocals, inst = _separate(wav, lambda p: phase(p, 0.22))

        inst_cur = inst.copy()
        total = len(edits)
        for i, e in enumerate(edits):
            f0 = 0.3 + 0.55 * i / total
            s_rel, t_rel = e["start_s"] - off, e["end_s"] - off
            opts = {k: e.get(k) for k in ("mode", "strength", "cfg_scale",
                                          "negative", "apg_scale")}
            for k in ("seed", "steps"):
                if job.get(k) is not None:
                    opts[k] = job[k]
            tagp = f"{i+1}/{total} {e.get('label') or ''}".strip()
            mode_e = (e.get("mode") or "inpaint").lower()
            if mode_e == "overlay":
                # overlay v2 — 문맥 조건 차분 추출: t2a 고립 생성은 곡의 음색·
                # 그루브를 몰라 '안 어울리는' 소리를 냈다(실사고). 대신
                # ① 구간±문맥을 inpaint (곡 팔레트를 물려받은 결과)
                # ② 원본에 없던 성분만 스펙트럼 차로 추출
                # ③ 원본 위에 얹기 — 어울림은 inpaint 가, 보존은 차분이 담당.
                a, b = int(s_rel * SR), min(int(t_rel * SR), inst_cur.shape[1])
                wa = max(0.0, s_rel - A2A_CTX_S)
                wb = min(inst_cur.shape[1] / SR, t_rel + A2A_CTX_S)
                ia, ib = int(wa * SR), int(wb * SR)
                seg = np.ascontiguousarray(inst_cur[:, ia:ib])
                ra, rb = int((s_rel - wa) * SR), int((t_rel - wa) * SR)
                opts2 = dict(opts)
                opts2["mode"] = "inpaint"
                gen = _inpaint(seg, s_rel - wa, t_rel - wa, e["prompt"], opts2,
                               lambda p: phase(f"{tagp} · {p}", f0))
                gen = _align_len(gen, seg)
                gen, lag = _phase_align(gen, seg, ra, rb)
                if lag:
                    _log(f"  위상 정렬: {lag * 1000.0 / SR:+.0f}ms")
                elem = _spectral_new(gen[:, ra:rb], seg[:, ra:rb])
                ref = inst_cur[:, a:b]
                ref_rms = float(np.sqrt(np.mean(ref ** 2))) or 1e-4
                e_rms = float(np.sqrt(np.mean(elem ** 2)))
                if e_rms < 0.02 * ref_rms:
                    _log("  overlay: 모델이 추가한 성분이 미미 — 이 연산은 건너뜀")
                    continue
                # strength = 얹는 크기: 0.25 은은 · 0.4 또렷 · 0.55 전면
                lvl = min(max(float(e.get("strength") or 0.4), 0.15), 0.9)
                elem *= (ref_rms / max(e_rms, 0.05 * ref_rms)) * lvl * 1.2
                n_el = min(elem.shape[1], b - a)
                xf0 = int(0.03 * SR)
                if n_el > 2 * xf0:              # 요소 양끝 페이드 (클릭 방지)
                    r0 = np.linspace(0.0, 1.0, xf0, dtype=np.float32)
                    elem[:, :xf0] *= r0
                    elem[:, n_el - xf0:n_el] *= r0[::-1]
                inst_cur[:, a:a + n_el] = inst_cur[:, a:a + n_el] + elem[:, :n_el]
                _log(f"  overlay: 문맥 추가분 RMS {e_rms:.4f} → ×{lvl * 1.2:.2f} 로 합성"
                     f" (원본 무변경)")
                continue
            # 생성은 항상 '구간 ± 문맥 창'만 — a2a 는 좁게(충실도), inpaint 는
            # 넓게(앞뒤 흐름). 창 밖은 어차피 손대지 않고, 전곡 생성은 느린 데다
            # 마스크 비율이 논문 평가 범위(2~20%)를 벗어나 품질이 떨어진다.
            a2a = mode_e == "a2a"
            ctx = A2A_CTX_S if a2a else INPAINT_CTX_S
            dur_cur = inst_cur.shape[1] / SR
            wa = max(0.0, s_rel - ctx) if ctx > 0 else 0.0
            wb = min(dur_cur, t_rel + ctx) if ctx > 0 else dur_cur
            ia, ib = int(wa * SR), int(wb * SR)
            seg = np.ascontiguousarray(inst_cur[:, ia:ib])
            ra, rb = int((s_rel - wa) * SR), int((t_rel - wa) * SR)
            gen = _inpaint(seg, s_rel - wa, t_rel - wa, e["prompt"], opts,
                           lambda p: phase(f"{tagp} · {p}", f0))
            gen = _align_len(gen, seg)
            if not e.get("fill"):
                gen, lag = _phase_align(gen, seg, ra, rb)
                if lag:
                    _log(f"  위상 정렬: {lag * 1000.0 / SR:+.0f}ms")
                gen = _match_spectrum(gen, seg)     # 유도發 어두움 보정 (EQ 매칭)
                gen = _match_rms(gen, seg, ra, rb)
            inst_cur[:, ia:ib] = _splice(seg, gen, s_rel - wa, t_rel - wa)

        phase("합성 (원본 스플라이스 + 보컬)", 0.9)
        mixed = inst_cur + vocals
        # 수정 구간만 mixed — 밖은 분리·재합성조차 거치지 않은 진짜 원본.
        # 보컬 게이트: 구간에 보컬이 사실상 없으면 스템(블리드 포함)을 얹지 않는다.
        result_win = wav
        for e in edits:
            s_rel, t_rel = e["start_s"] - off, e["end_s"] - off
            a, b = int(s_rel * SR), min(int(t_rel * SR), wav.shape[1])
            v_rms = float(np.sqrt(np.mean(vocals[:, a:b] ** 2))) if b > a else 0.0
            m_rms = float(np.sqrt(np.mean(wav[:, a:b] ** 2))) if b > a else 0.0
            layer = mixed
            if v_rms < max(1e-3, VOCAL_MIN_RATIO * m_rms):
                layer = inst_cur
                _log(f"  보컬 게이트: {s_rel:.1f}~{t_rel:.1f}s 보컬 미검출 — 반주만 합성")
            result_win = _splice(result_win, layer, s_rel, t_rel)
        if off or result_win.shape[1] < full.shape[1]:
            out = full.copy()
            a = int(off * SR)
            out[:, a:a + result_win.shape[1]] = result_win
            result = out
        else:
            result = result_win
        peak = float(np.abs(result).max() or 1.0)
        if peak > 1.0:
            result = result / peak * 0.999

        phase("인코딩", 0.95)
        job["result_b64"] = _encode_wav(result)
        # 보컬 스템 동봉 — 클라이언트가 사이드카로 저장해 두면 다음 체이닝
        # 리터치에서 재분리 없이 재사용한다. 윈도우 크롭이 적용된 긴 곡은
        # 스템이 전곡을 못 덮으므로 동봉하지 않는다 (오재사용 = 보컬 소실).
        if job.get("keep_vocals", True) and dur <= MAX_AUDIO_S:
            job["stem_b64"] = _encode_wav(vocals)
        job["sr"] = SR
        job["status"], job["phase"], job["progress"] = "done", "완료", 1.0
        _log(f"잡 {job['id'][:8]} 완료 — {time.time()-t0:.0f}s, "
             f"{total}건 {lo:.1f}~{hi:.1f}s")
    except Exception as e:
        oom = False
        try:
            import torch
            oom = isinstance(e, torch.cuda.OutOfMemoryError)
            if oom:
                torch.cuda.empty_cache()
        except Exception:
            pass
        job["status"] = "failed"
        job["error"] = "cuda_oom" if oom else f"{type(e).__name__}: {e}"
        job["phase"] = "실패"
        _log(f"잡 {job['id'][:8]} 실패: {job['error']}\n{traceback.format_exc(limit=3)}")
    finally:
        job["elapsed_s"] = round(time.time() - t0, 1)
        _unload()


def _worker():
    while True:
        job = None
        with _lock:
            if _queue:
                job = _queue.pop(0)
                job["status"] = "running"
        if job is None:
            time.sleep(1)
            continue
        _run_job(job)
        with _lock:                              # 오래된 결과 정리 (메모리)
            done = [j for j in _jobs.values() if j["status"] in ("done", "failed")]
            for j in sorted(done, key=lambda x: x["created"])[:-RESULT_KEEP]:
                j.pop("result_b64", None)
                j.pop("stem_b64", None)


class EditReq(BaseModel):
    audio_b64: str
    vocals_b64: str = None       # v4: 이전 리터치의 보컬 스템 — 있으면 재분리 생략
    edits: list = None           # [{start_s,end_s,prompt[,mode,strength,cfg_scale,fill,label,negative]}]
    start_s: float = None        # ↓ 구형 단건 호환
    end_s: float = None
    prompt: str = ""
    keep_vocals: bool = True
    seed: int = None
    steps: int = None
    cfg_scale: float = None      # 미지정 시 7.0 (프롬프트 준수 강도)
    mode: str = "inpaint"        # "inpaint"=다시 그리기 | "a2a"=원본 유지 변형
    strength: float = None       # a2a 노이즈 강도 (문서: 0.1 밀접 · 0.5 중간 혼합)


def _busy() -> bool:
    return bool(_queue) or any(j["status"] == "running" for j in _jobs.values())


@app.get("/health")
def health():
    fa = _flash_attn_status()
    return {"status": "ok", "version": VERSION, "cuda": _cuda(),
            "model_loaded": _sa3 is not None, "max_audio_s": MAX_AUDIO_S,
            "busy": _busy(),
            "modes": ["inpaint", "a2a", "overlay"],   # 클라이언트 기능 게이트
            "flash_attn": fa["ok"],
            "flash_attn_info": fa.get("version") or fa.get("error", ""),
            "queue": len(_queue) + sum(1 for j in _jobs.values()
                                       if j["status"] == "running")}


@app.post("/unload")
def unload_model():
    """모델 언로드 — 야간 심사(MF)에 GPU를 양보할 때 맥 중재자가 호출.
    리터치 잡이 진행/대기 중이면 409(busy) — 진행 중 잡은 깨뜨리지 않는다."""
    with _lock:
        if _busy():
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=409, content={"error": "busy"})
    _drop_models()
    _log("모델 언로드 — GPU를 MF(야간 심사)에 양보. 다음 리터치 잡에서 자동 재로드")
    return {"ok": True, "model_loaded": False}


@app.post("/diag")
def diag():
    """원격 진단 — 최소 생성부터 인자를 하나씩 얹어 어느 단계가 소리를
    망가뜨리는지 가른다 (맥에서 호출·분석). busy 면 409."""
    with _lock:
        if _busy():
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=409, content={"error": "busy"})
    import inspect

    import torch
    model = _load_sa3()
    try:
        sig = str(inspect.signature(model.generate))
    except Exception as e:
        sig = f"(조회 불가: {e})"
    try:
        import stable_audio_3 as _sa
        pkg_ver = getattr(_sa, "__version__", "?")
    except Exception:
        pkg_ver = "?"
    info = {"generate_signature": sig[:2000], "package_version": pkg_ver,
            "torch": torch.__version__, "model_class": type(model).__name__,
            "flash_attn": _flash_attn_status()}

    def to_np(out):
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "cpu"):
            out = out.float().cpu().numpy()
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:
            out = out[0]
        if out.ndim == 1:
            out = np.stack([out, out])
        return out

    prompt = "upbeat instrumental K-pop dance, 130 BPM, four-on-the-floor kick, bright synth"
    t = np.linspace(0, 10, 10 * SR, dtype=np.float32)
    music_like = (0.3 * np.sin(2 * np.pi * 220 * t)
                  * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.1666 * t))))
    seg = np.stack([music_like, music_like])
    cases = {
        "bare": dict(prompt=prompt, duration=10),
        "steps50": dict(prompt=prompt, duration=10, steps=50),
        "kw_sample_size": dict(prompt=prompt, duration=10,
                               sample_size=int(10 * SR)),
        "kw_cfg_neg": dict(prompt=prompt, duration=10, cfg_scale=7.0,
                           negative_prompt=NEG_PROMPT),
        "a2a_low": dict(prompt=prompt, duration=10,
                        init_audio=(SR, torch.from_numpy(seg)),
                        init_noise_level=0.1),
        "inpaint_mid": dict(prompt=prompt, duration=10,
                            inpaint_audio=(SR, torch.from_numpy(seg)),
                            inpaint_mask_start_seconds=4.0,
                            inpaint_mask_end_seconds=6.0),
    }
    results = {}
    for name, kwargs in cases.items():
        t0 = time.time()
        try:
            out = to_np(model.generate(**kwargs))
            results[name] = {"ok": True, "elapsed_s": round(time.time() - t0, 2),
                             "shape": list(out.shape),
                             "audio_b64": _encode_wav(out)}
            _log(f"diag {name}: OK {out.shape} {results[name]['elapsed_s']}s")
        except Exception as e:
            results[name] = {"ok": False,
                             "error": f"{type(e).__name__}: {str(e)[:300]}"}
            _log(f"diag {name}: 실패 {results[name]['error']}")
    _unload()
    return {"info": info, "results": results}


@app.post("/edit")
def edit(r: EditReq):
    if r.edits:
        edits = []
        for e in r.edits[:24]:
            try:
                edits.append({
                    "start_s": float(e.get("start_s", e.get("start"))),
                    "end_s": float(e.get("end_s", e.get("end"))),
                    "prompt": str(e.get("prompt") or "").strip(),
                    "mode": str(e.get("mode") or "inpaint"),
                    "strength": e.get("strength"),
                    "cfg_scale": e.get("cfg_scale"),
                    "fill": bool(e.get("fill")),
                    "label": str(e.get("label") or ""),
                    "negative": e.get("negative"),
                    "apg_scale": e.get("apg_scale"),
                })
            except (TypeError, ValueError):
                raise HTTPException(400, "edits 형식 오류 — start_s/end_s/prompt 필수")
        if any(not e["prompt"] for e in edits):
            raise HTTPException(400, "prompt가 비어 있는 예약이 있습니다")
    elif r.prompt.strip() and r.start_s is not None and r.end_s is not None:
        edits = [{"start_s": float(r.start_s), "end_s": float(r.end_s),
                  "prompt": r.prompt.strip(), "mode": r.mode,
                  "strength": r.strength, "cfg_scale": r.cfg_scale,
                  "fill": False, "label": "", "negative": None}]
    else:
        raise HTTPException(400, "edits 또는 start_s/end_s/prompt가 필요합니다")
    jid = uuid.uuid4().hex
    job = {"id": jid, "status": "queued", "phase": "대기열", "progress": 0.0,
           "created": time.time(), "audio_b64": r.audio_b64, "edits": edits,
           "vocals_b64": r.vocals_b64,
           "keep_vocals": r.keep_vocals, "seed": r.seed, "steps": r.steps}
    with _lock:
        _jobs[jid] = job
        _queue.append(job)
    _log(f"잡 접수 {jid[:8]} — {len(edits)}건 "
         f"{min(e['start_s'] for e in edits):.1f}~{max(e['end_s'] for e in edits):.1f}s, "
         f"첫 프롬프트: {edits[0]['prompt'][:70]}")
    return {"job_id": jid}


@app.get("/jobs/{jid}")
def job_status(jid: str):
    j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return {k: j.get(k) for k in
            ("id", "status", "phase", "progress", "error", "elapsed_s")}


@app.get("/jobs/{jid}/result")
def job_result(jid: str):
    j = _jobs.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    if j["status"] != "done":
        raise HTTPException(409, f"잡 상태: {j['status']}")
    if "result_b64" not in j:
        raise HTTPException(410, "결과가 만료되었습니다 — 다시 실행하세요")
    out = {"audio_b64": j["result_b64"], "sr": j["sr"]}
    if j.get("stem_b64"):     # v4: 보컬 스템 — 클라이언트가 체이닝용으로 저장
        out["vocals_b64"] = j["stem_b64"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    # 실행 환경 자가 복구 — flash-attn 깨짐('지지직' 글리치)과 uv sync 사고로
    # torch 가 CPU 빌드가 된 것(cuda=False)을 모두 감지해 자동 설치한다.
    # run.bat / run_sa3.bat 어느 쪽으로 떠도 여기를 지나므로 별도 절차 불필요.
    fa = _flash_attn_status()
    if not fa["ok"] or not _cuda():
        _log(f"⚠ 실행 환경 이상 (flash-attn={'OK' if fa['ok'] else '없음'}, "
             f"cuda={_cuda()}) — 자동 복구를 시도합니다…")
        try:
            from ensure_flash_attn import ensure
            st = ensure(_log)
            if st == "restart":
                _log("환경 복구 설치 완료 — 서버를 자동 재시작합니다")
                try:
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                except Exception as e:
                    _log(f"자동 재시작 실패({type(e).__name__}: {e}) — "
                         "이 창을 닫고 run.bat 을 다시 실행하세요")
                    raise SystemExit(1)
            if st:
                _FA = None                      # 상태 캐시 리셋 후 재판정
                fa = _flash_attn_status()
        except SystemExit:
            raise
        except Exception as e:
            _log(f"자동 복구 실행 오류({type(e).__name__}: {e})")
    if not fa["ok"]:
        _log(f"⚠⚠ flash-attn 여전히 불가 ({fa.get('error')}) — README "
             "'지지직' 트러블슈팅 절의 수동 절차가 필요합니다")
    else:
        _log(f"flash-attn OK (v{fa.get('version')})")
    threading.Thread(target=_worker, daemon=True).start()
    _log(f"SA3 리터치 서버 {VERSION} — {args.host}:{args.port} "
         f"(cuda={_cuda()}, unload_each={UNLOAD_EACH} — GPU 양보는 맥 중재자가 /unload 로)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
