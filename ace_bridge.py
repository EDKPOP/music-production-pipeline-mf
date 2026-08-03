"""ACE-Step 1.5 브리지 — 공식 REST 서버(8001)를 송캠프 리터치 계약(8600)으로 통역.

왜 브리지인가: 맥 본체(songcamp/postprod/retouch.py)는 sa3-v4 계약
(/health /edit /jobs/{id} /jobs/{id}/result, audio_b64 + edits 배열)을 쓴다.
ACE 공식 서버는 자체 계약(/release_task → /query_result → /v1/audio)이므로,
이 브리지가 통역을 맡아 맥 쪽은 주소만 바꾸면 되게 한다.

안전 장치: ACE repaint 가 마스크 밖을 얼마나 보존하는지는 미검증이다 —
브리지는 반환 오디오에서 **마스크 구간만** 잘라 원본 위에 등전력 크로스페이드로
스플라이스한다 (마스크 밖 원본 보존을 모델과 무관하게 구조적으로 보장).

원격 실측: POST /diag 로 업스트림 상태·순수 생성·리페인트 자가 테스트를,
{"raw_task": {...}, "src_audio_b64": ...} 로 ACE 원 API 를 그대로 두들길 수
있다 — 맥의 Claude 가 원격으로 품질 실험을 돌리는 통로.

실행: run_ace.bat (공식 서버 + 이 브리지를 함께 기동)
"""
import argparse
import base64
import io
import json
import os
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

VERSION = "ace-v3-fullmix-stems"
SR = 44100                    # 송캠프 계약 SR — ACE(48k) 결과를 여기로 리샘플
UPSTREAM = os.environ.get("ACE_UPSTREAM", "http://127.0.0.1:8001").rstrip("/")
# 모델 선택 — 실측(2026-08-03, 같은 시드·같은 구간)이 문서 티어 권장을 뒤집음:
#   xl-base: 평탄도 0.102·538Hz(잡음·먹먹) / turbo 계열: 0.057·1080Hz·2배 빠름.
# repaint 워크로드에선 turbo 가 우세 — XL 은 이득 없음. (gs 는 base 전용)
MODEL = os.environ.get("ACE_MODEL", "acestep-v15-turbo")
STEPS = int(os.environ.get("ACE_STEPS", 8))      # turbo 권장 8
ACE_DIR = os.environ.get("ACE_DIR", "")          # 설정 시 브리지가 upstream 수명 관리
_proc = None                                     # 관리 중인 acestep-api 프로세스
XFADE_S = 0.10
MAX_AUDIO_S = 600.0
RESULT_KEEP = 5

app = FastAPI(title="ace-bridge", docs_url=None)
_jobs = {}
_queue = []
_lock = threading.Lock()


def _log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ── 오디오 유틸 (sa3_server 와 동일 규약) ─────────────────────────
def _decode_wav(b64: str):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32",
                       always_2d=True)
    wav = data.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != SR:
        wav = _resample(wav, sr, SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _decode_any(raw: bytes):
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = data.T
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != SR:
        wav = _resample(wav, sr, SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _resample(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """48k(ACE) → 44.1k(계약) — MF venv 에 torchaudio 가 없을 수 있어 폴백 체인."""
    try:
        import torch
        import torchaudio
        return torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), sr_from, sr_to).numpy()
    except Exception:
        pass
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr_from, sr_to)
        return resample_poly(wav, sr_to // g, sr_from // g,
                             axis=1).astype(np.float32)
    except Exception:
        pass
    _log("⚠ torchaudio/scipy 없음 — 선형 보간 리샘플 (고역 품질 소폭 저하)")
    n_new = int(round(wav.shape[1] * sr_to / sr_from))
    x_old = np.linspace(0.0, 1.0, wav.shape[1])
    x_new = np.linspace(0.0, 1.0, n_new)
    return np.stack([np.interp(x_new, x_old, ch) for ch in wav]).astype(np.float32)


def _encode_wav(wav: np.ndarray) -> str:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav.T, -1.0, 1.0), SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _wav_bytes(wav: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav.T, -1.0, 1.0), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _match_spectrum(gen: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """마스크 구간 EQ 매칭 — ACE repaint 는 원본보다 크게 어둡게 나온다
    (실측 센트로이드 659~916Hz vs 원본 1749). 생성 구간의 장기 스펙트럼
    기울기를 원본 구간에 맞추면 원본급 밝기로 회복 (916→1801Hz, 평탄도
    0.039→0.016, 반영 유지 실측). 원본 대비 80% 이상 밝으면 무개입."""
    if os.environ.get("ACE_EQ_MATCH", "1") == "0":
        return gen
    n = min(gen.shape[1], ref.shape[1])
    if n < SR // 2 or float(np.sqrt(np.mean(ref[:, :n] ** 2))) < 1e-3:
        return gen
    G = np.abs(np.fft.rfft(gen[:, :n].mean(0)))
    R = np.abs(np.fft.rfft(ref[:, :n].mean(0)))
    w = max(64, len(G) // 40)
    Gs = np.convolve(G, np.ones(w) / w, mode="same")
    Rs = np.convolve(R, np.ones(w) / w, mode="same")
    fr = np.arange(len(G), dtype=np.float32)
    c_g = float((Gs ** 2 * fr).sum() / max((Gs ** 2).sum(), 1e-9))
    c_r = float((Rs ** 2 * fr).sum() / max((Rs ** 2).sum(), 1e-9))
    if c_g >= 0.95 * c_r:
        # 문턱 0.8→0.95 (실사고: 반주 기준 83%가 통과해 최종 믹스가 52~61%
        # 밝기로 먹먹 — repaint 는 거의 항상 어둡게 나오므로 적극 보정)
        return gen
    _log(f"  EQ 매칭: 센트로이드 {c_g / c_r * 100:.0f}% → 원본 기울기로 보정")
    k = np.clip(Rs / np.maximum(Gs, 1e-9), 0.5, 4.0)
    spec = np.fft.rfft(gen, axis=1)
    k2 = np.interp(np.linspace(0, 1, spec.shape[1]), np.linspace(0, 1, len(k)), k)
    out = np.fft.irfft(spec * k2, n=gen.shape[1], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _spectral_new(gen: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """생성물에서 '원본에 없던 성분'만 추출 — overlay(요소 추가)의 핵심.
    |G|-|O| 정류 후 G 위상으로 복원 (sa3 검증: 추가 성분 120배 우세 추출).
    repaint 가 구간을 재작곡해버리는 문제를 우회해, 원본 편곡을 그대로 두고
    캡션의 새 요소만 얹는다."""
    import numpy as _np
    n_fft, hop = 2048, 512
    win = _np.hanning(n_fft).astype(_np.float32)
    def stft(x):
        pads = _np.pad(x, (n_fft // 2, n_fft // 2))
        frames = _np.lib.stride_tricks.sliding_window_view(pads, n_fft)[::hop]
        return _np.fft.rfft(frames * win, axis=1)
    out = []
    for ch in range(gen.shape[0]):
        G = stft(gen[ch]); O = stft(orig[ch])
        m = min(G.shape[0], O.shape[0]); G, O = G[:m], O[:m]
        mag = _np.maximum(_np.abs(G) - _np.abs(O), 0.0)
        S = mag * _np.exp(1j * _np.angle(G))
        # overlap-add 역변환
        y = _np.zeros(m * hop + n_fft, dtype=_np.float32)
        wsum = _np.zeros_like(y)
        fr = _np.fft.irfft(S, n=n_fft, axis=1).astype(_np.float32) * win
        for i in range(m):
            y[i*hop:i*hop+n_fft] += fr[i]
            wsum[i*hop:i*hop+n_fft] += win ** 2
        y = y[n_fft//2:n_fft//2+gen.shape[1]] / _np.maximum(
            wsum[n_fft//2:n_fft//2+gen.shape[1]], 1e-6)
        out.append(y[:gen.shape[1]])
    return _np.stack(out).astype(_np.float32)


def _splice(original: np.ndarray, generated: np.ndarray,
            start_s: float, end_s: float) -> np.ndarray:
    """마스크 구간만 generated — 등전력 크로스페이드 (sa3_server 와 동일)."""
    n = min(original.shape[1], generated.shape[1])
    out = original[:, : original.shape[1]].copy()
    gen = generated[:, :n]
    a, b = int(start_s * SR), min(int(end_s * SR), n)
    if b <= a:
        return out
    out[:, a:b] = gen[:, a:b]
    xf = int(XFADE_S * SR)
    theta = np.linspace(0.0, np.pi / 2, xf, dtype=np.float32)
    g_in, g_out = np.sin(theta), np.cos(theta)
    lo = max(a - xf, 0)
    if a - lo > 0:
        gi, go = g_in[-(a - lo):], g_out[-(a - lo):]
        out[:, lo:a] = original[:, lo:a] * go + gen[:, lo:a] * gi
    hi = min(b + xf, n)
    if hi - b > 0:
        gi, go = g_in[: hi - b], g_out[: hi - b]
        out[:, b:hi] = gen[:, b:hi] * go + original[:, b:hi] * gi
    return out


def _phase_align(gen, ref, a, b, max_lag_s=0.045):
    """경계 위상 정렬 (sa3-v4.2 와 동일) — 페이징 방지."""
    n = gen.shape[1]
    max_lag = int(max_lag_s * SR)
    win = int(0.25 * SR)
    corr = None
    for edge in (a, b):
        lo, hi = max(edge - win, 0), min(edge + win, n, ref.shape[1])
        if hi - lo < max_lag * 3:
            continue
        r = ref[:, lo:hi].mean(0)
        g = gen[:, lo:hi].mean(0)
        if float(np.sqrt((r ** 2).mean())) < 1e-3:
            continue
        L = hi - lo
        c = np.correlate(r, g, "full")[L - 1 - max_lag: L + max_lag]
        corr = c if corr is None else corr + c
    if corr is None or not len(corr):
        return gen, 0
    lag = int(np.argmax(corr)) - max_lag
    if lag == 0:
        return gen, 0
    out = np.empty_like(gen)
    if lag > 0:
        out[:, lag:] = gen[:, : n - lag]
        out[:, :lag] = gen[:, :1]
    else:
        out[:, : n + lag] = gen[:, -lag:]
        out[:, n + lag:] = gen[:, -1:]
    return out, lag


# ── ACE 업스트림 호출 ─────────────────────────────────────────────
def _ace_task(data: dict, src_wav: np.ndarray = None, timeout_s: float = 300,
              phase=lambda p: None) -> np.ndarray:
    """release_task → query_result 폴링 → /v1/audio 다운로드 → (2,T)@44.1k.

    src_wav 가 있으면 multipart(src_audio)로 함께 올린다 (repaint/cover 용).
    """
    fields = {k: str(v) for k, v in data.items() if v is not None}
    fields.setdefault("audio_format", "wav")
    fields.setdefault("batch_size", "1")
    fields.setdefault("inference_steps", str(STEPS))
    if MODEL:
        fields.setdefault("model", MODEL)
    files = None
    if src_wav is not None:
        files = {"src_audio": ("src.wav", _wav_bytes(src_wav), "audio/wav")}
    r = requests.post(f"{UPSTREAM}/release_task", data=fields, files=files,
                      timeout=600)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"ACE release_task 오류: {body['error']}")
    tid = body["data"]["task_id"]
    phase(f"ACE 잡 {tid[:8]} 대기")
    t0 = time.time()
    while True:
        time.sleep(2)
        q = requests.post(f"{UPSTREAM}/query_result",
                          json={"task_id_list": [tid]}, timeout=30).json()
        row = (q.get("data") or [{}])[0]
        st = row.get("status")
        if st == 1:
            break
        if st == 2:
            raise RuntimeError(f"ACE 잡 실패: {str(row)[:600]}")
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"ACE 잡 시간 초과({timeout_s:.0f}s)")
    res = row.get("result")
    if isinstance(res, str):
        res = json.loads(res)
    file_url = (res or [{}])[0].get("file", "")
    if not file_url:
        raise RuntimeError(f"ACE 결과에 파일 없음: {str(res)[:300]}")
    if file_url.startswith("/"):
        file_url = UPSTREAM + file_url
    audio = requests.get(file_url, timeout=120)
    audio.raise_for_status()
    return _decode_any(audio.content)


def _spawn_upstream(log=print) -> bool:
    """관리 모드: acestep-api 를 자식 프로세스로 기동 (ACE_DIR 필요).

    공식 unload API 가 없으므로 MF 와의 GPU 교대는 프로세스 시작/종료로
    구현한다 — kill 이 곧 VRAM 완전 해제다."""
    global _proc
    if not ACE_DIR:
        return False
    if _proc is not None and _proc.poll() is None:
        return True
    import subprocess
    env = dict(os.environ)
    env.setdefault("ACESTEP_API_PORT", UPSTREAM.rsplit(":", 1)[-1])
    env.setdefault("SERVER_NAME", "127.0.0.1")
    env.setdefault("ACESTEP_CONFIG_PATH", MODEL)
    env.setdefault("ACESTEP_INIT_LLM", "true")   # thinking(repaint 품질 재료)에 필요
    _log(f"upstream 기동: uv run acestep-api ({MODEL}, LM 없음) @ {ACE_DIR}")
    _proc = subprocess.Popen(["uv", "run", "acestep-api"], cwd=ACE_DIR, env=env)
    return True


def _kill_upstream(log=print):
    global _proc
    _MODEL_READY["ok"] = False
    if _proc is not None and _proc.poll() is None:
        _log("upstream 종료 — GPU 를 MF(심사)에 양보")
        _proc.terminate()
        try:
            _proc.wait(timeout=30)
        except Exception:
            _proc.kill()
    _proc = None


def _ensure_up(log=print, timeout_s: float = 300) -> bool:
    """업스트림 살아있음 보장 — 관리 모드면 필요 시 스폰하고 health 를 기다린다."""
    if _upstream_ok()["ok"]:
        return True
    if not _spawn_upstream(log):
        return False
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        if _upstream_ok()["ok"]:
            return True
    return False


_demucs = None


def _separate(wav, phase):
    """demucs 2-스템 (sa3_server 와 동일 규약) — 스템 완전 자동화의 핵심.
    사이드카가 없으면 여기서 분리한다. inst = 원본 − vocals (합 보존)."""
    global _demucs
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if _demucs is None:
        phase("demucs(htdemucs) 로딩")
        _demucs = get_model("htdemucs")
        _demucs.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    phase("보컬/반주 자동 분리 (demucs)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(wav)[None]
    ref = t.mean(0)
    t = (t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        srcs = apply_model(_demucs, t.to(dev), device=dev,
                           split=True, overlap=0.25, progress=False)[0]
    srcs = srcs * (ref.std() + 1e-8) + ref.mean()
    vocals = srcs[_demucs.sources.index("vocals")].cpu().numpy()
    return vocals.astype(np.float32)


def _upstream_ok() -> dict:
    try:
        h = requests.get(f"{UPSTREAM}/health", timeout=4).json()
        return {"ok": True, "detail": h.get("data") or h}
    except Exception as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}


def _models() -> list:
    try:
        d = requests.get(f"{UPSTREAM}/v1/models", timeout=10).json().get("data")
        if isinstance(d, dict):
            return d.get("models") or []
        return d or []
    except Exception:
        return []


_MODEL_READY = {"ok": False}


def _ensure_model(phase=lambda p: None) -> bool:
    """DiT 모델 초기화 보장 — ACE 서버는 /v1/init 전엔 모델이 비어 있고
    /release_task 가 무한 대기한다 (실측: models=[] + POST 120s 타임아웃).
    기동 시·첫 잡 전에 호출한다. 다운로드 포함 최대 20분 폴링."""
    if _MODEL_READY["ok"] or _models():
        _MODEL_READY["ok"] = True
        return True
    phase("ACE 모델 초기화 요청 — 최초엔 다운로드·로드로 수 분")
    _log(f"/v1/init 호출 (model={MODEL}, init_llm=True — thinking 용)")
    try:
        r = requests.post(f"{UPSTREAM}/v1/init",
                          json={"model": MODEL, "init_llm": True},
                          timeout=1800)
        _log(f"/v1/init 응답: {str(r.text)[:200]}")
    except Exception as e:
        _log(f"/v1/init 예외({type(e).__name__}: {e}) — 폴링으로 준비 확인")
    for _ in range(120):
        if _models():
            _MODEL_READY["ok"] = True
            phase("ACE 모델 준비 완료")
            _log("ACE 모델 준비 완료")
            return True
        time.sleep(10)
    return False


def _boot_watch():
    # 관리 모드(ACE_DIR)에서는 자동 스폰하지 않는다 — GPU 는 기본적으로 MF 의
    # 것이고, ACE 는 /load(리터치 시작) 때만 올라온다 (교대 규칙)
    if ACE_DIR:
        return
    """기동 도우미 — 업스트림(공식 서버)이 뜰 때까지 기다렸다가 모델 초기화.
    브리지를 먼저 켜도 순서 문제가 없다."""
    for _ in range(360):                # 최대 1시간 (최초 모델 다운로드 감안)
        if _upstream_ok()["ok"]:
            break
        time.sleep(10)
    _ensure_model()


# ── 잡 워커 (sa3 계약과 동일한 수명주기) ──────────────────────────
def _run_job(job: dict):
    def phase(p, prog=None):
        job["phase"] = p
        if prog is not None:
            job["progress"] = prog
        _log(f"잡 {job['id'][:8]}: {p}")

    t0 = time.time()
    try:
        if not _ensure_model(lambda p: phase(p, 0.03)):
            raise RuntimeError("ACE 모델 초기화 실패 — ace-api 창을 확인하세요")
        phase("오디오 디코드", 0.05)
        wav = _decode_wav(job.pop("audio_b64"))
        dur = wav.shape[1] / SR
        edits = job["edits"]
        for e in edits:
            s, t = float(e["start_s"]), float(e["end_s"])
            if not (0 <= s < t <= dur + 0.5):
                raise ValueError(f"구간이 곡 길이를 벗어남: {s}~{t}s (곡 {dur:.1f}s)")
            e["start_s"], e["end_s"] = s, min(t, dur)
        if not _ensure_up(timeout_s=420):
            raise RuntimeError("ACE 서버(upstream) 기동 실패 — ace-bridge 창 확인")
        if not _ensure_model(lambda p: phase(p, 0.04)):
            raise RuntimeError("ACE 모델 초기화 실패 — ace-bridge 창 확인")
        # ── v3 아키텍처: 분리는 입력이 아니라 '출력'에서 ──
        # 모델에는 항상 풀믹스(보컬 포함)를 들려준다 — 곡의 진짜 문맥(보컬
        # 라인)에 맞춰 작곡하게 하기 위함 (실사고: 분리 반주만 들려주면
        # '맥락 무시'·보컬 부조화). 보컬 보존은 생성된 구간에서 가짜 보컬을
        # demucs 로 제거하고 원본 보컬을 되돌려 얹는 방식으로 달성한다.
        need_preserve = (any(not e.get("vocal_edit") for e in edits)
                         and job.get("keep_vocals", True))
        vocals = None
        vb64 = job.pop("vocals_b64", None)
        if need_preserve:
            if vb64:
                phase("원본 보컬 스템 재사용", 0.08)
                vocals = _decode_wav(vb64)
            else:
                vocals = _separate(wav, lambda p: phase("원본 " + p, 0.1))
            if vocals.shape[1] < wav.shape[1]:
                vocals = np.concatenate(
                    [vocals, np.zeros((2, wav.shape[1] - vocals.shape[1]),
                                      np.float32)], axis=1)
            vocals = np.ascontiguousarray(vocals[:, :wav.shape[1]])
        mix_cur = wav.copy()
        vocal_edited = False
        total = len(edits)
        for i, e in enumerate(edits):
            f0 = 0.15 + 0.75 * i / total
            tagp = f"{i+1}/{total} {e.get('label') or ''}".strip()
            v_edit = bool(e.get("vocal_edit"))
            mode_e = (e.get("mode") or "repaint").lower()
            phase(f"{tagp} · ACE repaint(풀믹스 문맥) "
                  f"{e['start_s']:.1f}~{e['end_s']:.1f}s", f0)
            data = {"task_type": "repaint", "prompt": e["prompt"],
                    "repainting_start": e["start_s"],
                    "repainting_end": e["end_s"],
                    "audio_duration": round(dur, 2)}
            if v_edit:
                if e.get("lyrics"):
                    data["lyrics"] = e["lyrics"]
                if e.get("vocal_language"):
                    data["vocal_language"] = e["vocal_language"]
            if "turbo" not in MODEL:
                cs = float(e.get("cfg_scale") or 2.0)
                data["guidance_scale"] = round(min(max(3.0 + cs * 2.0, 4.0), 9.0), 1)
            if e.get("bpm"):
                data["bpm"] = int(float(e["bpm"]))
            if job.get("seed") is not None:
                data["seed"] = int(job["seed"])
                data["use_random_seed"] = "false"
            gen = _ace_task(data, src_wav=mix_cur,
                            phase=lambda p: phase(f"{tagp} · {p}", f0))
            if gen.shape[1] < mix_cur.shape[1]:
                gen = np.concatenate([gen, mix_cur[:, gen.shape[1]:]], axis=1)
            gen = gen[:, : mix_cur.shape[1]]
            a = int(e["start_s"] * SR)
            b = min(int(e["end_s"] * SR), mix_cur.shape[1])
            gen, lag = _phase_align(gen, mix_cur, a, b)
            if lag:
                _log(f"  위상 정렬: {lag * 1000.0 / SR:+.0f}ms")
            if v_edit:
                # 보컬 편집 — 생성 풀믹스(새 가창)를 그대로 채택
                fixed = _match_spectrum(gen[:, a:b], mix_cur[:, a:b])
                gen = gen.copy()
                gen[:, a:a + fixed.shape[1]] = fixed
                mix_cur = _splice(mix_cur, gen, e["start_s"], e["end_s"])
                vocal_edited = True
                continue
            # 반주 경로 — 생성 구간에서 '가짜 보컬'을 분리 제거
            gen_reg = np.ascontiguousarray(gen[:, a:b])
            if vocals is not None:
                gen_voc = _separate(
                    gen_reg, lambda p: phase(f"{tagp} · 생성 구간 {p}", f0))
                gen_inst = gen_reg - gen_voc
            else:
                gen_inst = gen_reg
            if mode_e == "overlay":
                # 요소 추가 — 원본에 없던 성분만 추출해 기존 믹스 위에 합성
                elem = _spectral_new(gen_inst, mix_cur[:, a:b])
                ref_rms = float(np.sqrt(np.mean(mix_cur[:, a:b] ** 2))) or 1e-4
                e_rms = float(np.sqrt(np.mean(elem ** 2)))
                if e_rms < 0.02 * ref_rms:
                    _log("  overlay: 추가 성분 미미 — 건너뜀")
                    continue
                lvl = min(max(float(e.get("strength") or 0.4), 0.15), 0.9)
                elem *= (ref_rms / max(e_rms, 0.05 * ref_rms)) * lvl * 1.2
                xf0 = int(0.03 * SR)
                if elem.shape[1] > 2 * xf0:
                    r0 = np.linspace(0.0, 1.0, xf0, dtype=np.float32)
                    elem[:, :xf0] *= r0
                    elem[:, -xf0:] *= r0[::-1]
                mix_cur = mix_cur.copy()
                mix_cur[:, a:a + elem.shape[1]] += elem
                _log(f"  overlay: 요소만 합성 ×{lvl * 1.2:.2f} (기존 편곡 무변경)")
                continue
            # repaint(재작곡) — 새 반주 + 원본 보컬 재합성 (보컬 게이트)
            if vocals is not None:
                orig_inst_reg = wav[:, a:b] - vocals[:, a:b]
                gen_inst = _match_spectrum(gen_inst, orig_inst_reg)
                v_rms = float(np.sqrt(np.mean(vocals[:, a:b] ** 2)))
                m_rms = float(np.sqrt(np.mean(wav[:, a:b] ** 2)))
                region = (gen_inst if v_rms < max(1e-3, 0.05 * m_rms)
                          else gen_inst + vocals[:, a:b])
            else:
                region = _match_spectrum(gen_inst, wav[:, a:b])
            gen_full = gen.copy()
            gen_full[:, a:a + (b - a)] = region[:, : b - a]
            mix_cur = _splice(mix_cur, gen_full, e["start_s"], e["end_s"])
        cur = mix_cur
        peak = float(np.abs(cur).max() or 1.0)
        if peak > 1.0:
            cur = cur / peak * 0.999
        phase("인코딩", 0.95)
        job["result_b64"] = _encode_wav(cur)
        if vocals is not None and not vocal_edited:
            # 스템 사이드카 순환 — 보컬을 편집했다면 스템이 낡았으므로 생략
            # (다음 리터치에서 자동 재분리가 정합성을 보장)
            job["stem_b64"] = _encode_wav(vocals)
        job["sr"] = SR
        job["status"], job["phase"], job["progress"] = "done", "완료", 1.0
        _log(f"잡 {job['id'][:8]} 완료 — {time.time()-t0:.0f}s, {total}건")
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {e}"
        job["phase"] = "실패"
        _log(f"잡 {job['id'][:8]} 실패: {job['error']}\n"
             f"{traceback.format_exc(limit=3)}")
    finally:
        job["elapsed_s"] = round(time.time() - t0, 1)


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
        with _lock:
            done = [j for j in _jobs.values() if j["status"] in ("done", "failed")]
            for j in sorted(done, key=lambda x: x["created"])[:-RESULT_KEEP]:
                j.pop("result_b64", None)
                j.pop("stem_b64", None)


def _busy() -> bool:
    return bool(_queue) or any(j["status"] == "running" for j in _jobs.values())


class EditReq(BaseModel):
    audio_b64: str
    edits: list = None
    keep_vocals: bool = True     # 호환 필드
    vocals_b64: str = None       # 보컬 스템 — 있으면 반주만 repaint (보컬 보존)
    seed: int = None
    steps: int = None


@app.get("/health")
def health():
    up = _upstream_ok()
    models = _models() if up["ok"] else []
    if models:
        _MODEL_READY["ok"] = True
    return {"status": "ok" if up["ok"] else "upstream_down",
            "version": VERSION, "cuda": up["ok"],   # 업스트림 living = GPU 가동
            "model_loaded": bool(models),
            "ace_models": [m.get("name") for m in models if isinstance(m, dict)],
            "max_audio_s": MAX_AUDIO_S,
            "busy": _busy(), "modes": ["inpaint", "a2a", "overlay", "repaint"],
            "flash_attn": True, "flash_attn_info": "(ace)",
            "engine": "ace-step-1.5", "upstream": UPSTREAM,
            "upstream_detail": up["detail"],
            "queue": len(_queue) + sum(1 for j in _jobs.values()
                                       if j["status"] == "running")}


@app.post("/unload")
def unload():
    """GPU 중재 — XL 모델은 MF 와 공존 불가. 관리 모드에서는 upstream 프로세스를
    종료해 VRAM 을 완전 해제한다 (공식 unload API 부재의 우회). busy 면 409."""
    with _lock:
        if _busy():
            return JSONResponse(status_code=409, content={"error": "busy"})
    if not ACE_DIR:
        return {"ok": True, "model_loaded": True,
                "note": "비관리 모드 — upstream 을 직접 내릴 수 없음"}
    _kill_upstream()
    return {"ok": True, "model_loaded": False}


@app.post("/load")
def load():
    """리터치 준비 — upstream 기동 + 모델 초기화 (XL 로드 수십 초~수 분)."""
    if not _ensure_up(timeout_s=420):
        return JSONResponse(status_code=503, content={"error": "upstream 기동 실패"})
    if not _ensure_model():
        return JSONResponse(status_code=503, content={"error": "모델 초기화 실패"})
    return {"ok": True, "model_loaded": True, "model": MODEL}


@app.post("/edit")
def edit(r: EditReq):
    edits = []
    for e in (r.edits or [])[:24]:
        try:
            edits.append({
                "start_s": float(e.get("start_s", e.get("start"))),
                "end_s": float(e.get("end_s", e.get("end"))),
                "prompt": str(e.get("prompt") or "").strip(),
                "lyrics": e.get("lyrics"),
                "bpm": e.get("bpm"),
                "audio_cover_strength": e.get("audio_cover_strength"),
                "cfg_scale": e.get("cfg_scale"),
                "mode": str(e.get("mode") or "repaint"),
                "strength": e.get("strength"),
                "vocal_edit": bool(e.get("vocal_edit")),
                "vocal_language": e.get("vocal_language"),
                "label": str(e.get("label") or ""),
            })
        except (TypeError, ValueError):
            return JSONResponse(status_code=400,
                                content={"detail": "edits 형식 오류"})
    if not edits or any(not e["prompt"] for e in edits):
        return JSONResponse(status_code=400,
                            content={"detail": "prompt 가 있는 edits 필요"})
    jid = uuid.uuid4().hex
    job = {"id": jid, "status": "queued", "phase": "대기열", "progress": 0.0,
           "created": time.time(), "audio_b64": r.audio_b64, "edits": edits,
           "vocals_b64": r.vocals_b64, "keep_vocals": r.keep_vocals,
           "seed": r.seed}
    with _lock:
        _jobs[jid] = job
        _queue.append(job)
    _log(f"잡 접수 {jid[:8]} — {len(edits)}건, 첫 프롬프트: {edits[0]['prompt'][:70]}")
    return {"job_id": jid}


@app.get("/jobs/{jid}")
def job_status(jid: str):
    j = _jobs.get(jid)
    if not j:
        return JSONResponse(status_code=404, content={"detail": "job not found"})
    return {k: j.get(k) for k in
            ("id", "status", "phase", "progress", "error", "elapsed_s")}


@app.get("/jobs/{jid}/result")
def job_result(jid: str):
    j = _jobs.get(jid)
    if not j:
        return JSONResponse(status_code=404, content={"detail": "job not found"})
    if j["status"] != "done":
        return JSONResponse(status_code=409, content={"detail": j["status"]})
    if "result_b64" not in j:
        return JSONResponse(status_code=410, content={"detail": "결과 만료"})
    out = {"audio_b64": j["result_b64"], "sr": j["sr"]}
    if j.get("stem_b64"):
        out["vocals_b64"] = j["stem_b64"]
    return out


class DiagReq(BaseModel):
    audio_b64: str = None        # 실곡 repaint 테스트용 (없으면 합성음)
    start_s: float = 3.0
    end_s: float = 6.0
    prompt: str = ("Rapid chopped drum-and-bass breakbeat fill, fast snare "
                   "rolls, busy percussion. 130 BPM.")
    raw_task: dict = None        # ACE 원 API 패스스루 — 원격 실측 만능 통로
    src_audio_b64: str = None    # raw_task 의 src_audio


@app.post("/diag")
def diag(r: DiagReq):
    """원격 자가 테스트 — 업스트림 상태·순수 생성·repaint 를 한 번에.

    raw_task 가 있으면 그것만 그대로 ACE 에 던지고 결과를 돌려준다
    (맥의 Claude 가 파라미터 실험을 자유롭게 돌리는 통로)."""
    if _busy():
        return JSONResponse(status_code=409, content={"error": "busy"})
    if not _ensure_model():
        return JSONResponse(status_code=503,
                            content={"error": "ACE 모델 초기화 실패 — ace-api 창 확인"})
    info = {"upstream": UPSTREAM, "health": _upstream_ok()}
    try:
        info["models"] = requests.get(f"{UPSTREAM}/v1/models", timeout=8).json().get("data")
    except Exception as e:
        info["models"] = f"조회 실패: {type(e).__name__}"
    out = {"info": info, "results": {}}

    def run_case(name, data, src=None):
        t0 = time.time()
        try:
            wav = _ace_task(data, src_wav=src, timeout_s=240)
            out["results"][name] = {"ok": True,
                                    "elapsed_s": round(time.time() - t0, 1),
                                    "shape": list(wav.shape),
                                    "audio_b64": _encode_wav(wav)}
            _log(f"diag {name}: OK {wav.shape} {out['results'][name]['elapsed_s']}s")
        except Exception as e:
            out["results"][name] = {"ok": False,
                                    "error": f"{type(e).__name__}: {str(e)[:300]}"}
            _log(f"diag {name}: 실패 {out['results'][name]['error']}")

    if r.raw_task:
        src = _decode_wav(r.src_audio_b64) if r.src_audio_b64 else None
        run_case("raw", dict(r.raw_task), src=src)
        return out
    # 기본 배터리: 순수 생성 → repaint(합성음 또는 보낸 실곡)
    run_case("t2a", {"task_type": "text2music", "prompt": r.prompt,
                     "audio_duration": 10, "lyrics": "[instrumental]"})
    if r.audio_b64:
        src = _decode_wav(r.audio_b64)
    else:
        t = np.arange(int(10 * SR)) / SR
        beat = (0.4 * np.sin(2 * np.pi * 220 * t)
                * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 2.1666 * t))))
        src = np.stack([beat, beat]).astype(np.float32)
    run_case("repaint", {"task_type": "repaint", "prompt": r.prompt,
                         "repainting_start": r.start_s,
                         "repainting_end": r.end_s,
                         "audio_duration": round(src.shape[1] / SR, 2)},
             src=src)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_boot_watch, daemon=True).start()
    up = _upstream_ok()
    _log(f"ACE 브리지 {VERSION} — {args.host}:{args.port} → 업스트림 {UPSTREAM} "
         f"({'연결됨' if up['ok'] else '대기 — acestep-api 가 뜨면 자동 연결·모델 초기화'})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
