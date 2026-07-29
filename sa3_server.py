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
                           "start_s","end_s","prompt"[, "keep_vocals"=true,
                           "seed","steps","cfg_scale"]} → {"job_id"}
  GET  /jobs/{id}       → {"status":"queued|running|done|failed","phase",
                           "progress":0~1,"elapsed_s"[,"error"]}
  GET  /jobs/{id}/result→ {"audio_b64": 결과 전체 곡 wav b64, "sr":44100}
  OOM 시 잡 error = "cuda_oom" (본체가 이 문자열로 안내 분기)

GPU 메모리: 이 PC는 MF 8B(~16GB)가 상주하므로 SA3(~6.5GB)+demucs는
기본적으로 잡이 끝나면 언로드한다 (SA3_KEEP_LOADED=1 로 상주 전환).

라이선스: Stable Audio 3 는 Stability AI Community License — 비상업 용도 OK.
"""
import argparse
import base64
import io
import os
import threading
import time
import traceback
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VERSION = "sa3-v2-arbiter"
SR = 44100
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
    phase(f"구간 재생성 (Stable Audio 3, {start_s:.1f}~{end_s:.1f}s / {dur:.0f}s)")
    kwargs = dict(
        # 공식 시그니처는 (sample_rate, audio) 순서 — README 예제(torchaudio.load
        # 반환 순서)와 반대다. 뒤집으면 'int has no attribute to' 로 즉사 (실기 재현)
        inpaint_audio=(SR, torch.from_numpy(inst)),
        inpaint_mask_start_seconds=float(start_s),
        inpaint_mask_end_seconds=float(end_s),
        prompt=prompt,
        duration=dur,
        # 기본 sample_size 는 ~120s 상한 — 전곡 컨텍스트가 조용히 잘리지 않게
        # 곡 길이(+여유)만큼 명시한다 (모델 상한 380s 안쪽)
        sample_size=int(min(dur + 8.0, 380.0) * SR),
    )
    for k in ("seed", "steps", "cfg_scale"):     # 선택 파라미터 — API가 모르면 제거
        if extra.get(k) is not None:
            kwargs[k] = extra[k]
    try:
        out = model.generate(**kwargs)
    except TypeError as e:
        _log(f"⚠ 선택 파라미터 미지원({e}) — 기본 파라미터로 재시도")
        for k in ("seed", "steps", "cfg_scale"):
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
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    lo = max(a - xf, 0)
    if a - lo > 0:                               # 들어가는 경계
        r = ramp[-(a - lo):]
        out[:, lo:a] = original[:, lo:a] * (1 - r) + gen[:, lo:a] * r
    hi = min(b + xf, n)
    if hi - b > 0:                               # 나오는 경계
        r = ramp[: hi - b]
        out[:, b:hi] = gen[:, b:hi] * (1 - r) + original[:, b:hi] * r
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
        start, end = float(job["start_s"]), float(job["end_s"])
        if not (0 <= start < end <= dur + 0.5):
            raise ValueError(f"구간이 곡 길이를 벗어남: {start}~{end}s (곡 {dur:.1f}s)")
        end = min(end, dur)

        # 모델 상한 초과 곡 → 마스크를 중심에 둔 윈도우만 모델에 보낸다
        off = 0.0
        full = wav
        if dur > MAX_AUDIO_S:
            seg = end - start
            off = max(0.0, min(start - (MAX_AUDIO_S - seg) / 2, dur - MAX_AUDIO_S))
            a = int(off * SR)
            wav = wav[:, a:a + int(MAX_AUDIO_S * SR)]
            _log(f"  곡 {dur:.0f}s > {MAX_AUDIO_S:.0f}s — 윈도우 {off:.1f}s~ 적용")

        phase("보컬/반주 분리 (demucs)", 0.15)
        if job.get("keep_vocals", True):
            vocals, inst = _separate(wav, lambda p: phase(p, 0.25))
        else:
            vocals, inst = np.zeros_like(wav), wav

        phase("구간 재생성 (Stable Audio 3)", 0.45)
        gen_inst = _inpaint(inst, start - off, end - off, job["prompt"],
                            job, lambda p: phase(p, 0.5))

        phase("합성 (원본 스플라이스 + 보컬)", 0.88)
        new_inst = _splice(inst, gen_inst, start - off, end - off)
        mixed = new_inst + vocals
        # 마스크 밖은 분리·재합성조차 거치지 않은 진짜 원본으로 되돌린다
        result_win = _splice(wav, mixed, start - off, end - off)
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
        job["sr"] = SR
        job["status"], job["phase"], job["progress"] = "done", "완료", 1.0
        _log(f"잡 {job['id'][:8]} 완료 — {time.time()-t0:.0f}s, "
             f"구간 {start:.1f}~{end:.1f}s, 프롬프트: {job['prompt'][:60]}")
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


class EditReq(BaseModel):
    audio_b64: str
    start_s: float
    end_s: float
    prompt: str
    keep_vocals: bool = True
    seed: int = None
    steps: int = None
    cfg_scale: float = None


def _busy() -> bool:
    return bool(_queue) or any(j["status"] == "running" for j in _jobs.values())


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "cuda": _cuda(),
            "model_loaded": _sa3 is not None, "max_audio_s": MAX_AUDIO_S,
            "busy": _busy(),
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


@app.post("/edit")
def edit(r: EditReq):
    if not r.prompt.strip():
        raise HTTPException(400, "prompt가 비어 있습니다")
    jid = uuid.uuid4().hex
    job = {"id": jid, "status": "queued", "phase": "대기열", "progress": 0.0,
           "created": time.time(), "audio_b64": r.audio_b64,
           "start_s": r.start_s, "end_s": r.end_s, "prompt": r.prompt.strip(),
           "keep_vocals": r.keep_vocals, "seed": r.seed, "steps": r.steps,
           "cfg_scale": r.cfg_scale}
    with _lock:
        _jobs[jid] = job
        _queue.append(job)
    _log(f"잡 접수 {jid[:8]} — {r.start_s:.1f}~{r.end_s:.1f}s, "
         f"프롬프트: {r.prompt[:80]}")
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
    return {"audio_b64": j["result_b64"], "sr": j["sr"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    threading.Thread(target=_worker, daemon=True).start()
    _log(f"SA3 리터치 서버 {VERSION} — {args.host}:{args.port} "
         f"(cuda={_cuda()}, unload_each={UNLOAD_EACH} — GPU 양보는 맥 중재자가 /unload 로)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
