"""SAO-Instruct 편집 서버(8700) — 유형③(악기·음색 교체, 멜로디 유지) 전용.

왜 이 서버인가 (docs/retouch_workflow_redesign.md, 2026-08-04):
- ACE repaint 는 구간 재작곡이라 "신스→통기타, 나머지 유지" 같은 외과적
  교체가 원리적으로 불가능하고, 멜로디 스템 입력 시 순수성이 깨진다(실측).
- SAO-Instruct(ETH, NeurIPS 2025)는 Stable Audio Open(44.1kHz) 위에서
  자연어 지시 편집을 하는 공개 웨이트 모델 — 원본 라텐트에서 출발하므로
  보존 성향이 생성형 repaint 보다 유리할 것으로 기대(실측 전 미입증).

전략 — 스템 격리 + 창(window) 편집:
1. 마스크 구간 ± 문맥을 최대 45s 창으로 잘라낸다 (SAO 길이 한계 ~47s).
2. mode=stem_edit: 창을 demucs 4스템 분리 → 대상 스템만 지시 편집 →
   창믹스 − 원본스템 + 새스템 재합성 (나머지 성분 1:1 보존).
   mode=edit: 창 풀믹스를 직접 편집 (비교 실측용).
3. 편집된 마스크 구간만 원곡에 등전력 크로스페이드로 스플라이스.

계약: sa3/ace 브리지와 동일 (/health /load /unload /edit /jobs/{id}
/jobs/{id}/result, audio_b64 + edits 배열). /diag 로 원격 실측.
SAO_MOCK=1 이면 모델 없이 수명주기·재합성 수학을 검증할 수 있다.

실행: _sao_start.bat (원클릭 — 클론·환경·모델 다운로드·기동)
"""
import base64
import io
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

VERSION = "sao-v1-stemedit"
SR = 44100
SAO_DIR = os.environ.get("SAO_DIR", "")      # sao-instruct 레포 경로
HF_ID = os.environ.get("SAO_HF_ID", "disco-eth/sao-instruct")
MOCK = os.environ.get("SAO_MOCK", "0") == "1"
WIN_MAX_S = 45.0                             # SAO 생성 길이 한계(~47s) 안쪽
WIN_CTX_S = 8.0                              # 마스크 양옆 문맥
XFADE_S = 0.10
RESULT_KEEP = 5

app = FastAPI(title="sao-server", docs_url=None)
_jobs = {}
_queue = []
_lock = threading.Lock()
_model = None
_demucs = None


def _log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ── 오디오 유틸 (ace_bridge 와 동일 규약) ─────────────────────────
def _resample(wav, sr_from, sr_to):
    if sr_from == sr_to:
        return wav
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
    n_new = int(round(wav.shape[1] * sr_to / sr_from))
    x_old = np.linspace(0.0, 1.0, wav.shape[1])
    x_new = np.linspace(0.0, 1.0, n_new)
    return np.stack([np.interp(x_new, x_old, ch) for ch in wav]).astype(np.float32)


def _decode_wav(b64):
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


def _encode_wav(wav):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav.T, -1.0, 1.0), SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _splice(original, generated, start_s, end_s):
    n = min(original.shape[1], generated.shape[1])
    out = original.copy()
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
        out[:, lo:a] = original[:, lo:a] * g_out[-(a - lo):] \
            + gen[:, lo:a] * g_in[-(a - lo):]
    hi = min(b + xf, n)
    if hi - b > 0:
        out[:, b:hi] = gen[:, b:hi] * g_out[: hi - b] \
            + original[:, b:hi] * g_in[: hi - b]
    return out


def _match_spectrum(gen, ref):
    """장기 스펙트럼 기울기 매칭 — ace_bridge 와 동일 (밝기 회복)."""
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
        return gen
    _log(f"  EQ 매칭: 센트로이드 {c_g / c_r * 100:.0f}% → 원본 기울기로 보정")
    k = np.clip(Rs / np.maximum(Gs, 1e-9), 0.5, 4.0)
    spec = np.fft.rfft(gen, axis=1)
    k2 = np.interp(np.linspace(0, 1, spec.shape[1]), np.linspace(0, 1, len(k)), k)
    out = np.fft.irfft(spec * k2, n=gen.shape[1], axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def _rolloff(x, q=0.95):
    m = x.mean(axis=0) if x.ndim == 2 else x
    if m.shape[-1] < 4096:
        return 0.0
    S = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(m.shape[-1], 1.0 / SR)
    c = np.cumsum(S)
    if c[-1] <= 0:
        return 0.0
    return float(f[min(int(np.searchsorted(c, q * c[-1])), len(f) - 1)])


def _separate_4(wav, phase):
    global _demucs
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    if _demucs is None:
        phase("demucs(htdemucs) 로딩")
        _demucs = get_model("htdemucs")
        _demucs.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    phase("4스템 자동 분리 (demucs)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(wav)[None]
    ref = t.mean(0)
    t = (t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        srcs = apply_model(_demucs, t.to(dev), device=dev,
                           split=True, overlap=0.25, progress=False)[0]
    srcs = srcs * (ref.std() + 1e-8) + ref.mean()
    out = {}
    for i, name in enumerate(_demucs.sources):
        st = srcs[i].cpu().numpy().astype(np.float32)
        if st.shape[1] < wav.shape[1]:
            st = np.concatenate(
                [st, np.zeros((2, wav.shape[1] - st.shape[1]), np.float32)],
                axis=1)
        out[name] = np.ascontiguousarray(st[:, :wav.shape[1]])
    return out


# ── SAO-Instruct 모델 ─────────────────────────────────────────────
_LOAD_ERR = {"err": ""}


def _strip_ckpt_paths(node):
    """config 내 pretransform_ckpt_path 참조 제거 — HF 레포에 없는
    vae_model.ckpt 를 열려다 죽는 것을 막는다 (가중치는 model.pt 에 포함)."""
    if isinstance(node, dict):
        node.pop("pretransform_ckpt_path", None)
        for v in node.values():
            _strip_ckpt_paths(v)
    elif isinstance(node, list):
        for v in node:
            _strip_ckpt_paths(v)


def _load_model(phase=lambda p: None) -> bool:
    global _model
    if MOCK or _model is not None:
        return True
    if SAO_DIR and SAO_DIR not in sys.path:
        sys.path.insert(0, SAO_DIR)
    import json as _json
    try:
        import torch
        from model.sao_instruct import SAOInstruct
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        phase("SAO-Instruct 모델 로드 (최초엔 HF 다운로드 수 분)")
        _log(f"모델 로드: {HF_ID}")
        # 수동 조립이 기본 경로다 — from_pretrained 는 게이트 저장소
        # (stabilityai/stable-audio-open-1.0 의 vae_model.ckpt)를 당겨
        # 403 으로 죽는다 (실측 2026-08-04). 그들의 open()(cp949 함정)도
        # 우회해 nn.Module 을 직접 조립한다.
        import torch.nn as _nn
        from huggingface_hub import hf_hub_download
        from stable_audio_tools import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
        import pyloudnorm as _pyln
        cfg_p = hf_hub_download(HF_ID, "config.json")
        ckpt_p = hf_hub_download(HF_ID, "model.pt")
        with open(cfg_p, encoding="utf-8") as f:
            mc = _json.load(f)
        _strip_ckpt_paths(mc)
        m = SAOInstruct.__new__(SAOInstruct)
        _nn.Module.__init__(m)
        m.model = create_model_from_config(mc)
        sd = load_ckpt_state_dict(ckpt_p)
        miss = m.model.load_state_dict(sd, strict=False)
        if miss.missing_keys or miss.unexpected_keys:
            _log(f"  state_dict: missing {len(miss.missing_keys)} / "
                 f"unexpected {len(miss.unexpected_keys)} (strict=False)")
        m.sample_rate = m.model.sample_rate
        m.loudnorm_meter = _pyln.Meter(m.sample_rate)
        m = m.eval().to(dev)
        if not hasattr(m, "device"):
            m.device = dev          # edit_audio 가 self.device 를 참조
        _model = m
        _LOAD_ERR["err"] = ""
        _log("모델 준비 완료")
        return True
    except Exception as e:
        _LOAD_ERR["err"] = (f"{type(e).__name__}: {str(e)[:400]}\n"
                            + traceback.format_exc(limit=8))
        _log(f"모델 로드 실패: {_LOAD_ERR['err']}")
        return False


def _unload_model():
    global _model
    if _model is not None:
        _model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        _log("모델 언로드 — VRAM 해제")


def _edit_clip(wav, instruction, cfg, noise, phase):
    """창(≤45s)을 지시문으로 편집해 같은 길이(2,T)@44.1k 로 반환."""
    if MOCK:
        # 모의: 880Hz 톤을 얹어 '편집됨'을 주파수로 표시 (계약 검증용)
        t = np.arange(wav.shape[1]) / SR
        return (wav * 0.9
                + 0.1 * np.sin(2 * np.pi * 880 * t)[None, :]).astype(np.float32)
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "win.wav")
        sf.write(p, wav.T, SR)
        phase("SAO-Instruct 편집 추론")
        clips = _model.edit_audio(
            instructions=[str(instruction)], audio_path=p,
            encode_audio=True, cfg_scale=float(cfg),
            encoded_audio_noise=float(noise))
    c = clips[0]
    try:
        c = c.detach().cpu().numpy()
    except AttributeError:
        c = np.asarray(c)
    if c.ndim == 1:
        c = np.stack([c, c])
    elif c.shape[0] > 2:
        c = c[:2]
    m_sr = int(getattr(_model, "sample_rate", SR))
    c = _resample(c.astype(np.float32), m_sr, SR)
    if c.shape[1] < wav.shape[1]:
        c = np.concatenate(
            [c, np.zeros((2, wav.shape[1] - c.shape[1]), np.float32)], axis=1)
    return np.ascontiguousarray(c[:, : wav.shape[1]], dtype=np.float32)


# ── 잡 처리 ───────────────────────────────────────────────────────
def _run_job(job):
    def phase(p, prog=None):
        job["phase"] = p
        if prog is not None:
            job["progress"] = prog
        _log(f"잡 {job['id'][:8]}: {p}")

    t0 = time.time()
    try:
        if not _load_model(lambda p: phase(p, 0.03)):
            raise RuntimeError("SAO-Instruct 모델 로드 실패 — sao-server 창 확인")
        phase("오디오 디코드", 0.05)
        mix = _decode_wav(job.pop("audio_b64"))
        dur = mix.shape[1] / SR
        edits = job["edits"]
        total = len(edits)
        for i, e in enumerate(edits):
            f0 = 0.10 + 0.85 * i / total
            tagp = f"{i+1}/{total} {e.get('label') or ''}".strip()
            s, t = float(e["start_s"]), min(float(e["end_s"]), dur)
            if t - s <= 0.2:
                continue
            if t - s > WIN_MAX_S:
                raise ValueError(
                    f"구간 {t - s:.0f}s > 한계 {WIN_MAX_S:.0f}s — SAO 는 긴"
                    " 구간을 못 다룹니다. 구간을 나눠 예약하세요")
            # 창 = 마스크 ± 문맥 (한계 안쪽)
            ctx = min(WIN_CTX_S, max(0.0, (WIN_MAX_S - (t - s)) / 2))
            w0 = max(0.0, s - ctx)
            w1 = min(dur, t + ctx)
            wa, wb = int(w0 * SR), int(w1 * SR)
            win = np.ascontiguousarray(mix[:, wa:wb])
            a_r, b_r = int((s - w0) * SR), int((t - w0) * SR)
            mode_e = str(e.get("mode") or "stem_edit").lower()
            stem_name = str(e.get("stem") or "other").lower()
            strength = min(max(float(e.get("strength") or 0.5), 0.15), 0.9)
            cfg = float(e.get("cfg_scale") or 6.0)
            noise = float(e.get("noise") if e.get("noise") is not None
                          else round(2.0 + strength * 4.0, 1))
            phase(f"{tagp} · {mode_e}({stem_name}) 창 {w0:.1f}~{w1:.1f}s", f0)
            if mode_e == "stem_edit":
                stems = _separate_4(win, lambda p: phase(f"{tagp} · {p}", f0))
                if stem_name not in stems:
                    raise ValueError(f"알 수 없는 스템: {stem_name}")
                st = stems[stem_name]
                new_st = _edit_clip(st, e["prompt"], cfg, noise,
                                    lambda p: phase(f"{tagp} · {p}", f0))
                reg = _match_spectrum(new_st[:, a_r:b_r], st[:, a_r:b_r])
                r_o = float(np.sqrt(np.mean(st[:, a_r:b_r] ** 2)))
                r_m = float(np.sqrt(np.mean(win[:, a_r:b_r] ** 2))) or 1e-4
                r_g = float(np.sqrt(np.mean(reg ** 2)))
                target = r_o if r_o > 0.05 * r_m else 0.4 * r_m
                if r_g > 1e-6:
                    reg = reg * min(target / r_g, 4.0)
                st_full = st.copy()
                st_full[:, a_r:a_r + reg.shape[1]] = reg
                new_st = _splice(st, st_full, s - w0, t - w0)
                new_win = win - st + new_st
                ro_o, ro_g = _rolloff(st[:, a_r:b_r]), _rolloff(reg)
            else:                       # edit — 창 풀믹스 직접 편집(비교 실측용)
                new_win = _edit_clip(win, e["prompt"], cfg, noise,
                                     lambda p: phase(f"{tagp} · {p}", f0))
                reg = _match_spectrum(new_win[:, a_r:b_r], win[:, a_r:b_r])
                new_win = new_win.copy()
                new_win[:, a_r:a_r + reg.shape[1]] = reg
                ro_o, ro_g = _rolloff(win[:, a_r:b_r]), _rolloff(reg)
            job.setdefault("gate", []).append(
                {"label": e.get("label") or "", "stem": stem_name,
                 "rolloff_orig": round(ro_o), "rolloff_gen": round(ro_g)})
            _log(f"  밝기 게이트: 원본 {ro_o:.0f}Hz → 생성 {ro_g:.0f}Hz"
                 + (" ⚠ 어두움" if ro_o > 0 and ro_g < 0.5 * ro_o else ""))
            # 창을 원곡 좌표로 되돌려 마스크 구간만 스플라이스
            gen_full = mix.copy()
            gen_full[:, wa:wa + new_win.shape[1]] = new_win
            mix = _splice(mix, gen_full, s, t)
        peak = float(np.abs(mix).max() or 1.0)
        if peak > 1.0:
            mix = mix / peak * 0.999
        phase("인코딩", 0.95)
        job["result_b64"] = _encode_wav(mix)
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


def _busy():
    return bool(_queue) or any(j["status"] == "running" for j in _jobs.values())


class EditReq(BaseModel):
    audio_b64: str
    edits: list = None
    keep_vocals: bool = True     # 계약 호환 필드 (스템 격리라 자체 무의미)
    vocals_b64: str = None
    seed: int = None


@app.get("/health")
def health():
    cuda = False
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        pass
    return {"status": "ok", "version": VERSION, "cuda": cuda or MOCK,
            "model_loaded": MOCK or _model is not None,
            "load_error": _LOAD_ERR["err"][:2400],
            "max_audio_s": 600.0, "busy": _busy(),
            "modes": ["stem_edit", "edit"],
            "flash_attn": True, "flash_attn_info": "(sao)",
            "engine": "sao-instruct", "hf_id": HF_ID, "mock": MOCK,
            "queue": len(_queue) + sum(1 for j in _jobs.values()
                                       if j["status"] == "running")}


@app.post("/load")
def load():
    if not _load_model():
        return JSONResponse(status_code=503, content={
            "error": "모델 로드 실패", "detail": _LOAD_ERR["err"][:4000]})
    return {"ok": True, "model_loaded": True}


@app.post("/unload")
def unload():
    with _lock:
        if _busy():
            return JSONResponse(status_code=409, content={"error": "busy"})
    _unload_model()
    return {"ok": True, "model_loaded": False}


@app.post("/edit")
def edit(r: EditReq):
    edits = []
    for e in (r.edits or [])[:24]:
        try:
            edits.append({
                "start_s": float(e.get("start_s", e.get("start"))),
                "end_s": float(e.get("end_s", e.get("end"))),
                "prompt": str(e.get("prompt") or "").strip(),
                "mode": str(e.get("mode") or "stem_edit"),
                "stem": e.get("stem"),
                "strength": e.get("strength"),
                "cfg_scale": e.get("cfg_scale"),
                "noise": e.get("noise"),
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
           "created": time.time(), "audio_b64": r.audio_b64, "edits": edits}
    with _lock:
        _jobs[jid] = job
        _queue.append(job)
    _log(f"잡 접수 {jid[:8]} — {len(edits)}건, 첫 지시: {edits[0]['prompt'][:70]}")
    return {"job_id": jid}


@app.get("/jobs/{jid}")
def job_status(jid: str):
    j = _jobs.get(jid)
    if not j:
        return JSONResponse(status_code=404, content={"detail": "job not found"})
    return {k: j.get(k) for k in
            ("id", "status", "phase", "progress", "error", "elapsed_s", "gate")}


@app.get("/jobs/{jid}/result")
def job_result(jid: str):
    j = _jobs.get(jid)
    if not j:
        return JSONResponse(status_code=404, content={"detail": "job not found"})
    if j["status"] != "done":
        return JSONResponse(status_code=409, content={"detail": j["status"]})
    if "result_b64" not in j:
        return JSONResponse(status_code=410, content={"detail": "결과 만료"})
    return {"audio_b64": j["result_b64"], "sr": j["sr"]}


class DiagReq(BaseModel):
    instruction: str = "make the synth sound like an acoustic guitar"
    audio_b64: str = None
    cfg_scale: float = 6.0
    noise: float = 4.0
    stem: str = None             # 지정 시 해당 스템만 분리해 편집 (순수성 실측)


@app.post("/diag")
def diag(r: DiagReq):
    """원격 실측 통로 — 지시·오디오·파라미터를 그대로 모델에 두들긴다."""
    if _busy():
        return JSONResponse(status_code=409, content={"error": "busy"})
    if not _load_model():
        return JSONResponse(status_code=503, content={"error": "모델 로드 실패"})
    t0 = time.time()
    try:
        if r.audio_b64:
            wav = _decode_wav(r.audio_b64)
        else:
            t = np.arange(int(10 * SR)) / SR
            wav = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2).astype(
                np.float32)
        wav = wav[:, : int(WIN_MAX_S * SR)]
        if r.stem:
            stems = _separate_4(wav, lambda p: None)
            wav = stems.get(r.stem.lower())
            if wav is None:
                return JSONResponse(status_code=400,
                                    content={"error": f"스템 없음: {r.stem}"})
        out = _edit_clip(wav, r.instruction, r.cfg_scale, r.noise,
                         lambda p: None)
        return {"ok": True, "elapsed_s": round(time.time() - t0, 1),
                "rolloff_in": round(_rolloff(wav)),
                "rolloff_out": round(_rolloff(out)),
                "audio_b64": _encode_wav(out)}
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
            "trace": traceback.format_exc(limit=3)})


@app.post("/update")
def update():
    """원격 자가 업데이트 — git pull 후 프로세스 종료(래퍼 bat 루프가 새 코드로
    재기동). 맥의 Claude 가 수정 배포를 사람 손 없이 돌리기 위한 통로."""
    if _busy():
        return JSONResponse(status_code=409, content={"error": "busy"})
    import subprocess
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", repo, "pull"], capture_output=True,
                             text=True, timeout=120)
        msg = (out.stdout + out.stderr).strip()[-400:]
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"error": f"{type(e).__name__}: {e}"})
    _log(f"/update: {msg} — 3초 후 재시작")
    threading.Timer(3.0, lambda: os._exit(0)).start()
    return {"ok": True, "git": msg, "restarting": True}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    threading.Thread(target=_worker, daemon=True).start()
    _log(f"SAO-Instruct 서버 {VERSION} — {args.host}:{args.port} "
         f"(레포 {SAO_DIR or '미지정'}, 모델 {HF_ID}"
         f"{', MOCK' if MOCK else ''}) — 모델은 /load 또는 첫 잡에서 로드")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
