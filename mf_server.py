"""Music Flamingo 추론 서버 — Windows 11 + NVIDIA GPU 전용.

자율 송캠프(music-production-pipeline) 본체(맥)에서 게이트② A&R 심사를
HTTP로 위임받아, NVIDIA GPU에서 Music Flamingo(8B)로 채점해 돌려준다.

실행:  python mf_server.py            (기본 0.0.0.0:8400)
       python mf_server.py --port 8500

프로토콜 (본체의 HttpCritic 과 계약):
  GET  /health → {"status":"ok","model_loaded":bool,"device":str}
  POST /       → {"mode":"rubric"|"compare","prompt":str,
                  "audio_b64":str[, "audio_b64_b":str]} → 채점 JSON

라이선스: Music Flamingo 는 NVIDIA OneWay Noncommercial — 비상업 용도 전용.
"""
import argparse
import base64
import json
import re
import os
import tempfile
import threading

import uvicorn
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="mf-server", docs_url=None)
_model = _processor = None
# GPU 점유 락 — 심사 요청과 /load·/unload가 겹치지 않게. 심사 중 언로드가
# 들어오면 즉시 409(busy)로 거절한다 (진행 중 잡을 깨뜨리지 않는 것이 계약).
_gpu_lock = threading.Lock()
MODEL_ID = "nvidia/music-flamingo-2601-hf"
MAX_NEW_TOKENS = 448  # 상세 루브릭 (heard A/B/C·첫인상·전개·타깃 청중)


def _load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoProcessor
        try:  # 신형 transformers 전용 클래스 우선
            from transformers import \
                MusicFlamingoForConditionalGeneration as MFClass
        except ImportError:
            from transformers import \
                AudioFlamingo3ForConditionalGeneration as MFClass
        if not torch.cuda.is_available():
            print("⚠ CUDA GPU가 감지되지 않았습니다 — CPU로 돌면 곡당 수십 분이 걸립니다.")
            print("  NVIDIA 드라이버와 CUDA용 PyTorch 설치를 확인하세요 (README 참고).")
        print(f"Music Flamingo 로딩: {MODEL_ID} (최초 실행 시 ~16GB 다운로드)…")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:   # SDPA = 메모리 효율 어텐션 — 전곡(수 분) 오디오의 어텐션이
               # eager로는 시퀀스² 로 폭발한다 (24GB에서 16GB+ 단일 할당 실측)
            _model = MFClass.from_pretrained(
                MODEL_ID, device_map="auto", torch_dtype=dtype,
                attn_implementation="sdpa")
            print("어텐션: sdpa (메모리 효율)")
        except (TypeError, ValueError) as e:
            print(f"⚠ sdpa 미지원({e}) — 기본 어텐션으로 로드 (긴 오디오 OOM 위험)")
            _model = MFClass.from_pretrained(
                MODEL_ID, device_map="auto", torch_dtype=dtype)
        try:
            _model.generation_config.max_length = 16384  # 전곡 구조 분석용
        except Exception:
            pass
        print(f"로딩 완료. device={_model.device}")
    return _model, _processor


def _parse_json(text, fallback=None):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return fallback if fallback is not None else {"raw": text[:500]}


def _extract_rubric(text: str) -> dict:
    """중괄호가 안 닫힌 유사-JSON에서 점수·근거를 정규식으로 복구."""
    out = {}
    for k in ("hook", "production", "structure", "vocal"):
        m = re.search(
            rf'"{k}"\s*:\s*\{{\s*"score"\s*:\s*([0-9.]+)'
            rf'(?:\s*,\s*"evidence"\s*:\s*"([^"]*)")?', text)
        if not m:
            return {}
        out[k] = {"score": float(m.group(1)), "evidence": m.group(2) or ""}
    for k in ("one_line_note", "first_impression", "development",
              "target_audience"):
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)"', text)
        if m:
            out[k] = m.group(1)
    m = re.search(r'"heard"\s*:\s*"([^"]*)"', text)
    if m:
        out["heard"] = m.group(1)
    else:  # 객체형 heard {"A":..,"B":..,"C":..}
        m = re.search(r'"heard"\s*:\s*\{(.*?)\}', text, re.S)
        if m:
            out["heard"] = {kk: vv for kk, vv in
                            re.findall(r'"([ABC])"\s*:\s*"([^"]*)"', m.group(1))}
    if not out.get("heard"):
        out["heard"] = "(JSON 복구 — heard 원문 일부 유실)"
    return out


def _ask(prompt: str, audio_paths: list, gen: dict = None,
         mode: str = "rubric") -> dict:
    import torch
    model, processor = _load()
    content = [{"type": "text", "text": prompt}]
    for p in audio_paths:
        content.append({"type": "audio", "path": p})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    # dtype 주의: float64만 float32로 강등. float32를 bf16으로 내리면 안 됨 —
    # 오디오 인코더 layer_norm 이 float32 입력을 기대해
    # "expected scalar type Float but found BFloat16" 로 죽는다.
    # CUDA에서는 모델이 내부에서 정밀도를 알아서 관리한다.
    for k in list(inputs.keys()):
        v = inputs[k]
        if torch.is_tensor(v) and v.dtype == torch.float64:
            inputs[k] = v.to(torch.float32)
    inputs = inputs.to(model.device)
    # autocast 필수: 모델에 bf16 모듈(conv1d)과 fp32 고정 모듈(layer_norm)이
    # 섞여 있어 입력을 한쪽에 맞추면 반대쪽이 죽는다 — autocast 가 연산별로
    # conv 는 bf16, layer_norm 은 fp32 로 자동 캐스팅해 양쪽을 만족시킨다.
    import contextlib
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if torch.cuda.is_available() else contextlib.nullcontext())
    # greedy(do_sample=False)는 이 모델에서 템플릿 복사·반복 루프에 빠진다
    # ("no lyrics" 무한 반복 실측) → 약한 샘플링 + 반복 페널티가 정답.
    # 결정성은 3회 채점 중앙값이 대신 확보한다.
    gk = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": True,
          "temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.15,
          "no_repeat_ngram_size": 8, "use_cache": True}
    gk.update(gen or {})
    fn = _enforcer(processor.tokenizer, mode)
    if fn is not None:
        gk["prefix_allowed_tokens_fn"] = fn
        gk.pop("no_repeat_ngram_size", None)  # 스키마 강제와 충돌 방지
    with ctx:
        out = model.generate(**inputs, **gk)
    text = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"── 생성 원문 ({len(text)}자): {text[:300]}{'…' if len(text) > 300 else ''}")
    if mode == "text":  # 평문 모드 — 정규화·스키마 강제 없음, 전문 그대로
        return {"text": text}
    if mode == "structure":  # 구조 분석은 루브릭 정규화를 거치지 않음
        out = _parse_json(text, {})
        if isinstance(out, dict):
            out["_raw"] = text[:800]
        return out
    return _normalize(_parse_json(text, {}), text)


_CRIT = ("hook", "production", "structure", "vocal")

# ── JSON 스키마 강제 디코딩 (lm-format-enforcer — 설치 시 자동 활성) ──
# 토큰 선택 단계에서 스키마에 맞는 토큰만 허용 → 포맷 이탈이 원천 불가능.
#   pip install lm-format-enforcer
_C = {"type": "object",
      "properties": {"score": {"type": "number", "minimum": 1, "maximum": 5},
                     "evidence": {"type": "string", "maxLength": 140}},
      "required": ["score", "evidence"]}
RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "heard": {"type": "object",
                  "properties": {k: {"type": "string", "maxLength": 120}
                                 for k in ("A", "B", "C")},
                  "required": ["A", "B", "C"]},
        "hook": _C, "production": _C, "structure": _C, "vocal": _C,
        "first_impression": {"type": "string", "maxLength": 200},
        "development": {"type": "string", "maxLength": 200},
        "one_line_note": {"type": "string", "maxLength": 200},
        "target_audience": {"type": "string", "maxLength": 80},
    },
    "required": ["heard", "hook", "production", "structure", "vocal",
                 "first_impression", "development", "one_line_note",
                 "target_audience"],
}
STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "bpm": {"type": "number"},
        "sections": {"type": "array", "minItems": 3, "maxItems": 20,
                     "items": {"type": "object",
                               "properties": {
                                   "label": {"type": "string",
                                             "enum": ["intro", "verse",
                                                      "pre-chorus", "chorus",
                                                      "bridge", "instrumental",
                                                      "outro"]},
                                   "start": {"type": "number"},
                                   "end": {"type": "number"}},
                               "required": ["label", "start", "end"]}},
    },
    "required": ["bpm", "sections"],
}
_ENFORCER = {}


def _enforcer(tokenizer, mode: str):
    """모드별 prefix_allowed_tokens_fn (없으면 None — 자유 생성 + 복구 파서)."""
    if mode in _ENFORCER:
        return _ENFORCER[mode]
    fn = None
    schema = {"rubric": RUBRIC_SCHEMA, "structure": STRUCTURE_SCHEMA}.get(mode)
    if schema is not None:
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import \
                build_transformers_prefix_allowed_tokens_fn
            fn = build_transformers_prefix_allowed_tokens_fn(
                tokenizer, JsonSchemaParser(schema))
            print("✓ JSON 스키마 강제 디코딩 활성 (lm-format-enforcer)")
        except ImportError:
            print("… lm-format-enforcer 미설치 — 자유 생성 + 복구 파서로 동작"
                  " (pip install lm-format-enforcer 권장)")
        except Exception as e:
            print(f"… 스키마 강제 비활성({type(e).__name__}) — 자유 생성으로 동작")
    _ENFORCER[mode] = fn
    return fn


def _normalize(p, text: str) -> dict:
    """모델 출력의 흔한 변형을 스키마로 정규화 + 실패 시 정규식 복구.

    흡수하는 변형: 대문자 키(Hook), scores 중첩({"scores":{...}}),
    점수만 숫자로 온 경우("hook": 4), 마크다운 펜스 안 JSON.
    항상 _raw(원문 일부)를 동봉해 본체에서 실패 원인을 볼 수 있게 한다.
    """
    if not isinstance(p, dict):
        p = {}
    q = {}
    for k, v in p.items():
        q[k.lower().strip() if isinstance(k, str) else k] = v
    if isinstance(q.get("scores"), dict):
        for k, v in q["scores"].items():
            q.setdefault(str(k).lower().strip(), v)
    for k in _CRIT:
        v = q.get(k)
        if isinstance(v, (int, float)) or (isinstance(v, str) and
                                           v.replace(".", "").isdigit()):
            q[k] = {"score": float(v), "evidence": ""}
        elif isinstance(v, dict) and "score" not in v and "value" in v:
            q[k] = {"score": v["value"], "evidence": v.get("evidence", "")}
    ok = all(isinstance(q.get(k), dict) and q[k].get("score") is not None
             for k in _CRIT)
    if not ok:
        rep = _extract_rubric(text)
        if rep:
            rep["_raw"] = text[:1000]
            return rep
    q["_raw"] = text[:1000]
    return q


class Req(BaseModel):
    mode: str
    prompt: str
    audio_b64: str = ""            # 단일 클립 (compare A / 구버전 호환)
    audio_b64_b: str = ""          # compare B
    audio_b64s: list = []          # 다지점 발췌 (rubric — A/B/C 순)
    audio_name: str = "a.mp3"
    gen: dict = {}                 # 생성 파라미터 오버라이드 (temperature 등)


def _enforcer_available() -> bool:
    try:
        import lmformatenforcer  # noqa
        return True
    except ImportError:
        return False


@app.get("/health")
def health():
    import torch
    return {"status": "ok", "version": "v7-arbiter", "model_loaded": _model is not None,
            "max_audio_s": MAX_AUDIO_S,
            "cuda": torch.cuda.is_available(),
            "enforcer": _enforcer_available(),   # False면 스키마 불일치가 잦아진다
            "busy": _gpu_lock.locked(),          # 심사 진행 중 여부 (중재자 참조)
            "device": (str(_model.device) if _model is not None else "unloaded")}


@app.post("/load")
def load_model():
    """모델 예열 — GPU 중재자(맥)가 야간 심사 시작 전에 호출.
    이미 로드돼 있으면 no-op. 로딩은 수십 초~1분 (동기)."""
    if not _gpu_lock.acquire(timeout=2):
        return JSONResponse(status_code=409, content={"error": "busy"})
    try:
        _load()
        return {"ok": True, "model_loaded": True}
    finally:
        _gpu_lock.release()


@app.post("/unload")
def unload_model():
    """모델 언로드 — 트랙 작업실 리터치(SA3)에 GPU를 양보할 때 맥이 호출.
    심사 요청이 진행 중이면 409(busy) — 진행 중 잡은 절대 깨뜨리지 않는다.
    언로드 후에도 심사 요청이 오면 자동으로 다시 로드된다 (정합성 유지)."""
    global _model, _processor
    if not _gpu_lock.acquire(timeout=2):
        return JSONResponse(status_code=409, content={"error": "busy"})
    try:
        if _model is None:
            return {"ok": True, "model_loaded": False}
        _model = _processor = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("모델 언로드 — GPU를 SA3(구간 리터치)에 양보. 다음 심사 요청 시 자동 재로드")
        return {"ok": True, "model_loaded": False}
    finally:
        _gpu_lock.release()


MAX_AUDIO_S = int(os.environ.get("MF_MAX_AUDIO_S", "420"))   # 서버 자기방어 상한


def _clamp_wav(raw: bytes, max_s: int) -> bytes:
    """RIFF/WAVE를 max_s초로 절단 (청크 워커 — ffmpeg의 LIST 등 부가 청크 대응).

    클라이언트(ffmpeg -ac 1 -ar 16000 s16le)가 보내는 어떤 배치든 data 청크를
    찾아 자르고 RIFF/data 크기를 보정한다. 파싱 불가 형식은 그대로 통과.
    """
    import struct
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return raw
    pos, sr, ch, bits, dpos, dlen = 12, 0, 0, 0, None, 0
    try:
        while pos + 8 <= len(raw):
            cid = raw[pos:pos + 4]
            csz = struct.unpack_from("<I", raw, pos + 4)[0]
            if cid == b"fmt " and csz >= 16:
                ch = struct.unpack_from("<H", raw, pos + 8 + 2)[0]
                sr = struct.unpack_from("<I", raw, pos + 8 + 4)[0]
                bits = struct.unpack_from("<H", raw, pos + 8 + 14)[0]
            elif cid == b"data":
                dpos, dlen = pos + 8, min(csz, len(raw) - pos - 8)
                break
            pos += 8 + csz + (csz & 1)
    except struct.error:
        return raw
    if not dpos or not sr or not ch or not bits:
        return raw
    bps = sr * ch * (bits // 8)
    keep = max_s * bps
    if bps <= 0 or dlen <= keep:
        return raw
    end = dpos + keep
    out = bytearray(raw[:end])
    struct.pack_into("<I", out, 4, end - 8)        # RIFF chunk size
    struct.pack_into("<I", out, dpos - 4, keep)    # data chunk size
    print(f"⚠ 오디오 {dlen / bps:.0f}s > 상한 {max_s}s — 앞 {max_s}s로 절단")
    return bytes(out)


@app.post("/")
def handle(r: Req):
    import os
    b64s = r.audio_b64s or [b for b in [r.audio_b64, r.audio_b64_b] if b]
    paths = []
    _gpu_lock.acquire()   # 심사 중 /unload 진입 차단 (unload는 2초 대기 후 409)
    try:
        for b64 in b64s:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.write(_clamp_wav(base64.b64decode(b64), MAX_AUDIO_S))
            f.close()
            paths.append(f.name)
        # 클라이언트가 발췌들을 무음 간격으로 합친 '단일 오디오'를 보낸다
        # (MF 프로세서는 텍스트:오디오 1:1 제약) — 프롬프트에 구조 설명 포함됨
        allowed = {"temperature", "top_p", "repetition_penalty",
                   "no_repeat_ngram_size", "max_new_tokens", "do_sample"}
        try:
            return _ask(r.prompt, paths,
                        {k: v for k, v in (r.gen or {}).items() if k in allowed},
                        mode=r.mode)
        except Exception as e:
            import torch
            oom = isinstance(e, torch.cuda.OutOfMemoryError) or                 "out of memory" in str(e).lower()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if oom:   # 명확한 신호로 반환 — 클라이언트가 더 짧은 오디오로 강등
                print(f"✗ CUDA OOM — 캐시 비움. 클라이언트 강등 유도: {str(e)[:150]}")
                return JSONResponse(status_code=507,
                                    content={"error": "cuda_oom",
                                             "detail": str(e)[:300]})
            raise
    finally:
        _gpu_lock.release()
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   # 요청 간 단편화 누적 방지
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-preload", action="store_true",
                    help="첫 요청 때 모델 로딩 (기본은 기동 시 미리 로딩)")
    args = ap.parse_args()
    # 실행 환경 자가 복구 — uv sync 사고로 torch 가 CPU 빌드가 됐으면
    # CUDA 빌드 재설치 후 자동 재시작 (sa3_server 와 같은 메커니즘 —
    # 동시 기동 시 파일 락으로 SA3 쪽 설치와 직렬화된다)
    try:
        import torch as _t
        _cuda_ok = _t.cuda.is_available()
    except Exception:
        _cuda_ok = False
    if not _cuda_ok:
        print("⚠ CUDA torch 가 아닙니다 — 자동 복구를 시도합니다…")
        try:
            import sys as _sys

            from ensure_flash_attn import ensure_cuda_torch
            if ensure_cuda_torch(print) == "restart":
                print("환경 복구 완료 — 서버를 자동 재시작합니다")
                os.execv(_sys.executable, [_sys.executable] + _sys.argv)
        except SystemExit:
            raise
        except Exception as _e:
            print(f"자동 복구 실행 오류({type(_e).__name__}: {_e}) — CPU로 계속")
    if not args.no_preload:
        _load()
    print(f"mf-server 대기 중 — http://{args.host}:{args.port}  (헬스체크: /health)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
