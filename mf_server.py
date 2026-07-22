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
import tempfile

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="mf-server", docs_url=None)
_model = _processor = None
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
        _model = MFClass.from_pretrained(
            MODEL_ID, device_map="auto",
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available()
                         else torch.float32))
        try:
            _model.generation_config.max_length = 4096
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


def _ask(prompt: str, audio_paths: list) -> dict:
    import torch
    model, processor = _load()
    content = [{"type": "text", "text": prompt}]
    for p in audio_paths:
        content.append({"type": "audio", "path": p})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    dtype = getattr(model, "dtype", None)  # 피처 dtype 정합 (float64 방지)
    if dtype is not None:
        for k in list(inputs.keys()):
            v = inputs[k]
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(dtype)
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False, use_cache=True)
    text = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    parsed = _parse_json(text, {})
    if '"hook"' in text and not (isinstance(parsed, dict) and parsed.get("hook")):
        repaired = _extract_rubric(text)
        if repaired:
            return repaired
    return parsed


class Req(BaseModel):
    mode: str
    prompt: str
    audio_b64: str = ""            # 단일 클립 (compare A / 구버전 호환)
    audio_b64_b: str = ""          # compare B
    audio_b64s: list = []          # 다지점 발췌 (rubric — A/B/C 순)
    audio_name: str = "a.mp3"


@app.get("/health")
def health():
    import torch
    return {"status": "ok", "model_loaded": _model is not None,
            "cuda": torch.cuda.is_available(),
            "device": (str(_model.device) if _model is not None else "unloaded")}


@app.post("/")
def handle(r: Req):
    import os
    b64s = r.audio_b64s or [b for b in [r.audio_b64, r.audio_b64_b] if b]
    paths = []
    try:
        for b64 in b64s:
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            f.write(base64.b64decode(b64))
            f.close()
            paths.append(f.name)
        # 클라이언트가 발췌들을 무음 간격으로 합친 '단일 오디오'를 보낸다
        # (MF 프로세서는 텍스트:오디오 1:1 제약) — 프롬프트에 구조 설명 포함됨
        return _ask(r.prompt, paths)
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-preload", action="store_true",
                    help="첫 요청 때 모델 로딩 (기본은 기동 시 미리 로딩)")
    args = ap.parse_args()
    if not args.no_preload:
        _load()
    print(f"mf-server 대기 중 — http://{args.host}:{args.port}  (헬스체크: /health)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
