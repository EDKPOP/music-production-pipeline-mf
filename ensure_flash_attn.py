"""flash-attn 자동 복구 — 감지 → 환경 태그 산출 → 맞는 휠 다운로드·설치 → 재검증.

medium 모델의 '지지직' 글리치(= flash-attn 미설치/깨짐, 공식 README Troubleshooting)를
사람 손 없이 고친다. sa3_server.py 가 기동 시 임포트가 깨져 있으면 자동 호출한다.

1) 현재 환경의 파이썬(cpXY) · torch(x.y) · CUDA(cuXYZ) · 플랫폼 태그 산출
2) 후보 릴리스 저장소들의 GitHub API 에서 태그가 정확히 일치하는 휠 검색
   (정확 일치 우선, 같은 CUDA 메이저는 차선 — 어떤 걸 골랐는지 항상 로그)
3) pip --no-deps 로 설치 (torch 등 기존 의존성은 절대 건드리지 않음) → 임포트 재검증

실패해도 서버는 뜬다 — 경고와 수동 절차 안내만 남긴다 (조용한 실패 금지).
주의: `uv sync` 는 자동 실행하지 않는다 — lock 이 torch/CUDA 빌드를 갈아치울 수
있어 위험하다. 사람이 의존성을 갱신할 때만 `uv sync --inexact` 를 쓸 것
(--inexact 가 없으면 pyproject 밖의 flash-attn 이 도로 지워진다).
"""
import json
import re
import subprocess
import sys
import urllib.request

# 순서 = 신뢰 순. mjun0812 은 리눅스 중심이지만 릴리스에 win_amd64 가 섞이면 잡힌다.
REPOS = [
    "mjun0812/flash-attention-prebuild-wheels",
    "kingbri1/flash-attention",       # 윈도우 휠 배포 실적
    "bdashore3/flash-attention",      # 윈도우 휠 배포 실적 (구)
    "Dao-AILab/flash-attention",      # 공식 (리눅스 위주)
]


def _tags():
    import torch
    py = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    tv = torch.__version__.split("+")[0].split(".")
    torch_tag = f"torch{tv[0]}.{tv[1]}"
    cu = (torch.version.cuda or "").replace(".", "")      # "12.6" → "126"
    plat = "win_amd64" if sys.platform == "win32" else "linux_x86_64"
    return py, torch_tag, cu, plat


def _http_json(url, log):
    """GitHub API JSON — requests(certifi) → urllib+certifi → 비검증 폴백.
    일부 파이썬 배포는 루트 인증서가 미연결이라 urlopen 이 SSL 오류를 낸다.
    목록 조회만 최후에 비검증 허용 (휠 '설치'는 pip 가 자체 인증서로 검증)."""
    try:
        import requests
        return requests.get(url, timeout=30,
                            headers={"User-Agent": "songcamp-sa3"}).json()
    except ImportError:
        pass
    req = urllib.request.Request(url, headers={"User-Agent": "songcamp-sa3"})
    try:
        import ssl
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return json.load(r)
    except Exception:
        import ssl
        log("  (인증서 검증 불가 — 목록 조회만 비검증으로 진행)")
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return json.load(r)


def _release_wheels(repo, log):
    rels = _http_json(f"https://api.github.com/repos/{repo}/releases?per_page=15", log)
    out = []
    for rel in rels:
        for a in rel.get("assets", []):
            n = a.get("name", "")
            if n.endswith(".whl"):
                out.append((n, a.get("browser_download_url", "")))
    return out


def _fa_version(name):
    m = re.match(r"flash_attn-(\d+(?:\.\d+)*)", name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)


def _score(name, py, torch_tag, cu, plat):
    """휠 이름 적합도 — None=불가, 숫자 클수록 우선."""
    n = name.lower()
    if plat not in n or py not in n:
        return None
    # torch 태그: 'torch2.7' 은 'torch2.7.0/1' 도 흡수, 'torch2.71' 오인은 (\D) 로 차단
    if not re.search(re.escape(torch_tag) + r"(\D|$)", n):
        return None
    if f"cu{cu}" in n:
        return 2                       # CUDA 정확 일치
    if cu and f"cu{cu[:2]}" in n:      # 같은 메이저 (cu12x) — 대개 호환, 차선
        return 1
    return None


def _pip_install(url, log):
    cmds = [
        [sys.executable, "-m", "pip", "install", "--no-deps", url],
        ["uv", "pip", "install", "--python", sys.executable, url],
    ]
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode == 0:
                return True
            log(f"  설치 실패({cmd[0]}): {(r.stderr or r.stdout)[-300:]}")
        except FileNotFoundError:
            continue
        except Exception as e:
            log(f"  설치 오류({cmd[0]}): {type(e).__name__}: {e}")
    return False


def _import_ok():
    for k in [k for k in sys.modules if k == "flash_attn" or k.startswith("flash_attn.")]:
        del sys.modules[k]             # 깨진 부분 임포트 잔재 제거 후 재시도
    import importlib
    importlib.invalidate_caches()
    import flash_attn
    from flash_attn import flash_attn_func   # noqa: F401 — 실기능까지 검증
    return getattr(flash_attn, "__version__", "?")


def ensure(log=print) -> bool:
    """flash-attn 이 동작하면 True. 깨져 있으면 자동 설치 시도 후 결과 반환."""
    try:
        v = _import_ok()
        log(f"flash-attn OK (v{v})")
        return True
    except Exception as e:
        log(f"flash-attn 손상 감지({type(e).__name__}: {e}) — 자동 복구 시작")
    try:
        py, torch_tag, cu, plat = _tags()
    except Exception as e:
        log(f"  환경 태그 산출 실패({type(e).__name__}) — torch 설치 상태 확인 필요")
        return False
    if not cu:
        log("  CUDA 빌드 torch 가 아닙니다 — flash-attn 휠 복구 불가 (CPU torch?)")
        return False
    log(f"  환경 태그: {py} · {torch_tag} · cu{cu} · {plat}")

    best = None                        # (score, fa_version, name, url)
    for repo in REPOS:
        try:
            wheels = _release_wheels(repo, log)
        except Exception as e:
            log(f"  {repo} 릴리스 조회 실패({type(e).__name__}) — 다음 저장소")
            continue
        for name, url in wheels:
            s = _score(name, py, torch_tag, cu, plat)
            if s is None:
                continue
            key = (s, _fa_version(name))
            if best is None or key > (best[0], best[1]):
                best = (s, _fa_version(name), name, url, repo)
        if best and best[0] == 2:      # 정확 일치를 찾았으면 더 안 뒤진다
            break
    if not best:
        log("  ✗ 맞는 휠을 찾지 못했습니다 — README 'flash-attn' 절의 수동 절차 필요"
            f" (필요 태그: {py}·{torch_tag}·cu{cu}·{plat})")
        return False
    s, _, name, url, repo = best
    log(f"  휠 선택({repo}{', CUDA 메이저 일치' if s == 1 else ''}): {name}")
    if not _pip_install(url, log):
        log("  ✗ 휠 설치 실패 — README 수동 절차를 확인하세요")
        return False
    try:
        v = _import_ok()
        log(f"  ✓ flash-attn v{v} 복구 완료")
        log("  (참고: 이후 의존성 갱신은 `uv sync --inexact` — 아니면 도로 지워짐)")
        return True
    except Exception as e:
        log(f"  설치는 됐지만 임포트 실패({type(e).__name__}: {e}) — "
            "서버 재시작 후 다시 확인하세요")
        return False


if __name__ == "__main__":
    ok = ensure()
    sys.exit(0 if ok else 1)
