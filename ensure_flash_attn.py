"""SA3 실행 환경 자동 복구 — CUDA torch + flash-attn 을 사람 손 없이 되살린다.

medium 모델의 '지지직' 글리치(= flash-attn 미설치/깨짐)와, `uv sync` 사고로
torch 가 CPU 빌드로 바뀌는 것(윈도우 PyPI torch = CPU 전용)을 모두 복구한다.
sa3_server.py 가 기동 시 자동 호출한다 — run.bat / run_sa3.bat 어느 쪽이든.

순서:
1) torch 가 CPU 빌드면: nvidia-smi 로 드라이버 CUDA 확인 → PyTorch 공식
   CUDA 인덱스(download.pytorch.org/whl/cuXXX)에서 같은 버전 재설치
2) flash-attn 이 없으면: 파이썬(cpXY)·torch(x.y)·CUDA(cuXYZ) 태그로 휠
   저장소들에서 정확히 맞는 휠 선택 → pip --no-deps 설치
3) 뭔가 설치했으면 "restart" 반환 — 서버가 스스로 재시작해 새 환경 적용

torch 정보는 임포트가 아니라 설치 메타데이터에서 읽는다 — 이 프로세스에
이미 로드된 (구)torch 와 무관하게 정확하다.

실패해도 서버는 뜬다 — 경고와 수동 절차 안내만 남긴다 (조용한 실패 금지).
주의: `uv sync` 는 절대 자동 실행하지 않는다 — lock 이 CUDA torch 를 CPU 로
갈아치우는 것이 바로 이 사고의 원인이다. 의존성 갱신은 사람이
`uv sync --inexact` 로만, 그 후 run.bat 이 어긋난 것을 도로 복구한다.
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

# PyTorch 공식 휠 인덱스의 CUDA 태그 후보 — 드라이버가 지원하는 최고부터
CUDA_INDEXES = [130, 129, 128, 126, 124, 121, 118]


def _dist_version(name):
    """설치 메타데이터의 버전 — 임포트 없이, 방금 설치한 것도 정확히 반영."""
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _torch_meta():
    """(기본버전, cu태그) — 예: ("2.7.1", "128") / CPU 빌드면 ("2.7.1", "")."""
    v = _dist_version("torch")
    if not v:
        return None, ""
    base = v.split("+")[0]
    cu = v.split("+cu")[1] if "+cu" in v else ""
    return base, re.sub(r"\D.*$", "", cu)


def _tags():
    py = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    base, cu = _torch_meta()
    tv = (base or "0.0").split(".")
    torch_tag = f"torch{tv[0]}.{tv[1]}"
    plat = "win_amd64" if sys.platform == "win32" else "linux_x86_64"
    return py, torch_tag, cu, plat


def _driver_cuda(log):
    """nvidia-smi 의 'CUDA Version: 12.8' → 128. GPU/드라이버 미검출 시 None."""
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                           timeout=20)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout or "")
        if m:
            return int(m.group(1)) * 10 + int(m.group(2))
    except Exception as e:
        log(f"  nvidia-smi 실행 불가({type(e).__name__})")
    return None


def _fix_cuda_torch(log):
    """CPU torch → 같은 버전의 CUDA 빌드로 재설치. 성공 시 True."""
    base, cu = _torch_meta()
    if base is None:
        log("  torch 자체가 설치돼 있지 않습니다 — README 설치 절차 필요")
        return False
    drv = _driver_cuda(log)
    if not drv:
        log("  NVIDIA 드라이버/GPU 미검출 — CUDA torch 복구 불가")
        return False
    ta = _dist_version("torchaudio")
    log(f"  CPU 빌드 torch v{base} 감지 — CUDA 재설치 시작 "
        f"(드라이버 CUDA {drv // 10}.{drv % 10}, 수 GB 다운로드라 몇 분 걸립니다)")
    pins = [f"torch=={base}"] + ([f"torchaudio=={ta.split('+')[0]}"] if ta else
                                 ["torchaudio"])
    for exact in (True, False):       # 버전 고정 우선, 전 인덱스 실패 시 최신
        for cu_idx in CUDA_INDEXES:
            if cu_idx > drv:
                continue
            idx = f"https://download.pytorch.org/whl/cu{cu_idx}"
            spec = pins if exact else ["torch", "torchaudio"]
            log(f"  시도: cu{cu_idx} 인덱스, {'버전 고정' if exact else '최신'}")
            if _run_install(["--index-url", idx] + spec, log, timeout=3600):
                nb, ncu = _torch_meta()
                if ncu:
                    log(f"  ✓ CUDA torch v{nb}+cu{ncu} 재설치 완료")
                    return True
    log("  ✗ CUDA torch 재설치 실패 — README 수동 절차 필요")
    return False


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


def _run_install(args, log, timeout=1800):
    """pip → uv pip 순으로 설치 시도. args 는 'install' 뒤에 붙는 인자들."""
    cmds = [
        [sys.executable, "-m", "pip", "install"] + args,
        ["uv", "pip", "install", "--python", sys.executable] + args,
    ]
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout)
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


def _find_fa_wheel(py, torch_tag, cu, plat, log):
    best = None                        # (score, fa_version, name, url, repo)
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
    return best


def ensure(log=print):
    """환경이 온전하면 True, 복구 설치를 했으면 "restart"(서버 재시작 필요),
    복구 실패면 False.

    "restart" 인 이유: 이 프로세스엔 이미 (구)torch 가 로드돼 있을 수 있어
    새로 설치한 CUDA torch/flash-attn 은 프로세스를 새로 띄워야 적용된다.
    """
    changed = False

    # ① torch — CPU 빌드(uv sync 사고)면 CUDA 빌드로 재설치
    base, cu = _torch_meta()
    if base is None:
        log("  torch 미설치 — README 설치 절차 필요")
        return False
    if not cu:
        if not _fix_cuda_torch(log):
            return False
        changed = True

    # ② flash-attn — 없거나 깨졌으면 맞는 휠 설치
    fa_broken = _dist_version("flash-attn") is None
    if not fa_broken and not changed:
        try:                           # 설치는 돼 있는데 임포트가 깨진 경우 탐지
            v = _import_ok()
            log(f"flash-attn OK (v{v})")
            return True
        except Exception as e:
            log(f"flash-attn 손상 감지({type(e).__name__}: {e}) — 재설치")
            fa_broken = True
    if fa_broken:
        py, torch_tag, cu, plat = _tags()
        log(f"  flash-attn 휠 탐색 — 환경 태그: {py} · {torch_tag} · cu{cu} · {plat}")
        best = _find_fa_wheel(py, torch_tag, cu, plat, log)
        if not best:
            log("  ✗ 맞는 휠을 찾지 못했습니다 — README 'flash-attn' 절의 수동 "
                f"절차 필요 (필요 태그: {py}·{torch_tag}·cu{cu}·{plat})")
            return "restart" if changed else False
        s, _, name, url, repo = best
        log(f"  휠 선택({repo}{', CUDA 메이저 일치' if s == 1 else ''}): {name}")
        if not _run_install(["--no-deps", url], log):
            log("  ✗ 휠 설치 실패 — README 수동 절차를 확인하세요")
            return "restart" if changed else False
        changed = True

    if changed:
        log("  ✓ 환경 복구 설치 완료 — 새 torch/flash-attn 적용을 위해 재시작 필요")
        log("  (참고: 이후 의존성 갱신은 `uv sync --inexact` — 그냥 sync 하면 "
            "CPU torch 로 도로 돌아갑니다)")
        return "restart"
    return True


if __name__ == "__main__":
    st = ensure()
    print(f"결과: {st}")
    sys.exit(0 if st else 1)
