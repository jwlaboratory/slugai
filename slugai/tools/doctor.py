"""``slugai doctor``: check a Mac's setup before a job."""

from __future__ import annotations

import platform
import shutil
import socket
import sys
from typing import Optional

from .. import __version__
from ..models import apply_models_dir, model_files, resolve_model_path
from ..net.discovery import DEFAULT_PORT


def _check(ok: Optional[bool], label: str, detail: str = "") -> bool:
    mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
    print(f"[{mark}] {label}{': ' + detail if detail else ''}")
    return ok is not False


def main(args) -> int:
    apply_models_dir(args.models_dir)
    good = True
    print(f"slugai {__version__} doctor on {platform.node()} ({platform.platform()}, {platform.machine()})")
    py = sys.version_info
    good &= _check(py >= (3, 10), "python", f"{py.major}.{py.minor}.{py.micro}")

    is_mac = platform.system() == "Darwin"
    apple = is_mac and platform.machine() == "arm64"
    _check(True if apple else None, "apple silicon", "yes" if apple else "no — hosts need an Apple-silicon Mac for MLX; the fake backend still works")

    try:
        import mlx.core as mx

        dev = str(mx.default_device())
        info = {}
        try:
            info = mx.device_info()
        except Exception:
            try:
                info = mx.metal.device_info()
            except Exception:
                pass
        rec = info.get("max_recommended_working_set_size")
        mem = info.get("memory_size")
        detail = f"mlx {mx.__version__}, device {dev}"
        if mem:
            detail += f", {mem / 1e9:.0f} GB unified, {rec / 1e9:.0f} GB recommended working set"
        good &= _check("gpu" in dev or not apple, "mlx", detail)
    except ImportError:
        _check(False if apple else None, "mlx", "not installed (pip install 'slugai[mlx]')")
        if apple:
            good = False
    try:
        import mlx_lm

        _check(True, "mlx-lm", mlx_lm.__version__)
    except ImportError:
        _check(False if apple else None, "mlx-lm", "not installed")
        if apple:
            good = False

    for mod in ("numpy", "zeroconf", "textual"):
        try:
            m = __import__(mod)
            _check(True, mod, getattr(m, "__version__", ""))
        except ImportError:
            good &= _check(False, mod, "missing")

    from ..net.transport import local_ip

    ip = local_ip()
    _check(ip != "127.0.0.1", "network", f"this Mac is {ip}" if ip != "127.0.0.1" else "no LAN address found")
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", DEFAULT_PORT))
        _check(True, "port", f"{DEFAULT_PORT} is free")
    except OSError:
        _check(None, "port", f"{DEFAULT_PORT} in use (another slugai host? use --port)")
    finally:
        s.close()
    if is_mac:
        _check(True if shutil.which("caffeinate") else None, "caffeinate", "available" if shutil.which("caffeinate") else "not found; the Mac may sleep mid-job")

    if args.model:
        path = resolve_model_path(args.model)
        if path is None:
            good &= _check(False, "model", f"{args.model} not staged; run: slugai prep-model {args.model}")
        else:
            mf = model_files(path)
            ok = mf["config"] and bool(mf["weights"]) and mf["tokenizer"]
            good &= _check(ok, "model", f"{path} — {mf['weight_bytes'] / 1e9:.2f} GB in {len(mf['weights'])} shard(s)" + ("" if ok else " (missing config/weights/tokenizer)"))
            try:
                from ..backends import get_backend

                be = get_backend("auto")
                info = be.model_info(args.model)
                sp = info.spec
                total = sp.num_layers * sp.layer_bytes + sp.embed_bytes + sp.head_bytes
                _check(True, "model spec", f"{sp.num_layers} layers, {total / 1e9:.1f} GB of weights, hidden {sp.hidden_size}")
                free = be.free_memory()
                fits = total <= free
                _check(True if fits else None, "fits on this Mac alone", f"{'yes' if fits else 'no'} ({free / 1e9:.1f} GB free) — " + ("bench single will work" if fits else "needs the ring (or bench streaming)"))
            except Exception as e:
                _check(None, "model spec", f"could not inspect: {e}")
    print("all good" if good else "problems found")
    return 0 if good else 1
