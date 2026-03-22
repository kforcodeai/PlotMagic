from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import ComplianceEngine


def load_env_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def terminate_process(name: str, process: subprocess.Popen[bytes], timeout_sec: float = 5.0) -> None:
    if process.poll() is not None:
        return
    print(f"[stack] stopping {name} (pid={process.pid})")
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"[stack] force killing {name} (pid={process.pid})")
        process.kill()
        process.wait()


def warm_ingest() -> None:
    engine = ComplianceEngine(ROOT)
    print("[stack] warming indexes: kerala/municipality")
    engine.ingest(state="kerala", jurisdiction_type="municipality")
    print("[stack] warming indexes: kerala/panchayat")
    engine.ingest(state="kerala", jurisdiction_type="panchayat")
    print("[stack] warm ingest complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PlotMagic FastAPI + Streamlit together.")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", default=8000, type=int)
    parser.add_argument("--api-reload", action="store_true", help="Enable uvicorn auto-reload.")
    parser.add_argument("--ui-host", default="0.0.0.0")
    parser.add_argument("--ui-port", default=8501, type=int)
    parser.add_argument("--no-load-env", action="store_true", help="Do not auto-load .env from repo root.")
    parser.add_argument(
        "--warm-ingest",
        action="store_true",
        help="Ingest Kerala municipality and panchayat before starting services.",
    )
    args = parser.parse_args()

    if not args.no_load_env:
        load_env_file(ROOT / ".env", override=False)

    if args.warm_ingest:
        warm_ingest()

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    if args.api_reload:
        api_cmd.append("--reload")

    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.address",
        args.ui_host,
        "--server.port",
        str(args.ui_port),
    ]

    print(f"[stack] starting api: {' '.join(api_cmd)}")
    api_proc = subprocess.Popen(api_cmd, cwd=ROOT, env=os.environ.copy())
    time.sleep(0.8)

    print(f"[stack] starting ui: {' '.join(ui_cmd)}")
    ui_proc = subprocess.Popen(ui_cmd, cwd=ROOT, env=os.environ.copy())

    stopping = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        signame = signal.Signals(signum).name
        print(f"[stack] received {signame}, shutting down...")
        terminate_process("ui", ui_proc)
        terminate_process("api", api_proc)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while True:
            api_rc = api_proc.poll()
            ui_rc = ui_proc.poll()

            if api_rc is not None:
                if stopping:
                    break
                print(f"[stack] api exited with code {api_rc}")
                terminate_process("ui", ui_proc)
                raise SystemExit(api_rc)
            if ui_rc is not None:
                if stopping:
                    break
                print(f"[stack] ui exited with code {ui_rc}")
                terminate_process("api", api_proc)
                raise SystemExit(ui_rc)

            time.sleep(0.4)
    finally:
        terminate_process("ui", ui_proc)
        terminate_process("api", api_proc)


if __name__ == "__main__":
    main()
