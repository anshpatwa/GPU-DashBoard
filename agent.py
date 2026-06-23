"""
Minimal GPU agent (FastAPI) for the GPU Dashboard.

Runs on a GPU server and exposes ONLY the data the dashboard needs:
    GET /api/gpus  ->  {"ok": true, "servers": [{"name", "gpus": [...]}]}

It's the GPU-reading half of main.py with none of the dashboard UI, so the file
stays small. The full dashboard (HTML + multi-server aggregation) lives on the
aggregator (Render); a GPU server only needs to serve its own numbers.

Run:
    pip install fastapi uvicorn nvidia-ml-py
    uvicorn agent:app --host 127.0.0.1 --port 8900
"""
from __future__ import annotations

import shutil
import socket
import subprocess
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="GPU Agent")


# Try NVML first (fast, clean); fall back to parsing nvidia-smi. Same approach as
# main.py, so the agent reports identical numbers.
try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _read_gpus_nvml() -> list[dict]:
    gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        try:
            temp: Optional[int] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            power: Optional[float] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
        except Exception:
            power = None
        mem_used = mem.used / 1024**2
        mem_total = mem.total / 1024**2
        gpus.append(
            {
                "index": i,
                "name": _decode(pynvml.nvmlDeviceGetName(h)),
                "gpu_util": int(util.gpu),
                "mem_used": round(mem_used, 1),
                "mem_total": round(mem_total, 1),
                "mem_util": round(mem_used / mem_total * 100, 1) if mem_total else 0.0,
                "temperature": temp,
                "power": power,
            }
        )
    return gpus


def _to_number(token: str):
    token = token.strip()
    if not token or token.startswith("[") or token.lower() == "n/a":
        return None
    return float(token)


def _read_gpus_smi() -> list[dict]:
    fields = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    gpus = []
    for line in out.splitlines():
        idx, name, util, used, total, temp, power = (c.strip() for c in line.split(","))
        used_n = _to_number(used) or 0.0
        total_n = _to_number(total) or 0.0
        temp_n = _to_number(temp)
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "gpu_util": int(_to_number(util) or 0),
                "mem_used": round(used_n, 1),
                "mem_total": round(total_n, 1),
                "mem_util": round(used_n / total_n * 100, 1) if total_n else 0.0,
                "temperature": int(temp_n) if temp_n is not None else None,
                "power": _to_number(power),
            }
        )
    return gpus


def read_gpus() -> list[dict]:
    if _NVML_OK:
        return _read_gpus_nvml()
    if shutil.which("nvidia-smi"):
        return _read_gpus_smi()
    raise RuntimeError(
        "No NVIDIA GPU access: install `nvidia-ml-py` or ensure nvidia-smi is on PATH."
    )


@app.get("/api/gpus")
def api_gpus():
    """The only route — this machine's GPU stats as JSON, in the dashboard's shape."""
    try:
        return {
            "ok": True,
            "servers": [{"name": socket.gethostname(), "gpus": read_gpus(), "online": True}],
        }
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
