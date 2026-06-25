"""
GPU Utilization Dashboard — FastAPI only.

Built to monitor NVIDIA H100 / H200 server nodes (8+ GPUs), but the exact same
file runs on a laptop with a single GPU because NVML / nvidia-smi expose an
identical interface on every NVIDIA card. Test locally, deploy unchanged.

Serves two things from one app:
  GET /            -> live HTML dashboard (auto-refreshes in the browser)
  GET /api/gpus    -> JSON with per-GPU utilization + VRAM stats

Run locally:
    pip install -r requirements.txt
    uvicorn website_gpu_main:app --reload --port 8900
    # open http://127.0.0.1:8900

Run on the H100/H200 server (reachable from your machine):
    uvicorn website_gpu_main:app --host 0.0.0.0 --port 8900
    # open http://<server-ip>:8900   (keep it behind a VPN/firewall)
"""

from __future__ import annotations

import hmac
import json
import os
import shutil
import socket
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import Body, FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="GPU Utilization Dashboard")


# ---------------------------------------------------------------------------
# GPU reading
# ---------------------------------------------------------------------------
# Try NVML first (the C library nvidia-smi itself uses): fast and gives clean
# numbers. If the python binding isn't installed, shell out to nvidia-smi.
# Both paths return the same list-of-dicts shape, so the rest of the app and
# the H100/H200 server don't care which one is active.

try:
    import pynvml  # provided by the `nvidia-ml-py` package

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _decode(value) -> str:
    """nvidia-ml-py returns str on new versions, bytes on old ones."""
    return value.decode() if isinstance(value, bytes) else value


def _read_gpus_nvml() -> list[dict]:
    gpus = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)

        # Temperature / power exist on H100/H200 and most cards, but guard them.
        try:
            temp: Optional[int] = pynvml.nvmlDeviceGetTemperature(
                h, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temp = None
        try:
            power: Optional[float] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
        except Exception:
            power = None

        mem_used = mem.used / 1024**2   # bytes -> MiB
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
    fields = (
        "index,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw"
    )
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
    """Return a list of per-GPU stat dicts, or raise if no GPU tooling exists."""
    if _NVML_OK:
        return _read_gpus_nvml()
    if shutil.which("nvidia-smi"):
        return _read_gpus_smi()
    raise RuntimeError(
        "No NVIDIA GPU access: install `nvidia-ml-py` or ensure nvidia-smi is on PATH."
    )


# ---------------------------------------------------------------------------
# Other servers (optional — for showing several servers on one dashboard)
# ---------------------------------------------------------------------------
# List each OTHER server's base URL (its own running dashboard) in a
# `servers.txt` file next to this file — one URL per line — or in the
# GPU_REMOTES env var (comma-separated). This machine then fetches each one's
# /api/gpus and merges them in. Every remote just runs this same app normally;
# nothing special is needed there.
#
#   servers.txt example:
#       https://brooklyn-consider-precise-helen.trycloudflare.com
#       http://10.0.0.42:8800

_SERVERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers.txt")

# Token that protects the add/remove actions. Viewing is always public; only
# changing the server list needs this. Set it when you launch the app:
#     ADMIN_TOKEN="some-secret" python -m uvicorn website_gpu_main:app ...
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def _normalize_url(raw: str) -> str:
    """Clean a user-entered URL; only http(s) is allowed (blocks file:// etc.)."""
    url = (raw or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url if url.startswith(("http://", "https://")) else ""


def _file_remotes() -> list[str]:
    """The server URLs stored in servers.txt (the ones the UI manages)."""
    if not os.path.exists(_SERVERS_FILE):
        return []
    out = []
    with open(_SERVERS_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line.rstrip("/"))
    return list(dict.fromkeys(out))


def _write_file_remotes(urls: list[str]) -> None:
    with open(_SERVERS_FILE, "w") as fh:
        for u in urls:
            fh.write(u + "\n")


def read_remotes() -> list[str]:
    """All remotes to poll: env GPU_REMOTES + servers.txt, http(s) only."""
    env = [u.strip().rstrip("/") for u in os.environ.get("GPU_REMOTES", "").split(",") if u.strip()]
    urls = [u for u in (env + _file_remotes()) if u.startswith(("http://", "https://"))]
    return list(dict.fromkeys(urls))


def fetch_remote(url: str) -> list[dict]:
    """Pull another server's GPU list. On failure, return it marked offline so
    it still shows on the dashboard instead of silently disappearing. Each entry
    is tagged with `source` (the URL) so the UI can offer a remove button."""
    try:
        req = urllib.request.Request(url + "/api/gpus", headers={"User-Agent": "gpu-dashboard"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        out = [dict(srv, online=True, source=url) for srv in data.get("servers", [])]
        return out or [{"name": url, "gpus": [], "online": False, "error": "no data", "source": url}]
    except Exception as exc:
        return [{"name": url.split("//")[-1], "gpus": [], "online": False, "error": str(exc), "source": url}]


def fetch_all_remotes(urls: list[str]) -> list[dict]:
    if not urls:
        return []
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        for servers in pool.map(fetch_remote, urls):
            out.extend(servers)
    return out


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/gpus")
def api_gpus(demo: int = 0):
    """JSON snapshot the dashboard polls once per second.

    Returns one entry per server: this machine's own GPUs first, then any
    servers listed in servers.txt / GPU_REMOTES. `?demo=N` pads the LOCAL GPU
    list to N entries to preview the multi-GPU super-card layout.
    """
    servers: list[dict] = []
    errors: list[str] = []

    # this machine's own GPUs (skip gracefully if it has none / is a pure hub)
    try:
        gpus = read_gpus()
        simulated = False
        if demo and gpus:
            real_count = len(gpus)
            while len(gpus) < demo:
                clone = dict(gpus[len(gpus) % real_count])
                clone["index"] = len(gpus)
                gpus.append(clone)
            simulated = len(gpus) > real_count
        servers.append(
            {
                "name": socket.gethostname() + (" (demo)" if simulated else ""),
                "gpus": gpus,
                "online": True,
                "local": True,  # this host — the UI won't show a remove button
            }
        )
    except Exception as exc:
        errors.append(f"local: {exc}")

    # any other servers listed in servers.txt / GPU_REMOTES
    servers.extend(fetch_all_remotes(read_remotes()))

    if not servers:
        return JSONResponse(
            {"ok": False, "error": "; ".join(errors) or "no GPUs found"}, status_code=500
        )
    return {"ok": True, "servers": servers}


# ---------------------------------------------------------------------------
# Managing the server list (add / remove) — protected by the admin token
# ---------------------------------------------------------------------------
def _check_admin(token: Optional[str]):
    if not _ADMIN_TOKEN:
        return False, "Adding servers is disabled. Restart the dashboard with an ADMIN_TOKEN set."
    if not token or not hmac.compare_digest(token, _ADMIN_TOKEN):
        return False, "Wrong admin token."
    return True, ""


@app.get("/api/servers")
def list_servers():
    """The remotes currently in servers.txt, and whether management is enabled."""
    return {"remotes": _file_remotes(), "admin_enabled": bool(_ADMIN_TOKEN)}


@app.post("/api/servers")
def add_server(
    payload: dict = Body(...),
    x_admin_token: Optional[str] = Header(default=None),
):
    ok, msg = _check_admin(x_admin_token)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=403)
    url = _normalize_url(payload.get("url", ""))
    if not url:
        return JSONResponse({"ok": False, "error": "Enter a valid http(s) URL."}, status_code=400)
    remotes = _file_remotes()
    if url not in remotes:
        remotes.append(url)
        _write_file_remotes(remotes)
    return {"ok": True, "remotes": _file_remotes()}


@app.post("/api/servers/remove")
def remove_server(
    payload: dict = Body(...),
    x_admin_token: Optional[str] = Header(default=None),
):
    ok, msg = _check_admin(x_admin_token)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=403)
    url = _normalize_url(payload.get("url", ""))
    _write_file_remotes([u for u in _file_remotes() if u != url])
    return {"ok": True, "remotes": _file_remotes()}


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard():
    # no-store so browsers always fetch the latest UI (avoids stale cached layout)
    return HTMLResponse(HTML_PAGE, headers={"Cache-Control": "no-store"})


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Dashboard</title>
<style>
  :root { --text:#eef2f9; --muted:#9aa4b2; --line:rgba(255,255,255,0.10); }
  * { box-sizing: border-box; }
  html, body { overflow-x:hidden; }
  body {
    margin:0; min-height:100vh; color:var(--text);
    font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
    background:
      radial-gradient(1100px circle at 12% 6%, rgba(99,102,241,0.20), transparent 42%),
      radial-gradient(900px circle at 88% 10%, rgba(20,184,166,0.16), transparent 42%),
      radial-gradient(1000px circle at 50% 110%, rgba(168,85,247,0.18), transparent 50%),
      linear-gradient(165deg, #0b0e16 0%, #05060a 60%, #020308 100%);
    background-attachment: fixed;
  }
  /* slow-drifting ambient light behind the glass — the "liquid" */
  body::before, body::after {
    content:""; position:fixed; z-index:-1; border-radius:50%;
    filter:blur(90px); opacity:.5; pointer-events:none;
  }
  body::before { width:520px; height:520px; top:-120px; left:-90px;
    background:radial-gradient(circle, rgba(79,70,229,.55), transparent 70%);
    animation:drift1 19s ease-in-out infinite alternate; }
  body::after { width:560px; height:560px; bottom:-170px; right:-120px;
    background:radial-gradient(circle, rgba(168,85,247,.45), transparent 70%);
    animation:drift2 23s ease-in-out infinite alternate; }
  @keyframes drift1 { from{transform:translate(0,0) scale(1)} to{transform:translate(130px,90px) scale(1.15)} }
  @keyframes drift2 { from{transform:translate(0,0) scale(1)} to{transform:translate(-110px,-70px) scale(1.12)} }

  /* left sidebar — vertical brand, scaffold for future nav */
  .sidebar {
    position:fixed; top:0; left:0; bottom:0; width:124px; z-index:20;
    display:flex; flex-direction:column; align-items:center; gap:22px; padding:26px 0;
    background:rgba(255,255,255,0.045);
    backdrop-filter:blur(24px) saturate(160%); -webkit-backdrop-filter:blur(24px) saturate(160%);
    border-right:1px solid var(--line);
    box-shadow:8px 0 30px rgba(0,0,0,.35);
    overflow:hidden;
    transition:width .3s cubic-bezier(.4,0,.2,1);
  }
  .logo { width:44px; height:44px; border-radius:13px; display:grid; place-items:center;
    background:linear-gradient(145deg, rgba(99,102,241,.65), rgba(168,85,247,.55));
    box-shadow:0 8px 20px rgba(99,102,241,.45), inset 0 1px 0 rgba(255,255,255,.45); }
  .vtitle { writing-mode:vertical-rl; text-orientation:mixed; transform:rotate(180deg);
    flex:1; white-space:nowrap; display:flex; align-items:center; justify-content:center;
    font-size:16px; font-weight:700; letter-spacing:6px; text-transform:uppercase;
    background:linear-gradient(0deg,#ffffff 0%,#8b93ff 55%,#a855f7 100%);
    -webkit-background-clip:text; background-clip:text; color:transparent;
    filter:drop-shadow(0 2px 10px rgba(129,140,248,.35)); }
  .sidefoot { writing-mode:vertical-rl; font-size:11px; letter-spacing:3px; text-transform:uppercase;
    color:var(--muted); display:flex; align-items:center; gap:9px; }
  .sidefoot .dot { width:7px; height:7px; border-radius:50%; background:#3fb950;
    box-shadow:0 0 9px #3fb950; animation:pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .main { margin-left:124px; min-height:100vh; transition:margin .32s; }
  .overlay { display:none; }

  header {
    position:sticky; top:0; z-index:10; display:flex; align-items:center; gap:14px;
    padding:18px 28px; border-bottom:1px solid var(--line);
    background:rgba(255,255,255,0.03);
    backdrop-filter:blur(20px) saturate(150%); -webkit-backdrop-filter:blur(20px) saturate(150%);
  }
  header #status { color:var(--muted); font-size:13px; }
  header .spacer { flex:1; }
  .iconbtn { display:none; font-size:18px; line-height:1; color:var(--text); cursor:pointer;
    width:38px; height:38px; align-items:center; justify-content:center; border-radius:10px;
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.16); backdrop-filter:blur(10px); }

  .grid { padding:24px 28px; }

  /* desktop only: lay server cards side by side so they use the width
     instead of stacking full-width with empty space on the right */
  @media (min-width:821px) {
    .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(430px, 1fr));
            gap:22px; align-items:start; }
    .server { margin:0; }
  }

  /* mobile: sidebar collapses to just the icon square; ☰ expands it */
  @media (max-width:820px) {
    .sidebar { width:120px; transform:translateX(-100%); align-items:center;
      padding:24px 0; gap:22px; z-index:30; box-shadow:8px 0 44px rgba(0,0,0,.6); }
    .main { margin-left:0; }
    .iconbtn { display:inline-flex; }
    body.nav-open .sidebar { transform:translateX(0); }
    body.nav-open .overlay { display:block; position:fixed; inset:0; z-index:20;
      background:rgba(0,0,0,.55); backdrop-filter:blur(2px); }

    /* make the content fit narrow screens — one card per row, tighter spacing */
    header { padding:14px 16px; gap:10px; }
    header #status { font-size:12px; }
    .grid { padding:16px 14px; }
    .server-head { padding:14px 16px; }
    .server-gpus { grid-template-columns:minmax(0,1fr); gap:14px; padding:16px; }
    .card { padding:18px; }
    #addPanel { padding:14px 16px; }
    #addPanel #srvUrl { min-width:0; }
  }

  /* super-card: a faint glass frame around a server's GPUs */
  .server {
    margin:0 0 24px; border:1px solid var(--line); border-radius:22px; overflow:hidden;
    background:rgba(255,255,255,0.025);
    backdrop-filter:blur(14px) saturate(140%); -webkit-backdrop-filter:blur(14px) saturate(140%);
    box-shadow:0 12px 40px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.10);
  }
  .server-head { display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:18px 22px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  .server-head h2 { margin:0; font-size:15px; font-weight:600; }
  .server-head .agg { display:flex; gap:8px; flex-wrap:wrap; }
  .server-gpus { display:grid; gap:18px; padding:22px;
    grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); }

  /* liquid glass GPU card */
  .card {
    position:relative; overflow:hidden; padding:22px; border-radius:20px;
    border:1px solid rgba(255,255,255,0.14);
    background:linear-gradient(150deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03) 60%);
    backdrop-filter:blur(22px) saturate(180%); -webkit-backdrop-filter:blur(22px) saturate(180%);
    box-shadow:0 10px 30px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.25), inset 0 -1px 1px rgba(0,0,0,.25);
    transition:transform .25s ease, box-shadow .25s ease;
  }
  .card::before { content:""; position:absolute; inset:0; z-index:0; pointer-events:none; border-radius:inherit;
    background:linear-gradient(140deg, rgba(255,255,255,0.18), rgba(255,255,255,0) 40%); }
  .card::after { content:""; position:absolute; top:-45%; left:-25%; width:90%; height:90%; z-index:0; pointer-events:none;
    background:radial-gradient(circle, rgba(255,255,255,0.14), transparent 65%); }
  .card > * { position:relative; z-index:1; }
  .card:hover { transform:translateY(-3px);
    box-shadow:0 18px 46px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.32); }
  .card h2 { margin:0 0 3px; font-size:15px; font-weight:600; }

  .sub { color:var(--muted); font-size:12px; }
  .card .sub { margin-bottom:18px; }

  .metric { margin-bottom:16px; }
  .metric .row { display:flex; justify-content:space-between; font-size:13px; margin-bottom:7px; }
  .metric .row b { font-weight:600; }
  .bar { height:11px; border-radius:8px; overflow:hidden; background:rgba(255,255,255,0.07);
    box-shadow:inset 0 1px 2px rgba(0,0,0,.5), inset 0 -1px 0 rgba(255,255,255,.05); }
  .bar > i { display:block; height:100%; width:0; border-radius:8px;
    transition:width .5s cubic-bezier(.4,0,.2,1), background .4s ease; }

  .pills { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
  .pill { font-size:12px; color:var(--muted); border-radius:20px; padding:5px 12px;
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); backdrop-filter:blur(6px); }
  .pill b { color:#fff; }

  #error { padding:16px 30px; color:#ff7b72; font-size:14px; display:none; }

  .btn { font:inherit; font-size:13px; color:var(--text); cursor:pointer; border-radius:10px; padding:8px 16px;
    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.16);
    backdrop-filter:blur(10px); transition:background .2s, border-color .2s; }
  .btn:hover { background:rgba(255,255,255,0.15); border-color:rgba(255,255,255,0.30); }
  #addPanel { display:none; gap:8px; align-items:center; flex-wrap:wrap; padding:16px 30px;
    background:rgba(255,255,255,0.03); border-bottom:1px solid var(--line); backdrop-filter:blur(16px); }
  #addPanel input { font:inherit; font-size:13px; color:var(--text); border-radius:10px; padding:9px 13px;
    background:rgba(0,0,0,0.30); border:1px solid rgba(255,255,255,0.14); }
  #addPanel input:focus { outline:none; border-color:rgba(99,102,241,0.7); }
  #addPanel #srvUrl { flex:1; min-width:280px; }
  #addMsg { font-size:13px; color:var(--muted); }
  .rm { cursor:pointer; color:var(--muted); border-radius:8px; font-size:12px; padding:3px 9px;
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); }
  .rm:hover { color:#ff7b72; border-color:#ff7b72; }
</style>
</head>
<body>
  <aside class="sidebar" id="sidebar">
    <div class="logo" id="logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">
        <rect x="6" y="6" width="12" height="12" rx="2"/>
        <path d="M9 1.5v3M15 1.5v3M9 19.5v3M15 19.5v3M1.5 9h3M1.5 15h3M19.5 9h3M19.5 15h3"/>
      </svg>
    </div>
    <div class="vtitle">GPU&nbsp;Utilization&nbsp;Dashboard</div>
    <div class="sidefoot"><span class="dot"></span>live</div>
  </aside>
  <div class="overlay" id="overlay"></div>
  <main class="main">
    <header>
      <button class="iconbtn" id="navToggle" aria-label="Toggle sidebar">☰</button>
      <span id="status">connecting…</span>
      <span class="spacer"></span>
      <button class="btn" id="addBtn">＋ Add server</button>
    </header>
    <div id="addPanel">
      <input id="srvUrl" placeholder="Server URL  e.g.  https://server-b.trycloudflare.com  or  http://10.0.0.42:8800">
      <input id="srvToken" type="password" placeholder="admin token">
      <button class="btn" id="srvAdd">Add</button>
      <span id="addMsg"></span>
    </div>
    <div id="error"></div>
    <div class="grid" id="grid"></div>
  </main>

<script>
const grid = document.getElementById('grid');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');

// mobile sidebar toggle — keeps the sidebar out of the way on small screens
const navToggle = document.getElementById('navToggle');
const overlay = document.getElementById('overlay');
navToggle.onclick = () => document.body.classList.toggle('nav-open');
overlay.onclick = () => document.body.classList.remove('nav-open');
document.getElementById('logo').onclick = () => document.body.classList.toggle('nav-open');

// colour by how busy something is
function colour(p) {
  if (p < 50) return '#3fb950';   // green
  if (p < 80) return '#d29922';   // amber
  return '#f85149';               // red
}

// show big VRAM (H100 80GB, H200 141GB) as GiB, not 144384 MiB
const gib = (mib) => (mib / 1024).toFixed(1);

function metric(label, percent, detail) {
  const c = colour(percent);
  return `
    <div class="metric">
      <div class="row"><span>${label}</span><b>${detail}</b></div>
      <div class="bar"><i style="width:${percent}%;background:linear-gradient(90deg,${c}88,${c});box-shadow:inset 0 1px 0 rgba(255,255,255,.5), 0 0 14px ${c}66"></i></div>
    </div>`;
}

function card(g) {
  const pills = [];
  if (g.temperature != null) pills.push(`<span class="pill">Temp <b>${g.temperature}°C</b></span>`);
  if (g.power != null)       pills.push(`<span class="pill">Power <b>${g.power} W</b></span>`);
  return `
    <div class="card">
      <h2>GPU ${g.index} — ${g.name}</h2>
      <div class="sub">${gib(g.mem_used)} / ${gib(g.mem_total)} GiB VRAM</div>
      ${metric('GPU Utilization', g.gpu_util, g.gpu_util + '%')}
      ${metric('VRAM Utilization', g.mem_util, g.mem_util + '%')}
      <div class="pills">${pills.join('')}</div>
    </div>`;
}

// server-level summary shown in the super card's header
function aggPills(gpus) {
  if (!gpus.length) return '';
  const used  = gpus.reduce((a, g) => a + g.mem_used, 0);
  const total = gpus.reduce((a, g) => a + g.mem_total, 0);
  const avg   = Math.round(gpus.reduce((a, g) => a + g.gpu_util, 0) / gpus.length);
  return `
    <span class="pill"><b>${gpus.length}</b> GPU${gpus.length > 1 ? 's' : ''}</span>
    <span class="pill">Avg load <b>${avg}%</b></span>
    <span class="pill">VRAM <b>${gib(used)} / ${gib(total)} GiB</b></span>`;
}

// the "super card": one server wrapping all its GPU cards
function serverCard(s) {
  const online = s.online !== false;
  const header = online
    ? aggPills(s.gpus)
    : `<span class="pill" style="color:#f85149;border-color:#f85149">offline</span>`;
  const body = online
    ? s.gpus.map(card).join('')
    : `<div class="sub" style="padding:4px 0">No data — ${s.error || 'unreachable'}</div>`;
  // remotes (not this host) can be removed from the dashboard
  const remove = (!s.local && s.source)
    ? `<button class="rm" onclick="removeServer('${s.source}')">✕ remove</button>` : '';
  return `
    <div class="server">
      <div class="server-head">
        <h2>🖥 ${s.name}</h2>
        <div class="agg">${header}${remove}</div>
      </div>
      <div class="server-gpus">${body}</div>
    </div>`;
}

// ----- Add / remove servers (token saved in this browser only) -----
const addBtn = document.getElementById('addBtn');
const addPanel = document.getElementById('addPanel');
const srvUrl = document.getElementById('srvUrl');
const srvToken = document.getElementById('srvToken');
const addMsg = document.getElementById('addMsg');

addBtn.onclick = () => {
  const open = addPanel.style.display === 'flex';
  addPanel.style.display = open ? 'none' : 'flex';
  if (!open) { srvToken.value = localStorage.getItem('gpuAdminToken') || ''; srvUrl.focus(); }
};

async function adminPost(path, url) {
  const token = srvToken.value.trim();
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Admin-Token': token },
    body: JSON.stringify({ url })
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) throw new Error(data.error || ('HTTP ' + res.status));
  localStorage.setItem('gpuAdminToken', token);   // remember token after a success
  return data;
}

document.getElementById('srvAdd').onclick = async () => {
  addMsg.textContent = 'Adding…';
  try {
    await adminPost('/api/servers', srvUrl.value);
    srvUrl.value = '';
    addMsg.textContent = '✓ Added';
    tick();
    setTimeout(() => { addMsg.textContent = ''; }, 2500);
  } catch (e) {
    addMsg.textContent = '✕ ' + e.message;
  }
};

async function removeServer(url) {
  if (!confirm('Remove this server from the dashboard?\\n' + url)) return;
  try { await adminPost('/api/servers/remove', url); tick(); }
  catch (e) { alert('Could not remove: ' + e.message); }
}

// forward ?demo=N from the page URL to the API so you can preview multi-GPU
const demo = new URLSearchParams(location.search).get('demo');
const apiUrl = '/api/gpus' + (demo ? '?demo=' + encodeURIComponent(demo) : '');

async function tick() {
  try {
    const res = await fetch(apiUrl);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);
    errorEl.style.display = 'none';
    grid.innerHTML = data.servers.map(serverCard).join('');
    const gpuCount = data.servers.reduce((a, s) => a + s.gpus.length, 0);
    statusEl.textContent = data.servers.length + ' server(s) · ' + gpuCount +
      ' GPU(s) · live · ' + new Date().toLocaleTimeString();
  } catch (e) {
    errorEl.textContent = 'Error: ' + e.message;
    errorEl.style.display = 'block';
    statusEl.textContent = 'disconnected';
  }
}

tick();
setInterval(tick, 1000);   // refresh once per second
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("website_gpu_main:app", host="0.0.0.0", port=8900, reload=True)
