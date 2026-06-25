#!/usr/bin/env bash
#
# One-shot setup for a new GPU server.
#
#   curl -sSL https://raw.githubusercontent.com/AnshPatwa/GPU-DashBoard/main/setup.sh | bash
#
# Or, if the repo is already cloned:
#   bash setup.sh
#
# Idempotent: safe to re-run. Won't double-start the app or tunnel.
# At the end it prints the public trycloudflare.com URL to paste into the
# dashboard (admin token: GGFC).

set -euo pipefail

REPO_URL="https://github.com/AnshPatwa/GPU-DashBoard.git"
REPO_DIR="$HOME/gpu-dashboard"
PORT=8900
APP_LOG="$REPO_DIR/app.log"
TUNNEL_LOG="$REPO_DIR/tunnel.log"
CLOUDFLARED="$HOME/cloudflared"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m   ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[1;33m   ! %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m   ✗ %s\033[0m\n' "$*" >&2; exit 1; }

say "1/6  Checking prerequisites"
command -v python3 >/dev/null || die "python3 not found — try: sudo apt install -y python3 python3-venv"
command -v git     >/dev/null || die "git not found — try: sudo apt install -y git"
command -v wget    >/dev/null || die "wget not found — try: sudo apt install -y wget"
command -v nvidia-smi >/dev/null || warn "nvidia-smi not on PATH — the agent will return 'no GPU access' until NVIDIA drivers are installed"
ok "python3 / git / wget present"

say "2/6  Cloning or updating repo at $REPO_DIR"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
  ok "repo updated"
else
  git clone "$REPO_URL" "$REPO_DIR"
  ok "repo cloned"
fi
cd "$REPO_DIR"

say "3/6  Setting up Python venv + dependencies"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
ok "deps installed"

say "4/6  Downloading cloudflared (if missing)"
if [ ! -x "$CLOUDFLARED" ]; then
  wget -q -O "$CLOUDFLARED" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CLOUDFLARED"
  ok "downloaded $($CLOUDFLARED --version 2>&1 | head -1)"
else
  ok "already present: $($CLOUDFLARED --version 2>&1 | head -1)"
fi

say "5/6  Starting the GPU service on 127.0.0.1:$PORT"
if pgrep -f "uvicorn gpu_fastapi:app.*--port $PORT" >/dev/null; then
  ok "already running (pid $(pgrep -f "uvicorn gpu_fastapi:app.*--port $PORT" | head -1))"
else
  nohup .venv/bin/python -m uvicorn gpu_fastapi:app --host 127.0.0.1 --port "$PORT" \
    > "$APP_LOG" 2>&1 &
  sleep 3
  ok "started (pid $!)"
fi

# verify locally
if ! curl -sf -m 5 "http://127.0.0.1:$PORT/api/gpus" >/dev/null; then
  warn "local /api/gpus did not respond — see $APP_LOG"
fi

say "6/6  Starting cloudflared tunnel -> http://localhost:$PORT"
if pgrep -f "cloudflared tunnel --url http://localhost:$PORT" >/dev/null; then
  ok "tunnel already running (pid $(pgrep -f "cloudflared tunnel --url http://localhost:$PORT" | head -1))"
else
  : > "$TUNNEL_LOG"   # truncate so we read a fresh URL
  nohup "$CLOUDFLARED" tunnel --url "http://localhost:$PORT" > "$TUNNEL_LOG" 2>&1 &
  ok "tunnel started (pid $!) — waiting for public URL..."
fi

# wait up to ~30s for the trycloudflare URL to appear
URL=""
for _ in $(seq 1 15); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 2
done

echo
if [ -z "$URL" ]; then
  warn "Could not detect tunnel URL yet. Last lines of $TUNNEL_LOG:"
  tail -15 "$TUNNEL_LOG"
  exit 1
fi

# final external check
EXT_OK=no
if curl -sf -m 10 "$URL/api/gpus" >/dev/null; then EXT_OK=yes; fi

cat <<EOF

================================================================
  GPU DASHBOARD AGENT IS LIVE
================================================================
  Public URL : $URL
  Local API  : http://127.0.0.1:$PORT/api/gpus
  External OK: $EXT_OK

  NEXT STEP — add this server to the dashboard:
    1. Open  https://gpu-dashboard-96zn.onrender.com
    2. Click '+ Add server'  (top-right)
    3. Server URL  : $URL
       (paste the base URL only — no '/api/gpus' suffix)
    4. Admin token : GGFC
    5. Click Add
================================================================

  Logs:
    app    -> $APP_LOG
    tunnel -> $TUNNEL_LOG

  Note: this URL stays valid only while the cloudflared process
  is alive. After a reboot you'll get a new URL — re-run this
  script and paste the new URL into the dashboard.
EOF
