#!/usr/bin/env bash
# Bring up (or tear down) the full rmedia stack — the two vLLM servers, the
# FastAPI backend, and the Bun frontend — health-gated and detached.
#
#   DB=transcripts_v2.lance bash scripts/serve-all.sh up
#   bash scripts/serve-all.sh down
#
# Idempotent: any service already healthy on its port is left running. The two
# vLLM servers are started SEQUENTIALLY (embed fully up before rerank) — starting
# both at once trips vLLM's GPU memory-profiling race. See docs/REPRODUCE.md.
#
# Knobs (env): DB (default transcripts_v2.lance), VLLM_GPU (0), BACKEND_PORT (8000),
# FRONTEND_PORT (5274), LOG_DIR (/tmp).
set -uo pipefail
cd "$(dirname "$0")/.."

DB=${DB:-transcripts_v2.lance}
VLLM_GPU=${VLLM_GPU:-0}
# vLLM ports — overridable (e.g. EMBED_PORT=8011 when :8001 is squatted by
# another dev stack); forwarded to the make targets and the backend env below.
EMBED_PORT=${EMBED_PORT:-8001}
RERANK_PORT=${RERANK_PORT:-8002}
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5274}
LOG_DIR=${LOG_DIR:-/tmp}

log() { printf '  %s\n' "$*"; }

healthy() {  # $1=port $2=path → 0 if HTTP 200
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$1/$2" 2>/dev/null)" = 200 ]
}

wait_healthy() {  # $1=port $2=path $3=label $4=timeout_s
  local i=0
  until healthy "$1" "$2"; do
    i=$((i + 1))
    [ "$i" -ge "${4:-120}" ] && { log "✗ $3 not healthy after ${4:-120}s — see the log"; return 1; }
    sleep 1
  done
  log "✓ $3 healthy (:$1)"
}

pids_on() {  # $1=port → pids listening on it
  ss -ltnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
}

start_detached() {  # $1=name $2=logfile  rest=command string
  local name=$1 logf=$2
  shift 2
  setsid bash -c "$*" >"$logf" 2>&1 </dev/null &
  log "→ started $name (detached) → $logf"
}

up() {
  echo "rmedia stack ↑  (DB=$DB, vLLM GPU=$VLLM_GPU)"

  if healthy "$EMBED_PORT" health; then
    log "• vLLM embed already up (:$EMBED_PORT)"
  else
    start_detached "vLLM embed" "$LOG_DIR/rmedia-embed.log" "VLLM_GPU=$VLLM_GPU EMBED_PORT=$EMBED_PORT make embed-server"
    wait_healthy "$EMBED_PORT" health "vLLM embed" 300 || exit 1
  fi

  if healthy "$RERANK_PORT" health; then
    log "• vLLM rerank already up (:$RERANK_PORT)"
  else
    start_detached "vLLM rerank" "$LOG_DIR/rmedia-rerank.log" "VLLM_GPU=$VLLM_GPU RERANK_PORT=$RERANK_PORT make rerank-server"
    wait_healthy "$RERANK_PORT" health "vLLM rerank" 300 || exit 1
  fi

  if healthy "$BACKEND_PORT" api/health; then
    log "• backend already up (:$BACKEND_PORT)"
  else
    start_detached "backend" "$LOG_DIR/rmedia-backend.log" "DB=$DB BACKEND_PORT=$BACKEND_PORT MEDIA_EMBED_URL=http://127.0.0.1:$EMBED_PORT MEDIA_RERANK_URL=http://127.0.0.1:$RERANK_PORT make backend"
    wait_healthy "$BACKEND_PORT" api/health "backend" 120 || exit 1
  fi

  if healthy "$FRONTEND_PORT" ''; then
    log "• frontend already up (:$FRONTEND_PORT)"
  else
    start_detached "frontend" "$LOG_DIR/rmedia-frontend.log" "BACKEND_PORT=$BACKEND_PORT FRONTEND_PORT=$FRONTEND_PORT make frontend"
    wait_healthy "$FRONTEND_PORT" '' "frontend" 180 || exit 1
  fi

  echo "stack up → http://localhost:$FRONTEND_PORT"
}

down() {
  echo "rmedia stack ↓"
  for p in "$FRONTEND_PORT" "$BACKEND_PORT" "$RERANK_PORT" "$EMBED_PORT"; do
    for pid in $(pids_on "$p"); do
      kill "$pid" 2>/dev/null && log "killed pid $pid (:$p)"
    done
  done
}

case "${1:-up}" in
  up) up ;;
  down) down ;;
  *)
    echo "usage: $0 [up|down]"
    exit 2
    ;;
esac
