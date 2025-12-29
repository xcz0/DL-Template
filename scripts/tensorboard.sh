#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/tensorboard.sh [--logdir DIR] [--port 6006] [--host 0.0.0.0]

Default logdir selection order:
  1) --logdir
  2) ./saved_models/tutorial5/tensorboards
  3) Latest ./outputs/*/*/tensorboard (if present)

Examples:
  scripts/tensorboard.sh --logdir saved_models/tutorial5/tensorboards
  PORT=6007 scripts/tensorboard.sh
EOF
}

LOGDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --logdir)
      LOGDIR="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --host)
      HOST="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$LOGDIR" ]]; then
  if [[ -d "saved_models/tutorial5/tensorboards" ]]; then
    LOGDIR="saved_models/tutorial5/tensorboards"
  else
    # Fallback: pick newest outputs/*/*/tensorboard if any
    newest=""
    while IFS= read -r -d '' d; do
      newest="$d"
    done < <(find outputs -maxdepth 3 -type d -name tensorboard -print0 2>/dev/null | xargs -0 -I{} stat -c "%Y {}" {} 2>/dev/null | sort -n | awk '{print $2}' | tr '\n' '\0')

    if [[ -n "$newest" ]]; then
      LOGDIR="$newest"
    fi
  fi
fi

if [[ -z "$LOGDIR" ]] || [[ ! -d "$LOGDIR" ]]; then
  echo "No logdir found. Provide --logdir, or download tutorial logs (scripts/download.sh)." >&2
  exit 1
fi

echo "TensorBoard logdir: $LOGDIR"

# Prefer tensorboard CLI if available; fallback to python -m
if command -v tensorboard >/dev/null 2>&1; then
  exec tensorboard --logdir "$LOGDIR" --port "$PORT" --host "$HOST"
else
  exec python -m tensorboard.main --logdir "$LOGDIR" --port "$PORT" --host "$HOST"
fi
