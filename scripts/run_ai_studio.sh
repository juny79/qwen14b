#!/usr/bin/env bash
# 로컬 전용 AI Studio를 실행합니다. GPU 작업은 사용자가 별도로 관리하세요.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${AI_STUDIO_HOST:-127.0.0.1}"
PORT="${AI_STUDIO_PORT:-8080}"
if [ "$HOST" != "127.0.0.1" ] && [ "$HOST" != "localhost" ]; then
  echo "[차단] 개발 MVP는 localhost에만 바인딩할 수 있습니다: $HOST" >&2
  exit 2
fi

if [ -x .venv-infer/bin/python ] && .venv-infer/bin/python -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  PYTHON=".venv-infer/bin/python"
elif [ -x .venv-vllm/bin/python ] && .venv-vllm/bin/python -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  PYTHON=".venv-vllm/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen14b}"
echo "AI Studio: http://$HOST:$PORT"
echo "외부 GPU 프로세스/vLLM 작업과의 조정은 수동으로 확인해야 합니다."
exec "$PYTHON" -m uvicorn apps.ai_studio.app:app --host "$HOST" --port "$PORT"
