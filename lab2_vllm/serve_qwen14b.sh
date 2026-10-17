#!/usr/bin/env bash
# 2단계: Qwen2.5-14B OpenAI 호환 API 서버
# 터미널 A: bash lab2_vllm/serve_qwen14b.sh
# 터미널 B: python lab2_vllm/test_api.py
# API는 localhost에만 바인딩합니다. 원격 접근은 SSH 터널을 사용하세요.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/load_env.sh
load_project_env .env

if [ ! -x .venv-vllm/bin/vllm ]; then
  echo "[실패] vLLM 환경이 없습니다. 먼저 bash scripts/setup_vllm_env.sh 를 실행하세요."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[실패] nvidia-smi를 찾을 수 없습니다. GPU 할당과 드라이버를 확인하세요."
  exit 1
fi

MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen14b}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-lab-secret-change-me}"

if [ "$API_KEY" = "lab-secret-change-me" ]; then
  echo "[경고] 기본 실습 API 키를 사용합니다. localhost/SSH 터널 실습에만 사용하세요."
fi

export VLLM_API_KEY="$API_KEY"
echo "모델: $MODEL | served name: $SERVED_NAME | port: $PORT"
echo "GPU 점유를 확인한 뒤 서버를 시작합니다. 기존 GPU 학습/추론 프로세스와 동시에 실행하지 마세요."
nvidia-smi

exec .venv-vllm/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --api-key "$VLLM_API_KEY"
