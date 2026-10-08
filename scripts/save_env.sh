#!/usr/bin/env bash
# 재현성 기록: 패키지 목록, GPU/MIG 정보, OS 정보를 스냅샷 폴더에 저장합니다.
# 사용법: bash scripts/save_env.sh [메모]
#   예) bash scripts/save_env.sh "lab2 vLLM 성공 직후"
# 결과: logs/snapshot_YYYYmmdd_HHMMSS/
# 주의: 비밀값 유출을 막기 위해 환경 변수 전체(env)는 저장하지 않습니다.

set -u
cd "$(dirname "$0")/.."

TS="$(date +%Y%m%d_%H%M%S)"
OUT="logs/snapshot_${TS}"
mkdir -p "$OUT"

# 메모
echo "${1:-(메모 없음)}" > "$OUT/note.txt"

# 1) 패키지 목록 (가상환경별)
for VENV in .venv-infer .venv-vllm; do
  NAME="${VENV#.venv-}"
  if [ -x "$VENV/bin/python" ]; then
    "$VENV/bin/python" -m pip freeze > "$OUT/requirements-${NAME}.txt" 2>&1 \
      && echo "[OK] $VENV 패키지 목록 저장" \
      || echo "[실패] $VENV pip freeze"
    "$VENV/bin/python" --version > "$OUT/python-${NAME}.txt" 2>&1
  else
    echo "(없음) $VENV 가 없어 건너뜀" | tee "$OUT/requirements-${NAME}.txt"
  fi
done

# 2) GPU / MIG 정보
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi        > "$OUT/gpu-environment.txt" 2>&1
  nvidia-smi -L     > "$OUT/gpu-list.txt" 2>&1
  nvidia-smi -q     > "$OUT/gpu-query.txt" 2>&1
  echo "[OK] nvidia-smi 정보 저장"
else
  echo "[실패] nvidia-smi 없음" | tee "$OUT/gpu-environment.txt"
fi

# 3) PyTorch 가 보는 실제 GPU 메모리 (MIG 할당량 확인용)
if [ -x .venv-infer/bin/python ]; then
  if .venv-infer/bin/python - > "$OUT/torch-gpu.txt" 2>&1 <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} | 메모리 {p.total_memory / 1024**3:.1f} GiB | SM {p.multi_processor_count}")
PY
  then
    echo "[OK] PyTorch GPU 정보 저장"
  else
    echo "[실패] PyTorch GPU 정보 수집 실패 - $OUT/torch-gpu.txt 확인"
  fi
else
  echo "(없음) .venv-infer 가 없어 PyTorch GPU 정보 건너뜀" | tee "$OUT/torch-gpu.txt"
fi

# 4) OS / 재현에 필요한 일부 환경 변수만
{
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo "kernel: $(uname -r)"
  cat /etc/os-release 2>/dev/null
  echo
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<설정되지 않음>}"
  echo "HF_HOME=${HF_HOME:-<설정되지 않음>}"
} > "$OUT/system.txt"

echo
echo "[완료] 스냅샷 저장 위치: $OUT"
ls -1 "$OUT"