#!/usr/bin/env bash
# 0단계: A100 MIG 40GB 실습 환경 점검
# 사용법: bash scripts/00_check_env.sh
# 결과: logs/env_report.txt
# 주의: 이 스크립트는 정보만 읽으며 GPU 할당/CUDA_VISIBLE_DEVICES 를 변경하지 않습니다.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs
REPORT="logs/env_report.txt"

section() { echo; echo "===== $1 ====="; }

{
  section "점검 시각"
  date

  section "OS"
  cat /etc/os-release 2>/dev/null || echo "[확인 불가] /etc/os-release"

  section "Python"
  python3 --version || echo "[실패] python3 없음"

  section "CUDA_VISIBLE_DEVICES (읽기 전용)"
  echo "${CUDA_VISIBLE_DEVICES:-<설정되지 않음>}"

  section "nvidia-smi"
  nvidia-smi || echo "[실패] nvidia-smi 실행 불가 - 드라이버/MIG 할당 확인 필요"

  section "GPU / MIG 장치 목록 (nvidia-smi -L)"
  nvidia-smi -L || true

  section "드라이버 정보"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv 2>/dev/null || true
  echo "※ MIG 환경에서는 위 memory.total 이 전체 GPU 용량으로 보일 수 있습니다."
  echo "  실제 할당 메모리는 아래 PyTorch 점검 결과로 확인하세요."

  section "GPU 점유 프로세스"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
    || echo "(조회 불가 - MIG 환경에서는 프로세스 목록이 보이지 않을 수 있음)"

  section "RAM"
  free -h

  section "디스크 (현재 폴더 기준)"
  df -h .
  AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ -z "$AVAIL_GB" ]; then
    echo "[확인 불가] 여유 공간을 읽지 못했습니다 - 위 df 결과를 직접 확인하세요"
  elif [ "$AVAIL_GB" -lt 100 ]; then
    echo "[경고] 여유 공간 ${AVAIL_GB}GB - 여러 모델 저장 시 100GB 이상 권장"
  else
    echo "[OK] 여유 공간 ${AVAIL_GB}GB"
  fi

  section "Hugging Face 캐시 위치"
  HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
  echo "$HF_CACHE"
  du -sh "$HF_CACHE" 2>/dev/null || echo "(아직 캐시 없음)"

  section "인터넷 연결 (huggingface.co)"
  CODE=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 10 https://huggingface.co 2>/dev/null)
  if [ "$CODE" = "200" ]; then
    echo "[OK] HTTP $CODE"
  else
    echo "[경고] HTTP ${CODE:-연결 실패} - 모델 다운로드가 불가할 수 있음 (프록시/방화벽 확인)"
  fi

  section "PyTorch CUDA 인식"
  if [ -x .venv-infer/bin/python ]; then
    .venv-infer/bin/python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[{i}] {p.name} | 메모리 {p.total_memory / 1024**3:.1f} GiB | SM {p.multi_processor_count}")
    print("bf16 지원:", torch.cuda.is_bf16_supported())
    free, total = torch.cuda.mem_get_info()
    print(f"현재 사용 가능 메모리: {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB")
PY
  else
    echo "(.venv-infer 없음 - scripts/setup_infer_env.sh 실행 후 다시 점검하세요)"
  fi
} 2>&1 | tee "$REPORT"

echo
echo "점검 결과 저장: $REPORT"