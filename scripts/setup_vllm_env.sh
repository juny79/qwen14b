#!/usr/bin/env bash
# vLLM API 서버 전용 가상환경(.venv-vllm) 생성
# 사용법: bash scripts/setup_vllm_env.sh
# 주의: vLLM 은 호환되는 torch 를 함께 설치하므로 torch 를 따로 설치하지 않습니다.
#       호스트 NVIDIA 드라이버는 변경하지 않습니다.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

VENV=".venv-vllm"

if [ -x "$VENV/bin/python" ] && [ -f "$VENV/bin/activate" ]; then
  echo "[안내] $VENV 가 이미 있습니다. 재사용합니다. (새로 만들려면 rm -rf $VENV 후 다시 실행)"
else
  if [ -d "$VENV" ]; then
    echo "[안내] 불완전한 $VENV 폴더를 발견해 다시 만듭니다."
    rm -rf "$VENV"
  fi
  if ! python3 -m venv "$VENV"; then
    rm -rf "$VENV"
    echo "[실패] 가상환경 생성 실패."
    echo "  Ubuntu/Debian 이라면 python3-venv (또는 python3.X-venv) 패키지가 필요합니다. 관리자에게 설치를 요청하세요."
    exit 1
  fi
  echo "[OK] $VENV 생성"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo
echo "===== 1) vLLM 및 호환 Transformers 설치 ====="
# vLLM 전용 venv에 vLLM을 설치한 뒤, 이번 실습에서 확인한 Transformers 4.x로 고정합니다.
# vLLM 0.7.3이 Transformers 5.x를 자동 선택하면 ProcessorMixin import가 실패할 수 있습니다.
# 호스트 드라이버는 변경하지 않습니다. MIG 인스턴스에서 런타임 검증은 별도로 필요합니다.
pip install vllm==0.7.3
pip install "transformers==4.51.3"

echo
echo "===== 2) 설치 점검 (모델은 로딩하지 않음) ====="
CHECK_FAILED=0
python - <<'PY' || CHECK_FAILED=1
import sys
failed = []
print("python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
    ok = torch.cuda.is_available()
    print("cuda available:", ok)
    if ok:
        p = torch.cuda.get_device_properties(0)
        print(f"device: {p.name} | 메모리 {p.total_memory / 1024**3:.1f} GiB")
    else:
        print("[경고] vLLM 환경의 torch 가 GPU 를 인식하지 못했습니다.")
        print("  - nvidia-smi 의 'CUDA Version' 이 vLLM 휠의 CUDA 빌드보다 낮을 수 있습니다.")
        print("  - 드라이버를 변경하지 말고, 드라이버가 지원하는 CUDA 빌드의 vLLM 버전을 확인하세요.")
        failed.append("cuda")
except Exception as e:
    print(f"[실패] torch: {e}")
    failed.append("torch")

try:
    import vllm
    print(f"[OK] vllm {vllm.__version__}")
except Exception as e:
    print(f"[실패] vllm: {e}")
    failed.append("vllm")

if failed:
    print("[점검 실패 항목]", ", ".join(failed))
    sys.exit(1)
PY

pip freeze > logs/requirements-vllm.txt
echo
echo "패키지 목록 저장: logs/requirements-vllm.txt"
if [ "$CHECK_FAILED" -ne 0 ]; then
  echo "[실패] 설치 점검에서 문제가 발견됐습니다. 위 [경고]/[실패] 메시지를 확인한 뒤 다시 실행하세요."
  exit 1
fi
echo "[완료] $VENV 준비 완료"
echo "활성화: source $VENV/bin/activate"