#!/usr/bin/env bash
# 추론·RAG·QLoRA·VL 실습용 가상환경(.venv-infer) 생성
# 사용법: bash scripts/setup_infer_env.sh
# 주의: 호스트 NVIDIA 드라이버는 변경하지 않습니다.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

VENV=".venv-infer"

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "[실패] python3 venv 모듈이 없습니다. 관리자에게 python3-venv 설치를 요청하세요."
  exit 1
fi

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
echo "===== 1) PyTorch 설치 ====="
# 보고된 환경: NVIDIA 드라이버 535.183.06 / CUDA 12.2.
# cu121 PyTorch 휠을 사용해 설치할 CUDA 런타임을 드라이버 지원 범위에 맞춥니다.
# 이 조합은 인스턴스에서 실행 검증된 것은 아니며, 설치 후 CUDA 점검으로 확인합니다.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo
echo "===== 2) 추론·RAG·학습 패키지 설치 ====="
# Transformers 5.x와 일부 PEFT/Sentence Transformers 버전 사이의 import
# 호환성 문제를 피하기 위해 이번 실습은 Transformers 4.x로 고정합니다.
pip install \
  "transformers==4.51.3" accelerate safetensors sentencepiece \
  "peft==0.15.2" bitsandbytes \
  "datasets==3.6.0" "huggingface-hub==0.30.2" \
  "sentence-transformers==3.4.1" faiss-cpu \
  openai requests httpx numpy

echo
echo "===== 3) VL 실습 패키지 설치 ====="
pip install pillow qwen-vl-utils

echo
echo "===== 4) 설치 점검 ====="
CHECK_FAILED=0
python - <<'PY' || CHECK_FAILED=1
import sys
import torch
failed = []
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} | 메모리 {p.total_memory / 1024**3:.1f} GiB")
    print("bf16 지원:", torch.cuda.is_bf16_supported())
else:
    print("[경고] PyTorch 가 GPU 를 인식하지 못했습니다.")
    print("  - nvidia-smi 의 'CUDA Version' 을 확인하고, 그 이하 CUDA 빌드의 torch 를 설치하세요.")
    print("    예) pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print("  - 드라이버를 임의로 변경하지 마세요.")
    failed.append("cuda")

for name in ["transformers", "accelerate", "peft", "bitsandbytes", "datasets",
             "sentence_transformers", "faiss", "openai", "qwen_vl_utils"]:
    try:
        mod = __import__(name)
        print(f"[OK] {name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"[실패] {name}: {e}")
        failed.append(name)

if failed:
    print("[점검 실패 항목]", ", ".join(failed))
    sys.exit(1)
PY

pip freeze > logs/requirements-infer.txt
echo
echo "패키지 목록 저장: logs/requirements-infer.txt"
if [ "$CHECK_FAILED" -ne 0 ]; then
  echo "[실패] 설치 점검에서 문제가 발견됐습니다. 위 [경고]/[실패] 메시지를 확인한 뒤 다시 실행하세요."
  exit 1
fi
echo "[완료] $VENV 준비 완료"
echo "활성화: source $VENV/bin/activate"