#!/usr/bin/env bash
# 이 실습에서 내가 시작한 GPU 작업(vLLM 서버, 추론/학습/VL 스크립트)을 정상 종료합니다.
# 사용법: bash scripts/stop_gpu_jobs.sh          # 목록 확인 후 y 입력 시 종료
#         bash scripts/stop_gpu_jobs.sh --yes    # 확인 없이 종료
#         bash scripts/stop_gpu_jobs.sh --self-test  # 아무것도 종료하지 않고 대상 패턴만 점검
# 원칙: 내 계정의 실습 프로세스만 대상으로 하며, SIGTERM 만 보냅니다(강제 종료 안 함).

set -u
cd "$(dirname "$0")/.."

# 명령줄이 python(또는 vllm 실행 파일)으로 시작하는 실습 프로세스만 대상으로 합니다.
# vim/tail/grep 등으로 실습 파일을 열어 둔 프로세스는 대상에서 제외됩니다.
PY_PREFIX='^[^ ]*python[0-9.]*( -[^ ]+)*'
PATTERN="^(${PY_PREFIX#^} )?[^ ]*vllm serve( |\$)"
PATTERN="$PATTERN|${PY_PREFIX} -m vllm\.entrypoints"
PATTERN="$PATTERN|${PY_PREFIX} ([^ ]*/)?lab(1_infer|4_qlora|6_vl)/[^ ]*\.py"
ME="$(id -un)"
ASSUME_YES=0
[ "${1:-}" = "--yes" ] && ASSUME_YES=1

# --self-test: 예시 명령줄을 패턴에 대조만 합니다. 프로세스를 조회·종료하지 않습니다.
if [ "${1:-}" = "--self-test" ]; then
  FAIL=0
  check() {  # $1=기대값(match|skip) $2=명령줄
    if printf '%s\n' "$2" | grep -Eq "$PATTERN"; then GOT=match; else GOT=skip; fi
    if [ "$GOT" = "$1" ]; then
      echo "[OK]   $GOT : $2"
    else
      echo "[FAIL] 기대 $1 / 실제 $GOT : $2"
      FAIL=1
    fi
  }
  echo "===== 종료 대상이어야 하는 명령줄 ====="
  check match ".venv-vllm/bin/python3 .venv-vllm/bin/vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000"
  check match ".venv-vllm/bin/vllm serve Qwen/Qwen2.5-14B-Instruct"
  check match "/home/u/qwen/.venv-vllm/bin/python -m vllm.entrypoints.openai.api_server --model x"
  check match "python lab4_qlora/train_qlora.py"
  check match "python -u lab1_infer/infer_qwen14b.py"
  check match ".venv-infer/bin/python /home/u/qwen/lab6_vl/vl_receipt.py --image a.jpg"
  echo
  echo "===== 제외되어야 하는 명령줄 ====="
  check skip "vim lab4_qlora/train_qlora.py"
  check skip "tail -f lab4_qlora/train.log"
  check skip "grep vllm serve README.md"
  check skip "less lab2_vllm/serve_qwen14b.sh"
  check skip "bash scripts/stop_gpu_jobs.sh --yes"
  check skip "python lab3_rag/rag_chat.py --question test"
  check skip "python lab5_bench/bench.py --concurrency 1 2 4"
  echo
  if [ "$FAIL" -eq 0 ]; then
    echo "[OK] 모든 패턴 점검 통과"
  else
    echo "[실패] 기대와 다른 항목이 있습니다. 이 스크립트로 프로세스를 종료하지 말고 결과를 공유해 주세요."
  fi
  exit "$FAIL"
fi

echo "===== GPU 점유 현황 (참고용) ====="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
  || echo "(조회 불가 - MIG 환경에서는 프로세스 목록이 보이지 않을 수 있음)"

echo
echo "===== 이 실습에서 시작한 내 프로세스 (사용자: $ME) ====="
PIDS="$(pgrep -u "$ME" -f "$PATTERN" || true)"

if [ -z "$PIDS" ]; then
  echo "(대상 프로세스 없음)"
  echo
  echo "※ 다른 사용자/다른 작업이 GPU 를 쓰고 있을 수 있습니다. 위 GPU 현황을 확인하세요."
  exit 0
fi

# shellcheck disable=SC2086
ps -o pid,etime,cmd -p $(echo $PIDS | tr ' ' ',')

if [ "$ASSUME_YES" -ne 1 ]; then
  echo
  read -r -p "위 프로세스를 종료할까요? [y/N] " ANS
  case "$ANS" in
    y|Y|yes|YES) ;;
    *) echo "취소했습니다."; exit 0 ;;
  esac
fi

echo
echo "SIGTERM 전송..."
for PID in $PIDS; do
  kill -TERM "$PID" 2>/dev/null && echo "  -> $PID"
done

# 최대 30초 대기
for _ in $(seq 1 30); do
  REMAIN=""
  for PID in $PIDS; do
    kill -0 "$PID" 2>/dev/null && REMAIN="$REMAIN $PID"
  done
  [ -z "$REMAIN" ] && break
  sleep 1
done

echo
if [ -n "$REMAIN" ]; then
  echo "[경고] 30초 후에도 종료되지 않은 프로세스:$REMAIN"
  echo "  상태를 확인한 뒤 필요할 때만 직접 강제 종료하세요: kill -9$REMAIN"
  exit 1
fi

echo "[OK] 대상 프로세스가 모두 종료됐습니다."
echo "GPU 메모리 반환은 몇 초 걸릴 수 있습니다. 다음 작업 전 nvidia-smi 로 확인하세요."