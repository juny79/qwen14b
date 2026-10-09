# Qwen2.5 A100 MIG 실습 진행 보고서 v3

- **보고서 버전:** v3 (v1·v2를 보존하는 새 스냅샷)
- **기준 시점:** v3 작성 시점의 작업 공간 파일 및 v2에 기록된 사용자 공유 실행 결과
- **프로젝트:** G-NAHPM-40 A100 MIG 40GB 오픈소스 AI 실습
- **현재 진행:** 0~3단계는 v2에서 실행 완료로 확인됨. 4단계 QLoRA 어댑터 산출물은 작업 공간에서 확인되나 학습 로그와 평가 결과는 확인되지 않음. 5단계 성능 측정은 시작 전.
- **검증 범위:** v1·v2에 적힌 실행 결과는 당시 사용자 공유 내용에 근거한다. v3에서 새로 확인한 사실은 현재 작업 공간에 존재하는 파일과 그 내용에 한정한다. 파일 존재만으로 원격 GPU 인스턴스에서의 실행 성공이나 품질을 추가로 단정하지 않는다.

> v1과 v2는 당시 진행 상태를 보존한 보고서다. 본 보고서도 다음 진행 보고서로 덮어쓰지 말고 `docs/progress_report_v4.md` 등 새 버전으로 추가한다.

## 1. 전체 진행 요약

| 단계 | 내용 | 현재 상태 | 근거 및 비고 |
|---|---|---|---|
| 0 | 환경 점검 및 Python 환경 구성 | 완료 (v2 확인) | A100 MIG 3g.40gb, CUDA 인식 및 BF16 지원 확인 기록 |
| 1 | Qwen2.5-14B 기본 추론 | 완료 (v2 확인) | 4개 과제 및 참고 자동 점검 4/4, peak allocated 27.58 GiB 기록 |
| 2 | vLLM OpenAI 호환 API | 완료 (v2 확인) | vLLM 0.7.3 서버, `/v1/models`, 한국어 completion 성공 기록 |
| 3 | 문서 RAG | 완료 (v2 확인) | 문서 1개 인덱싱, 검색 점검 3/3, 근거 답변·답변 보류 기록 |
| 4 | Qwen2.5-7B QLoRA | 산출물 확인, 실행 세부 및 평가 미확인 | 어댑터 설정·가중치·토크나이저 파일 존재. 로그/평가 결과는 이 작업 공간에서 확인되지 않음 |
| 5 | vLLM API 성능 측정 | 미실행 / 진행 예정 | 벤치마크 코드 작성됨. GPU 상태 확인 후 수행 |
| 6 | VL 이미지 분석 | 미실행 (v2 기준) | 다른 GPU 작업이 없을 때 진행 |

## 2. 4단계 QLoRA 현황

### 작업 공간에서 확인한 산출물

`outputs/qwen25-7b-qlora/`에 다음 파일이 있다.

- `adapter_model.safetensors`: LoRA 어댑터 가중치 파일
- `adapter_config.json`: PEFT LoRA 설정
- 토크나이저 관련 파일: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` 등

어댑터 설정에서 확인한 주요 항목:

| 항목 | 확인값 |
|---|---|
| 베이스 모델 | `Qwen/Qwen2.5-7B-Instruct` |
| 어댑터 형식 | LoRA / `CAUSAL_LM` |
| rank (`r`) | 16 |
| alpha | 32 |
| dropout | 0.05 |
| 대상 모듈 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

`lab4_qlora/train_qlora.py`의 기본 실행 설정은 NF4 4-bit 양자화, BF16 연산, double quantization, 최대 길이 512, 최대 20 steps, batch size 1 및 gradient accumulation 8이다. 학습 데이터는 `lab4_qlora/data/train.jsonl`에 있는 소규모 예제 8개이며, 기본 설정의 20 steps는 파이프라인 점검용이다. 이 설정 정보는 코드에서 확인한 내용이며, 실제 실행 시 동일 옵션을 사용했는지는 기록만으로 확인되지 않는다.

### 확인되지 않은 사항과 해석상 주의

- 학습 실행 로그, 실제 step/loss, 실행 시간 및 GPU 메모리 사용량은 현재 확인된 자료에 없다.
- `lab4_qlora/eval_adapter.py`는 베이스 모델과 어댑터 적용 모델을 3개 프롬프트로 비교하도록 작성되어 있지만, 실행 결과 및 통과 개수는 확인되지 않았다.
- 따라서 어댑터 파일은 존재하지만, 이 보고서에서는 학습의 재현성이나 어댑터 적용 효과·성능 향상을 검증 완료로 표시하지 않는다.
- 이 데이터와 평가 사례는 데모 수준이다. 성능 향상을 주장하려면 학습에 사용하지 않은 별도 테스트셋과 사람 평가가 필요하다.

## 3. 5단계 성능 측정 준비 및 측정 범위

`lab5_bench/bench.py`는 vLLM chat completion API에 비스트리밍 요청을 보내 동시성별 완료 지연과 출력 처리량을 기록하도록 작성되어 있다.

- 기본 동시성: 1, 2, 4
- 기본 반복 수: 각 동시성 수준별 4회
- 워밍업: 1회 수행하며 통계에서 제외
- 기본 생성 상한: 요청당 128 tokens
- 기록 지표: 평균·중앙값·p95·최대 지연, 성공/실패 수, 전체 출력 토큰, aggregate output tokens/s
- 기본 CSV: `logs/bench_results.csv` (기존 파일이 있으면 이어서 기록)
- API 기본 주소/모델: `http://127.0.0.1:8000/v1`, `qwen14b`; API 키는 환경변수 `VLLM_API_KEY`를 우선 사용하고 미설정 시 스크립트 기본값 사용

이 스크립트는 **비스트리밍 완료 지연**을 측정하며 첫 토큰 지연(TTFT), 장시간 안정성, 반복 실험에 따른 분산을 충분히 평가하지 않는다. 기본 요청 수가 적으므로 결과는 실습 환경의 간단한 비교값으로 해석해야 하며, 일반적인 성능 보증으로 해석하지 않는다. 재실행 시 기존 CSV에 결과가 추가되므로 새 측정과 과거 결과를 구분할 필요가 있다.

## 4. 다음 실행 전 확인 사항

1. QLoRA 학습 프로세스가 끝났는지 확인하고, vLLM 또는 다른 GPU 프로세스와 겹치지 않도록 한다. 현재 서버/프로세스가 실행 중인지 여부는 이 보고서에서 확인되지 않았다.
2. `nvidia-smi`로 MIG 장치와 GPU 프로세스 상태를 확인한다. 관리자 할당 MIG 및 `CUDA_VISIBLE_DEVICES`는 변경하지 않는다.
3. `.venv-vllm` 환경에서 vLLM 서버를 시작하고 서버 로그의 기동 완료 메시지를 확인한다. 서버는 기존 원칙대로 `127.0.0.1`에만 바인딩한다.
4. 서버 API 키, served model name, `/v1/models` 응답을 확인하고 단일 요청으로 동작을 점검한 후 성능 측정을 수행한다.
5. 기본 측정 결과를 검토하고 CSV, 서버 로그, 패키지 버전, 실행 옵션을 보존한다. 민감정보와 API 키는 보고서나 저장소에 기록하지 않는다.
6. 측정이 끝나면 성능 결과를 별도 기록하고 서버를 종료해 GPU 자원을 반환한다.

예정된 기본 실행은 다음과 같다.
echo '--- 메모리 구성 ---'
grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SReclaimable|SUnreclaim|Shmem|Dirty|Writeback):' /proc/meminfo

echo '--- 메모리 사용량이 큰 프로세스 ---'
ps -eo pid,comm,rss --sort=-rss | head -15

echo '--- 커널 slab 요약 ---'
slabtop -o 2>/dev/null | head -20 || echo 'slabtop을 사용할 수 없습니다'

echo '--- 컨테이너 메모리 제한 및 사용량 (파일이 있을 때) ---'
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.stat; do
  if [ -r "$f" ]; then echo "### $f"; cat "$f"; fi
donebash lab2_vllm/serve_qwen14b.shbash lab2_vllm/serve_qwen14b.shbash lab2_vllm/serve_qwen14b.shbash lab2_vllm/serve_qwen14b.shbash lab2_vllm/serve_qwen14b.sh
```bash
# 터미널 A: vLLM 서버 실행
bash lab2_vllm/serve_qwen14b.sh

# 터미널 B: 서버 준비를 확인한 후 성능 측정
source .venv-infer/bin/activate
python lab5_bench/bench.py --concurrency 1 2 4
```

실행에 앞서 해당 시점의 `README.md`, 서버 실행 스크립트, GPU 상태를 다시 확인한다. 측정 실패나 미실행을 성공으로 간주하지 않는다.

## 5. 이어받기 참고

- 기존 환경 및 1~3단계의 세부 실행 수치는 `docs/progress_report_v2.md`에 보존되어 있다.
- 생성 파일과 로그는 `.gitignore` 설정상 `outputs/`, `logs/` 등이 Git에서 제외된다. 결과를 공유하거나 백업할 때 어댑터·로그의 용량 및 민감정보를 검토한다.
- 5단계 이후 결과는 본 보고서를 수정하지 않고 새 버전 보고서에 실제 실행 결과와 검증 범위를 함께 기록한다.
