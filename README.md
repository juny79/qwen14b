# A100 MIG 40GB 오픈소스 AI 실습 (Qwen2.5)

> 이 코드는 G-NAHPM-40(A100 MIG 40GB) 실습 가이드(`docs/project_guide.md`)를 바탕으로 작성했습니다.
> **해당 인스턴스에서 실행해 검증한 코드가 아닙니다.** 드라이버, 패키지 버전, MIG 프로파일에 따라 수정이 필요할 수 있습니다.

## 실습 순서

```text
0. 환경 점검 → 1. Qwen 14B 추론 → 2. vLLM API → 3. 문서 RAG → 4. Qwen 7B QLoRA → 5. 성능 측정 → (추가) 6. VL 이미지 분석
```

| 단계 | 폴더 | 사용 환경 | GPU 점유 | 결과물 |
|---|---|---|---|---|
| 0 | `scripts/` | 시스템 | - | `logs/env_report.txt` |
| 1 | `lab1_infer/` | `.venv-infer` | 14B BF16 (~30GB) | 한국어 응답, 최대 메모리 |
| 2 | `lab2_vllm/` | `.venv-vllm` | vLLM 서버 | OpenAI 호환 API |
| 3 | `lab3_rag/` | `.venv-infer` (임베딩은 CPU) | 2단계 서버 사용 | 근거 표시 챗봇 |
| 4 | `lab4_qlora/` | `.venv-infer` | 7B 4bit 학습 | LoRA 어댑터 |
| 5 | `lab5_bench/` | `.venv-infer` | 2단계 서버 사용 | 지연·처리량 CSV |
| 6 | `lab6_vl/` | `.venv-infer` | VL-7B | 영수증 JSON |

## 웹앱 개발 시작점

`apps/ai_studio/`에 로컬 개발용 AI Studio MVP를 추가했습니다. 탭에서 사내 문서 도우미, 이미지 생성, 마스크 이미지 수정을 선택할 수 있습니다. Wan2.2 동영상 기능은 실제 MIG 검증 전까지 UI에서 비활성화되어 있습니다. 설치·실행 방법과 알려진 제한은 [`apps/ai_studio/README.md`](apps/ai_studio/README.md)를 확인하세요.

> 이 웹앱은 개발 프로토타입입니다. SSO, 문서 ACL, 운영 GPU 스케줄링, 데이터 자동 삭제가 없으므로 `127.0.0.1`에서만 실행하고 비민감 시험 자료만 사용하세요. 실제 A100 MIG에서 아직 검증하지 않았습니다.

## 핵심 원칙

1. **GPU 작업은 한 번에 하나만 실행합니다.** 1단계 추론, 2단계 vLLM 서버, 4단계 학습, 6단계 VL 모델을 동시에 실행하지 않습니다.
2. 짧은 입력과 낮은 동시성으로 시작하고, 측정 후 단계적으로 확장합니다.
3. 관리자가 지정한 `CUDA_VISIBLE_DEVICES`와 MIG 할당은 변경하지 않습니다.
4. API는 `127.0.0.1`에만 바인딩하고, 원격 접근은 SSH 터널을 사용합니다.

## 빠른 시작

```bash
# 0. 환경 점검 및 셸 스크립트 문법/종료 패턴 확인
bash scripts/00_check_env.sh
for f in scripts/00_check_env.sh scripts/setup_infer_env.sh scripts/setup_vllm_env.sh scripts/stop_gpu_jobs.sh scripts/save_env.sh; do bash -n "$f" || exit 1; done
bash scripts/stop_gpu_jobs.sh --self-test

# 가상환경 생성 (최초 1회)
bash scripts/setup_infer_env.sh
bash scripts/setup_vllm_env.sh

# 1. 14B 기본 추론
source .venv-infer/bin/activate
python lab1_infer/infer_qwen14b.py
deactivate

# 2. vLLM 서버 (터미널 A)
bash lab2_vllm/serve_qwen14b.sh
# 2. API 테스트 (터미널 B)
source .venv-infer/bin/activate
python lab2_vllm/test_api.py

# 3. RAG (서버 실행 상태에서)
python lab3_rag/build_index.py
python lab3_rag/rag_chat.py --question "국내 출장 숙박비 한도는 얼마인가요?"
python lab3_rag/eval_rag.py

# 4. QLoRA (반드시 vLLM 서버 종료 후)
bash scripts/stop_gpu_jobs.sh
python lab4_qlora/train_qlora.py
python lab4_qlora/eval_adapter.py

# 5. 성능 측정 (학습 프로세스 종료 확인 후 터미널 A에서 vLLM 재실행)
bash lab2_vllm/serve_qwen14b.sh          # 터미널 A
python lab5_bench/bench.py --concurrency 1 2 4   # 터미널 B

# 6. VL 이미지 분석 (vLLM 서버 등 다른 GPU 작업 종료 후)
#    개인정보가 없는 영수증 이미지를 lab6_vl/samples/receipt.jpg 로 직접 넣어 주세요.
bash scripts/stop_gpu_jobs.sh
python lab6_vl/vl_receipt.py --image lab6_vl/samples/receipt.jpg

# 재현성 스냅샷
bash scripts/save_env.sh "실습 완료"
```

## 재현성 기록

```bash
bash scripts/save_env.sh
```