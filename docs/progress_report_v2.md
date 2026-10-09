# Qwen2.5 A100 MIG 실습 진행 보고서 v2

- **보고서 버전:** v2 (v1을 보존하는 누적 진행 보고서)
- **기준 시점:** 2026-10-09, 사용자가 공유한 실행 결과 기준
- **프로젝트:** G-NAHPM-40 A100 MIG 40GB 오픈소스 AI 실습
- **현재 진행:** 준비 완료, 1단계 추론 완료, vLLM API 완료, 3단계 RAG 완료. QLoRA 전 GPU 서버 종료 확인 대기.
- **검증 범위:** 실행 완료로 표시한 항목은 사용자가 해당 A100 MIG 인스턴스에서 실행해 공유한 결과로 확인했다. 나머지는 미실행 상태다.

> v1은 당시의 진행 상태를 기록한 스냅샷으로 보존한다. 이후 실습 결과는 이 파일에 덮어쓰지 않고 v3 등 새 버전으로 추가한다.

## 1. 진행 요약

| 단계 | 내용 | 상태 | 확인된 결과 |
|---|---|---|---|
| 0 | 환경 점검 및 Python 환경 구성 | 완료 | A100 MIG 3g.40gb, CUDA 인식, BF16 지원 확인 |
| 1 | Qwen2.5-14B 기본 추론 | 완료 | 4개 과제 실행, 참고 검사 4/4, peak allocated 27.58 GiB |
| 2 | vLLM OpenAI 호환 API | 완료 | vLLM 0.7.3 서버 구동, `/v1/models` HTTP 200, 생성 API 한국어 응답 |
| 3 | 문서 RAG | 완료 | 1개 문서 인덱싱, 검색 점검 3/3, 근거 답변 및 답변 보류 확인 |
| 4 | Qwen2.5-7B QLoRA | 대기 | vLLM 서버 종료 및 GPU 반환 확인 후 진행 예정 |
| 5 | API 성능 측정 | 미실행 | 4단계 후 vLLM 재시작해 진행 예정 |
| 6 | VL 이미지 분석 | 미실행 | 별도 GPU 작업이 없을 때 진행 예정 |

## 2. 확인된 환경

| 항목 | 확인 결과 |
|---|---|
| OS / Python | Ubuntu 22.04.5 LTS / Python 3.10.14 |
| GPU | NVIDIA A100 80GB PCIe, MIG `3g.40gb`, 42 SM |
| MIG 메모리 | `nvidia-smi`: 40448 MiB, PyTorch: 39.5 GiB |
| NVIDIA 드라이버 | 535.183.06 (`nvidia-smi` CUDA 표시 12.2) |
| RAM / 디스크 | RAM 96 GiB, 점검 당시 디스크 여유 254 GB |
| Hugging Face 연결 | HTTP 200 |

관리자 할당 MIG와 `CUDA_VISIBLE_DEVICES`는 변경하지 않았다.

## 3. 가상환경 상태

### `.venv-infer`

- PyTorch 2.5.1+cu121, CUDA build 12.1; CUDA 인식 True; BF16 지원 True.
- 확인된 패키지: Transformers 4.51.3, PEFT 0.15.2, Sentence Transformers 3.4.1, torchvision 0.20.1+cu121.
- 주요 모듈 import 성공, `pip check`: `No broken requirements found.`
- 초기 Transformers/datasets/huggingface-hub 호환 오류를 패키지 버전 조정으로 해결했다.

### `.venv-vllm`

- vLLM 0.7.3, Transformers 4.51.3.
- PyTorch 2.5.1+cu124, CUDA build 12.4; CUDA 인식 True; MIG 39.5 GiB 확인.
- 초기 Transformers 5.19.0 환경에서 `ProcessorMixin` import 오류가 있었다. Transformers 4.51.3 설치 후 `pip check`와 `import vllm`이 성공했다.
- 이후 실제 vLLM 모델 로딩, 서버 기동 및 API 요청도 성공했다.

## 4. 1단계: 기본 추론 결과

- 모델: `Qwen/Qwen2.5-14B-Instruct`, BF16, 단일 MIG 장치.
- 결과 파일: `logs/lab1_results.json`.
- 네 과제(문서 요약, 회의록 할 일, JSON, 근거 없는 질문 답변 보류) 실행 및 참고 자동 점검 4/4 통과.
- 최대 할당 메모리 27.58 GiB, 최대 예약 메모리 27.61 GiB, OOM 없음.
- 주의: Transformers가 sliding-window attention과 eager 구현 관련 경고를 출력했다. 짧은 실습 입력은 완료됐지만 긴 컨텍스트 동작은 검증되지 않았다.

## 5. 2단계: vLLM API 결과

- 모델: `Qwen/Qwen2.5-14B-Instruct`; served name `qwen14b`.
- 서버 주소: `http://127.0.0.1:8000`; API 키 인증 사용.
- vLLM 로그: 가중치 27.5681 GB, MIG 총 메모리 39.50 GiB, GPU utilization 예산 35.55 GiB, KV cache 7.45 GiB.
- 로그에 `Application startup complete` 확인.
- `GET /v1/models` 응답: HTTP 200, `qwen14b` 모델 노출.
- `lab2_vllm/test_api.py`: 모델 목록 성공, 한국어 chat completion 성공, 사용량 71 tokens (prompt 49, completion 22).
- 서버는 최초 터미널이 종료된 뒤 내려간 적이 있다. 이후 `nohup`으로 로그를 `logs/vllm_server.log`, PID를 `logs/vllm_server.pid`에 기록해 시작한 뒤 API 테스트에 성공했다.
- **현재 서버 프로세스가 아직 실행 중인지, QLoRA 전에 종료됐는지는 미확인이다.** 다음 GPU 학습 전 종료 스크립트와 `nvidia-smi`로 확인할 것.

## 6. 3단계: RAG 결과

- 임베딩 모델: `intfloat/multilingual-e5-small`, CPU 실행.
- 인덱싱 문서: `lab3_rag/docs/travel_policy.txt` 1개, 1개 청크.
- 인덱스 산출물: `lab3_rag/index/` (Git 제외 대상).
- 검색 회귀 점검: 기대 문서 포함 3/3 통과. cosine 내적 유사도 예: 숙박비 0.910, 출장 신청 0.860, 영수증 기한 0.837.
- 근거가 있는 질문 “국내 출장 숙박비 한도는 얼마인가요?”에 “1박 120,000원”이라고 답하고 `[1]` 인용 표시.
- 문서에 값이 없는 “해외 출장 일비는 얼마인가요?”에 문서에 기재되지 않았다고 답하고 `[1]` 인용 표시.
- 이 검증은 단일 짧은 실습 문서 대상이다. 검색 3/3은 검색 품질의 일반적 보증이나 생성 답변의 전체 정확도 인증이 아니다.

## 7. 주요 실행 파일 및 산출물

| 경로 | 용도 / 현재 상태 |
|---|---|
| `README.md`, `.gitignore` | 실습 안내 및 생성물 제외 규칙 작성 |
| `scripts/00_check_env.sh` | 환경 점검 실행 완료 |
| `scripts/setup_infer_env.sh` | 추론 환경 설치; 호환 패키지 버전 조정 반영 |
| `scripts/setup_vllm_env.sh` | vLLM 환경 설치; Transformers 4.51.3 고정 반영 |
| `scripts/stop_gpu_jobs.sh` | 본인 실습 프로세스 종료 보조; 현재 QLoRA 전에 사용 필요 |
| `scripts/save_env.sh` | 재현성 스냅샷 스크립트 작성; 실행 결과 미공유 |
| `lab1_infer/` | 14B 기본 추론 코드 및 과제; 실행 완료 |
| `lab2_vllm/` | 서버 및 API 테스트; 실행 완료 |
| `lab3_rag/` | 샘플 문서, FAISS 인덱싱, RAG, 검색 점검; 실행 완료 |
| `lab4_qlora/` | 7B QLoRA 코드·예제 데이터·평가 코드; 아직 미실행 |
| `lab5_bench/bench.py` | API 성능 측정 코드; 아직 미실행 |
| `lab6_vl/vl_receipt.py` | 이미지 분석 코드; 아직 미실행 |
| `logs/lab1_results.json` | 1단계 결과 |
| `logs/vllm_server.log`, `logs/vllm_server.pid` | nohup vLLM 서버 로그 및 PID 기록 |

`.gitignore`는 가상환경, `logs/`, `outputs/`, RAG 인덱스, VL 샘플 디렉터리, `.env`를 제외한다. Git 저장소 초기화는 시도 전 `fatal: not a git repository`로 확인됐으며, 현재까지 `git init` 및 GitHub 원격 연결 완료 여부는 확인되지 않았다.

## 8. 다음 단계 및 안전 확인

1. `bash scripts/stop_gpu_jobs.sh`로 vLLM 서버 종료 확인.
2. `nvidia-smi`에서 다른 GPU 프로세스가 없고 MIG 사용량이 반환됐는지 확인.
3. `.venv-infer`에서 7B QLoRA 학습 실행 및 어댑터 평가.
4. 학습 종료 후 vLLM을 재기동하여 동시성 1·2·4 성능 측정.
5. 다른 GPU 작업이 없을 때 VL 이미지 실습.
6. 실행 로그와 필요한 설정을 검토한 뒤 GitHub에 커밋할 항목을 선정한다. API 키, 개인 정보, 영수증 이미지, 대용량 모델/인덱스/로그는 커밋하지 않는다.

GPU 추론과 학습은 동시에 실행하지 않는다. 다음 보고서는 이 파일을 덮어쓰지 않고 `docs/progress_report_v3.md`로 생성한다.
