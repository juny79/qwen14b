# Qwen2.5 A100 MIG 실습 진행 보고서 v1

- **보고서 버전:** v1
- **기준 시점:** 2026-10-08, 사용자가 공유한 실행 로그 기준
- **프로젝트:** G-NAHPM-40 A100 MIG 40GB 오픈소스 AI 실습
- **진행 상태:** 환경 구성 및 1단계 기본 추론 완료. vLLM 패키지 설치/import 확인 완료. API 서버 실행은 대기 중.
- **검증 범위:** 본 보고서는 인스턴스에서 사용자가 실행해 공유한 로그와 결과만 실행 확인 사실로 기록한다. 나머지 실습 코드는 아직 해당 인스턴스에서 검증되지 않았다.

## 1. 요약

A100 MIG 3g.40gb 환경에서 추론용 Python 환경을 구성하고 Qwen2.5-14B-Instruct를 BF16으로 실행했다. 요약, 회의록 할 일 추출, JSON 구조 출력, 근거 없는 질문에 대한 답변 보류의 4개 과제가 모두 실행됐으며 참고용 자동 점검도 통과했다. 최대 GPU 할당 메모리는 27.58 GiB, 최대 예약 메모리는 27.61 GiB였다.

별도의 vLLM 환경도 구성했다. vLLM 0.7.3을 import할 때 처음에는 Transformers 5.19.0과의 호환성 오류가 발생했으나 Transformers 4.51.3으로 조정한 뒤 `pip check`와 vLLM import가 통과했다. **vLLM API 서버 기동 및 API 요청은 아직 실행되지 않았다.**

## 2. 확인된 인스턴스

| 항목 | 확인 결과 |
|---|---|
| 운영체제 | Ubuntu 22.04.5 LTS |
| Python | 3.10.14 |
| GPU | NVIDIA A100 80GB PCIe, MIG 활성화 |
| MIG 프로파일 | `3g.40gb`, 42 SM |
| MIG 메모리 | `nvidia-smi`: 40448 MiB, PyTorch: 39.5 GiB |
| NVIDIA 드라이버 | 535.183.06 |
| `nvidia-smi` CUDA 표시 | 12.2 |
| RAM | 96 GiB (점검 당시 available 약 94 GiB) |
| 디스크 여유 | 254 GB |
| Hugging Face 연결 | HTTP 200 |
| 환경 점검 당시 GPU 프로세스 | 없음 |

관리자가 할당한 MIG 장치나 `CUDA_VISIBLE_DEVICES`는 변경하지 않았다.

## 3. 가상환경 및 패키지 상태

### 추론 환경 `.venv-infer`

| 패키지/항목 | 확인 결과 |
|---|---|
| PyTorch | 2.5.1+cu121, CUDA build 12.1 |
| CUDA 인식 | True |
| GPU 메모리 | 39.5 GiB |
| BF16 | 지원 |
| Transformers | 4.51.3 |
| PEFT | 0.15.2 |
| Sentence Transformers | 3.4.1 |
| torchvision | 0.20.1+cu121 |
| `pip check` | `No broken requirements found.` |
| 주요 모듈 import | torch, transformers, peft, sentence_transformers, torchvision, qwen_vl_utils 성공 |

초기에는 Transformers 5.x 및 datasets/huggingface-hub 조합에서 PEFT, Sentence Transformers import 오류가 발생했다. 호환 버전으로 조정한 뒤 import와 의존성 검사를 통과했다.

### vLLM 환경 `.venv-vllm`

| 패키지/항목 | 확인 결과 |
|---|---|
| vLLM | 0.7.3, import 성공 |
| Transformers | 4.51.3 |
| PyTorch | 2.5.1+cu124, CUDA build 12.4 |
| CUDA 인식 | True |
| 장치 | NVIDIA A100 MIG 3g.40gb, 39.5 GiB |

초기 vLLM import는 `ProcessorMixin` import 오류로 실패했다. Transformers를 4.51.3으로 조정한 뒤 `pip check`가 통과하고 `vllm` import가 성공했다. 아직 모델을 vLLM으로 로딩하지 않았으므로 서버 실행 가능성은 미확인이다.

## 4. 1단계: Qwen2.5-14B 기본 추론

- 모델: `Qwen/Qwen2.5-14B-Instruct`
- 정밀도: BF16, 단일 MIG 장치
- 입력/생성: chat template 적용, 결정적 생성 설정
- 실행 과제: 전체 4개 완료
- 결과 파일: `logs/lab1_results.json`

| 과제 | 결과 | 참고 자동 점검 |
|---|---|---|
| 사내 공지 요약 | 숙박비 한도와 신청·정산 기한 요약 | 통과 |
| 회의록 할 일 | 세 담당자의 업무와 기한 추출 | 통과 |
| JSON 출력 | 한도 120000원, 신청 3일, 영수증 7일 값 출력 | 통과 |
| 근거 없는 질문 보류 | 해외 출장 일비 질문에 문서에서 확인 불가 응답 | 통과 |

전체 실행 메모리: 최대 할당 27.58 GiB, 최대 예약 27.61 GiB. 실제 생성 결과를 확인했으며 OOM은 발생하지 않았다.

### 남은 경고 및 주의

실행 시 다음 경고가 남았다.

`Sliding Window Attention is enabled but not implemented for 'eager'; unexpected results may be encountered.`

현재 짧은 실습 입력의 생성은 완료됐으나, 긴 문맥에서의 sliding-window attention 정확성은 검증하지 않았다. 이 경고를 해결했다고 간주하지 않고 긴 입력 실험 전까지 기록해 둔다. 이전 실행에서 나타난 sampling 설정 및 attention mask 관련 경고는 코드 수정 후 공유된 전체 실행 로그에서 더 이상 보이지 않았다.

## 5. 생성된 주요 프로젝트 파일

| 경로 | 목적 | 진행 상태 |
|---|---|---|
| `README.md`, `.gitignore` | 실습 안내, 생성 파일 제외 | 작성 |
| `scripts/00_check_env.sh` | OS/GPU/MIG/디스크/인터넷 점검 | 실행 완료 |
| `scripts/setup_infer_env.sh` | 추론·RAG·QLoRA·VL 환경 설치 | 설치 완료, 호환 버전 조정 |
| `scripts/setup_vllm_env.sh` | vLLM 전용 환경 설치 및 점검 | 설치 완료, Transformers 호환성 조정 반영 |
| `scripts/stop_gpu_jobs.sh` | 본인 실습 GPU 프로세스 종료 보조 | 작성; self-test 결과 미기록 |
| `scripts/save_env.sh` | 재현성 스냅샷 기록 | 작성; 인스턴스 실행 미확인 |
| `lab1_infer/tasks.py` | 4개 기본 추론 과제·참고 검사 | 실행 완료 |
| `lab1_infer/infer_qwen14b.py` | 모델 추론·메모리 기록 | 전체 4개 과제 실행 완료 |
| `lab2_vllm/serve_qwen14b.sh` | vLLM API 서버 실행 | 작성; 서버 미기동 |
| `lab2_vllm/test_api.py` | API 호출 테스트 | 작성; 미실행 |
| `lab3_rag/` | TXT·FAISS 기반 RAG | 작성; 미실행 |
| `lab4_qlora/` | 7B QLoRA 및 비교 평가 | 작성; 미실행 |
| `lab5_bench/bench.py` | API 동시성 성능 측정 | 작성; 미실행 |
| `lab6_vl/vl_receipt.py` | 영수증 이미지 분석 | 작성; 미실행 |

## 6. 다음 단계

1. vLLM API 서버 실행(`lab2_vllm/serve_qwen14b.sh`) 및 로그 확인
2. 새 터미널에서 API 테스트(`lab2_vllm/test_api.py`)
3. 서버를 이용한 문서 RAG 인덱싱·질의·검색 평가
4. vLLM 서버를 종료한 뒤 7B QLoRA 학습·베이스 모델 비교
5. 서버를 재기동해 동시성 1·2·4 성능 측정
6. GPU 작업을 종료한 뒤 VL 이미지 분석

각 GPU 실험은 동시에 실행하지 않는다. 다음 보고서는 본 문서를 덮어쓰지 않고 `docs/progress_report_v2.md`로 추가해, 이후의 실행 결과와 변경 사항을 기록한다.
