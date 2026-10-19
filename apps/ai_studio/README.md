# 사내 AI Studio — 로컬 개발 MVP

현재 앱은 다음 기능을 탭으로 제공하는 로컬 개발 MVP입니다.

- 문서 RAG 질문/답변, 근거 표시 및 RAG 인덱스 생성
- vLLM API 연결 확인과 API 기반 성능 측정
- Qwen 기본 추론, QLoRA 학습 및 어댑터 평가
- VL 영수증 JSON 추출
- FLUX.1-schnell 이미지 생성, SDXL 마스크 인페인팅
- Wan2.2 짧은 동영상 생성 worker(경로는 구현되어 있으나, 저장소/체크포인트 설정과 실제 MIG 검증 전에는 사용할 수 없음)

GPU 작업은 이 웹 프로세스 안에서 한 번에 하나씩 실행되고 상태/결과를 제공합니다. 상단의 **GPU 작업 허용** 스위치는 기본적으로 꺼져 있으며, 켜면 앱이 GPU를 사용하는 새 작업을 제출할 수 있습니다. 끄면 새 작업 제출이 차단되고 아직 실행을 시작하지 않은 대기 작업도 시작되지 않습니다. 이미 실행 중인 작업이나 별도 vLLM/다른 프로세스의 GPU 사용을 중단·제어하지는 않습니다. 이 스위치는 GPU를 실제로 감지하거나 CUDA를 활성화하는 설정이 아니므로, 모델 의존성·GPU 가용성은 별도로 확인해야 합니다. vLLM API 서버는 별도로 준비해야 합니다. 개별 작업의 모델 다운로드, CUDA 호환성 및 MIG 실행은 시작 전에 확인해야 합니다.

> **중요:** 이것은 로컬 프로토타입이지 운영용 웹앱이 아닙니다. SSO, 사용자별 ACL, CSRF 방어, 악성 파일 검사, 영속 작업 큐, 다중 프로세스 GPU lease, 데이터 보존/삭제 관리가 없습니다. `127.0.0.1`에서만 실행하고 합성·비민감 자료만 사용하세요. 별도 vLLM 서버와 동시에 GPU를 쓰지 마세요.

## 시작 전

- 할당된 MIG GPU 및 사용 가능 VRAM을 확인합니다. `CUDA_VISIBLE_DEVICES`나 MIG 설정은 변경하지 않습니다.
- 애플리케이션은 승인된 모델 ID만 호출합니다. 첫 이미지 요청에서 Hugging Face 모델이 캐시에 없으면 다운로드가 시작될 수 있으므로, 운영망에서는 사전에 승인된 revision을 별도로 가져와 오프라인 캐시/아티팩트로 관리하세요.
- 이 저장소의 `.venv-infer`가 없거나 패키지 조합을 변경하기 싫다면 별도 가상환경에서 CUDA 호환 PyTorch를 구성한 뒤 `requirements-web.txt`를 설치합니다. requirements는 PyTorch/CUDA wheel을 강제로 선택하지 않습니다.
- `lab3_rag/index/`를 준비하고 OpenAI-compatible vLLM을 `127.0.0.1:8000`에서 실행해야 업무지원 채팅이 동작합니다. 임베딩만 확인하고 vLLM이 없으면 채팅 API는 오류를 반환합니다.

## 인증 및 환경 설정

Hugging Face 접근 토큰은 `.env`에 저장하지 않습니다. 모델 페이지에서 접근 조건에 동의하고 계정 권한을 받은 뒤, 같은 Linux 사용자 계정으로 Hugging Face CLI에 로그인합니다. 토큰은 Hugging Face 사용자 캐시에 저장되며, AI Studio도 같은 사용자로 실행해야 이를 사용할 수 있습니다.

```bash
.venv-vllm/bin/hf auth login
```

앱과 로컬 vLLM의 실행 설정은 프로젝트 루트 `.env`에 둘 수 있습니다. 아래는 예시이며 `VLLM_API_KEY`를 임의의 비밀 값으로 교체하세요. Hugging Face 토큰이나 실제 비밀 키를 문서·채팅에 붙여 넣지 마세요.

```dotenv
AI_STUDIO_HOST=127.0.0.1
AI_STUDIO_PORT=8081
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
SERVED_MODEL_NAME=qwen14b
VLLM_PORT=8000
VLLM_API_KEY=replace-with-a-private-local-secret
QWEN_MODEL=Qwen/Qwen2.5-14B-Instruct
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.90
WAN22_REPO=/absolute/path/to/Wan2.2
WAN22_CKPT=/absolute/path/to/Wan2.2-TI2V-5B
WAN22_PYTHON=/absolute/path/to/Wan2.2/.venv-wan/bin/python
```

`run_ai_studio.sh`가 이 설정을 읽습니다. `WAN22_REPO`는 `generate.py`가 있는 Wan2.2 저장소, `WAN22_CKPT`는 TI2V-5B 체크포인트, `WAN22_PYTHON`은 Wan2.2 의존성(특히 PyTorch와 FlashAttention 2)이 설치된 실행 Python입니다. AI Studio 자체의 Python과 Wan2.2 Python은 달라도 되며, 동영상 worker는 `WAN22_PYTHON`으로 생성기를 실행합니다. 셸에서 명시한 환경변수는 `.env` 값보다 우선합니다. `.env`는 Git에 추가하지 말고 파일 권한도 제한하세요.

```bash
chmod 600 .env
```

`.env`는 간단한 `KEY=value` 형식으로 작성하고, 위 allow-list에 없는 변수나 셸 명령은 로드되지 않습니다. Hugging Face 토큰은 저장소에 추가하지 않습니다.

## 실행

```bash
# 새 터미널에서 실행: 외부 네트워크에 바인딩하지 않음
bash scripts/run_ai_studio.sh
```

브라우저는 `AI_STUDIO_PORT` 설정값(기본값 `8080`)의 localhost 프록시 주소를 엽니다. API 점검은 `/api/health`, `/api/features`에서 합니다. 첫 모델 호출은 가중치 다운로드와 초기 로딩으로 오래 걸릴 수 있습니다.

## 웹앱 기능 사용

`실습 기능` 탭에서 RAG 인덱스 재생성, vLLM API 테스트, Qwen 14B 추론, QLoRA 학습/평가, QLoRA 업무 분류 초안, vLLM 벤치마크, VL 영수증 분석을 작업으로 제출할 수 있습니다. 작업 결과 파일을 브라우저에서 내려받을 수 있습니다. 분류 초안은 저장된 어댑터(`AI_STUDIO_ADAPTER`, 기본 `outputs/qwen25-7b-qlora`)를 사용하며 학습 데이터가 작으므로 반드시 사람이 검토하고 자동 라우팅에 사용하지 마세요. 영수증 분석은 비민감 테스트 이미지로만 수행하세요.

`이미지 생성·수정` 탭은 GPU 작업 허용 스위치가 켜져 있을 때 FLUX.1-schnell (4 steps, CPU offload)과 SDXL Inpainting (1024×1024, 흰색 마스크 영역 수정)을 제출할 수 있습니다. 스위치는 앱 내부의 작업 제출을 통제할 뿐, GPU 프로세스 종료나 외부 작업 조정 기능은 아닙니다.

`동영상` 탭은 Wan2.2 저장소·체크포인트·Python 경로가 설정되어야 활성화됩니다. `.env`에 `WAN22_REPO`, `WAN22_CKPT`, `WAN22_PYTHON`을 설정하고 AI Studio를 재시작하세요. `WAN22_PYTHON`은 최소한 `torch`와 `flash_attn`을 import할 수 있어야 합니다. 앱은 작업 시작 전에 이 두 모듈을 검사하고, Wan2.2 생성기를 지정된 Python으로 실행합니다. 기본 크기는 `1280*704`이며, 첫 로딩은 오래 걸릴 수 있습니다. 다른 GPU 작업을 종료하고 짧은 17프레임 테스트부터 시작하세요. 3g.40gb MIG에서의 실제 성능과 메모리 여유는 별도로 검증해야 합니다. 동영상은 새 T2V/I2V 생성이며 기존 영상 편집 기능은 아닙니다.

RAG 채팅에는 기존 `lab3_rag` 인덱스 및 로컬 vLLM이 필요합니다. 실습 작업에서 vLLM API 테스트/벤치마크를 실행하려면 API 서버가 별도로 실행 중이어야 합니다. QLoRA 학습·평가, HF 추론 및 VL 작업 전에는 vLLM 및 다른 GPU 프로세스를 사용자가 직접 종료하고 GPU가 비어 있는지 확인하세요. 앱 내 대기열은 별도 vLLM 프로세스나 다른 사용자의 GPU 점유를 감지·조정하지 않습니다.

## 주의/다음 단계

- 이 프로세스의 단일 작업 큐는 별도 vLLM, 다른 OS 프로세스, 다른 사용자 GPU 작업과 조율하지 않습니다. 현 MVP에서는 미디어 생성 전 vLLM 등 다른 GPU 작업을 중지해야 합니다.
- 작업 정보는 메모리에만 있고 산출물은 `apps/ai_studio/var/jobs/` 아래 저장됩니다. 서버 재시작 시 상태 조회가 사라집니다. 운영 전 작업 DB, 소유자 검증, 만료/자동 삭제, 백업/복구, queue/lease가 필요합니다.
- 인페인팅용 원본/마스크 파일은 결과 폴더에 남습니다. 프로토타입에서는 민감한 이미지를 업로드하지 말고 실행 후 해당 작업 폴더를 삭제하세요.
- 운영 전 SSO/RBAC 및 문서 ACL, 업로드 형식 검증, 바이러스/콘텐츠 검사, rate limit, 감사 정책, license review, 보안 테스트를 추가해야 합니다.
- `requirements-web.txt`는 개발용 시작점이지 재현 가능한 lockfile이 아닙니다. 실제 GPU 환경에서 CUDA·PyTorch·Transformers·Diffusers의 호환 조합을 고정하고 별도 검증하세요.
