# 사내 AI Studio — 로컬 개발 MVP

현재 앱은 기능 선택 UI, 기존 `lab3_rag` FAISS 인덱스와 localhost vLLM을 사용하는 문서 Q&A, FLUX.1-schnell 이미지 생성, SDXL 마스크 인페인팅 작업 제출/진행률/결과 다운로드를 제공합니다. Wan2.2는 모델 후보와 기능 선택 UI만 포함하며, 실제 inference worker는 검증 전이라 비활성화했습니다.

> **중요:** 이것은 로컬 프로토타입이지 운영용 웹앱이 아닙니다. SSO, 사용자별 ACL, CSRF 방어, 악성 파일 검사, 영속 작업 큐, 다중 프로세스 GPU lease, 데이터 보존/삭제 관리가 없습니다. `127.0.0.1`에서만 실행하고 합성·비민감 자료만 사용하세요. 별도 vLLM 서버와 동시에 GPU를 쓰지 마세요.

## 시작 전

- 할당된 MIG GPU 및 사용 가능 VRAM을 확인합니다. `CUDA_VISIBLE_DEVICES`나 MIG 설정은 변경하지 않습니다.
- 애플리케이션은 승인된 모델 ID만 호출합니다. 첫 이미지 요청에서 Hugging Face 모델이 캐시에 없으면 다운로드가 시작될 수 있으므로, 운영망에서는 사전에 승인된 revision을 별도로 가져와 오프라인 캐시/아티팩트로 관리하세요.
- 이 저장소의 `.venv-infer`가 없거나 패키지 조합을 변경하기 싫다면 별도 가상환경에서 CUDA 호환 PyTorch를 구성한 뒤 `requirements-web.txt`를 설치합니다. requirements는 PyTorch/CUDA wheel을 강제로 선택하지 않습니다.
- `lab3_rag/index/`를 준비하고 OpenAI-compatible vLLM을 `127.0.0.1:8000`에서 실행해야 업무지원 채팅이 동작합니다. 임베딩만 확인하고 vLLM이 없으면 채팅 API는 오류를 반환합니다.

## 실행

```bash
# 가상환경 활성화 후; PyTorch의 CUDA 조합을 확인한 다음 웹 의존성을 설치
python -m pip install -r apps/ai_studio/requirements-web.txt

# 새 터미널에서 실행: 외부 네트워크에 바인딩하지 않음
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export SERVED_MODEL_NAME=qwen14b
uvicorn apps.ai_studio.app:app --host 127.0.0.1 --port 8080
```

브라우저는 `http://127.0.0.1:8080`을 엽니다. API 점검은 `/api/health`, `/api/features`에서 합니다. 첫 모델 호출은 가중치 다운로드와 초기 로딩으로 오래 걸릴 수 있습니다.

## 초기 기능/검증 범위

1. **업무지원**: 기존 실습 인덱스와 로컬 모델 API를 연결합니다. 현재 실습 인덱스의 접근제어는 없으므로 인덱스에 민감 자료를 넣지 마세요.
2. **이미지 생성**: FLUX.1-schnell, 4 steps, CPU offload, 작업 프로세스 내 GPU 작업 한 개.
3. **이미지 수정**: SDXL Inpainting, 원본과 mask 업로드, 1024×1024로 리사이즈, 흰색 마스크 영역 수정.
4. **동영상**: `Wan-AI/Wan2.2-TI2V-5B`를 후보로 노출하되 실행은 비활성화. 공식 추론 코드와 모델 버전을 고정하고 MIG에서 peak VRAM, 영상 길이/해상도, 생성시간, OOM을 검증한 뒤에만 worker를 붙입니다. T2V/I2V 생성은 기존 영상의 일관성 보장 편집 기능과 다릅니다.

## 주의/다음 단계

- 이 프로세스의 단일 작업 큐는 별도 vLLM, 다른 OS 프로세스, 다른 사용자 GPU 작업과 조율하지 않습니다. 현 MVP에서는 미디어 생성 전 vLLM 등 다른 GPU 작업을 중지해야 합니다.
- 작업 정보는 메모리에만 있고 산출물은 `apps/ai_studio/var/jobs/` 아래 저장됩니다. 서버 재시작 시 상태 조회가 사라집니다. 운영 전 작업 DB, 소유자 검증, 만료/자동 삭제, 백업/복구, queue/lease가 필요합니다.
- 인페인팅용 원본/마스크 파일은 결과 폴더에 남습니다. 프로토타입에서는 민감한 이미지를 업로드하지 말고 실행 후 해당 작업 폴더를 삭제하세요.
- 운영 전 SSO/RBAC 및 문서 ACL, 업로드 형식 검증, 바이러스/콘텐츠 검사, rate limit, 감사 정책, license review, 보안 테스트를 추가해야 합니다.
- `requirements-web.txt`는 개발용 시작점이지 재현 가능한 lockfile이 아닙니다. 실제 GPU 환경에서 CUDA·PyTorch·Transformers·Diffusers의 호환 조합을 고정하고 별도 검증하세요.
