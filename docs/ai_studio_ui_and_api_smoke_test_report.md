# AI Studio 화면·서버 상태 확인 보고서

- **문서 유형:** 개발용 스모크 테스트 보고서
- **대상:** Elice 터널을 통한 AI Studio 웹앱
- **접속 기준 주소:** `https://pmrkdtmaxyahbnfm.tunnel.elice.io/proxy/8081/`
- **테스트 범위:** 웹 화면 로딩, `/api/health`, `/api/features`
- **기록 기준:** 사용자가 실제로 확인한 화면 및 API 응답과 저장소의 구현을 바탕으로 작성

## 1. 목적 및 범위

AI Studio가 Elice 프록시를 통해 정상적으로 제공되는지와, 기본 상태·기능 정보 API가 응답하는지를 확인했다. 이 검사는 화면과 서버의 기본 연결을 확인하는 스모크 테스트이며, 모델 추론이나 GPU 작업의 성공 여부를 검증하지 않는다.

## 2. 접속 경로 문제와 조치

초기에는 웹앱 HTML은 표시되지만 CSS 디자인이 적용되지 않는 현상이 있었다. 앱이 정적 파일과 API를 `/static/...`, `/api/...`처럼 도메인 루트 기준 절대 경로로 요청하고 있었고, Elice 프록시의 실제 앱 경로가 `/proxy/8081/`이어서 요청 경로가 프록시 경로를 벗어날 수 있었다.

이를 해결하기 위해 프런트엔드에서 현재 앱 기본 경로를 기준으로 정적 파일 및 API URL을 만들도록 수정했다. HTML에는 상대 경로 기반을 지정하고, JavaScript의 상태 확인·기능 확인·작업 요청·결과 조회 URL도 앱 기본 경로에 상대적으로 구성했다. 브라우저 캐시가 변경된 파일을 재사용하지 않도록 정적 파일 버전도 갱신했다.

이후 사용자가 8081 프록시 주소에서 디자인이 정상 표시되는 것을 확인했다.

## 3. 수행 절차 및 결과

### 3.1 AI Studio 화면

**접속 주소**

`https://pmrkdtmaxyahbnfm.tunnel.elice.io/proxy/8081/`

**확인 절차**

1. 브라우저에서 주소를 열거나 새로고침한다.
2. AI Studio 화면과 스타일이 적용됐는지 확인한다.
3. 왼쪽 메뉴에서 문서 도우미, 이미지 스튜디오, 동영상 생성, 모델 실험실 탭을 선택해 화면 전환을 확인한다.

**관찰 결과**

- 사용자가 8081 주소에서 웹앱이 정상적으로 표시된다고 확인했다.
- 그 전에는 HTML만 보이고 디자인이 적용되지 않았으며, 상대 경로 수정 뒤 정상 표시가 확인됐다.
- 탭별 GPU 기능 실행까지 확인한 것은 아니다.

**판정:** 화면 제공 및 스타일 로딩 **통과** (사용자 확인 기준)

### 3.2 서버 상태 API (`/api/health`)

**접속 주소**

`https://pmrkdtmaxyahbnfm.tunnel.elice.io/proxy/8081/api/health`

**수행 절차**

1. 주소를 브라우저에서 연다.
2. 화면에 표시되는 JSON 응답에서 `status`와 `note`를 확인한다.

**구현상 응답 내용**

```json
{
  "status": "development",
  "note": "Local-only prototype; no SSO/ACL or cross-process GPU lock."
}
```

**해석**

- `status: development`는 AI Studio 서버가 HTTP 요청에 응답하고 있음을 나타낸다.
- `note`는 개발용 프로토타입의 제한을 알린다. SSO/접근 제어가 없고, 앱 프로세스 밖의 GPU 작업과 앱 내부 작업 큐가 서로 조정되지 않는다.
- 이 API는 서버 생존 여부 확인용이다. vLLM 연결, RAG 응답, 모델 다운로드, GPU 가용성 또는 추론 성공을 보증하지 않는다.

**판정:** 서버 상태 API 응답 **통과** (앱 페이지가 정상 표시되고 해당 API가 응답하는 기준)

### 3.3 기능·모델 상태 API (`/api/features`)

**접속 주소**

`https://pmrkdtmaxyahbnfm.tunnel.elice.io/proxy/8081/api/features`

**수행 절차**

1. 주소를 브라우저에서 연다.
2. JSON의 `labs`와 `models` 배열에서 기능 키, `enabled` 값, 관련 안내를 확인한다.

**사용자가 확인한 응답**

```json
{
  "support": true,
  "labs": [
    {"key": "rag-index", "enabled": true},
    {"key": "vllm-api-test", "enabled": true},
    {"key": "infer", "enabled": true},
    {"key": "qlora-train", "enabled": true},
    {"key": "qlora-eval", "enabled": true},
    {"key": "classify", "enabled": true},
    {"key": "benchmark", "enabled": true},
    {"key": "vl-receipt", "enabled": true}
  ],
  "models": [
    {
      "key": "flux-schnell",
      "model_id": "black-forest-labs/FLUX.1-schnell",
      "task": "text-to-image",
      "enabled": true
    },
    {
      "key": "sdxl-inpaint",
      "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
      "task": "masked-inpainting",
      "enabled": true
    },
    {
      "key": "wan22-ti2v",
      "model_id": "Wan-AI/Wan2.2-TI2V-5B",
      "task": "text-or-image-to-video",
      "enabled": false,
      "note": "Requires local Wan2.2 official repository and checkpoint"
    }
  ]
}
```

**해석 및 판정**

- `support: true`와 `enabled: true`는 기능이 앱에 등록되어 있거나 코드의 조건상 활성화되어 있음을 나타낸다. 실제 서비스 연결 및 GPU 실행을 시험한 결과는 아니다.
- `infer`, `qlora-train`, `qlora-eval`, `classify`, `vl-receipt`는 `.venv-infer/bin/python` 실행 파일의 존재를 기준으로 활성화될 수 있다. 각 모델, 패키지, 체크포인트, VRAM이 준비됐다는 뜻은 아니다.
- `rag-index`, `vllm-api-test`, `benchmark`의 활성화 표시만으로 인덱스 생성이나 vLLM 연결을 보장하지 않는다.
- FLUX와 SDXL의 `enabled: true`는 앱에서 해당 옵션을 제공한다는 뜻이다. 첫 실행 시 모델 다운로드가 필요하거나 실행 환경에 따라 실패할 수 있다.
- Wan2.2는 로컬 공식 저장소와 체크포인트가 설정되지 않아 `enabled: false`로 표시됐다. 이는 현재 설정 기준으로 기대되는 상태다.

**판정:** 기능·모델 상태 API 응답 **통과**. Wan2.2 비활성 상태는 설정 미완료를 올바르게 표시한 것으로 판단한다. 실제 작업 실행은 별도 검증 필요.

## 4. 포트별 관찰

- `8081`: 기존 AI Studio HTML 응답이 확인됐으며, 상대 경로 처리 수정 후 스타일이 적용된 화면을 사용자가 확인했다. 이후 접속 기준 포트로 사용한다.
- `18080`: 브라우저에서 `connect ECONNREFUSED 0.0.0.0:18080`가 표시됐다. 해당 시점에 프록시가 연결할 수 있는 서버가 그 포트에 없거나 해당 포트로 연결되지 않았다는 의미다. 본 테스트의 정상 접속 주소가 아니므로 사용하지 않는다.

## 5. 종합 결과 및 미검증 항목

| 검사 항목 | 결과 | 비고 |
|---|---|---|
| 8081 웹앱 화면 및 스타일 로딩 | 통과 | 사용자가 정상 표시 확인 |
| `/api/health` 기본 응답 | 통과 | 개발 상태 응답은 GPU 준비 여부를 보장하지 않음 |
| `/api/features` JSON 응답 | 통과 | 기능별 `enabled`는 실제 실행 검증과 다름 |
| 문서 RAG 실제 답변 | 미검증 | 인덱스 및 vLLM 서버 필요 |
| vLLM API 테스트·벤치마크 | 미검증 | vLLM 서버 필요 |
| Qwen 추론·QLoRA·VL | 미검증 | 모델·패키지·GPU 조건 확인 필요 |
| FLUX 이미지 생성·SDXL 인페인팅 | 미검증 | 모델 다운로드·GPU 조건 확인 필요 |
| Wan2.2 동영상 생성 | 실행 불가 상태 | 로컬 저장소와 체크포인트 미설정으로 비활성 |

## 6. 후속 기능 테스트 권고

1. 먼저 브라우저에서 `/api/health`와 `/api/features`가 같은 `/proxy/8081/` 접두 경로 아래에서 열리는지 확인한다.
2. 실 GPU 테스트 전 `nvidia-smi` 등으로 GPU 사용 상태를 확인한다.
3. GPU를 사용하는 추론·학습·미디어 작업은 한 번에 하나만 실행한다. 별도 vLLM 서버를 함께 사용해야 하는 기능은 서버 사용 조건을 확인한다.
4. 부담이 적은 연결 테스트부터 시작하고, 이미지 생성이나 학습 등 오래 걸리고 VRAM을 많이 쓰는 작업은 준비가 확인된 뒤 시험한다.
5. 테스트에는 비민감·합성 자료만 사용한다. 이 MVP에는 로그인, 문서 접근 제어, 업로드 악성 파일 검사 및 작업 간 GPU 조정이 없다.
