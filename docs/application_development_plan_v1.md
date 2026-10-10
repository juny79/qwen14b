# G-NAHPM-40 A100 MIG 기반 AI 프로그램·웹앱 개발 종합 계획서 v1

- **대상:** G-NAHPM-40 A100 MIG `3g.40gb`
- **목표:** 지금까지 검증한 추론, vLLM, RAG, QLoRA, API 벤치마크, VL 실습을 내부 업무지원 프로그램과 웹앱으로 발전시킨다.
- **성격:** 기술 계획 및 제한 파일럿 제안. SLA, 외부 공개, 자동 업무처리 승인을 의미하지 않는다.
- **근거:** `docs/progress_report_v1.md`~`v9.md`의 사용자 공유 결과. 운영 전 환경과 성능 재측정이 필요하다.

> 단일 MIG GPU와 작은 실습 데이터의 결과를 서비스 품질 보증으로 확대 해석하지 않는다. 모델·데이터 라이선스, 개인정보, 클라우드 정책은 배포 전에 담당자가 확인한다.

## 1. 요약

우선 **사내 문서 근거형 RAG 웹앱**을 MVP로 만든다. 현재 RAG 실습은 1개 문서·1개 청크와 검색 점검 3건이므로, 출처 표시·권한·다중 문서·문서 갱신을 충분히 검증한 뒤 제한된 파일럿에 제공한다.

다음은 **업무 요청 분류 보조**다. QLoRA 비교는 3개 데모에서 베이스 0/3, 어댑터 3/3 exact-match였으나, 학습 데이터 8개·유사 평가 3개 및 attention/generation 경고가 있었다. 일반화 증거가 아니며 사람 검토 전 초안으로만 쓴다.

**영수증/VL**은 단일 이미지에서 JSON 파싱과 출력 내부 금액 합계를 확인했으나 `val.json`에는 텍스트 전사/필드 정답이 없다. 정확도는 미검증이므로 정산 자동처리는 금지하고 원본 대조와 사용자 승인을 요구한다.

GPU는 단일 MIG 인스턴스다. 14B vLLM, QLoRA, VL을 동시에 실행하지 않는다. MVP는 RAG + vLLM 하나만 GPU에 올리고, 모델 작업 전환은 기존 서버 종료와 GPU 반환을 확인한 뒤 수행한다. 동시 멀티모달 운영이 필요하면 별도 GPU가 선행 조건이다.

## 2. 검증된 기술 기준 및 제약

| 항목 | 확인된 관찰값 | 설계 반영 |
|---|---|---|
| GPU | A100 80GB PCIe의 MIG `3g.40gb`, 42 SM, 장치 표시 40448 MiB | 할당 MIG와 `CUDA_VISIBLE_DEVICES` 유지. 동시 GPU 작업 금지 |
| RAM/cgroup | 호스트 96 GiB, 관찰 당시 cgroup limit 약 96 GiB | 호스트 MemAvailable과 cgroup memory.current/events/pressure 확인 |
| 메모리 캐시 | 한 시점 file/cache 약 78.5 GiB, anon 약 1.1 GiB, available 약 94 GiB | 파일 캐시를 프로세스 누수로 단정하지 않음. 압박 없이 `drop_caches` 금지 |
| OS/도구 | Ubuntu 22.04.5, Python 3.10.14, NVIDIA driver 535.183.06 | 검증 환경을 잠그고 업그레이드는 별도 스테이징에서 시험 |
| `.venv-infer` | PyTorch 2.5.1+cu121, Transformers 4.51.3, PEFT 0.15.2 | RAG/HF/QLoRA/VL 환경. vLLM과 분리 |
| `.venv-vllm` | vLLM 0.7.3, PyTorch 2.5.1+cu124, Transformers 4.51.3 | 검증된 패키지 조합 유지; 임의로 환경 섞지 않음 |
| 14B HF 추론 | BF16 peak allocated 27.58 GiB | 단일 GPU 작업으로 취급 |
| 14B vLLM | 가중치 27.5681 GB, MIG total 39.50 GiB, vLLM 예산 35.55 GiB, KV cache 7.45 GiB; context 4096, max seqs 2, utilization .90 | 초기 용량 기준. context/concurrency 변경은 별도 시험 |
| API 부하시험 | concurrency 1/2/4, 각 4회; 처리량 26.159/50.734/50.557 tokens/s, 성공 12/12 | 작은 표본/128 token 상한 결과. SLA로 간주 금지 |
| RAG | multilingual-e5-small CPU 임베딩, 문서·청크 각 1, 검색 점검 3/3 | 다중 문서·갱신·권한을 재검증 |
| QLoRA | 7B NF4 adapter; 데모 3건 base 0/3, adapter 3/3 | 학습 8건·유사 평가 3건·실행 경고; 일반화 근거 아님 |
| VL | Qwen2.5-VL-7B-Instruct, 단일 이미지 peak 15.80 GiB | vLLM과 동시 실행 금지, 정확도 미검증 |
| 데이터 주석 | `words` points/orientation/language, 크기; 텍스트 전사·필드 정답 없음 | test split 보존, 의미 필드 정확도 수치 금지 |

운영 직전에 GPU, GPU process, RAM/cgroup, 디스크 여유, package/model revision을 다시 확인한다. 과거 관찰값은 현재 자원 보장치가 아니다.

### 재사용 코드

`lab1_infer/` 기본 프롬프트, `lab2_vllm/` API 서버, `lab3_rag/` FAISS RAG, `lab4_qlora/` adapter 학습·평가, `lab5_bench/` 비스트리밍 부하 테스트, `lab6_vl/` 이미지 JSON 추출, `scripts/` 점검 도구를 참고한다. 코드는 교육용이며 인증, ACL, 파일 관리, request queue, 예외 복구, 보안, 통합테스트를 더하기 전 인터넷에 공개하지 않는다.

## 3. 제품 기능 범위

### 3.1 우선 MVP: 문서 기반 사내 도우미

- SSO 로그인과 사용자/그룹별 문서 권한 적용.
- 승인된 정책·업무 문서 업로드, 버전·owner·유효일 관리.
- 문서 파싱 → 청킹 → CPU embedding → index build/test → 검증된 index 원자 교체.
- 질문에 근거 문서명, 버전, 페이지/절 또는 chunk 위치를 보여준다.
- 근거가 부족하면 추측하지 않고 답변 보류 및 담당자 확인을 안내한다.
- 사용자 피드백 및 문서 갱신 절차를 제공한다.

### 3.2 후속 기능: 업무 요청 분류 보조

- category, priority, 한 줄 요약을 schema가 고정된 JSON으로 제안.
- 허용 category 외 값, 파싱 오류, 낮은 confidence는 자동 라우팅하지 않고 사람에게 전달.
- 티켓·결재 등 외부 action은 초안과 미리보기를 제공하고 담당자가 승인.
- 비밀번호/개인정보/보안 요청 정책은 prompt뿐 아니라 backend 검증으로 적용.
- 운영 전에 독립 holdout, 충분한 라벨, macro-F1·confusion matrix와 안전성 평가 확보.

### 3.3 제한 기능: 영수증/VL

- 형식·크기·픽셀 제한과 파일 검사 후 임시 처리.
- merchant/date/total/items는 제안으로 표시하고 미확인 필드 및 원본 대조 안내 제공.
- 사용자 검토·수정 전에는 저장, 지급, 회계 반영을 하지 않는다.
- 임시 이미지 자동 삭제를 기본으로 하고 장기 보존은 별도 승인.
- 텍스트 정답, field accuracy, 개인정보 처리 승인이 없으면 실제 정산 기능을 제공하지 않는다.

## 4. 목표 아키텍처

1. **Web UI:** 사내 SSO, RAG 채팅, citation panel, 문서 관리, feedback, 운영 상태.
2. **Backend/API:** 인증·인가, 입력 제한, rate limit, request ID, schema 검증, 업무 시스템 연결.
3. **Inference gateway:** OpenAI-compatible client에서 localhost vLLM으로 접속. API key는 서버 측 secret만 사용.
4. **RAG service:** parse/chunk, metadata ACL, FAISS retrieval, citation validation, answer abstention. embedding은 우선 CPU.
5. **GPU job controller:** 단일 GPU lease/semaphore, 예약·상태·취소. vLLM 중 VL/QLoRA 실행 금지.
6. **Persistence:** 사용자/권한/문서 metadata/index version/feedback DB와 파일·index storage를 분리, 백업·복구 관리.
7. **Observability:** latency/status/token/실패/queue/GPU·RAM·cgroup·disk/model·index version. 원문·secret·PII는 metric label/log에 넣지 않음.

흐름은 사용자 인증/ACL → 입력 정책검사 → 검색 또는 모델 실행 → JSON/citation 검증 → UI 응답·피드백. VL 파일은 임시 저장 후 삭제. confidence 낮음, 근거 없음, 형식 오류에는 사람 검토로 넘긴다.

## 5. API 및 데이터 원칙

- API는 `/api/v1` 등 버전을 고정한다. 응답에 `answer`, `citations`, `abstained`, `request_id`, `model_version`, `index_version` 등 추적 정보를 포함한다.
- category/VL JSON을 server-side JSON Schema로 검증한다. retry 횟수·token budget을 제한하고 계속 실패하면 오류/사람 검토를 반환한다.
- 내부 reasoning은 노출·저장하지 않는다. 대화 로그는 최소화하고 보존기간·삭제·내보내기를 정의한다.
- 문서 metadata: id, display name, version, owner, ACL group, effective date, source, checksum, ingestion status.
- index는 버전 디렉터리에서 생성·검증 후 원자 교체. 실패 시 직전 버전으로 rollback.
- 검색 전에 ACL 적용. 문서 내 prompt injection은 권한 지시가 아니라 신뢰할 수 없는 콘텐츠로 취급한다.
- API request size, context, output tokens, timeout, retry, queue를 제한한다.

## 6. 보안·개인정보·신뢰성

- 외부 사용자 제공 전 SSO, 최소권한 RBAC, TLS reverse proxy, CSRF/XSS 방어, CORS allowlist, upload 검사 및 보안 리뷰 필수.
- vLLM을 `0.0.0.0`에 직접 노출하지 않는다. 기본 실습 API key는 운영에 금지. 키는 secret manager/protected env에 두고 repo, browser, log에 넣지 않는다.
- 사용자/팀 ACL을 검색 전에 집행해 다른 권한 문서가 snippet·제목·답변으로 노출되지 않게 한다.
- 모델·데이터 라이선스, 문서/영수증 PII 처리근거, cloud region, 암호화, 보존·삭제·감사정책을 책임자가 승인한다.
- 금액/기한/보안/개인정보 결정은 원문 근거와 사람 승인 없이 자동 실행하지 않는다.
- readiness/health, graceful shutdown, 제한된 backoff, circuit breaker, 취소·복구 절차를 둔다.
- 모델 weights, outputs, logs, FAISS index, 사용자 이미지, secret은 Git에 올리지 않는다. `datasets/` 이용권한을 별도 확인한다.

## 7. 개발 로드맵과 수용 기준

### Phase 0 — 업무·데이터 승인

대상 업무, 사용자, owner, 허용 문서, ACL 그룹, PII 등급, 보존기간, 사람 승인 흐름을 확정한다. 사용 중인 모델/데이터의 라이선스 및 클라우드 정책도 확인한다.

**완료 기준:** 제품 담당자와 보안/개인정보 책임자의 승인, RAG 평가 질문·정답 근거 초안 승인.

### Phase 1 — RAG·인덱스 서비스화

`lab3_rag` 코드를 재사용 가능한 API/service로 분리하고, 다중 문서 ingestion, metadata ACL, 버전 인덱스, citation, abstention, 오류·재시도·rollback을 추가한다. 평가용 질문은 업무 담당자가 만든다.

**완료 기준:** 문서 생성/수정/삭제 lifecycle test, ACL 누출 test, 근거 있는 답의 citation 확인, 근거가 없는 질문의 보류, 프로세스 재시작과 이전 index 복구 test 통과.

### Phase 2 — Backend와 웹 UI MVP

SSO 인증, RAG API, 채팅/출처 패널, 문서 관리자 기능, feedback, request ID를 개발한다. DB·스토리지·reverse proxy는 실제 조직에서 지원하는 선택지를 따르며, 이 문서에서는 특정 공급자나 프레임워크를 강제하지 않는다.

**완료 기준:** 역할별 사용자 흐름, 잘못된 파일/입력, timeout·세션 만료·새로고침, 기본 접근성/브라우저 검증을 staging에서 통과.

### Phase 3 — 품질·보안·성능 검증

독립 검토 데이터에서 retrieval Recall@k, citation correctness, grounded answer correctness, abstention precision을 평가한다. API는 warm-up 후 충분한 반복으로 동시성별 지연, 처리량, 실패율을 재측정하고 장시간 stability를 따로 확인한다. 현재 실습의 4회 샘플 결과를 수용기준으로 재사용하지 않는다.

**완료 기준:** 업무 담당자가 사전에 수치 기준을 정하고 승인, security/privacy review 통과, backup/restore와 rollback drill 성공.

### Phase 4 — 제한 파일럿

소수 승인 사용자 및 승인된 소규모 문서 corpus로 파일럿. 근거 부재, 잘못된 출처, escalation, latency, queue, GPU/RAM/cgroup, 오류 및 사용자 피드백을 모니터링한다. 자동 action은 꺼 둔다.

**완료 기준:** 운영 담당자·업무 owner sign-off, 사용자 문의·오답 신고·문서 만료·장애·삭제 요청 절차 확인. 승인 후에만 확대한다.

### Phase 5 — 분류/VL 확장(별도 승인)

분류는 라벨 기준을 고정하고 비슷한 학습 사례와 분리된 독립 holdout으로 macro-F1/confusion matrix 및 안전성 평가를 만든 후 적용한다. VL은 OCR transcript 또는 영수증 필드 정답이 있는 평가 데이터를 확보하고 PII 처리를 승인받은 후 시작한다. 독립 GPU가 없으면 별도 작업창에서만 추론한다.

**완료 기준:** 업무 담당자의 category/field별 지표 승인, 오류 사례 분석, human override, 접근·보존·삭제 검증. 정답이 없으면 정량 정확도를 주장하지 않는다.

일정은 팀 규모, 인증/데이터 담당자의 승인 속도, 운영 플랫폼 선택에 따라 달라진다. Phase 0~4를 먼저 순차 진행하고 VL/분류의 시점은 별도 gate로 정한다.

## 8. 운영 방식과 배포 단계

### 개발/시험/운영 분리

- **개발:** 합성/비민감 문서로 UI/API 개발. 실제 PII는 사용 금지.
- **스테이징:** production과 유사한 패키지·권한·문서 인덱스에서 회귀, 용량, 보안 시험.
- **파일럿 운영:** 승인 사용자, 제한된 문서, 명시적 안내 및 사람 검토. model/index version 기록.
- 코드/의존성 변경은 lockfile과 환경 snapshot으로 추적한다. 새로운 CUDA/PyTorch/Transformers/vLLM 버전은 같은 GPU에서 별도 시험 후 승격한다.

### GPU 운영 규칙

1. vLLM/API가 GPU를 점유할 때 QLoRA/VL/HF inference를 실행하지 않는다.
2. 장시간 RAG 서비스가 필요하면 기본 resident model을 하나로 정하고, VL/훈련은 예약된 maintenance window로 분리한다.
3. 기동 전 MIG·GPU process·disk/RAM을 확인하고, 기동 후 readiness와 API smoke test를 한다.
4. 서버 종료 후 프로세스와 MIG memory 회수를 확인한다. 다른 사용자 작업은 종료하지 않는다.
5. 현재 관찰한 `max_num_seqs=2`, 4096 context, utilization .90에서 출발하되, 운영 concurrency와 prompt 크기는 실제 workload 부하 시험으로 확정한다.
6. 초기 서비스에서 최대 출력 토큰, 사용자별 동시 요청, 전체 queue 크기에 hard cap을 둔다. 큐 포화 시 오류/대기 안내를 제공한다.

## 9. 측정 지표 및 출시 gate

| 영역 | 측정 지표 | 통과 방식 |
|---|---|---|
| RAG 검색 | Recall@k, 근거 문서 적합성, 권한 밖 결과 수 | 실제 승인 문서의 독립 평가셋, ACL leak 0건 |
| 답변 | citation correctness, grounded answer, abstention precision | 업무 SME 검토, 미확인 답변을 정답으로 계산하지 않음 |
| API 품질 | p50/p95 latency, tokens/s, timeout/error, queue time | warm-up + 여러 반복·요청 길이·동시성 시험으로 팀 기준 설정 |
| 분류 | macro-F1, per-class recall, confusion matrix, rejection/escalation | 학습에서 분리된 대표 holdout 및 안전 테스트 |
| 영수증 | field exact/normalized match, OCR CER/WER, 누락률 | 정답 전사 포함 별도 승인 dataset 없으면 미산출 |
| 보안 | ACL leakage, secret scan, 업로드 abuse, 권한 테스트 | 출시 전 보안 책임자 review 및 test 기록 |
| 운영 | availability, restart recovery, backup restore, GPU/RAM/disk | staging drill 및 담당자 승인 |
| 사람 검토 | 수정률, override 비율, escalation, 오류 신고 | pilot 사용자 동의 및 개인정보 없는 집계 |

**출시 gate:** 기능 demo 성공만으로 배포하지 않는다. 제품 owner, 보안/개인정보 책임자, 운영 담당자가 품질·권한·보존·복구·비용을 승인해야 한다. AI 응답은 결정이 아닌 보조이며, 지급·접근권한 변경·결재 등 외부 action은 별도 승인.

## 10. 주요 위험과 완화책

| 위험 | 완화 |
|---|---|
| MIG 메모리 한계·GPU 경합 | GPU lease, 한 작업 원칙, context/token/concurrency 상한, OOM fallback/queue |
| RAG 근거 오류 또는 권한 누출 | retrieval ACL, citation 검증, 권한 부정 테스트, 답변 보류 |
| 작은 QLoRA 평가의 과대해석 | 독립 holdout, base/adapter 동일한 환경·프롬프트, generation/attention mask 고정, 외부 검토 |
| VL 숫자/일자 오인식 | 정답 데이터로 field metric 확보 전 제안 전용, 원본 대조·승인 필수 |
| 개인정보·비밀 유출 | 최소수집/보존, secret vault, access log, redaction, 암호화·삭제 절차 |
| 모델/라이브러리 버전 변화 | 환경 lock, artifact revision, regression suite, staged rollout/rollback |
| 단일 장비 장애·다운타임 | 초기 파일럿에 한계 명시, queue/maintenance 계획, 별도 백업·대체 자원 검토 |
| 비용/사용량 폭증 | token/파일 크기/요청수 한도, quota, 사용량 집계, 알림·kill switch |

## 11. 당장 시작할 실행 항목

1. 내부 MVP owner 및 문서 owner를 지정하고 실제 사용자 질문 30~50건과 답의 근거 문서를 준비한다(수량은 초기 제안이며 대표성과 SME 승인을 확인).
2. 각 문서의 사용자/그룹 ACL, 유효일, 민감도, 허용 목적을 정한다. 권한이 불명확한 자료는 인덱싱하지 않는다.
3. `lab3_rag`를 기준으로 multi-document ingestion/ACL/citation 테스트 계획을 작성하고, 현재 RAG 샘플을 production 품질로 간주하지 않는다.
4. `lab5_bench`를 개선할 때 결과를 행 단위로 기록하고 warm-up, 충분한 반복, input/output token 길이, p50/p95, errors, queue time을 포함한다. 기존 스크립트의 작은 표본은 운영 기준으로 사용하지 않는다.
5. 로그인·권한·보존·폐기·로그 정책이 승인되기 전 실사용자 데이터나 실제 영수증을 웹앱에 올리지 않는다.
6. 파일럿은 RAG만 우선 제공하고, 분류·VL은 별도 품질 및 개인정보 gate를 통과한 뒤 추가한다.

## 12. 의사결정이 필요한 항목

- 대상 부서, 우선 업무 및 업무상 오답 비용
- 승인 문서 및 평가 질문/정답 근거의 책임자
- 사내 SSO, reverse proxy, TLS, DB, storage, secret manager의 표준
- 대화·문서·영수증 보존기간 및 지역/규제 요구사항
- 제한 파일럿 사용자 수, 서비스 시간, 허용 latency/error 기준
- 모델/데이터의 라이선스, 상업/내부 사용 허용 범위
- 실제 멀티모달 동시 서비스가 필요할 경우 추가 GPU 예산과 GPU job scheduling 방식

위 결정이 완료되기 전에는 본 계획을 구현 승인이나 production readiness로 간주하지 않는다.

## 13. 최종 권고

A100 MIG 40GB를 활용한 내부 웹앱은 기술적으로 **RAG 중심의 제한 파일럿**부터 시작하는 것이 현실적이다. 현 시점의 검증은 모델 API와 작은 규모의 end-to-end 흐름을 보인 proof-of-concept 수준이다. 다중 문서·권한·독립 평가·운영 안정성을 검증한 뒤 분류 기능을 추가하고, receipt/VL은 정답 레이블과 개인정보 승인이 확보되기 전까지 사람 검토가 필요한 실험 기능으로 유지한다.

프로젝트 실습 결과와 계획 사이의 범위·측정·한계를 숨기지 않고 문서화하며, 어떤 자동화도 정확도와 권한 통제를 입증하기 전 실제 결재·지급·보안 조치를 실행하지 않는다.

---

**관련 보고서:** `docs/progress_report_v1.md`~`docs/progress_report_v9.md`  
**문서 상태:** 개발 계획 초안. 시스템 아키텍트, 업무 owner, 보안/개인정보 담당자 검토 후 개정.