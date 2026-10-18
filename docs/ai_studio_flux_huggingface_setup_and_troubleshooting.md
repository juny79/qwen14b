# AI Studio FLUX 이미지 생성: Hugging Face 연동 및 문제 해결 기록

- **대상 모델:** `black-forest-labs/FLUX.1-schnell`
- **범위:** Hugging Face 접근 승인·토큰 설정부터 AI Studio에서 이미지 생성까지
- **환경:** Elice 인스턴스, 프로젝트 `~/qwen`, AI Studio 포트 `8081`

> Hugging Face 토큰이나 비밀번호는 이 문서에 기록하지 않습니다. 토큰을 소스 코드, `.env`, 채팅에 붙여 넣지 마세요.

## 1. 세 가지 설정은 서로 다릅니다

1. **Hugging Face 웹사이트 모델 승인:** 모델 페이지에 `You have been granted access to this model`이 표시되면 웹 계정의 모델 접근 승인이 된 것입니다.
2. **서버 측 인증:** 모델을 내려받는 Linux 사용자에게 Hugging Face 토큰을 등록해야 합니다. 웹에서 승인받는 것만으로 AI Studio 서버가 로그인되지는 않습니다.
3. **AI Studio의 GPU 작업 허용:** 앱 내 작업 제출만 허용합니다. Hugging Face 인증, CUDA 설치, GPU 할당을 설정하지 않습니다. 서버 재시작 후에는 다시 꺼집니다.

## 2. 모델 접근 승인 및 토큰 권한

브라우저에서 Hugging Face에 로그인한 후 [FLUX.1-schnell 모델 페이지](https://huggingface.co/black-forest-labs/FLUX.1-schnell)를 엽니다. 이 사례에서는 `You have been granted access to this model`이 표시되어 접근 승인을 확인했습니다.

토큰은 **해당 접근 승인을 받은 Hugging Face 계정**에서 생성해야 합니다. Fine-grained 토큰을 쓸 경우 gated model 읽기와 repository content read 권한이 필요합니다. 이 사례의 `whoami()` 출력에는 `canReadGatedRepos: True`와 `repo.content.read`가 포함되어 있었습니다.

## 3. 토큰을 Linux 사용자 계정에 등록

“같은 Linux 계정”이란 Hugging Face 로그인 명령과 AI Studio 서버를 **같은 Linux 사용자 이름**으로 실행한다는 뜻입니다. 이 사례의 터미널 프롬프트는 `elicer@...:~/qwen$`였으므로 Linux 사용자는 `elicer`였습니다. Hugging Face 사용자 이름과 Linux 사용자 이름은 같을 필요가 없습니다.

프로젝트 루트 `~/qwen`에서 `.venv-infer/bin/hf auth login`을 시도했지만 해당 실행 파일이 없어 `No such file or directory`가 발생했습니다. 이 환경에서는 Python API로 로그인했습니다.

- 로그인: `.venv-infer/bin/python -c 'from huggingface_hub import login; login()'`
- 로그인 계정 확인: `.venv-infer/bin/python -c 'from huggingface_hub import whoami; print(whoami())'`

로그인 프롬프트에서 토큰을 입력합니다. 입력 글자가 화면에 보이지 않는 것은 정상입니다. 토큰을 명령줄에 직접 쓰지 마세요. 기본 설정에서 토큰은 프로젝트 파일이 아니라 해당 Linux 사용자의 Hugging Face 캐시에 저장됩니다. 이 사례의 캐시 경로는 `/home/elicer/.cache/huggingface/hub/`였습니다. `HF_HOME`을 따로 설정했다면 경로가 다를 수 있습니다.

로그인 중 `Add token as git credential?` 질문은 이미지 생성에 필요하지 않습니다. 이 사례에서는 `Y`를 선택해 Git helper 오류가 나왔지만, Hugging Face Hub 로그인에는 영향이 없었습니다. Git credential helper 설정을 추가할 필요는 없습니다. 다음 로그인에서 Git 저장 질문에는 `n`을 선택하면 됩니다.

## 4. 인증과 모델 접근 검증

다음 명령에서 권한이 승인된 Hugging Face 계정이 출력되는지 확인합니다.

`.venv-infer/bin/python -c 'from huggingface_hub import whoami; print(whoami())'`

그다음 gated 모델의 작은 파일을 실제로 내려받아 확인합니다.

`.venv-infer/bin/python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id="black-forest-labs/FLUX.1-schnell", filename="model_index.json"))'`

이 사례에서는 `model_index.json` 다운로드가 성공하고 `/home/elicer/.cache/huggingface/hub/...` 경로가 출력됐습니다. 따라서 계정 권한·토큰·모델 접근이 정상임을 확인했습니다. Git credential 경고와 모델 접근 인증은 별개입니다.

## 5. AI Studio 서버 시작과 포트

`scripts/run_ai_studio.sh`의 기본 포트는 `8080`입니다. 이 사례의 Elice 프록시 접속 포트는 `8081`이었으므로, 기본값으로 시작하면 원하는 포트에서 서버가 뜨지 않거나 다른 프로세스와 포트 충돌이 발생할 수 있었습니다.

프로젝트 루트에서 포트를 명시해 시작합니다.

`AI_STUDIO_PORT=8081 bash scripts/run_ai_studio.sh`

정상 시작 시 `AI Studio: http://127.0.0.1:8081`과 `Uvicorn running on http://127.0.0.1:8081`이 표시됩니다. 서버 실행 터미널은 계속 열어 둡니다. 별도 터미널에서 다음으로 확인합니다.

`curl -i http://127.0.0.1:8081/api/health`

정상이면 `HTTP/1.1 200 OK`와 JSON 응답이 나옵니다. `Connection refused`는 해당 포트에서 서버가 실행 중이지 않다는 뜻입니다. `address already in use`는 포트를 다른 프로세스가 사용 중이라는 뜻입니다. AI Studio 프로세스 확인에는 `ps -eo pid,args | grep '[u]vicorn.*--port 8081'`을 사용할 수 있습니다. 종료할 때는 `kill 12345`처럼 실제 PID 숫자를 입력하며, 꺾쇠괄호는 입력하지 않습니다.

포트를 매번 지정하지 않으려면 프로젝트 루트 `.env`에 `AI_STUDIO_PORT=8081`을 설정할 수 있습니다. 또는 매번 위처럼 환경변수를 명령 앞에 붙입니다.

## 6. 이미지 생성 절차

1. 브라우저에서 포트 `8081`의 AI Studio를 엽니다.
2. 상단 **GPU 작업 허용** 스위치를 켭니다. 서버를 재시작했다면 다시 켜야 합니다.
3. `이미지 생성·수정` 탭에서 일반 이미지 생성을 선택하고 프롬프트를 제출합니다. 인페인팅은 원본 이미지와 마스크가 필요한 별도 작업입니다.
4. 작업 상태와 오류를 확인합니다. 실패 시 화면의 오류 문구 또는 `/api/jobs/<작업ID>` 응답을 확인합니다.

앱의 FLUX worker는 `enable_model_cpu_offload()`를 사용하고 `num_inference_steps=4`로 생성합니다. GPU 허용 스위치는 실제 GPU 사용 여부를 보장하지 않습니다.

## 7. 확인된 오류와 해결

### `GatedRepoError: 401 Client Error`

웹사이트 접근 승인만으로는 서버 인증이 되지 않아 발생했습니다. 승인된 계정 토큰을 Linux 사용자 `elicer`의 Hugging Face 캐시에 등록한 뒤 `whoami()`와 `hf_hub_download()`가 성공해 해결됐습니다.

### `.venv-infer/bin/hf: No such file or directory`

가상환경에 `hf` 실행 파일이 없었습니다. `.venv-infer/bin/python`에서 `huggingface_hub.login()`을 호출하는 방법으로 진행했습니다.

### Git credential 저장 실패

로그인 때 Git 자격 증명 저장을 선택했지만 시스템에 Git credential helper가 없었습니다. Git 저장만 실패한 것이며 Hub 토큰 로그인과 모델 다운로드에는 영향이 없었습니다. 이미지 생성용으로 `git config --global credential.helper store`를 설정할 필요는 없습니다.

### `Unexpected token 'c', "connect EC..." is not valid JSON`

브라우저 코드가 서버 오류 텍스트(`connect ECONNREFUSED`)를 JSON으로 해석하려다 표시한 오류였습니다. 앞선 `Killed` 이후 AI Studio 서버가 종료되어 8081 연결이 거부됐습니다. 서버를 `AI_STUDIO_PORT=8081 bash scripts/run_ai_studio.sh`로 다시 시작한 뒤 요청이 처리됐습니다.

### `Killed` 및 서버 프로세스 종료

`Killed`는 프로세스가 강제 종료됐다는 뜻입니다. 메모리 부족(OOM)은 가능한 원인이지만, 로그 한 줄만으로 확정할 수는 없습니다. `/sys/fs/cgroup/memory.events`의 `oom_kill` 카운터를 확인해 판단합니다. 작업 상태 폴링 중 별도 PyTorch 진단 프로세스도 띄우지 말고, 생성 중 GPU 상태를 확인할 때는 가벼운 `nvidia-smi`를 사용합니다.

### 포트 기본값 `8080`과 실제 접속 포트 `8081` 불일치

별도 설정 없이 실행하면 스크립트가 `8080`을 사용합니다. Elice 프록시가 8081을 바라보는 경우 `AI_STUDIO_PORT=8081`을 지정하거나 `.env`에 설정해야 합니다.

## 8. 모델 다운로드·GPU·생성 시간 해석

첫 이미지 요청에서는 모델 가중치가 Hugging Face에서 다운로드됩니다. 로그의 `text_encoder`, `transformer`, `vae`, `Fetching 23 files`는 모델 파일 다운로드 및 캐시 작업입니다. 이번 사례에서는 모델 캐시가 `/home/elicer/.cache/huggingface/hub/` 아래에 생성됐습니다. 캐시를 지우지 않고 같은 사용자·캐시 경로를 사용하면 다음 요청에서 보통 전체 파일을 다시 받지 않습니다.

캐시 후에도 작업 제출 때마다 코드가 파이프라인을 새로 만들고 구성 요소를 로드하므로 로딩 시간이 듭니다. 실제 추론 진행은 `0/4`부터 `4/4`까지의 구간입니다. 기록된 생성에서는 이 구간이 약 1분 55초 걸렸습니다.

`enable_model_cpu_offload()`는 파이프라인 일부를 CPU 메모리에 두고 필요한 구성 요소를 GPU로 옮기는 방식입니다. CPU 메모리가 증가하는 것은 예상 가능한 현상이며, CPU offload 자체가 “GPU를 전혀 안 쓴다”는 뜻은 아닙니다. 그러나 로그만으로 실제 GPU 연산량을 판단할 수 없습니다.

이 환경의 진단에서 `.venv-vllm`의 `torch.cuda.is_available()`은 `True`, GPU 이름은 `NVIDIA A100 80GB PCIe MIG 3g.40gb`로 확인됐습니다. 이는 해당 Python 환경에서 CUDA 장치를 볼 수 있다는 뜻이지, 특정 이미지 작업의 GPU 활용률을 증명하는 것은 아닙니다. 추론 `0/4`~`4/4` 동안 별도 터미널에서 `nvidia-smi`로 확인합니다. 작업 중 다른 Python/PyTorch 테스트를 실행하면 GPU·시스템 메모리 사용량이 늘 수 있으므로 피합니다.

## 9. 프롬프트가 성별을 잘못 반영한 사례

한국어 프롬프트에 `한국의 여자 아이돌 모델`을 요청했으나 남성 이미지가 나온 사례가 있었습니다. 작업은 성공했지만 모델이 프롬프트의 세부 조건을 보장하지는 않았습니다. 한국어 이해와 4단계 빠른 생성의 한계가 영향을 줄 수 있습니다.

인물의 성별과 주요 특성을 프롬프트 앞부분에서 명확하고 구체적으로 설명하고, 영어 프롬프트도 시험할 수 있습니다. 예: `Photorealistic editorial fashion photograph of one adult Korean woman, a female K-pop idol, clearly feminine appearance, ... posing on a neon-lit Seoul street at night.` 현재 worker는 부정 프롬프트를 별도로 전달하지 않으므로 `No men` 등의 문구도 결과를 보장하지 않습니다. 시드나 프롬프트를 바꾸어 재시도합니다.

## 10. 산출물 경로 및 재시도

기본 산출물 디렉터리는 `apps/ai_studio/var/jobs/<작업ID>/`입니다. 일반 이미지 결과는 `result.png`입니다. 예를 들어 작업 ID가 `99593d32768b429c8770be386b575af8`이면 결과 경로는 다음과 같습니다.

`~/qwen/apps/ai_studio/var/jobs/99593d32768b429c8770be386b575af8/result.png`

`AI_STUDIO_DATA` 환경변수를 지정했다면 기본 경로 대신 그 위치에 저장됩니다. 이전 실패 작업은 자동으로 재실행되지 않으므로 새 작업을 제출합니다. 서버 재시작 후에는 앱 메모리의 기존 작업 상태가 사라질 수 있지만 이미 디스크에 저장된 파일은 기본적으로 남습니다.

## 11. 간단한 재현 체크리스트

- [ ] Hugging Face 모델 페이지에서 승인 상태 확인
- [ ] 서버를 실행하는 Linux 사용자로 로그인
- [ ] `whoami()`가 승인된 Hugging Face 계정을 표시
- [ ] `hf_hub_download(..., filename="model_index.json")` 성공
- [ ] AI Studio가 접속 주소와 같은 포트(이 사례는 8081)에서 실행
- [ ] `/api/health`가 200 응답
- [ ] AI Studio의 GPU 작업 허용 스위치 ON
- [ ] 새 이미지 작업을 제출하고 작업 상태 확인
- [ ] 생성 중에는 다른 PyTorch 테스트 대신 `nvidia-smi`로 GPU 관찰
- [ ] 결과는 `apps/ai_studio/var/jobs/<작업ID>/result.png`에서 확인
