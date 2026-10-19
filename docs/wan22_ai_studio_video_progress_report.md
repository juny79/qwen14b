# Wan2.2 및 AI Studio 동영상 생성 진행 보고서

- **문서 목적:** Wan2.2 직접 실행, AI Studio 연결, 가상환경 및 Source Control 정리 과정과 현재 남은 확인 사항을 기록한다.
- **대상 프로젝트:** `/home/elicer/qwen` AI Studio와 `/home/elicer/models/Wan2.2` Wan2.2 저장소
- **상태:** Wan2.2 CLI 생성은 성공. AI Studio 연결 코드는 수정 중이며, 최신 수정 코드로 실행한 최종 동영상 작업은 아직 확인되지 않음.
- **범위:** 사용자 공유 로그 및 현재 저장소의 코드·설정을 바탕으로 작성. GPU 성능이나 최종 AI Studio 성공을 추가로 검증한 보고서는 아님.

## 1. 요약

Wan2.2 TI2V-5B 체크포인트와 필수 VAE/T5 파일이 확인되었다. CUDA 컴파일러와 C++ 컴파일러도 사용할 수 있었다. Wan2.2 가상환경에 `flash-attn`을 설치한 뒤 17프레임, 5 denoising steps의 CLI smoke test가 성공해 MP4 결과를 생성했다.

AI Studio는 별도의 `.venv-vllm` Python으로 구동되며, 이 환경에는 `flash_attn`이 설치되어 있지 않았다. 따라서 AI Studio 영상 worker가 Wan2.2 전용 `.venv-wan` Python을 실행하도록 설정을 추가했다. 첫 시도에서 시스템 Python이 사용되는 원인을 찾아 코드도 수정했으나, 수정 후 새 프로세스로 재기동해 AI Studio 동영상 작업이 성공하는지 확인하는 단계가 남아 있다.

## 2. 체크포인트와 Wan2.2 CLI 준비

체크포인트 디렉터리 `/home/elicer/models/Wan2.2-TI2V-5B`에서 다음을 확인했다.

- `Wan2.2_VAE.pth`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `google/umt5-xxl/`
- `diffusion_pytorch_model-00001-of-00003.safetensors` 등 3개 가중치 shard와 index
- 체크포인트 디렉터리 약 32 GB

사용자 실행 기록상 `nvcc` 12.8과 `g++` 11.4가 설치되어 있었다. Wan2.2 Python 가상환경 `/home/elicer/models/Wan2.2/.venv-wan`에 `flash-attn 2.8.3.post2`가 빌드·설치되었으며, PyTorch는 `2.5.1+cu121`이다.

### CLI smoke test 결과

17프레임, `1280*704`, `--sample_steps 5`, CPU offload 설정으로 다음 프롬프트를 실행했다: “A small paper boat floating on a calm pond at sunrise.”

- sampling: 5/5 완료, 표시 시간 약 10초
- 전체 작업: 모델 로딩과 저장을 포함해 약 2분 37초
- 결과: `/home/elicer/models/Wan2.2/wan_ti2v_smoke.mp4`, 약 670 KB
- CLI 테스트는 성공적으로 완료됨

17프레임은 24fps 기준 재생 길이가 약 0.71초라는 뜻이며, 생성에 걸리는 시간이 0.71초라는 뜻은 아니다.

## 3. AI Studio 연결 변경

### 설정

프로젝트 `.env`에서 아래 경로를 설정하도록 안내했다. 실제 `.env` 파일은 로컬 설정이며 Git에 커밋하지 않는다.

- `WAN22_REPO=/home/elicer/models/Wan2.2`
- `WAN22_CKPT=/home/elicer/models/Wan2.2-TI2V-5B`
- `WAN22_PYTHON=/home/elicer/models/Wan2.2/.venv-wan/bin/python`

`scripts/load_env.sh`의 allow-list에 `WAN22_PYTHON`을 추가했다. 사용자가 직접 셸에서 `WAN22_PYTHON`으로 모듈을 검사했을 때 `torch 2.5.1+cu121`, `flash-attn 2.8.3.post2` import가 성공했다.

### 코드 변경

`apps/ai_studio/app.py`에서 다음을 변경했다.

1. 동영상 기능 활성화 여부 판단에 Wan Python 실행 파일 유무도 포함한다.
2. 영상 생성 직전에 설정된 Python에서 `torch`와 `flash_attn` import를 검사한다.
3. Wan 생성 subprocess는 AI Studio 자체 Python이 아니라 `WAN22_PYTHON`을 실행한다.
4. 가상환경의 `bin/python` 경로에 `Path.resolve()`를 적용하지 않도록 수정했다.

### 확인된 장애 원인

AI Studio 작업에서 `ModuleNotFoundError: No module named 'torch'`가 두 차례 발생했다. 설정된 경로를 확인하니 올바른 Wan 가상환경이었고, 해당 경로를 셸에서 직접 실행하면 import가 성공했다. 코드가 `WAN22_PYTHON`에 `Path.resolve()`를 호출해 가상환경의 `bin/python` 심볼릭 링크를 시스템 Python 경로로 변환하고, 그 결과 시스템 Python에서 `torch`를 찾으려 했던 것이 원인으로 확인되었다.

이를 막기 위해 worker에서는 실행 경로를 절대경로로 정규화하되 심볼릭 링크는 따라가지 않는 `Path.absolute()`를 사용하도록 수정했다. 수정 후 `apps/ai_studio/app.py`의 `py_compile`과 가상환경 경로로 import를 점검하도록 명령을 실행했다. 다만 터미널 실행 결과를 독립적으로 수집하지 못했고, 수정 코드로 AI Studio worker를 실행한 성공 로그도 아직 없다.

## 4. 실행시간 해석

CLI smoke test는 `--sample_steps 5`를 사용했다. 반면 현재 AI Studio의 `wan_video_worker`는 `--sample_steps`를 전달하지 않는다. 따라서 Wan2.2의 기본값(로그상 50 steps)이 사용될 수 있다. 이 때문에 AI Studio에서는 CLI smoke test보다 sampling 시간이 늘어날 수 있다. 추가로 각 작업에서 체크포인트를 다시 로드한다.

AI Studio UI는 작업 완료 전 subprocess의 진행 출력을 표시하지 않고 상태를 `running`으로 보여주는 구조다. 짧은 재생 길이가 짧은 실행 시간을 보장하지 않으며, smoke test용 스텝 수 제어 기능은 현재 UI/worker에 구현되어 있지 않다.

## 5. Source Control 정리

`.gitignore`에 다음 종류의 로컬/생성 파일 제외 규칙을 추가했다.

- 저장소 루트 및 하위의 `.venv*`, `venv*`, `env*`, `virtualenv*` 등 가상환경
- `site-packages`, `dist-packages`, `__pypackages__`, Python 캐시 및 테스트 산출물
- 로그, 모델 가중치, 체크포인트, 로컬 데이터/결과, 빌드 산출물

이미 Git이 추적 중인 파일은 `.gitignore`만으로 숨겨지지 않는다. 가상환경 및 패키지 설치 경로의 tracked 파일을 `git rm --cached`로 Git 인덱스에서 제거하는 명령을 시도했으나, 이후 staged 상태와 제거 범위를 `git status`/`git diff --cached`로 확인하지 못했다. 따라서 이 정리가 완결되었다고 간주하지 말고, Source Control에서 실제 staged 목록을 검토해야 한다. 의도된 가상환경 제거와 `.gitignore`만 포함된 것을 확인한 후에만 커밋한다. `--cached`는 로컬 설치 파일을 삭제하지 않지만, 삭제 내역은 커밋 전까지 Source Control에 표시된다.

작업 도중 `apps/ai_studio/README.md`와 `scripts/load_env.sh`에 기존 변경이 확인되었으므로, 관련 없는 변경을 정리/되돌리거나 가상환경 cleanup 커밋에 섞지 않는다.

## 6. 현재 상태와 남은 작업

| 항목 | 상태 |
|---|---|
| TI2V-5B 체크포인트 파일 | 사용자 출력으로 확인 완료 |
| `flash-attn` 설치 | Wan 가상환경에 설치 완료 |
| Wan2.2 CLI 17프레임/5 steps smoke test | 사용자 출력으로 MP4 생성 확인 완료 |
| AI Studio에 `WAN22_PYTHON` 설정 연결 | 코드 및 allow-list 수정 완료 |
| `Path.resolve()` 심볼릭 링크 문제 | 코드 수정 완료, 최신 서버에서 재검증 필요 |
| AI Studio 최신 코드 재기동 | 재기동 명령을 시도했지만 새 PID/최신 기동 로그를 검증하지 못함 |
| AI Studio를 통한 동영상 생성 | 아직 성공 확인 안 됨 |
| Git 인덱스에서 기존 환경 파일 제거 | 명령을 시도했으나 staged 목록 확인 및 커밋 미완료 |

### 다음 확인 순서

1. `/home/elicer/qwen/.env`의 `WAN22_REPO`, `WAN22_CKPT`, `WAN22_PYTHON` 경로를 확인한다.
2. Wan Python에서 `import torch, flash_attn`이 성공하는지 확인한다.
3. 기존 AI Studio 프로세스를 종료하고 최신 코드를 다시 실행한다. 새 PID와 `logs/ai_studio.log`의 기동 기록, `/api/health` 응답을 확인한다.
4. AI Studio 동영상 기능이 활성화되어 있는지 확인하고, 다른 GPU 작업을 종료한 상태에서 17프레임 T2V smoke test를 실행한다.
5. 작업 오류가 발생하면 새 job ID와 전문 오류를 기록한다. 성공하면 결과 MP4 경로/크기와 총 소요 시간을 기록한다.
6. Source Control에서 staged 삭제 경로를 검토한다. `.venv*` 및 패키지 설치 경로만 정리 대상인지 확인하고, 관련 없는 소스/문서 변경은 별도로 유지한다.

## 7. 사용 및 운영 주의

- AI Studio의 기본 작업 큐는 별도 vLLM 프로세스나 다른 사용자의 GPU 사용을 제어하지 않는다. 동영상 생성 전 다른 GPU 작업을 수동으로 중지/확인한다.
- AI Studio는 로컬 개발 MVP다. localhost에만 바인딩하고 민감 자료를 업로드하지 않는다.
- Wan 저장소/체크포인트는 외부 로컬 경로이며 Git에 추가하지 않는다.
- 5 steps는 smoke test용이다. 더 긴 스텝 수의 품질, VRAM, 실행시간은 별도로 확인해야 하며, 현재 AI Studio 기본 worker 동작은 직접 CLI 5-step 테스트와 같지 않다.
