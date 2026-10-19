#!/usr/bin/env python3
"""Local-only development MVP; not production-ready."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.ai_studio.lab_jobs import run_lab

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AI_STUDIO_DATA", APP_DIR / "var")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 12 * 1024 * 1024
MODELS = {
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    "sdxl-inpaint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "wan22-ti2v": "Wan-AI/Wan2.2-TI2V-5B",
}
app = FastAPI(title="사내 AI Studio (개발 MVP)", version="0.1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_gpu_lock = threading.Lock()
_gpu_state_lock = threading.Lock()
_gpu_enabled = False
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-job")


class GpuSettings(BaseModel):
    enabled: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=8)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields, updated_at=now())


def gpu_is_enabled() -> bool:
    with _gpu_state_lock:
        return _gpu_enabled


def require_gpu_enabled() -> None:
    if not gpu_is_enabled():
        raise HTTPException(status_code=409, detail="웹앱에서 GPU 사용을 먼저 허용하세요.")


def submit_gpu_job(job_id: str, worker, *args, requires_gpu: bool = True) -> None:
    def run() -> None:
        try:
            # This lock only serializes jobs in this process, not a separate vLLM server.
            with _gpu_lock:
                if requires_gpu and not gpu_is_enabled():
                    raise RuntimeError("GPU 사용이 비활성화되어 작업을 시작하지 않았습니다.")
                set_job(job_id, status="running", started_at=now())
                output = worker(*args)
            set_job(job_id, status="succeeded", result=f"/api/jobs/{job_id}/result",
                    result_name=Path(output).name)
        except Exception as exc:
            set_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    _executor.submit(run)


@app.get("/")
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "development", "note": "Local-only prototype; no SSO/ACL or cross-process GPU lock."}


@app.get("/api/gpu")
def get_gpu_setting():
    return {"enabled": gpu_is_enabled(), "scope": "AI Studio requests only; does not control external GPU processes"}


@app.put("/api/gpu")
def set_gpu_setting(settings: GpuSettings):
    global _gpu_enabled
    with _gpu_state_lock:
        _gpu_enabled = settings.enabled
    return {"enabled": settings.enabled,
            "note": "Turning GPU off blocks new app jobs; it does not stop a job already running."}


@app.get("/api/features")
def features():
    repo_value = os.getenv("WAN22_REPO", "")
    ckpt_value = os.getenv("WAN22_CKPT", "")
    python_value = os.getenv("WAN22_PYTHON", sys.executable)
    wan_repo = Path(repo_value).expanduser() if repo_value else Path("/__wan_repo_not_configured__")
    wan_ckpt = Path(ckpt_value).expanduser() if ckpt_value else Path("/__wan_ckpt_not_configured__")
    wan_python = Path(python_value).expanduser()
    wan_ready = ((wan_repo / "generate.py").is_file() and wan_ckpt.is_dir()
                 and wan_python.is_file() and os.access(wan_python, os.X_OK))
    return {"support": True, "labs": [
        {"key": "rag-index", "enabled": True},
        {"key": "vllm-api-test", "enabled": True},
        {"key": "infer", "enabled": (ROOT / ".venv-infer/bin/python").is_file()},
        {"key": "qlora-train", "enabled": (ROOT / ".venv-infer/bin/python").is_file()},
        {"key": "qlora-eval", "enabled": (ROOT / ".venv-infer/bin/python").is_file()},
        {"key": "classify", "enabled": (ROOT / ".venv-infer/bin/python").is_file()},
        {"key": "benchmark", "enabled": True},
        {"key": "vl-receipt", "enabled": (ROOT / ".venv-infer/bin/python").is_file()},
    ], "models": [
        {"key": "flux-schnell", "model_id": MODELS["flux-schnell"], "task": "text-to-image", "enabled": True},
        {"key": "sdxl-inpaint", "model_id": MODELS["sdxl-inpaint"], "task": "masked-inpainting", "enabled": True},
        {"key": "wan22-ti2v", "model_id": MODELS["wan22-ti2v"], "task": "text-or-image-to-video", "enabled": wan_ready,
         "note": "Requires local Wan2.2 official repository and checkpoint"},
    ]}


def wan_video_worker(prompt: str, image_path: Path | None, output_path: Path,
                     seed: int, frame_num: int) -> Path:
    repo = Path(os.getenv("WAN22_REPO", "")).resolve()
    checkpoint = Path(os.getenv("WAN22_CKPT", "")).resolve()
    script = repo / "generate.py"
    python_value = os.getenv("WAN22_PYTHON", sys.executable)
    # Keep the virtualenv's bin/python path intact; resolving its symlink can
    # launch the system interpreter and lose the virtualenv site-packages.
    wan_python = Path(python_value).expanduser().absolute()
    if not script.is_file() or not checkpoint.is_dir():
        raise RuntimeError("Set WAN22_REPO and WAN22_CKPT to approved local Wan2.2 paths")
    if not wan_python.is_file() or not os.access(wan_python, os.X_OK):
        raise RuntimeError(f"WAN22_PYTHON is not an executable Python: {wan_python}")
    probe = subprocess.run([str(wan_python), "-c", "import torch, flash_attn"],
                           cwd=repo, capture_output=True, text=True, check=False)
    if probe.returncode:
        detail = (probe.stderr or probe.stdout)[-1200:]
        raise RuntimeError(f"WAN22_PYTHON must import torch and flash_attn: {detail}")
    command = [str(wan_python), str(script), "--task", "ti2v-5B",
               "--size", "1280*704", "--ckpt_dir", str(checkpoint),
               "--offload_model", "True", "--convert_model_dtype", "--t5_cpu",
               "--prompt", prompt, "--frame_num", str(frame_num),
               "--base_seed", str(seed), "--save_file", str(output_path)]
    if image_path is not None:
        command.extend(["--image", str(image_path)])
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True,
                               timeout=int(os.getenv("WAN_TIMEOUT_SECONDS", "2400")),
                               check=False)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-1200:]
        raise RuntimeError(f"Wan worker exited {completed.returncode}: {tail}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Wan worker exited without creating an MP4")
    return output_path


@app.post("/api/media/video")
async def create_video_job(prompt: str = Form(...), seed: int = Form(42),
                           frame_num: int = Form(49), image: UploadFile | None = File(default=None)):
    require_gpu_enabled()
    prompt = prompt.strip()
    if not prompt or len(prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt must contain 1-2000 characters")
    if frame_num not in {17, 49, 81}:
        raise HTTPException(status_code=400, detail="Frame count must be 17, 49, or 81")
    if not features()["models"][2]["enabled"]:
        raise HTTPException(status_code=503, detail="Wan2.2 local repository/checkpoint is not configured")
    job_id = uuid.uuid4().hex
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True)
    source_path = None
    if image is not None:
        source_path = folder / "source"
        try:
            await save_upload(image, source_path)
            load_image(source_path, "RGB")
        except Exception:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
            raise
    output_path = folder / "result.mp4"
    job = {"id": job_id, "status": "queued", "operation": "video",
           "model": MODELS["wan22-ti2v"], "created_at": now(), "updated_at": now()}
    with _jobs_lock:
        _jobs[job_id] = job
    submit_gpu_job(job_id, wan_video_worker, prompt, source_path, output_path, seed, frame_num)
    return job


def run_rag(question: str, top_k: int) -> dict[str, Any]:
    runner = ROOT / ".venv-infer/bin/python"
    if not runner.is_file():
        raise RuntimeError("RAG용 .venv-infer 환경이 없습니다.")
    with _gpu_lock:
        completed = subprocess.run(
            [str(runner), str(APP_DIR / "rag_service.py")], cwd=ROOT,
            input=json.dumps({"question": question, "top_k": top_k}),
            capture_output=True, text=True, timeout=150, check=False)
    if completed.returncode:
        try:
            detail = json.loads(completed.stdout).get("error", "RAG 실행 실패")
        except Exception:
            detail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(detail)
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise RuntimeError("RAG worker가 JSON이 아닌 결과를 반환했습니다.") from exc


@app.post("/api/support/chat")
async def support_chat(request: ChatRequest):
    require_gpu_enabled()
    try:
        return await asyncio.to_thread(run_rag, request.question, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def load_image(path: Path, mode: str):
    from PIL import Image
    image = Image.open(path)
    image.load()
    if image.width * image.height > 16_000_000:
        raise ValueError("Image exceeds 16 megapixels")
    return image.convert(mode)


def flux_worker(prompt: str, out: Path, seed: int) -> Path:
    import torch
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(MODELS["flux-schnell"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    image = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=4,
                 max_sequence_length=256,
                 generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    image.save(out)
    del pipe
    return out


def inpaint_worker(prompt: str, source: Path, mask_path: Path, out: Path, seed: int) -> Path:
    import torch
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(MODELS["sdxl-inpaint"],
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_model_cpu_offload()
    image = load_image(source, "RGB").resize((1024, 1024))
    mask = load_image(mask_path, "L").resize((1024, 1024))
    result = pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=20,
                  guidance_scale=7.5, generator=torch.Generator("cpu").manual_seed(seed))
    result.images[0].save(out)
    del pipe
    return out


async def save_upload(upload: UploadFile, path: Path) -> None:
    total = 0
    with path.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Each upload is limited to 12 MB")
            target.write(chunk)
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload")


@app.post("/api/media/image")
async def create_image_job(operation: str = Form(...), prompt: str = Form(...),
                           seed: int = Form(42), image: UploadFile | None = File(default=None),
                           mask: UploadFile | None = File(default=None)):
    require_gpu_enabled()
    prompt = prompt.strip()
    if not prompt or len(prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt must contain 1-2000 characters")
    if operation not in {"generate", "inpaint"}:
        raise HTTPException(status_code=400, detail="Unsupported operation")
    if operation == "inpaint" and (image is None or mask is None):
        raise HTTPException(status_code=400, detail="Inpainting requires source image and mask")
    job_id = uuid.uuid4().hex
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True)
    out = folder / "result.png"
    try:
        if operation == "inpaint":
            source_path, mask_path = folder / "source", folder / "mask"
            await save_upload(image, source_path)
            await save_upload(mask, mask_path)
            # Validate files before scheduling GPU work.
            load_image(source_path, "RGB")
            load_image(mask_path, "L")
    except Exception:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        raise
    model_key = "flux-schnell" if operation == "generate" else "sdxl-inpaint"
    job = {"id": job_id, "status": "queued", "operation": operation,
           "model": MODELS[model_key], "created_at": now(), "updated_at": now()}
    with _jobs_lock:
        _jobs[job_id] = job
    if operation == "generate":
        submit_gpu_job(job_id, flux_worker, prompt, out, seed)
    else:
        submit_gpu_job(job_id, inpaint_worker, prompt, source_path, mask_path, out, seed)
    return job


@app.post("/api/labs/run")
async def create_lab_job(action: str = Form(...), request_text: str = Form(""),
                         image: UploadFile | None = File(default=None)):
    allowed = {"rag-index", "vllm-api-test", "infer", "qlora-train", "qlora-eval", "classify", "benchmark", "vl-receipt"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail="지원하지 않는 실습 기능입니다.")
    if action != "rag-index":
        require_gpu_enabled()
    if action == "vl-receipt" and image is None:
        raise HTTPException(status_code=400, detail="VL 영수증 분석에는 이미지가 필요합니다.")
    request_text = request_text.strip()
    if action == "classify" and not 1 <= len(request_text) <= 2000:
        raise HTTPException(status_code=400, detail="분류 요청은 1-2000자여야 합니다.")
    job_id = uuid.uuid4().hex
    folder = JOBS_DIR / job_id
    folder.mkdir(parents=True)
    image_path = None
    try:
        if image is not None:
            image_path = folder / "input-image"
            await save_upload(image, image_path)
            load_image(image_path, "RGB")
    except Exception:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        raise
    job = {"id": job_id, "status": "queued", "operation": action,
           "created_at": now(), "updated_at": now()}
    with _jobs_lock:
        _jobs[job_id] = job
    submit_gpu_job(job_id, run_lab, action, folder, image_path, request_text or None,
                   requires_gpu=action != "rag-index")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs or _jobs[job_id].get("status") != "succeeded":
            raise HTTPException(status_code=404, detail="Result not ready")
    name = _jobs[job_id].get("result_name", "")
    allowed_names = {"result.png", "result.mp4", "infer_results.json", "classification.json",
                     "qlora-adapter.tar.gz", "run.log", "bench_results.csv"}
    if name not in allowed_names:
        raise HTTPException(status_code=404, detail="Result file missing")
    result = (JOBS_DIR / job_id / name).resolve()
    if result.parent != (JOBS_DIR / job_id).resolve() or not result.is_file():
        raise HTTPException(status_code=404, detail="Result file missing")
    media_type = {"result.mp4": "video/mp4", "result.png": "image/png",
                  "infer_results.json": "application/json", "classification.json": "application/json",
                  "bench_results.csv": "text/csv",
                  "run.log": "text/plain", "qlora-adapter.tar.gz": "application/gzip"}.get(name, "application/octet-stream")
    return FileResponse(result, filename=f"ai-studio-{name}", media_type=media_type)
