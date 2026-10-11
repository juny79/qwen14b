"""Allow-listed adapters for the repository's local AI lab scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFER_PYTHON = ROOT / ".venv-infer" / "bin" / "python"
PYTHON = str(INFER_PYTHON if INFER_PYTHON.is_file() else Path(sys.executable))


def _run(command: list[str], log_path: Path, timeout: int = 14_400) -> None:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=timeout, check=False)
    log_path.write_text((completed.stdout or "") + "\n--- STDERR ---\n" +
                        (completed.stderr or ""), encoding="utf-8")
    if completed.returncode:
        tail = ((completed.stderr or completed.stdout or "")[-3000:])
        raise RuntimeError(f"실습 명령 실패(code={completed.returncode}): {tail}")


def run_lab(action: str, job_dir: Path, image_path: Path | None = None,
            request_text: str | None = None) -> Path:
    """Run only known repository scripts; never accepts arbitrary command arguments."""
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "run.log"
    if action == "rag-index":
        _run([PYTHON, str(ROOT / "lab3_rag/build_index.py")], log_path, timeout=3600)
        return log_path
    if action == "vllm-api-test":
        _run([PYTHON, str(ROOT / "lab2_vllm/test_api.py")], log_path, timeout=600)
        return log_path
    if action == "infer":
        result = job_dir / "infer_results.json"
        command = [PYTHON, str(ROOT / "lab1_infer/infer_qwen14b.py"), "--output", str(result)]
        _run(command, log_path)
        return result
    if action == "qlora-train":
        adapter_dir = job_dir / "adapter"
        command = [PYTHON, str(ROOT / "lab4_qlora/train_qlora.py"), "--output", str(adapter_dir)]
        _run(command, log_path)
        archive = job_dir / "qlora-adapter.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(adapter_dir, arcname="adapter")
        return archive
    if action == "classify":
        if not request_text or len(request_text) > 2000:
            raise ValueError("분류 요청은 1-2000자여야 합니다.")
        result = job_dir / "classification.json"
        completed = subprocess.run(
            [PYTHON, str(ROOT / "lab4_qlora/classify_request.py")], cwd=ROOT,
            input=json.dumps({"request": request_text}, ensure_ascii=False),
            capture_output=True, text=True, timeout=3600, check=False)
        log_path.write_text((completed.stdout or "") + "\n--- STDERR ---\n" +
                            (completed.stderr or ""), encoding="utf-8")
        if completed.returncode:
            raise RuntimeError((completed.stdout or completed.stderr)[-3000:])
        result.write_text(completed.stdout, encoding="utf-8")
        return result
    if action == "qlora-eval":
        adapter = Path(os.getenv("AI_STUDIO_ADAPTER", ROOT / "outputs/qwen25-7b-qlora")).resolve()
        if not adapter.is_dir():
            raise RuntimeError(f"QLoRA 어댑터 디렉터리가 없습니다: {adapter}")
        _run([PYTHON, str(ROOT / "lab4_qlora/eval_adapter.py"), "--adapter", str(adapter)], log_path)
        return log_path
    if action == "benchmark":
        result = job_dir / "bench_results.csv"
        _run([PYTHON, str(ROOT / "lab5_bench/bench.py"), "--concurrency", "1", "2", "4",
              "--output", str(result)], log_path, timeout=3600)
        return result
    if action == "vl-receipt":
        if image_path is None or not image_path.is_file():
            raise RuntimeError("영수증 이미지 파일을 지정해야 합니다.")
        _run([PYTHON, str(ROOT / "lab6_vl/vl_receipt.py"), "--image", str(image_path)],
             log_path, timeout=7200)
        return log_path
    raise ValueError("지원하지 않는 실습 작업입니다.")
