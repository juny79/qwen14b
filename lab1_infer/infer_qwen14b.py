#!/usr/bin/env python3
"""Qwen2.5 기본 추론 실습. 실행 예: python lab1_infer/infer_qwen14b.py"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tasks import SYSTEM_PROMPT, TASKS

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 단일 MIG 장치 기본 추론 실습")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF 모델 ID (메모리 검증은 Qwen2.5-7B-Instruct)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--task", choices=["all"] + [t["name"] for t in TASKS], default="all")
    parser.add_argument("--load-in-4bit", action="store_true", help="NF4 4bit 로딩 (bitsandbytes 필요)")
    parser.add_argument("--output", default="logs/lab1_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. nvidia-smi, MIG 할당, PyTorch CUDA 빌드를 확인하세요.")
    if not args.load_in_4bit and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU가 BF16을 지원하지 않습니다. --load-in-4bit로 확인하거나 환경을 점검하세요.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens는 1 이상이어야 합니다.")

    torch.cuda.reset_peak_memory_stats(0)
    print(f"모델: {args.model}")
    print(f"장치: {torch.cuda.get_device_name(0)} | BF16={not args.load_in_4bit} | 4bit={args.load_in_4bit}")
    print("모델 다운로드/로딩 중입니다. 첫 실행은 시간이 오래 걸릴 수 있습니다.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    load_kwargs = {
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        # Qwen2.5의 sliding-window 동작 경고를 피하고 해당 마스크를 적용합니다.
        "attn_implementation": "eager",
    }
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print("모델 로딩 완료.")

    selected = TASKS if args.task == "all" else [t for t in TASKS if t["name"] == args.task]
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "load_in_4bit": args.load_in_4bit,
        "max_new_tokens": args.max_new_tokens,
        "tasks": [],
    }
    for task in selected:
        print(f"\n===== {task['title']} =====")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["prompt"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda:0")
        attention_mask = torch.ones_like(encoded)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        answer_ids = output[0, encoded.shape[1]:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        passed, detail = task["check"](answer)
        print(answer)
        print(f"\n참고 점검: {'통과' if passed else '확인 필요'} — {detail}")
        results["tasks"].append({
            "name": task["name"], "title": task["title"], "answer": answer,
            "check_passed": passed, "check_detail": detail,
            "input_tokens": int(encoded.shape[1]),
            "output_tokens": int(answer_ids.shape[0]),
            "generation_seconds": round(elapsed, 3),
        })
        del encoded, attention_mask, output
        torch.cuda.empty_cache()

    results["peak_gpu_memory_gib"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)
    results["peak_reserved_memory_gib"] = round(torch.cuda.max_memory_reserved(0) / 1024**3, 2)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최대 할당 GPU 메모리: {results['peak_gpu_memory_gib']} GiB")
    print(f"최대 예약 GPU 메모리: {results['peak_reserved_memory_gib']} GiB")
    print(f"결과 저장: {out_path}")
    print("자동 점검은 참고용입니다. 응답 품질과 근거를 직접 확인하세요.")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        print(f"\n[CUDA OOM] GPU 메모리 부족: {exc}", file=sys.stderr)
        print("대응 순서: 다른 GPU 작업 확인 → task 하나만 실행(--task summary) → 출력 축소(--max-new-tokens 128) → 7B 모델로 검증(--model Qwen/Qwen2.5-7B-Instruct) → 4bit 검토(--load-in-4bit)", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"\n[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

# 모델 ID는 Hugging Face 접근/라이선스/네트워크 조건에 따라 사용이 달라질 수 있습니다.
# 해당 코드와 패키지 조합은 실습용 초안이며, A100 MIG 인스턴스에서 검증되지 않았습니다.
# GPU 메모리는 model.generate 외 CUDA 컨텍스트 메모리를 포함하지 않는 allocated 수치입니다.
# 'all' 실행 중 OOM이면 앞서 완료한 task 결과는 JSON에 저장되지 않습니다.