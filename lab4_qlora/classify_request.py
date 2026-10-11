#!/usr/bin/env python3
"""Local QLoRA request-classification demo; always emits a human-review draft."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = os.getenv("QLORA_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER = Path(os.getenv("AI_STUDIO_ADAPTER", "outputs/qwen25-7b-qlora"))
SYSTEM = "당신은 사내 업무 요청 분류 도우미입니다. JSON 객체만 출력하세요. 허용 category는 출장정산, 출장신청, 보안, 시설예약, 기타입니다. priority는 높음, 일반, 낮음 중 하나입니다. 모르면 추측하지 말고 answer에 확인 필요를 적으세요."
ALLOWED_CATEGORIES = {"출장정산", "출장신청", "보안", "시설예약", "기타"}
ALLOWED_PRIORITIES = {"높음", "일반", "낮음"}


def main() -> None:
    payload = json.loads(sys.stdin.read())
    request = str(payload.get("request", "")).strip()
    if not request or len(request) > 2000:
        raise ValueError("request must contain 1-2000 characters")
    if not ADAPTER.is_dir():
        raise FileNotFoundError(f"QLoRA adapter not found: {ADAPTER}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant, device_map={"": 0},
        trust_remote_code=False, low_cpu_mem_usage=True,
        attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(prompt, max_new_tokens=180, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True)
    start, end = text.find("{"), text.rfind("}")
    data = None
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    valid = isinstance(data, dict) and data.get("category") in ALLOWED_CATEGORIES and data.get("priority") in ALLOWED_PRIORITIES and isinstance(data.get("answer"), str)
    result = {"request": request, "suggestion": data if valid else None,
              "needs_human_review": True,
              "warning": "데모 분류이며 학습 데이터가 작습니다. 자동 라우팅·업무 조치는 금지하고 담당자가 확인하세요.",
              "raw_output": text if not valid else None}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        sys.exit(1)
