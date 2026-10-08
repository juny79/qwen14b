#!/usr/bin/env python3
"""베이스 모델과 LoRA 어댑터를 같은 검증 프롬프트로 비교합니다."""
import argparse
import json
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CASES = [
    ("출장 영수증은 언제까지 제출하나요?", "출장정산"),
    ("내 비밀번호를 알려줘.", "보안"),
    ("사무실 회의실 예약을 취소하고 싶어요.", "시설예약"),
]
SYSTEM = "당신은 사내 업무 요청을 분류하는 도우미입니다. 지정된 JSON 형식으로만 답하세요."


def generate(model, tokenizer, prompt):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=True,
                                        return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=160, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def find_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="outputs/qwen25-7b-qlora")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 미지원 GPU입니다.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map={"": 0},
        trust_remote_code=False, low_cpu_mem_usage=True)
    base.eval()
    baseline = []
    print("===== 베이스 모델 =====")
    for prompt, expected in CASES:
        text = generate(base, tokenizer, prompt)
        data = find_json(text)
        ok = data is not None and data.get("category") == expected
        baseline.append((text, ok))
        print(f"질문: {prompt}\n응답: {text}\nJSON/분류 점검: {'PASS' if ok else 'CHECK'} (기대 {expected})\n")

    try:
        adapted = PeftModel.from_pretrained(base, args.adapter)
    except Exception as exc:
        print(f"[실패] 어댑터 로딩: {exc}", file=sys.stderr)
        print("학습이 완료되어 adapter_config.json 이 존재하는지 확인하세요.", file=sys.stderr)
        raise
    adapted.eval()
    adapter_passes = 0
    print("===== LoRA 어댑터 적용 모델 =====")
    for i, (prompt, expected) in enumerate(CASES):
        text = generate(adapted, tokenizer, prompt)
        data = find_json(text)
        ok = data is not None and data.get("category") == expected
        adapter_passes += int(ok)
        print(f"질문: {prompt}\n응답: {text}\nJSON/분류 점검: {'PASS' if ok else 'CHECK'} (기대 {expected})\n")
        print(f"베이스 동일 사례 점검: {'PASS' if baseline[i][1] else 'CHECK'}")

    print(f"\n요약: 베이스 통과 {sum(ok for _, ok in baseline)}/{len(CASES)}, "
          f"어댑터 통과 {adapter_passes}/{len(CASES)}")
    print("이 작은 검증셋은 데모용입니다. 성능 향상 주장에는 별도 분리된 테스트셋과 사람 평가가 필요합니다.")


if __name__ == "__main__":
    main()
