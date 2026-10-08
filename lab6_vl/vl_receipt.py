#!/usr/bin/env python3
"""Qwen2.5-VL-7B로 영수증 이미지의 정보를 JSON으로 추출합니다."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-side", type=int, default=1024,
                        help="긴 변 최대 픽셀. 메모리 절약을 위해 기본 1024")
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"이미지를 찾을 수 없습니다: {args.image}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 미지원 GPU입니다.")

    torch.cuda.reset_peak_memory_stats(0)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        image.thumbnail((args.max_side, args.max_side))
        with tempfile.TemporaryDirectory(prefix="qwen-vl-") as tmp:
            resized = Path(tmp) / "resized.jpg"
            image.save(resized, quality=90)
            print(f"이미지 크기: {image.width}x{image.height}; 모델 로딩: {args.model}")
            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
                device_map={"": 0}, low_cpu_mem_usage=True,
                trust_remote_code=False)
            model.eval()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(resized)},
                    {"type": "text", "text": (
                        "영수증에서 상호명, 날짜(YYYY-MM-DD), 총액(숫자), 통화, "
                        "품목을 읽어 JSON 하나만 출력하세요. 읽을 수 없는 필드는 null로 두고 "
                        "추측하지 마세요. 형식: {\"merchant\":null,\"date\":null,"
                        "\"total\":null,\"currency\":null,\"items\":[]}"
                    )},
                ],
            }]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0].strip()
            print("\n모델 응답:\n" + answer)
            start, end = answer.find("{"), answer.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(answer[start:end + 1])
                    print("\nJSON 파싱: 성공")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                except json.JSONDecodeError as exc:
                    print(f"\nJSON 파싱: 실패 ({exc})")
            else:
                print("\nJSON 객체를 찾지 못했습니다. 응답을 직접 확인하세요.")
            print(f"최대 할당 GPU 메모리: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GiB")
            print("중요 금액/날짜는 원본과 대조하고, 없는 값을 만들어내지 않았는지 확인하세요.")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[CUDA OOM] {exc}", file=sys.stderr)
        print("다른 GPU 작업 종료 → 이미지 해상도 축소(--max-side 768) → 작업 입력 점검 순으로 대응하세요.", file=sys.stderr)
        sys.exit(2)
