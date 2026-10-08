#!/usr/bin/env python3
"""Qwen2.5-7B 4bit QLoRA 실습. GPU의 다른 작업을 먼저 종료하세요."""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA = Path(__file__).resolve().parent / "data" / "train.jsonl"


class ChatDataset(Dataset):
    def __init__(self, tokenizer, path, max_length):
        self.rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": row["system"]},
                 {"role": "user", "content": row["user"]}],
                tokenize=True, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(
                [{"role": "system", "content": row["system"]},
                 {"role": "user", "content": row["user"]},
                 {"role": "assistant", "content": row["assistant"]}],
                tokenize=True, add_generation_prompt=False)
            if len(prompt) >= max_length:
                print(f"[경고] {line_no}행 prompt가 최대 길이를 넘어 건너뜁니다.")
                continue
            # Chat template 구현 차이로 prompt가 full의 접두사가 아닐 때는 공통 접두 길이를 사용.
            prefix = 0
            for p, f in zip(prompt, full):
                if p != f:
                    break
                prefix += 1
            full = full[:max_length]
            labels = list(full)
            labels[:prefix] = [-100] * min(prefix, len(labels))
            if all(x == -100 for x in labels):
                raise ValueError(f"{line_no}행은 학습할 답변 토큰이 없습니다.")
            self.rows.append((list(full), labels))
        if not self.rows:
            raise ValueError("유효한 학습 샘플이 없습니다.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids, labels = self.rows[idx]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        width = max(x["input_ids"].numel() for x in items)
        ids, labels, masks = [], [], []
        for item in items:
            n = width - item["input_ids"].numel()
            ids.append(torch.nn.functional.pad(item["input_ids"], (0, n), value=self.pad_id))
            labels.append(torch.nn.functional.pad(item["labels"], (0, n), value=-100))
            masks.append(torch.tensor([1] * (width - n) + [0] * n, dtype=torch.long))
        return {"input_ids": torch.stack(ids), "labels": torch.stack(labels),
                "attention_mask": torch.stack(masks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", default="outputs/qwen25-7b-qlora")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20,
                        help="20은 파이프라인 확인용이며 성능 향상 증거가 아닙니다.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 사용할 수 없습니다. 환경을 먼저 점검하세요.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 미지원 GPU입니다. 이 실습 설정을 그대로 실행하지 마세요.")
    if not args.data.exists():
        raise SystemExit(f"데이터 없음: {args.data}")

    print(f"데이터: {args.data} | 베이스 모델: {args.model}")
    print("주의: vLLM 및 다른 GPU 작업이 종료됐는지 확인하세요.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map={"": 0},
        trust_remote_code=False, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    dataset = ChatDataset(tokenizer, args.data, args.max_length)
    print(f"유효 샘플: {len(dataset)} | 최대 길이: {args.max_length} | steps: {args.max_steps}")
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=PadCollator(tokenizer.pad_token_id))
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[CUDA OOM] {exc}", file=sys.stderr)
        print("다른 GPU 작업 종료 → --max-length 256 → 7B 단일 샘플 점검 순으로 대응하세요.", file=sys.stderr)
        raise
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[완료] LoRA 어댑터 저장: {out}")
    print("주의: 저장물은 어댑터입니다. 원본 베이스 모델이 별도로 필요합니다.")


if __name__ == "__main__":
    main()
