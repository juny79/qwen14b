#!/usr/bin/env python3
"""vLLM OpenAI 호환 API의 기본 연결/응답 확인."""
import os
import sys
from openai import OpenAI

base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
api_key = os.getenv("VLLM_API_KEY", "lab-secret-change-me")
model = os.getenv("SERVED_MODEL_NAME", "qwen14b")

client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
try:
    models = client.models.list()
    print("API 모델 목록:")
    for item in models.data:
        print(" -", item.id)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "간결하게 답하는 한국어 비서입니다."},
            {"role": "user", "content": "A100 MIG 실습 API 연결 테스트입니다. 한국어로 한 문장만 답하세요."},
        ],
        max_tokens=96,
        temperature=0,
    )
    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("응답 본문이 비어 있습니다.")
    print("\n응답:", answer)
    if response.usage:
        print("토큰 사용량:", response.usage.model_dump())
except Exception as exc:
    print(f"[실패] {type(exc).__name__}: {exc}", file=sys.stderr)
    print("서버 터미널, 포트 8000, VLLM_API_KEY, served model 이름을 확인하세요.", file=sys.stderr)
    sys.exit(1)
