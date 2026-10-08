#!/usr/bin/env python3
"""vLLM 비스트리밍 완료 지연/출력 처리량 측정. 첫 토큰 지연(TTFT)은 측정하지 않습니다."""
import argparse
import asyncio
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


async def one_request(client, url, headers, model, prompt, max_tokens, semaphore):
    started = time.perf_counter()
    async with semaphore:
        try:
            response = await client.post(url, headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
            })
            response.raise_for_status()
            data = response.json()
            elapsed = time.perf_counter() - started
            usage = data.get("usage", {})
            text = data["choices"][0]["message"].get("content") or ""
            completion_tokens = usage.get("completion_tokens")
            return {"ok": True, "seconds": elapsed,
                    "tokens": int(completion_tokens) if completion_tokens is not None else None,
                    "text_chars": len(text), "error": ""}
        except Exception as exc:
            return {"ok": False, "seconds": time.perf_counter() - started,
                    "tokens": None, "text_chars": 0,
                    "error": f"{type(exc).__name__}: {exc}"}


async def run_group(concurrency, requests_count, url, headers, model, prompt, max_tokens):
    timeout = httpx.Timeout(180.0, connect=10.0)
    limits = httpx.Limits(max_connections=max(4, concurrency))
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        return await asyncio.gather(*[
            one_request(client, url, headers, model, prompt, max_tokens, semaphore)
            for _ in range(requests_count)
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--requests-per-level", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", default="logs/bench_results.csv")
    args = parser.parse_args()
    if any(c < 1 for c in args.concurrency) or args.requests_per_level < 1:
        parser.error("동시성 및 요청 수는 1 이상이어야 합니다.")

    base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    url = base + "/chat/completions"
    headers = {"Authorization": "Bearer " + os.getenv("VLLM_API_KEY", "lab-secret-change-me")}
    model = os.getenv("SERVED_MODEL_NAME", "qwen14b")
    prompt = "A100 MIG 실습입니다. 출장 정산에서 근거를 확인해야 하는 이유를 한국어로 간단히 설명하세요."
    rows = []

    # 1회 워밍업. 성능 통계에서는 제외합니다.
    print("워밍업 요청...")
    warm = asyncio.run(run_group(1, 1, url, headers, model, prompt, args.max_tokens))[0]
    if not warm["ok"]:
        raise SystemExit(f"워밍업 실패: {warm['error']} (서버/키/모델 확인)")

    for concurrency in args.concurrency:
        print(f"\n동시성 {concurrency}: {args.requests_per_level}회 요청")
        batch_started = time.perf_counter()
        results = asyncio.run(run_group(concurrency, args.requests_per_level, url,
                                        headers, model, prompt, args.max_tokens))
        batch_wall = time.perf_counter() - batch_started
        good = [r for r in results if r["ok"]]
        latencies = [r["seconds"] for r in good]
        token_count = sum(r["tokens"] or 0 for r in good)
        elapsed_wall = batch_wall
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "concurrency": concurrency,
            "requests": len(results),
            "successes": len(good),
            "failures": len(results) - len(good),
            "mean_latency_s": round(statistics.mean(latencies), 3) if latencies else "",
            "median_latency_s": round(statistics.median(latencies), 3) if latencies else "",
            "p95_latency_s": round(percentile(latencies, 0.95), 3) if latencies else "",
            "max_latency_s": round(max(latencies), 3) if latencies else "",
            "total_output_tokens": token_count,
            "aggregate_output_tokens_per_s": round(token_count / elapsed_wall, 3) if elapsed_wall else 0,
            "max_tokens_per_request": args.max_tokens,
        }
        rows.append(row)
        print(row)
        for item in results:
            if not item["ok"]:
                print("  요청 실패:", item["error"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV 저장: {out}")
    print("이 측정은 비스트리밍 완료 지연이며 TTFT/장시간 안정성을 측정하지 않습니다.")


if __name__ == "__main__":
    main()
