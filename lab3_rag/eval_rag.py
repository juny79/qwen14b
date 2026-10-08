#!/usr/bin/env python3
"""간단한 검색 회귀 점검. 생성 답변의 정확성은 별도 수동 평가가 필요합니다."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index"
MODEL = "intfloat/multilingual-e5-small"
CASES = [
    ("국내 출장 숙박비 한도는 얼마인가요?", "travel_policy.txt"),
    ("출장 신청은 며칠 전에 해야 하나요?", "travel_policy.txt"),
    ("영수증 정산 기한은 언제인가요?", "travel_policy.txt"),
]


def main():
    index_path, meta_path = INDEX_DIR / "documents.faiss", INDEX_DIR / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("인덱스가 없습니다. 먼저 python lab3_rag/build_index.py 를 실행하세요.")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    encoder = SentenceTransformer(MODEL, device="cpu")
    passed = 0
    for question, expected_source in CASES:
        query = encoder.encode(["query: " + question], normalize_embeddings=True,
                               convert_to_numpy=True).astype("float32")
        scores, ids = index.search(query, min(3, len(metadata)))
        hits = [metadata[int(i)] for i in ids[0] if i >= 0]
        ok = any(h["source"] == expected_source for h in hits)
        passed += int(ok)
        print(f"{'PASS' if ok else 'CHECK'} | {question}")
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0:
                item = metadata[int(idx)]
                print(f"  {float(score):.3f} {item['source']}#{item['chunk']}: {item['text'][:120]}")
    print(f"\n검색 테스트: {passed}/{len(CASES)} 기대 파일 포함")
    print("주의: 기대 문서가 검색되어도 답변 정확도·인용 일치는 별도로 사람이 확인해야 합니다.")


if __name__ == "__main__":
    main()
