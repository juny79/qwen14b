#!/usr/bin/env python3
"""TXT 문서를 multilingual-e5 임베딩 후 FAISS 인덱스로 저장합니다 (임베딩은 CPU)."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
OUT = ROOT / "index"
MODEL = "intfloat/multilingual-e5-small"
CHUNK_CHARS = 700
OVERLAP = 100


def split_text(text):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - OVERLAP
    return chunks


def main():
    files = sorted(DOCS.glob("*.txt"))
    if not files:
        raise SystemExit(f"TXT 문서가 없습니다: {DOCS}")
    records = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_text(text)):
            records.append({"source": path.name, "chunk": i, "text": chunk})
    if not records:
        raise SystemExit("인덱싱할 내용이 없습니다.")

    print(f"임베딩 모델 로딩(CPU): {MODEL}")
    encoder = SentenceTransformer(MODEL, device="cpu")
    vectors = encoder.encode(
        ["passage: " + r["text"] for r in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    OUT.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT / "documents.faiss"))
    (OUT / "metadata.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[완료] {len(records)}개 청크를 {OUT} 에 저장했습니다.")


if __name__ == "__main__":
    main()
