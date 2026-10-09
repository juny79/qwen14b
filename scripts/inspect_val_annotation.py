#!/usr/bin/env python3
"""Summarize val annotation schema and print records linked to one image."""
import argparse
import json
from pathlib import Path


def find_records(value, target_name, path="$", seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        for key, item in value.items():
            candidate = Path(str(key)).name
            if candidate == target_name or Path(candidate).stem == Path(target_name).stem:
                yield f"{path}.{key}", item
        for key, item in value.items():
            yield from find_records(item, target_name, f"{path}.{key}", seen)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from find_records(item, target_name, f"{path}[{index}]", seen)


def describe_record(path, record):
    print(f"\n--- Record at {path} ---")
    if not isinstance(record, dict):
        print("record type:", type(record).__name__)
        return
    print("record fields/types:", {key: type(value).__name__ for key, value in record.items()})
    words = record.get("words")
    if isinstance(words, dict):
        print("words count:", len(words))
        field_types = {}
        for word in words.values():
            if isinstance(word, dict):
                for key, value in word.items():
                    field_types.setdefault(key, set()).add(type(value).__name__)
        print("all word fields/types:", {
            key: sorted(types) for key, types in sorted(field_types.items())
        })
        for word_id, word in list(words.items())[:3]:
            if isinstance(word, dict):
                print(f"words[{word_id}] fields/types:",
                      {key: type(value).__name__ for key, value in word.items()})
                points = word.get("points")
                if isinstance(points, list):
                    print(f"words[{word_id}].points: {len(points)} points")
            else:
                print(f"words[{word_id}] type:", type(word).__name__)
    else:
        print("words field is absent or not an object")
        print("record preview:", json.dumps(record, ensure_ascii=False)[:2000])

    for key in ("img_w", "img_h"):
        if key in record:
            print(f"{key}:", record[key])


def schema(value, depth=0):
    if isinstance(value, dict):
        if depth >= 3:
            return {"type": "object"}
        fields = list(value.items())[:12]
        return {"type": "object", "fields": {
            str(key): schema(item, depth + 1) for key, item in fields
        }, "additional_fields": max(0, len(value) - len(fields))}
    if isinstance(value, list):
        result = {"type": "array", "length": len(value)}
        if value:
            result["item"] = schema(value[0], depth + 1)
        return result
    return {"type": type(value).__name__}


def walk(value, path="$", target_stem=""):
    # Kept as a compatibility wrapper for callers of the initial inspector.
    yield from find_records(value, target_stem, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, default=Path("datasets/jsons/val.json"))
    parser.add_argument(
        "--image", default="drp.en_ko.in_house.selectstar_000307.jpg",
        help="image filename or stem to locate in the annotation",
    )
    args = parser.parse_args()
    if not args.annotation.is_file():
        raise SystemExit(f"Annotation file not found: {args.annotation}")

    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    print(f"File: {args.annotation} ({args.annotation.stat().st_size} bytes)")
    print("Root schema:")
    print(json.dumps(schema(data), ensure_ascii=False, indent=2))

    matches = list(find_records(data, Path(args.image).name))
    print(f"\nRecords matching {args.image}: {len(matches)}")
    for path, record in matches[:3]:
        describe_record(path, record)
    if not matches:
        print("No record is keyed by the image filename. Check whether the JSON uses image IDs instead.")


if __name__ == "__main__":
    main()
