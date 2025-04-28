#!/usr/bin/env python3
"""
Deep‑search every dict / list in a JSONL line and replace
  "input_text"  -> "text"
  "input_image" -> "image_url"
Usage:
    python fix_types_recursive.py /path/to/file.jsonl
Creates  <file>_fixed.jsonl  alongside the original.
"""

import argparse
import json
from pathlib import Path
from typing import Any

def replace_types(obj: Any, counters: dict[str, int]) -> None:
    """Recursively descend and fix in‑place."""
    if isinstance(obj, dict):
        # check this level
        if obj.get("type") == "input_text":
            obj["type"] = "text"
            counters["text"] += 1
        elif obj.get("type") == "input_image":
            obj["type"] = "image_url"
            counters["image"] += 1
        # then recurse
        for v in obj.values():
            replace_types(v, counters)
    elif isinstance(obj, list):
        for item in obj:
            replace_types(item, counters)

def fix_file(src: Path) -> Path:
    dst = src.with_stem(src.stem + "_fixed")
    counters = {"text": 0, "image": 0}

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_no}: invalid JSON") from e

            replace_types(obj, counters)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✔ Finished: wrote {dst}")
    print(f"  Replaced input_text  → text      : {counters['text']}")
    print(f"  Replaced input_image → image_url : {counters['image']}")
    return dst

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fix nested 'type' fields in a JSONL dataset.")
    ap.add_argument("file", type=Path, help="Path to the original .jsonl file")
    args = ap.parse_args()

    if not args.file.exists():
        ap.error(f"{args.file} does not exist.")
    fix_file(args.file)
