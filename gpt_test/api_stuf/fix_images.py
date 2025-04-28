#!/usr/bin/env python3
"""
Fix multimodal JSONL for OpenAI fine‑tune:

1) Change  "type": "input_image"  ➜  "type": "image_url"
2) Ensure   "image_url"  is a dict:  { "url": "https://…" }

It touches *only* items whose 'type' ends with "image".
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

# ------------------------------------------------------------------ #
# helpers                                                            #
# ------------------------------------------------------------------ #

def fix_image_item(item: Dict[str, Any]) -> bool:
    """
    In‑place repair of ONE image item.
    Returns True if any change was made.
    """
    changed = False

    # 1) normalise the type
    if item.get("type") == "input_image":
        item["type"] = "image_url"
        changed = True

    if item.get("type") != "image_url":
        return changed  # not an image item, nothing to do

    # 2) wrap string URL into dict
    url_field = item.get("image_url")

    if isinstance(url_field, str):
        item["image_url"] = {"url": url_field}
        changed = True
    elif isinstance(url_field, dict):
        # already dict; make sure it has a "url" key
        if "url" not in url_field:
            raise ValueError("found image_url dict without 'url'")
    else:
        raise ValueError("image_url must be string or dict with 'url'")

    return changed


def process_line(obj: Dict[str, Any]) -> int:
    """
    Apply `fix_image_item` to every candidate in this JSON object.
    Returns the number of items modified in the line.
    """
    fixes = 0
    for msg in obj.get("messages", []):
        content = msg.get("content")

        # content may be a list (image + text) or single dict
        if isinstance(content, list):
            for it in content:
                if isinstance(it, dict):
                    fixes += fix_image_item(it)
        elif isinstance(content, dict):
            fixes += fix_image_item(content)

    return fixes


# ------------------------------------------------------------------ #
# main                                                               #
# ------------------------------------------------------------------ #

def main(src: Path) -> None:
    dst = src.with_stem(src.stem + "_fixed")
    n_lines, n_fixes = 0, 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue  # skip empty lines
            n_lines += 1
            obj = json.loads(raw)
            n_fixes += process_line(obj)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✔ Finished: wrote {dst}")
    print(f"   processed {n_lines} lines, modified {n_fixes} image items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix image_url field in OpenAI multimodal JSONL")
    parser.add_argument("file", type=Path, help="path to your .jsonl file")
    args = parser.parse_args()

    if not args.file.exists():
        parser.error(f"{args.file} not found")
    main(args.file)
