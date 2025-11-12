import argparse
import json
import os
from typing import Dict, List


def get_directory_hierarchy(directory_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Return a mapping of each directory to its immediate sub-folders and files."""
    hierarchy: Dict[str, Dict[str, List[str]]] = {}
    for root, dirnames, filenames in os.walk(directory_path):
        hierarchy[root] = {
            "sub_folders": sorted(dirnames),
            "files": sorted(filenames),
        }
    return hierarchy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump directory hierarchy as JSON")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan")
    parser.add_argument("--out", type=str, required=False, help="Optional path to write JSON")
    args = parser.parse_args()

    data = get_directory_hierarchy(args.root)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote hierarchy to {args.out}")
    else:
        print(json.dumps(data, indent=2))

