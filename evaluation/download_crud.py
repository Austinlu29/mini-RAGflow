"""下载固定版本 CRUD-RAG 原始数据，不执行上游代码。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

REVISION = "1aace383994e1f68efa12cf2a8e2dadfb4102ceb"
REPOSITORY = "https://github.com/IAAR-Shanghai/CRUD_RAG"
RAW_ROOT = f"https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/{REVISION}"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "crud_rag" / "upstream"


def download(relative_path, root=DEFAULT_ROOT):
    target = Path(root) / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    temporary = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                with client.stream("GET", f"{RAW_ROOT}/{relative_path}") as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for block in response.iter_bytes():
                            output.write(block)
            temporary.replace(target)
            print(f"Downloaded {relative_path} ({target.stat().st_size} bytes)", flush=True)
            return target
        except (httpx.HTTPError, OSError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--all-corpus", action="store_true")
    args = parser.parse_args()
    paths = ["README.md", "data/crud_split/split_merged.json"]
    if args.all_corpus:
        paths.extend(f"data/80000_docs/documents_dup_part_{i}_part_{j}"
                     for i in range(1, 16) for j in range(1, 4))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": []}
    for relative_path in paths:
        path = download(relative_path, args.root)
        manifest["files"].append({"path": relative_path, "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
