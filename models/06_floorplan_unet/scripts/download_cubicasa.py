"""Download CubiCasa5K from Zenodo (~5.5 GB). NonCommercial license — see LICENSE_NOTES.md."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content"
EXPECTED_MD5 = "0ce0b203d1e3c125b51087b219bd23b9"
ZIP_PATH = ROOT / "data" / "cubicasa5k.zip"
EXTRACT_DIR = ROOT / "data"


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading ~5.5 GB → {ZIP_PATH}")
        print("This can take a long time. CC BY-NC — research use only.")
        with requests.get(URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            done = 0
            tmp = ZIP_PATH.with_suffix(".zip.part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r{100 * done / total:.1f}% ({done // (1 << 20)} MiB)", end="", flush=True)
            print()
            tmp.replace(ZIP_PATH)
    else:
        print(f"Zip present: {ZIP_PATH}")

    digest = md5_file(ZIP_PATH)
    print(f"md5={digest}")
    if digest != EXPECTED_MD5:
        print(f"WARNING: md5 mismatch (expected {EXPECTED_MD5}). Re-download recommended.")
        return 2

    # Zip usually contains top-level cubicasa5k/
    target = EXTRACT_DIR / "cubicasa5k"
    if not (target / "train.txt").exists() and not list(target.glob("**/train.txt")):
        print(f"Extracting → {EXTRACT_DIR}")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)
    else:
        print(f"Already extracted under {target}")

    # Normalize: if train.txt is nested, print hint
    trains = list(EXTRACT_DIR.rglob("train.txt"))
    print("Found train.txt at:", [str(p) for p in trains[:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
