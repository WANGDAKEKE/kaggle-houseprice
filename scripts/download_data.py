"""Download official competition files without submitting anything."""
from pathlib import Path
import os
import json
import hashlib
import argparse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / ".cache" / "kagglehub"))
COMPETITION = "house-prices-advanced-regression-techniques"
destination = ROOT / "data" / "raw"
destination.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--mirror", action="store_true", help="Public mirror fallback when Kaggle login is unavailable")
parser.add_argument("--zip", type=Path, help="Extract the official competition ZIP supplied by the user")
args = parser.parse_args()
sources = []
archive_details = {}
if args.zip:
    with zipfile.ZipFile(args.zip) as archive:
        for name in ["train.csv", "test.csv", "sample_submission.csv", "data_description.txt"]:
            (destination / name).write_bytes(archive.read(name))
    source = "official Kaggle ZIP supplied by user"
    archive_details = {"archive": str(args.zip.resolve()), "archive_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest()}
    print("Extracted official ZIP:", args.zip)
elif args.mirror:
    base = "https://raw.githubusercontent.com/manavkapadnis/House-prices-Advanced-regression-techniques/master/"
    other = "https://raw.githubusercontent.com/Rswcf/houseprices_kaggle/master/"
    for name in ["train.csv", "test.csv", "sample_submission.csv", "data_description.txt"]:
        with urllib.request.urlopen(base + name, timeout=60) as response:
            content = response.read()
        (destination / name).write_bytes(content)
        sources.append(base + name)
        if name in ["train.csv", "test.csv"]:
            with urllib.request.urlopen(other + name, timeout=60) as response:
                independent = response.read()
            # Normalize line endings for copies saved by different operating systems.
            if content.replace(b"\r\n", b"\n") != independent.replace(b"\r\n", b"\n"):
                raise ValueError(f"Independent mirror mismatch: {name}")
            sources.append(other + name)
        print(f"Downloaded and checked: {name} ({len(content)} bytes)", flush=True)
    source = "public GitHub mirrors; train/test match independently after CRLF normalization; official API returned 401"
else:
    import kagglehub
    path = kagglehub.competition_download(COMPETITION, output_dir=str(destination))
    print("Path to competition files:", path)
    source = "official Kaggle competition_download"
manifest = {
    "competition": COMPETITION,
    "source": source,
    "source_urls": sources,
    **archive_details,
    "files": {
        p.name: {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(destination.iterdir()) if p.is_file()
    },
}
(ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
