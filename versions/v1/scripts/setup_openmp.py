"""Install the signed LLVM OpenMP runtime locally for macOS wheel imports."""
from pathlib import Path
import urllib.request
import tarfile
import io
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
URL = "https://mac.r-project.org/openmp/openmp-19.1.5-darwin20-Release.tar.gz"
EXPECTED_SHA1 = "5b44175bcbaa334b0c57391482e068ea185c95a2"
destination = ROOT / ".runtime" / "lib"
destination.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(URL, timeout=60) as response:
    content = response.read()
assert hashlib.sha1(content).hexdigest() == EXPECTED_SHA1, "Download checksum mismatch"
with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
    library = archive.extractfile("usr/local/lib/libomp.dylib").read()
(destination / "libomp.dylib").write_bytes(library)
(ROOT / ".runtime" / "manifest.json").write_text(json.dumps({
    "url": URL, "archive_sha1": EXPECTED_SHA1,
    "library_sha256": hashlib.sha256(library).hexdigest(),
    "scope": "workspace only; no system installation",
}, indent=2) + "\n")
print("Local OpenMP runtime:", destination / "libomp.dylib")
