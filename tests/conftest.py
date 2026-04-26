import os
import tempfile
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

temp_root = ROOT / ".tmp_test"
temp_root.mkdir(parents=True, exist_ok=True)
temp_str = str(temp_root)
tempfile.tempdir = temp_str
os.environ["TMPDIR"] = temp_str
os.environ["TMP"] = temp_str
os.environ["TEMP"] = temp_str

torch_cache = temp_root / "torch-cache"
inductor_cache = torch_cache / "inductor"
triton_cache = torch_cache / "triton"
inductor_cache.mkdir(parents=True, exist_ok=True)
triton_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor_cache))
os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
