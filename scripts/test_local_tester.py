"""Smoke test untuk dquilane_local_test.py — pakai model hasil training sandbox."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/z/my-project/download/llm")

# copy model hasil test sandbox ke lokasi yang diharapkan
import shutil
src = Path("/home/z/my-project/download/llm/output_test/dquilane_balita.pt")
dst = Path("/home/z/my-project/download/llm/dquilane_balita.pt")
if src.exists() and not dst.exists():
    shutil.copy(src, dst)
    print(f"Copied model: {dst}")

# run dquilane_local_test dengan prompt custom
sys.argv = ["dquilane_local_test.py", "Indonesia adalah"]
exec(open("/home/z/my-project/download/llm/dquilane_local_test.py").read())
