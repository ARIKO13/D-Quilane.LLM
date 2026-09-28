#!/usr/bin/env python3
"""
D'QUILANE — Kaggle Push (versi interaktif).
Tinggal run, paste username + API key, selesai.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

LLM_DIR = Path("/home/z/my-project/download/llm")
KAGGLE_DIR = Path.home() / ".kaggle"


def main():
    print("=" * 60)
    print("  D'QUILANE — Push to Kaggle")
    print("=" * 60)
    print()

    # 1. minta kredensial
    print("Kredensial Kaggle kamu bisa dari:")
    print("  https://www.kaggle.com/settings -> Create New Token\n")
    username = input("Username Kaggle: ").strip()
    api_key = input("API Key     : ").strip()

    if not username or not api_key:
        print("[ERROR] Username & API key gak boleh kosong.")
        sys.exit(1)

    # 2. simpan kredensial
    KAGGLE_DIR.mkdir(exist_ok=True)
    cred = {"username": username, "key": api_key}
    cred_path = KAGGLE_DIR / "kaggle.json"
    cred_path.write_text(json.dumps(cred))
    cred_path.chmod(0o600)
    print(f"\n[OK] Kredensial disimpan ke {cred_path}")

    # 3. update kernel-metadata.json
    meta_path = LLM_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["id"] = f"{username}/d-quilane-llm-v0-1"
    meta["title"] = "d-quilane-llm-v0-1"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] Metadata updated: id={meta['id']}")

    # 4. push
    print("\n=== Pushing notebook to Kaggle... ===\n")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(LLM_DIR)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[STDERR]")
        print(result.stderr)
        sys.exit(result.returncode)

    print()
    print("=" * 60)
    print("  DONE! Notebook terupload.")
    print("=" * 60)
    print(f"URL: https://www.kaggle.com/code/{username}/d-quilane-llm-v0-1")
    print()
    print("Langkah selanjutnya:")
    print("  1. Buka URL di atas")
    print("  2. Settings (kanan) -> Accelerator -> GPU T4 x2")
    print("  3. Settings -> Internet -> On")
    print("  4. Klik 'Run All' (atau Save Version -> Save & Run All)")
    print("  5. Tunggu 5-15 menit")
    print("  6. Tab 'Output' untuk download model & tokenizer hasil training")


if __name__ == "__main__":
    main()
