#!/usr/bin/env python3
"""
D'QUILANE — Deploy ke GitHub Pages
=================================
Script otomatis deploy web tester D'QUILANE ke GitHub Pages.
"""
import os
import json
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Credentials (akan dihapus dari env setelah selesai)
GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_USER = "ARIKO13"
REPO_NAME = "D-Quilane.LLM"

# Files to deploy
WEB_SRC = Path("/home/z/my-project/download/llm/web")
NOTEBOOK_SRC = Path("/home/z/my-project/download/llm/dquilane_global.ipynb")
MODEL_CARD_SRC = Path("/home/z/my-project/download/llm/MODEL_CARD.md")

# Temp clone dir
TMP_DIR = Path("/tmp/dquilane-deploy")


def api_call(method, endpoint, data=None):
    """Call GitHub API."""
    url = f"https://api.github.com{endpoint}"
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "dquilane-deploy",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def run(cmd, **kwargs):
    """Run shell command, print + return."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, **{k: v for k, v in kwargs.items() if k != 'check'})
    if kwargs.get('check') and result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)
    return result


def main():
    if not GH_TOKEN:
        print("[ERROR] GH_TOKEN env var not set")
        sys.exit(1)

    print("=" * 60)
    print("  D'QUILANE — Deploy to GitHub Pages")
    print("=" * 60)
    print(f"User: {GH_USER}")
    print(f"Repo: {REPO_NAME}")

    # ---- 1. Cek apakah repo sudah ada ----
    print("\n[1/6] Check if repo exists...")
    status, repo = api_call("GET", f"/repos/{GH_USER}/{REPO_NAME}")
    if status == 200:
        print(f"  Repo exists: {repo['html_url']}")
        repo_exists = True
    elif status == 404:
        print("  Repo not found, will create...")
        repo_exists = False
    else:
        print(f"  Unexpected status {status}: {repo}")
        sys.exit(1)

    # ---- 2. Bikin repo kalau belum ada ----
    if not repo_exists:
        print("\n[2/6] Create repo...")
        status, resp = api_call("POST", "/user/repos", {
            "name": REPO_NAME,
            "description": "D'QUILANE Global - Multilingual LLM buatan sendiri dari nol (30+ bahasa, native script)",
            "private": False,
            "auto_init": True,  # bikin README otomatis
        })
        if status not in (201, 202):
            print(f"  ERROR creating repo: {status} {resp}")
            sys.exit(1)
        print(f"  Created: {resp['html_url']}")
        repo_url = resp['html_url']
        # tunggu bentar biar git siap
        import time
        time.sleep(3)
    else:
        repo_url = repo['html_url']

    # ---- 3. Clone repo ----
    print("\n[3/6] Clone repo...")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.parent.mkdir(exist_ok=True)

    clone_url = f"https://{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"
    result = run(f"git clone {clone_url} {TMP_DIR}", check=False)
    if result.returncode != 0:
        # kalau kosong, init baru
        print("  Clone failed (maybe empty repo). Init locally...")
        TMP_DIR.mkdir(exist_ok=True)
        run(f"cd {TMP_DIR} && git init && git remote add origin {clone_url}")
        # try fetching
        run(f"cd {TMP_DIR} && git fetch || true", check=False)

    # ---- 4. Copy file ke repo ----
    print("\n[4/6] Copy files...")
    # Setup git config
    run(f'cd {TMP_DIR} && git config user.email "{GH_USER}@users.noreply.github.com"')
    run(f'cd {TMP_DIR} && git config user.name "{GH_USER}"')

    # Pilih branch — kalau ada, pake; kalau gak, bikin main
    branches = run(f'cd {TMP_DIR} && git branch -a 2>&1', check=False).stdout or ""
    if "main" in branches:
        run(f"cd {TMP_DIR} && git checkout main 2>&1 || git checkout -b main", check=False)
    elif "master" in branches:
        run(f"cd {TMP_DIR} && git checkout master 2>&1 || git checkout -b master", check=False)
    else:
        run(f"cd {TMP_DIR} && git checkout -b main", check=False)

    # Hapus web/ lama kalau ada
    (TMP_DIR / "web").mkdir(exist_ok=True)
    # Copy semua file web
    for f in WEB_SRC.iterdir():
        if f.name in ("README.md", "corpus_multilingual.txt"):
            continue
        dest = TMP_DIR / "web" / f.name
        shutil.copy(f, dest)
        print(f"  copied: web/{f.name}")
    # bikin subfolder kaggle
    kaggle_dir = TMP_DIR / "kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    shutil.copy(NOTEBOOK_SRC, kaggle_dir / "dquilane_global.ipynb")
    print(f"  copied: kaggle/dquilane_global.ipynb")

    # Bikin README.md utama
    readme = """# D'QUILANE Global

Multilingual LLM (balita level) dibangun dari nol — **30+ bahasa dengan native script**.

## 🌍 Languages Supported

- Latin: Indonesia, English, Spanish, French, German, Italian, Portuguese, Polish, Czech, Swedish, Romanian, Tagalog, Catalan, Finnish, Norwegian, Dutch, Turkish, Vietnamese
- Cyrillic: Russian, Ukrainian
- CJK: Chinese, Japanese (Kanji)
- Hiragana + Katakana: Japanese
- Hangul: Korean
- Arabic script: Arabic
- Devanagari: Hindi
- Bengali, Thai, Greek, Hebrew

## 🌐 Live Web Tester

**https://ariko13.github.io/D-Quilane.LLM/**

## 📦 What's Inside

| Path | Description |
|------|-------------|
| `web/index.html` | Web UI (vanilla JS, no library) |
| `web/app.js` | JS inference engine (reimplements transformer from scratch) |
| `web/dquilane_model.json` | Model weights (7.4 MB, multilingual) |
| `kaggle/dquilane_global.ipynb` | Kaggle training notebook |

## 🧠 Architecture

- **Transformer decoder** with causal self-attention
- **Parameter sharing** (ALBERT-style): 1 block reused N times
- **SwiGLU** FFN
- **RMSNorm** (instead of LayerNorm)
- **Weight tying**: output head = token embedding
- **Char-level tokenizer** (handles any unicode character)

## 📊 Specs

- **Unique params**: ~250K (~1 MB fp32, no quantize)
- **Virtual depth**: 8 layers (parameter sharing)
- **Vocab**: ~556 chars (30+ scripts)
- **Effective params**: ~2M (if no sharing)
- **File size on disk**: ~7.4 MB (JSON)

## 🚀 How to Reproduce

1. Open `kaggle/dquilane_global.ipynb` in Kaggle
2. Settings: GPU T4 x2, Internet On
3. Run All (~5-15 min)
4. Download `output/dquilane_model.json`
5. Replace `web/dquilane_model.json` with new file
6. Push to GitHub → Pages auto-update

## 📄 License

MIT — bebas dipakai, modif, distribusi.

---

*Generated by D'QUILANE deploy script.*
"""
    (TMP_DIR / "README.md").write_text(readme)
    print("  created: README.md")

    # ---- 5. Commit & push ----
    print("\n[5/6] Commit & push...")
    run(f"cd {TMP_DIR} && git add -A")
    run(f'cd {TMP_DIR} && git commit -m "feat: D-QUILANE Global v0.4 - Multilingual LLM (30+ languages)" || echo "nothing to commit"')
    push_result = run(f"cd {TMP_DIR} && git push -u origin main 2>&1 || git push -u origin master 2>&1", check=False)
    print("  Push done" if push_result.returncode == 0 else "  Push may have failed, check log")

    # ---- 6. Aktifkan GitHub Pages ----
    print("\n[6/6] Activate GitHub Pages...")
    # Cek Pages status dulu
    status, pages = api_call("GET", f"/repos/{GH_USER}/{REPO_NAME}/pages")
    if status == 200:
        print(f"  Pages already enabled: {pages.get('html_url', 'N/A')}")
        page_url = pages.get('html_url')
    else:
        # Aktifin
        status, resp = api_call("POST", f"/repos/{GH_USER}/{REPO_NAME}/pages", {
            "source": {"branch": "main", "path": "/web"},
            "type": "legacy"
        })
        if status in (201, 202):
            print(f"  Pages enabled!")
            page_url = resp.get('html_url')
        else:
            print(f"  Failed: {status} {resp}")
            # coba lagi dengan master
            status, resp = api_call("POST", f"/repos/{GH_USER}/{REPO_NAME}/pages", {
                "source": {"branch": "master", "path": "/web"},
            })
            if status in (201, 202):
                print(f"  Pages enabled on master!")
                page_url = resp.get('html_url')
            else:
                print(f"  Still failed: {status} {resp}")
                page_url = None

    print("\n" + "=" * 60)
    print("  DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"Repo:  {repo_url}")
    print(f"Pages: {page_url or 'NOT ENABLED (manual needed)'}")
    if page_url:
        print(f"\n>>> Tunggu 1-2 menit, lalu buka URL di atas! <<<")
    print(f"\n⚠️  PENTING: Revoke PAT kamu sekarang:")
    print(f"   https://github.com/settings/tokens")
    print(f"   Cari token 'dquilane-deploy' → Delete")


if __name__ == "__main__":
    main()
