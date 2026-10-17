# D'QUILANE Base

Pure-Python inference engine + web scraper + browser UI. No dependencies, no PyTorch needed.

## Files

```
base/
├── gguf_infer.py    ← Pure-Python CPU inference (loads any GGUF)
├── web_search.py    ← Pure-Python web scraper (4 sources + fallback)
├── web/
│   ├── index.html   ← ChatGPT-style chat UI
│   └── app.js       ← GGUF binary parser + AION bypass in browser
└── docs/
    └── ARCHITECTURE.md  ← Full architecture spec
```

## Quick start

### CPU inference

```bash
python gguf_infer.py ../trained/model.gguf "Halo"
python gguf_infer.py ../trained/model.gguf "search: Jakarta hari ini"
python gguf_infer.py ../trained/model.gguf --interactive
```

### Web UI

Open `web/index.html` in a browser. It fetches `../trained/model.gguf` directly.

## Inference pipeline

```
Prompt
  ↓
[0] Web search trigger?
    (search:/cari:/google: prefix OR terbaru/sekarang/hari ini keywords)
    YES → fetch web → format → return
    NO  ↓
[1] AION bypass?
    (exact/substring/word-overlap match in engram cache)
    YES → return cached answer (0.04s)
    NO  ↓
[2] Transformer forward (24 effective layers)
    generate via top-k/top-p sampling + EntroCut
```

## Web search sources

| Source | Reliability | Best for |
|---|---|---|
| Wikipedia ID | Rate-limited | Factual Q&A, multilingual |
| Hacker News Algolia | Always works | Tech/news/programming |
| Bing HTML | Bot-blocked often | General search (best-effort) |
| DuckDuckGo HTML | Often times out | Fallback |

Cache: 24h TTL, JSON file at `trained/web_cache.json` (auto-generated).
