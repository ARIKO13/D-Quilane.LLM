# D'QUILANE Global — GitHub Pages Deployment

Web tester D'QUILANE siap di-deploy ke GitHub Pages. **No build step, no backend, no library** — pure HTML + JS yang bisa di-host gratis di GitHub Pages.

## 📦 File yang Perlu

```
web/
├── index.html              # UI + logic
├── app.js                  # JS inference engine (vanilla)
└── dquilane_model.json     # model JSON (hasil export dari Kaggle)
```

Total size ~3-5 MB (model JSON dominan).

## 🚀 Cara Deploy ke GitHub Pages

### Langkah 1 — Dapatkan model JSON dari Kaggle

1. Upload `dquilane_global.ipynb` ke Kaggle → Run All
2. Di tab Output, download `dquilane_model.json`
3. File ini berisi bobot model + tokenizer + config

### Langkah 2 — Bikin GitHub repo

```bash
# Bikin repo baru di github.com (nama bebas, mis: dquilane-llm)
git clone https://github.com/<username>/dquilane-llm.git
cd dquilane-llm

# Bikin folder web/
mkdir web
cd web

# Copy 3 file:
cp /path/ke/index.html .
cp /path/ke/app.js .
cp /path/ke/dquilane_model.json .
```

### Langkah 3 — Commit & push

```bash
cd ..
git add .
git commit -m "feat: D'QUILANE web tester"
git push origin main
```

### Langkah 4 — Aktifkan GitHub Pages

1. Buka repo kamu di GitHub
2. **Settings** → **Pages** (sidebar kiri)
3. Source: **Deploy from a branch**
4. Branch: `main` / folder: `/web` → Save
5. Tunggu 1-2 menit
6. URL: `https://<username>.github.io/dquilane-llm/`

## 🎯 Struktur Repo yang Disarankan

```
dquilane-llm/
├── README.md              # info repo
├── web/
│   ├── index.html         # web tester
│   ├── app.js
│   ├── dquilane_model.json
│   └── style.css (kalau mau dipisah)
├── kaggle/
│   └── dquilane_global.ipynb  # source notebook
└── LICENSE (optional)
```

## ⚙️ Customisasi

### Ganti tema warna
Edit `index.html` di bagian `<style>`:
```css
body {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);  /* ganti warna */
}
```

### Tambah prompt contoh
Edit `index.html`, cari `.example-chip`:
```html
<span class="example-chip" data-prompt="Bahasa baru">Baru</span>
```

### Naikkan model performance
Edit di `dquilane_global.ipynb` (cell 3):
```python
CONFIG = {
    'd_model':    192,    # naik dari 96
    'n_layers':   12,     # naik dari 6 (gratis! gak nambah size)
    'epochs':     5000,   # latih lebih lama
}
```
Lalu re-export JSON, push ulang ke GitHub.

## 🔧 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `404 Not Found` | Cek path file di repo, branch Pages setting |
| `CORS error` di browser console | Jangan pakai `file://`, harus pakai HTTP server |
| Generate lambat | File JSON terlalu gede? Kurangi d_model atau n_layers |
| Output random parah | Train lebih lama di Kaggle, atau naikkan dataset |
| Loading stuck | Cek console (F12), pastikan `dquilane_model.json` bisa di-fetch |

## 🌟 Optimasi (Optional)

### 1. Kompresi model JSON
```bash
# Model JSON ~3.5 MB. Bisa di-serve gzip di GitHub Pages otomatis.
# Browser bakal download ~700 KB (10x lebih cepat).
```

### 2. Lazy load model
Edit `index.html`, ganti `fetch('dquilane_model.json')` jadi load saat user pertama kali klik Generate:
```js
let modelPromise = null;
function getModel() {
    if (!modelPromise) {
        modelPromise = fetch('dquilane_model.json').then(r => r.json()).then(d => new DQuilane(d));
    }
    return modelPromise;
}
```

### 3. Pakai Web Worker biar gak block UI
Taruh `app.js` di worker thread, biar generate text gak freeze browser.

## 📊 Performa yang Diharapkan

| Device | Time per 80 tokens |
|--------|--------------------|
| PC modern (Chrome) | ~2-5 detik |
| Mac M1 (Safari) | ~1-3 detik |
| HP Android (Chrome) | ~8-15 detik |
| HP iPhone (Safari) | ~5-10 detik |

## 🎓 Penjelasan Teknis (untuk belajar)

Inference engine `app.js` reimplement PyTorch model di JavaScript:
- **Embedding**: lookup table (Float32Array)
- **RMSNorm**: `x * rsqrt(mean(x^2) + eps) * weight`
- **Attention**: multi-head causal, manual matmul
- **SwiGLU FFN**: `down(silu(gate(x)) * val(x))`
- **Weight tying**: output head = token embedding (gak di-export terpisah)
- **Sampling**: temperature + top-k + multinomial

Semua tanpa library — cuma vanilla JS Float32Array. Cocok buat belajar cara kerja transformer dari low-level.

## 📄 License

Bebas dipakai. Model D'QUILANE buatan kamu, source code punya kamu. Taruh MIT license kalau mau open source.
