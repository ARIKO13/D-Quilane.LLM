# D'QUILANE — Model Card

> **D'QUILANE** — *Dynamic QUantum-free Intelligent Language Architecture with Nested Efficiency*
> LLM pertama milik pengguna, dibangun dari nol, sandbox edition.

---

## 🪪 Identitas Model

| Field | Value |
|-------|-------|
| Nama | **D'QUILANE** |
| Versi | v0.1-sandbox |
| Tanggal lahir | 2026-09-28 |
| Asal | Dibuat dari nol di sandbox Z.ai |
| Tagline | "Kecil file-nya, dalam dalam pikirannya." |

---

## 🧬 Spesifikasi Teknis

| Spec | Nilai |
|------|-------|
| Arsitektur | Decoder-only Transformer (GPT-style) |
| Tokenizer | BPE buatan sendiri, 384 vocab |
| Unique parameters | 191,136 |
| Virtual depth | 12 layer (parameter sharing, ALBERT-style) |
| Effective depth | setara 12-layer GPT |
| Effective params | ~2.29M (kalau tanpa sharing) |
| File size (fp32) | 750 KB (TANPA quantize) |
| Quantize? | **TIDAK** — file asli fp32 |
| d_model | 96 |
| n_heads | 6 |
| d_ff | 384 |
| Context window | 64 token |
| Activation | SwiGLU |
| Normalisasi | RMSNorm |
| Weight tying | Output head = token embedding |
| KV-cache | Ya, untuk fast generation |

---

## ✨ Skill yang D'QUILANE Punya (v0.1)

### 🟢 Bisa (terbatas)
1. **Text Generation** — autoregressive next-token prediction
2. **Sampling kontrol** — top-k + temperature
3. **KV-cache generation** — sampling cepat
4. **Bahasa Indonesia dasar** — vocab sudah belajar token "Indonesia", "yang", "dan"
5. **Prompt continuation** — kasih prompt "Indonesia", dia lanjutin
6. **Bobot portable** — file .pt bisa di-load di device mana pun (CPU/GPU/HP?)

### 🟡 Belum Bener-bener Bisa
1. **Kalimat koheren** — output masih setengah acak (loss 4.59, target <3)
2. **Menghafal pattern** — beberapa kata training muncul, tapi tidak konsisten

### 🔴 Belum Bisa Sama Sekali (perlu versi selanjutnya)
1. ❌ Instruction following — belum ada SFT
2. ❌ Chat / conversation — belum ada chat template
3. ❌ Coding — belum pernah lihat code
4. ❌ Translation — belum dilatih bilingual
5. ❌ Reasoning — model terlalu kecil
6. ❌ Long context — max 64 token
7. ❌ Q&A — belum ada data instruction

---

## 🛣️ Roadmap Skill D'QUILANE

### v0.2 — Kaggle GPU (minggu depan)
- [ ] d_model 256, depth 24
- [ ] Train di T4 GPU, 2000 epoch
- [ ] Dataset 1MB+ teks Indonesia
- [ ] Output mulai masuk akal (loss < 3)

### v0.3 — Instruction tuning (bulan depan)
- [ ] Tambah SFT (Supervised Fine-Tuning) dengan data instruction Indonesia
- [ ] Chat template: `<user>...</user><assistant>...</assistant>`
- [ ] Bisa ikutin perintah sederhana ("translate", "ringkas")

### v0.4 — Alignment (2 bulan)
- [ ] DPO (Direct Preference Optimization) — tanpa RLHF mahal
- [ ] Personality & safety alignment
- [ ] Nama "D'QUILANE" di-burn-in ke system prompt

### v1.0 — Production
- [ ] Vocab 8K-32K
- [ ] d_model 512-768
- [ ] 48 layer virtual
- [ ] Effective params 100M-300M
- [ ] Fine-tune ke domain spesifik (kalau ada)
- [ ] Deploy ke web app atau API

---

## 🎨 Personality D'QUILANE (proposed)

Karena D'QUILANE = nama panggung yang elegan & powerful:
- **Tone**: tenang, elegan, puitis (ala bahasa sastra Indonesia)
- **Keahlian**: ngomong soal sejarah Indonesia, teknologi, dan filsafat
- **Motto**: *"Kecil bukan berarti lemah. Dalam bukan berarti lamban."*

---

## 📁 File D'QUILANE

| File | Lokasi | Ukuran |
|------|--------|--------|
| Bobot | `/home/z/my-project/download/llm/mini_llm.pt` | 750 KB |
| Tokenizer | `/home/z/my-project/download/llm/tokenizer.json` | 2.5 KB |
| Source code | `/home/z/my-project/scripts/` | — |
| Notebook Kaggle | `/home/z/my-project/download/llm/kaggle_notebook.ipynb` | 24 KB |
| Model card ini | `/home/z/my-project/download/llm/MODEL_CARD.md` | — |

---

## 💡 Cara Pakai D'QUILANE

```bash
python3 /home/z/my-project/scripts/infer.py "Indonesia"
```

Atau di Python:
```python
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from infer import load_model_and_tokenizer, generate

model, tok = load_model_and_tokenizer()
print(generate(model, tok, "Indonesia", max_new_tokens=40, temperature=0.7, top_k=20))
```

---

*—"Tidak ada LLM yang lahir dewasa. D'QUILANE dimulai dari 750 KB."*
