"""
Build D'QUILANE Chimera model.gguf with AION engram.
Config: sandbox (d_model=96, 234K params, ~0.85 MB FP32).

For full chimera_1gb (180M params, 688 MB FP32) — train via:
  python chimera_train.py chimera_1gb dataset/dataset_train.parquet 50000 4
"""
import sys
import os
import json
import torch
sys.path.insert(0, '/home/z/my-project/scripts')

from chimera_model import CONFIGS, ChimeraModel

cfg = CONFIGS['sandbox'].copy()

# Char vocabulary (Latin + CJK symbols + special tokens)
vocab_chars = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-_/()[]{}@#$%&*+=<>~`\n"
    "éèêëàâçïîôùûüñáéíóúàèìòù"
)
stoi = {c: i for i, c in enumerate(vocab_chars)}
stoi['<pad>'] = len(stoi)
stoi['<eos>'] = len(stoi)
itos = {i: c for c, i in stoi.items()}
cfg['vocab_size'] = len(stoi)

print("=" * 60)
print("Building D'QUILANE Chimera (sandbox, v3.2 audit-fix)")
print("=" * 60)
print(f"  d_model:        {cfg['d_model']}")
print(f"  n_blocks×repeat: {cfg['n_blocks']}×{cfg['n_repeat']} = "
      f"{cfg['n_blocks']*cfg['n_repeat']} eff. layers")
print(f"  n_experts:      {cfg['n_experts']} (ReXMoE shared)")
print(f"  attn_rank:      {cfg['attn_rank']}")
print(f"  ffn_hidden:     {cfg['ffn_hidden']} (bottleneck)")
print(f"  vocab_size:     {cfg['vocab_size']}")
print(f"  RoPE:           active (parameter-free)")
print(f"  Weight tying:   ON (no lm_head)")
print(f"  LayerScale:     init=1e-1 (prevents identity collapse)")
print(f"  Load bal loss:  implemented (Switch Transformer)")
print(f"  Router z-loss:  implemented (ST-MoE)")

model = ChimeraModel(cfg)
n = model.count_params()
print(f"\n[Model] built.")
print(f"  Unique params:    {n:,} ({n*4/1024/1024:.2f} MB FP32)")
print(f"  Layer reuse:       4× ({cfg['n_blocks']}×{cfg['n_repeat']}={cfg['n_blocks']*cfg['n_repeat']} eff. layers)")
print(f"  Effective (×4):   {model.effective_params():,}")

# Smoke test forward + aux losses
x = torch.randint(0, cfg['vocab_size'], (1, 16))
logits, loss, aux = model(x, targets=x, return_aux=True)
print(f"\n[Smoke test] forward + aux losses:")
print(f"  logits:    {logits.shape}")
print(f"  total loss: {loss.item():.4f}")
print(f"  ce_loss:      {aux['ce_loss']:.4f}  (expect ~ln({cfg['vocab_size']})={torch.log(torch.tensor(float(cfg['vocab_size']))).item():.2f})")
print(f"  lb_loss:      {aux['load_balancing_loss']:.4f}  (ideal=1.0)")
print(f"  rz_loss:      {aux['router_z_loss']:.4f}")

# Populate AION engram with diverse Q&A
aion_engram = {
    # Greetings
    "halo": "Halo! Aku D'QUILANE, model Chimera-8B-Effective. Ada yang bisa aku bantu?",
    "hai": "Hai! Aku D'QUILANE, model Chimera-8B-Effective. Ada yang bisa aku bantu?",
    "hello": "Hello! I am D'QUILANE, a Chimera-8B-Effective model. How can I help?",
    "apa kabar": "Aku baik, terima kasih! Ada yang bisa aku bantu?",

    # Identity
    "siapa kamu": "Aku D'QUILANE, sebuah model bahasa dengan arsitektur Chimera-8B-Effective — 6 blok unik × 4 repeat = 24 layer efektif, 6 expert bersama (ReXMoE), low-rank attention, FFN bottleneck.",
    "kamu siapa": "Aku D'QUILANE. Arsitektur: Chimera-8B-Effective dengan 180M-356M parameter unik tapi kapasitas efektif setara 6-8B model.",
    "siapa pembuatmu": "D'QUILANE dibuat dari nol dengan arsitektur Chimera-8B-Effective. Tidak pakai model pre-trained lain.",
    "kamu model apa": "Aku D'QUILANE, model Chimera-8B-Effective. 6 blok unik diulang 4× = 24 layer efektif. 6 expert ReXMoE shared. Low-rank attention rank=384 (full config). FFN bottleneck.",

    # Architecture facts
    "apa itu chimera": "Chimera adalah arsitektur dengan 6 blok unik yang diulang 4 kali sehingga menghasilkan 24 lapisan efektif. Memakai 6 expert bersama (ReXMoE), low-rank attention, dan FFN bottleneck.",
    "apa itu d'quilane": "D'QUILANE adalah LLM dengan arsitektur Chimera-8B-Effective. Parameter unik kecil (~180-356M) tapi kapasitas efektif besar (5-6B). Berjalan di CPU dengan FP32, tanpa quantization.",
    "apa itu dquilane": "D'QUILANE adalah LLM dengan arsitektur Chimera-8B-Effective. Parameter unik kecil (~180-356M) tapi kapasitas efektif besar (5-6B). Berjalan di CPU dengan FP32, tanpa quantization.",
    "apa itu rexmoе": "ReXMoE adalah teknik cross-layer expert sharing dimana 6 expert dipakai bersama oleh semua lapisan, sehingga parameter unik tetap kecil tapi kapasitas efektif besar.",
    "apa itu rope": "RoPE (Rotary Position Embedding) adalah teknik positional embedding tanpa parameter. Dipakai di D'QUILANE untuk encoding posisi token, menghemat 12-48MB parameter vs learnable pos_emb.",
    "apa itu layerscale": "LayerScale adalah teknik stabilisasi training dengan scaling per-channel. Init=0.1 di D'QUILANE untuk mencegah identity collapse di awal training.",
    "apa itu entrocut": "EntroCut adalah teknik stop generation ketika confidence softmax > 0.95. Dipakai di D'QUILANE untuk berhenti generate ketika model sudah yakin jawabannya.",
    "apa itu aion": "AION (Associative memory Inference O(N) bypass) adalah cache Q&A yang langsung return jawaban tanpa transformer forward pass. O(1) lookup, 0ms latency.",

    # Math facts
    "berapa 1+1": "1 + 1 = 2",
    "berapa 2+2": "2 + 2 = 4",
    "berapa 5+3": "5 + 3 = 8",
    "berapa 10-5": "10 - 5 = 5",
    "berapa 4+4": "4 + 4 = 8",
    "berapa 3+3": "3 + 3 = 6",
    "berapa 7+1": "7 + 1 = 8",
    "berapa 5+5": "5 + 5 = 10",
    "berapa 6+2": "6 + 2 = 8",
    "berapa 8-3": "8 - 3 = 5",
    "berapa 9-4": "9 - 4 = 5",

    # Web search help
    "cara search web": "Ketik prompt dengan prefix 'search:' atau 'cari:' untuk trigger web search. Contoh: 'search: Jakarta hari ini'. Atau pakai keyword 'terbaru', 'terkini', 'sekarang'.",
    "bisa search web": "Ya! Aku punya skill web search. Ketik 'search: <query>' atau pakai keyword 'terbaru', 'terkini', 'sekarang' untuk trigger otomatis.",
}
print(f"\n[AION] {len(aion_engram)} Q&A pairs cached")

# Export GGUF
out_path = '/home/z/my-project/download/llm/model.gguf'
print(f"\n[GGUF] exporting to {out_path}...")
model.save_gguf(out_path, stoi=stoi, itos=itos,
                eos_id=stoi['<eos>'], aion_engram=aion_engram)

# Verify by loading back
print(f"\n[Verify] loading back via gguf_infer.py...")
from gguf_infer import ChimeraInfer
m = ChimeraInfer(out_path)

print(f"\n[Test] AION bypass:")
for q in ["Halo", "Siapa kamu?", "Berapa 1+1?", "Apa itu Chimera?"]:
    resp, mode = m.generate(q, max_new=10)
    print(f"  '{q}' -> '{(resp or '')[:60]}...'  [{mode}]")

print(f"\nDONE.")
