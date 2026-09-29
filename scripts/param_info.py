"""Detail parameter breakdown untuk MiniGPT."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mini_gpt_llm import MiniGPT
import torch

CFG = {
    "vocab_size": 384,
    "d_model": 96,
    "n_heads": 6,
    "d_ff": 384,
    "num_layers": 12,
    "max_seq_len": 64,
}

model = MiniGPT(CFG)

print("=" * 70)
print("PARAMETER BREAKDOWN — MiniGPT (Parameter Sharing)")
print("=" * 70)

# Hitung per komponen
components = {
    "Token embedding (vocab x d_model)": CFG["vocab_size"] * CFG["d_model"],
    "Positional embedding (max_seq x d_model)": CFG["max_seq_len"] * CFG["d_model"],
    "Attention QKV (3 x d_model^2)": 3 * CFG["d_model"] * CFG["d_model"],
    "Attention output proj (d_model^2)": CFG["d_model"] * CFG["d_model"],
    "MLP w1 (d_model x d_ff)": CFG["d_model"] * CFG["d_ff"],
    "MLP w2 (d_ff x d_model)": CFG["d_ff"] * CFG["d_model"],
    "MLP w3 (d_model x d_ff)": CFG["d_model"] * CFG["d_ff"],
    "RMSNorm weights (3 x d_model)": 3 * CFG["d_model"],
    "Output head": 0,  # tied ke token embedding
}

actual_total = 0
print("\nKomponen unik (dipakai 1x di memori):")
print("-" * 70)
for name, n in components.items():
    print(f"  {name:<45} {n:>10,}  ({n*4/1024:>7.1f} KB)")
    actual_total += n
print("-" * 70)
print(f"  TOTAL UNIQUE: {actual_total:>10,}  ({actual_total*4/1024:>7.1f} KB)")
print(f"  Pytorch count: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "=" * 70)
print("EFFECTIVE PARAMS (kalau gak pakai sharing):")
print("=" * 70)
print(f"  Unique params x num_layers (12) virtual depth")
eff = actual_total * 12
print(f"  = {actual_total:,} x 12 = {eff:,} ({eff*4/1024/1024:.1f} MB equivalent)")
print(f"  Rasio efektif/aktual = 12x lipat")

print("\n" + "=" * 70)
print("PERBANDINGAN:")
print("=" * 70)
print(f"  Model ini (unik):     191,136    -> 750 KB")
print(f"  Model ini (virtual):  2,293,632   -> seakan 2.3M param")
print(f"  GPT-2 small:      124,000,000    -> 500 MB")
print(f"  GPT-3:        175,000,000,000    -> 700 GB")
print(f"  Llama 3 8B:     8,000,000,000    -> 16 GB (fp16)")
