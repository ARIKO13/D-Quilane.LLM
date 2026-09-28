"""
train.py — Training script utk mini GPT.
Singkat: ~30-60 detik di CPU sandbox.
Save model + tokenizer ke ../download/llm/
"""
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import local
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import MiniBPETokenizer
from mini_gpt_llm import MiniGPT

# ---------- config ----------
CORPUS_PATH = Path(__file__).parent / "corpus.txt"
OUT_DIR = Path("/home/z/my-project/download/llm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOK_PATH = OUT_DIR / "tokenizer.json"
MODEL_PATH = OUT_DIR / "mini_llm.pt"

CFG = {
    "vocab_size": 384,
    "d_model": 96,
    "n_heads": 6,
    "d_ff": 384,
    "num_layers": 12,         # virtual depth via parameter sharing
    "max_seq_len": 64,
}
EPOCHS = 80
BATCH = 32
LR = 3e-4
SEED = 42


# ---------- dataset ----------
class CharLevelLangDataset(Dataset):
    """Simple language modeling dataset dengan sliding window."""

    def __init__(self, text, tokenizer, seq_len):
        self.ids = tokenizer.encode(text)
        self.seq_len = seq_len
        # buat sample window per-token
        self.samples = []
        # padding-friendly: chunk jadi blocks of seq_len+1
        n_blocks = (len(self.ids) - 1) // (seq_len + 1)
        for b in range(n_blocks + 1):
            start = b * (seq_len + 1)
            end = start + seq_len + 1
            if end > len(self.ids):
                end = len(self.ids)
            if end - start < 2:
                break
            chunk = self.ids[start:end]
            if len(chunk) < seq_len + 1:
                chunk = chunk + [tokenizer.pad_id] * (seq_len + 1 - len(chunk))
            x = chunk[:-1]
            y = chunk[1:]
            # ignore pad dalam loss
            y = [t if t != tokenizer.pad_id else -100 for t in y]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main():
    torch.manual_seed(SEED)
    print("=" * 60)
    print("MINI LLM TRAINING — from scratch, sandbox")
    print("=" * 60)

    # 1. train tokenizer
    print("\n[1/4] Train tokenizer (BPE, vocab=384)...")
    text = CORPUS_PATH.read_text()
    tok = MiniBPETokenizer()
    tok.train(text, target_vocab=CFG["vocab_size"], verbose=False)
    tok.save(str(TOK_PATH))
    print(f"  saved tokenizer -> {TOK_PATH}")

    # 2. dataset
    print("\n[2/4] Build dataset...")
    ds = CharLevelLangDataset(text, tok, CFG["max_seq_len"])
    print(f"  dataset size: {len(ds)} samples")
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    # 3. model
    print("\n[3/4] Init model...")
    model = MiniGPT(CFG)
    model.info()
    n_params = model.count_params()
    print(f"  unique params: {n_params:,}  (~{n_params*4/1024:.1f} KB @ fp32)")
    print(f"  virtual depth: {CFG['num_layers']} (parameter sharing)")
    print(f"  effective (if no sharing): ~{n_params * CFG['num_layers']:,} params")

    # 4. train
    print("\n[4/4] Training...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    t0 = time.time()
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in dl:
            opt.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg = total_loss / max(n_batches, 1)
        losses.append(avg)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.4f}  t={elapsed:.1f}s")

    # save model
    model.save(str(MODEL_PATH))
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"File size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")
    print(f"Tokenizer size: {TOK_PATH.stat().st_size / 1024:.1f} KB")

    # quick sanity test
    print("\n--- generation test ---")
    from infer import generate
    prompts = [
        "Indonesia",
        "Belajar",
        "Teknologi",
    ]
    for p in prompts:
        out = generate(model, tok, p, max_new_tokens=40, temperature=0.7, top_k=20)
        print(f"[{p}] -> {out}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
