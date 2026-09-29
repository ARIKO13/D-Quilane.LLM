"""
SFT (Supervised Fine-Tuning) untuk D'QUILANE.
Setelah model pre-trained, fine-tune dengan data instruction biar bisa ikutin perintah.

Data format (contoh):
    Instruction: Translate to Indonesian
    Input: Hello world
    Output: Halo dunia

Alur:
    1. Load pre-trained model
    2. Siapkan SFT dataset
    3. Fine-tune beberapa epoch
    4. Save model baru (overrite atau .sft.pt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, math, time

# Load dari file pre-trained
CKPT_PATH = "output/dquilane_balita.pt"  # hasil training utama

def load_model(ckpt_path, device='cpu'):
    """Load model D'QUILANE + tokenizer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    CONFIG = ckpt['config']
    stoi = ckpt['stoi']
    itos = {int(k): v for k, v in ckpt['itos'].items()}

    # Recreate model class (paste dari training notebook)
    class Attention(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            assert d % h == 0
            self.h, self.dh = h, d // h
            self.qkv = nn.Linear(d, 3*d, bias=False)
            self.out = nn.Linear(d, d, bias=False)
        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, T, C)
            return self.out(out)

    class FFN(nn.Module):
        def __init__(self, d, d_ff):
            super().__init__()
            self.gate = nn.Linear(d, d_ff, bias=False)
            self.val  = nn.Linear(d, d_ff, bias=False)
            self.down = nn.Linear(d_ff, d, bias=False)
        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.val(x))

    class Block(nn.Module):
        def __init__(self, d, h, d_ff):
            super().__init__()
            self.norm1 = nn.RMSNorm(d)
            self.attn  = Attention(d, h)
            self.norm2 = nn.RMSNorm(d)
            self.ffn   = FFN(d, d_ff)
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class DQuilane(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
            self.pos_emb = nn.Embedding(cfg['max_seq'], cfg['d_model'])
            self.block = Block(cfg['d_model'], cfg['n_heads'], cfg['d_ff'])
            self.norm_f = nn.RMSNorm(cfg['d_model'])
            self.head = nn.Linear(cfg['d_model'], cfg['vocab_size'], bias=False)
            self.head.weight = self.tok_emb.weight
        def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for _ in range(self.cfg['n_layers']):
                x = self.block(x)
            x = self.norm_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, self.cfg['vocab_size']),
                                        targets.view(-1), ignore_index=-100)
            return logits, loss

    model = DQuilane(CONFIG).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model, CONFIG, stoi, itos


# ===== SFT DATASET =====
# Format: prompt + completion (kalau mau tambah instruction following)
SFT_EXAMPLES = [
    # Greeting patterns
    ("Halo", "Halo juga! Senang bertemu denganmu."),
    ("Hai", "Hai! Apa kabar?"),
    ("Hi", "Hi there! Bagaimana kabarmu?"),
    # Identity
    ("Siapa kamu?", "Aku D'QUILANE, LLM buatan Indonesia dari nol."),
    ("Nama kamu siapa?", "Namaku D'QUILANE. LLM multibahasa."),
    # Knowledge
    ("Apa itu Indonesia?", "Indonesia adalah negara kepulauan di Asia Tenggara."),
    ("Ibu kota Indonesia?", "Ibu kota Indonesia adalah Jakarta."),
    ("Bahasa resmi Indonesia?", "Bahasa resmi Indonesia adalah bahasa Indonesia."),
    # Translation (basic)
    ("Translate halo to English", "Hello"),
    ("Translate hello to Indonesian", "Halo"),
    ("Translate terima kasih to English", "Thank you"),
    ("Translate thank you to Indonesian", "Terima kasih"),
    # Math
    ("Satu tambah satu?", "Satu tambah satu sama dengan dua."),
    ("Berapa dua tambah dua?", "Dua tambah dua sama dengan empat."),
    # Misc
    ("Apa itu AI?", "AI adalah kecerdasan buatan, mesin yang meniru kecerdasan manusia."),
    ("Apa itu LLM?", "LLM adalah Large Language Model, model AI bahasa besar."),
    ("Terima kasih", "Sama-sama! Ada yang bisa aku bantu lagi?"),
    ("Goodbye", "Sampai jumpa! Semoga harimu menyenangkan."),
    ("Dadah", "Dadah! Sampai bertemu lagi."),
]


def sft_train(model, CONFIG, stoi, itos, examples, epochs=200, lr=1e-4, device='cpu'):
    """Fine-tune model dengan SFT examples."""
    PAD_ID = stoi.get('<pad>', 0)
    max_seq = CONFIG['max_seq']
    print(f"\n=== SFT Training ({len(examples)} examples, {epochs} epochs) ===")

    # Encode examples
    encoded = []
    for prompt, response in examples:
        # Format: prompt + " " + response + eos
        full = prompt + " " + response
        ids = [stoi[c] for c in full if c in stoi]
        if len(ids) > max_seq:
            ids = ids[:max_seq]
        # padding
        while len(ids) < max_seq:
            ids.append(PAD_ID)
        # target = shifted, but mask prompt portion with -100
        # we only compute loss on the response part
        prompt_len = len([c for c in prompt if c in stoi])
        targets = []
        for i in range(max_seq - 1):
            if i < prompt_len - 1:
                targets.append(-100)  # don't compute loss on prompt
            else:
                # target is next token
                next_id = ids[i + 1] if i + 1 < len(ids) else PAD_ID
                targets.append(next_id if next_id != PAD_ID else -100)
        encoded.append((ids[:-1], targets))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        # random shuffle
        import random
        random.shuffle(encoded)
        total_loss = 0
        for x, y in encoded:
            x = torch.tensor([x], dtype=torch.long, device=device)
            y = torch.tensor([y], dtype=torch.long, device=device)
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(encoded):.4f}  t={time.time()-t0:.1f}s")

    print(f"SFT done in {time.time()-t0:.1f}s")
    return model


def main():
    print("=" * 60)
    print("  D'QUILANE — SFT (Supervised Fine-Tuning)")
    print("=" * 60)

    if not os.path.exists(CKPT_PATH):
        print(f"[ERROR] Pre-trained model tidak ditemukan: {CKPT_PATH}")
        print("Run dquilane_global.ipynb dulu untuk training model utama.")
        return

    # Load model
    print(f"\nLoading pre-trained model: {CKPT_PATH}")
    model, CONFIG, stoi, itos = load_model(CKPT_PATH, device='cpu')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. params: {n_params:,}")

    # Fine-tune
    model = sft_train(model, CONFIG, stoi, itos, SFT_EXAMPLES, epochs=200)

    # Save fine-tuned model
    sft_path = CKPT_PATH.replace('.pt', '.sft.pt')
    torch.save({
        'state_dict': model.state_dict(),
        'config': CONFIG,
        'stoi': stoi,
        'itos': {int(k): v for k, v in itos.items()},
    }, sft_path)
    print(f"\nFine-tuned model saved: {sft_path}")
    print(f"Size: {os.path.getsize(sft_path)/1024:.1f} KB")

    # Test
    print("\n=== TEST AFTER SFT ===")
    model.eval()
    decode = lambda ids: ''.join(itos.get(int(i), '?') for i in ids if i != stoi.get('<pad>', 0))
    for prompt in ['Halo', 'Siapa kamu?', 'Ibu kota Indonesia?', 'Translate halo to English']:
        ids = [stoi[c] for c in prompt if c in stoi]
        idx = torch.tensor([ids], dtype=torch.long)
        # generate
        with torch.no_grad():
            for _ in range(40):
                x = idx if idx.size(1) <= CONFIG['max_seq'] else idx[:, -CONFIG['max_seq']:]
                logits, _ = model(x)
                logits = logits[:, -1, :]
                # greedy
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
        print(f"\n[{prompt}] -> {decode(idx[0].tolist())}")


if __name__ == "__main__":
    main()
