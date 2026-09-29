"""
mini-gpt-llm.py
===============
Mini GPT model dengan parameter sharing (ALBERT-style).

Ide:
- 1 set bobot transformer block (shared)
- Dipakai berulang-ulang ke depth virtual N (mis. N=12)
- Hasilnya: ukuran file kecil (~ratusan KB), tapi kedalaman efektif tinggi
- Inilah trik "parameter seakan besar, size tetap kecil tanpa quantize"

Tidak pakai library selain torch dan math.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention dengan implementasi from scratch."""

    def __init__(self, d_model: int, n_heads: int, attn_bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model harus kelipatan n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=attn_bias)
        self.proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x, kv_cache=None, pos=None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # B, T, 3C
        q, k, v = qkv.split(C, dim=-1)
        # reshape ke (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # KV-cache (untuk inference autoregressive)
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            if k_prev is not None:
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # causal mask
        T_full = k.size(2)
        # saat training T == T_full; saat generation T=1, mask tidak relevan
        causal = torch.tril(torch.ones(T, T_full, device=x.device, dtype=torch.bool))
        # (B, n_heads, T, T_full)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, n_heads, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out, new_cache


class SwiGLUMLP(nn.Module):
    """SwiGLU FFN — modern, lebih efisien dari MLP biasa."""

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """RMSNorm — lebih stabil & lebih sedikit param daripada LayerNorm."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return x_norm * self.weight


class SharedBlock(nn.Module):
    """
    1 block transformer (shared). Akan dipakai berulang-ulang
    utk depth virtual N. Inilah kunci 'parameter sharing' ALBERT-style.
    """

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = CausalSelfAttention(cfg["d_model"], cfg["n_heads"])
        self.norm2 = RMSNorm(cfg["d_model"])
        self.mlp = SwiGLUMLP(cfg["d_model"], cfg.get("d_ff"))

    def forward(self, x, kv_cache=None):
        # pre-norm
        a, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    """
    Mini GPT dari nol dengan parameter sharing.
    Bobot sebenarnya = 1 block * 1 embedding * 1 output head.
    Tapi kedalaman efektif = num_layers (virtual).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg["vocab_size"]
        self.d_model = cfg["d_model"]
        self.num_layers = cfg["num_layers"]  # virtual depth (parameter sharing)

        # token & positional embedding (learned)
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(cfg["max_seq_len"], self.d_model)

        # 1 shared block (dipakai N kali)
        self.block = SharedBlock(cfg)

        # final norm + output head (tie weights ke tok_emb utk hemat param)
        self.norm_f = RMSNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        new_caches = []
        for i in range(self.num_layers):
            # kalau ada cache per-layer, gunakan; jika tidak None
            cache = kv_caches[i] if kv_caches else None
            x, nc = self.block(x, kv_cache=cache)
            new_caches.append(nc)

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss, new_caches

    # ---------- generation ----------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 50, temperature: float = 1.0,
                 top_k: int | None = None, eos_id: int | None = None):
        self.eval()
        # inisialisasi cache per-layer (semua None)
        kv_caches = [None] * self.num_layers
        # forward seluruh prefix sekali utk isi cache
        for t in range(max_new_tokens):
            if t == 0:
                # forward seluruh prefix
                idx_cond = idx
            else:
                # forward hanya token baru (1 step), pakai cache
                idx_cond = idx[:, -1:]

            logits, _, kv_caches = self.forward(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx

    # ---------- persistence ----------
    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "cfg": self.cfg,
        }, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt["cfg"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    # ---------- info ----------
    def count_params(self):
        n = sum(p.numel() for p in self.parameters())
        # karena parameter sharing, "effective params" = n tapi dengan kedalaman num_layers
        return n

    def info(self):
        n = self.count_params()
        print(f"Model cfg: {self.cfg}")
        print(f"Actual unique params: {n:,} ({n*4/1024:.1f} KB @ fp32)")
        print(f"Virtual depth: {self.num_layers} layers (parameter sharing)")
        print(f"Vocab size: {self.vocab_size}")
        print(f"d_model: {self.d_model}")


if __name__ == "__main__":
    cfg = {
        "vocab_size": 384,
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 256,
        "num_layers": 8,        # virtual depth (sharing)
        "max_seq_len": 128,
    }
    m = MiniGPT(cfg)
    m.info()
    x = torch.randint(0, cfg["vocab_size"], (2, 16))
    logits, loss, _ = m(x, targets=x)
    print(f"logits shape: {logits.shape}, loss: {loss.item():.4f}")
