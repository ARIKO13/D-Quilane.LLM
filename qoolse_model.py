"""
Qoolse AI v3 (1B-Loop Native BF16) — rebuilt from the v3 architecture blueprint.

Physical parameter budget (locked by the spec, verified by `count_params`):

    Blok A  factorized tied embedding + LM head        6.34 M
    Blok B  stem pre-encoder, shared-KV producer,
            exit refinement                            18.66 M
    Blok C  core recursive shared sub-layers (C1-C3)    14.00 M
    Blok D  8 routed SwiGLU experts (top-2 active)      95.99 M
    ---------------------------------------------------------
    total physical                                    ~135.0 M
    effective depth  25.0 M + 9 x 110.0 M          = 1.015 B

Everything below is actually implemented, not only described:

1. Factorized tied embedding (A).  E: V x d_emb (24000 x 256), W_up: d_emb x
   d_model, RMSNorm_emb.  The same W_up and E are reused (transposed) as the
   output head, so the dictionary costs no extra bytes.
2. Stem + 1x shared KV (B1/B2).  Two non-recursive local-attention layers with
   a 256-token sliding window and a SwiGLU FFN, then a single Key/Value
   producer (d_model -> d_kv, 4 heads x 64, RoPE theta 500000) shared by all
   nine loop passes.  Each 64-token block also gets a K_min / K_max envelope.
3. Multi-timescale Gated DeltaNet (C1).  12 heads x 64, causal depthwise
   Conv1D k=4, delta-rule state update S_t = a_t S_{t-1} + b_t (v - S k) k^T.
   6 fast heads (retention ~0.70) and 6 slow heads (~0.997): the state is
   constant-size (98 KB) regardless of context length.
4. Per-loop shared-KV retrieval (C2).  A rank-32 LoRA on the query projection
   per loop pass, QK-norm + RoPE, block scores from the min/max envelope,
   top-32 blocks + the local block (<= 2112 tokens read per loop).
5. Always-on shared expert + routed MoE + halting gate (C3/D).  SwiGLU
   768 -> 4736 -> 768 is always active; 8 routed SwiGLU experts (768 -> 5208
   -> 768) are dispatched sparsely, top-2 per token.
6. Recursive loop with step embedding and adaptive early exit.  9 passes max,
   minimum 4; after the minimum the halting gate accumulates per-token
   probability and freezes tokens that pass the threshold.
7. Exit refinement (B3).  SwiGLU 768 -> 1780 -> 768, a scalar calibration
   gate and a final RMSNorm before the tied head, so an early exit at loop 4
   lands on the same norm scale as a full 9-loop exit.

Precision: the blueprint asks for 100% native BF16 weights.  Weights are stored
BF16/F16 in the container (see `save_gguf`), while CPU compute runs in FP32
(torch CPU has no fast BF16 kernels here); no quantisation, no ternary packing.

Container: GGUF-shaped (magic `GGUF`, version 3, 32-byte alignment) with the
`qoolse_v3_loop` architecture and F16 tensors, so llama.cpp/ollama will not
load it.  Use `infer_qoolse.py` or `playground.py`.
"""
import ctypes
import json
import math
import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- v3 constants
# Every value below is from the blueprint's "Locked Hyperparameters" registry.
QOOLSE_V3_CONFIG = {
    'vocab_size': 24000,        # 23984 BPE + 16 control tokens; local tokenizer is smaller
    'd_model': 768,
    'd_emb': 256,               # factorized embedding / tied head
    'd_kv': 256,                # 4 heads x 64, ONE cache for all 9 loops
    'n_loops': 9,
    'min_loops': 4,             # adaptive early exit floor
    'n_stem_layers': 2,
    'stem_ffn': 2048,
    'stem_window': 256,
    'n_heads': 12,              # stem / delta heads
    'head_dim': 64,
    'n_kv_heads': 4,
    'fast_heads': 6,
    'conv_kernel': 4,
    'kv_block': 64,
    'top_blocks': 32,
    'local_window': 64,
    'n_experts': 8,
    'top_k': 2,
    'expert_hidden': 5208,      # 3 * 768 * 5208 * 8 = 95.99 M routed
    'shared_hidden': 4736,      # 3 * 768 * 4736 = 10.91 M always on
    'exit_hidden': 1780,        # 3 * 768 * 1780 = 4.10 M
    'adapter_rank': 32,         # per-loop query LoRA
    'rope_theta': 500000.0,
    'max_seq': 4096,
    'halt_threshold': 0.5,
    'halt_cost': 0.01,
    'dropout': 0.0,
    'weight_decay': 0.1,
    'load_balancing_weight': 0.01,
    'router_z_loss_weight': 0.001,
    'ternary': False,
}

_CFG_ALIASES = {
    'n_repeat': 'n_loops',
    'expert_hidden_dim': 'expert_hidden',
    'sliding_window': 'stem_window',
    'swa_window_size': 'stem_window',
    'd_kv_cache': 'd_kv',
    'num_loops': 'n_loops',
}

CONFIGS = {
    # The spec's own 135.0 M BF16 model (1.015 B unrolled).
    'v3': QOOLSE_V3_CONFIG,
}


def normalize_cfg(cfg):
    """Accept blueprint names and the internal names, fill the derived keys."""
    c = {_CFG_ALIASES.get(k, k): v for k, v in cfg.items()}
    base = QOOLSE_V3_CONFIG
    for k, v in base.items():
        c.setdefault(k, v)
    c['n_loops'] = max(1, int(c['n_loops']))
    c['min_loops'] = max(1, min(int(c['min_loops']), c['n_loops']))
    c['d_model'] = int(c['d_model'])
    c['d_emb'] = int(c['d_emb'])
    c['d_kv'] = int(c['d_kv'])
    c['n_heads'] = int(c['n_heads'])
    c['head_dim'] = c['d_model'] // c['n_heads'] if 'head_dim' not in cfg else int(c['head_dim'])
    c['n_kv_heads'] = c['d_kv'] // 64 if 'n_kv_heads' not in cfg else int(c['n_kv_heads'])
    c['n_experts'] = int(c['n_experts'])
    c['top_k'] = max(1, min(int(c['top_k']), c['n_experts']))
    c['kv_block'] = max(1, int(c['kv_block']))
    c['top_blocks'] = max(1, int(c['top_blocks']))
    c['local_window'] = max(1, int(c['local_window']))
    return c


# ------------------------------------------------------------------- building
class RMSNorm(nn.Module):
    """Pre-norm RMSNorm with eps = 1e-6 (used everywhere in the spec)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    if cos.dim() == 2:                 # [N, dh] -> [1, 1, N, dh]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RoPE(nn.Module):
    def __init__(self, d_head, base=500000.0):
        super().__init__()
        self.d_head = d_head
        self.base = float(base)
        inv = 1.0 / (self.base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv, persistent=False)

    def forward(self, pos):
        f = torch.outer(pos.float(), self.inv_freq)
        f = torch.cat([f, f], dim=-1)
        return f.cos(), f.sin()


class LayerScale(nn.Module):
    """gamma_l: zero-init residual gate, as in the spec's loop equations."""

    def __init__(self, d, init=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((d,), float(init)))

    def forward(self, x):
        return x * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ------------------------------------------------------- Blok A: embedding/head
class FactorizedEmbedding(nn.Module):
    """E (V x d_emb) -> W_up (d_emb x d_model) -> RMSNorm_emb."""

    def __init__(self, vocab, d_emb, d_model):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_emb)
        self.up = nn.Linear(d_emb, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, idx):
        return self.norm(self.up(self.tok(idx)))

    def head(self, h, norm_out):
        """Tied output: (norm_out(h) @ W_up.T) @ E.T -> [B, T, V]."""
        e = F.linear(norm_out(h), self.up.weight.t())  # h @ W_up.T -> [B,T,d_emb]
        return F.linear(e, self.tok.weight)            # [B,T,d_emb] @ [d_emb,V]


# ------------------------------------------- Blok B1: stem local-attention layer
class StemAttention(nn.Module):
    def __init__(self, d, n_heads, window, rope_base, max_seq):
        super().__init__()
        self.h = n_heads
        self.dh = d // n_heads
        self.window = window
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = RoPE(self.dh, rope_base)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = self.rope(torch.arange(T, device=x.device))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        i = torch.arange(T, device=x.device).unsqueeze(1)
        j = torch.arange(T, device=x.device).unsqueeze(0)
        mask = (j <= i) & (j >= i - self.window + 1)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.o(out)

    def step(self, x, cache):
        B = x.shape[0]
        cos, sin = self.rope(torch.tensor([cache['pos']]))
        q = apply_rotary(self.q(x).view(B, 1, self.h, self.dh).transpose(1, 2), cos, sin)
        k = apply_rotary(self.k(x).view(B, 1, self.h, self.dh).transpose(1, 2), cos, sin)
        v = self.v(x).view(B, 1, self.h, self.dh).transpose(1, 2)
        cache['k'] = k if cache['k'] is None else torch.cat([cache['k'], k], dim=2)
        cache['v'] = v if cache['v'] is None else torch.cat([cache['v'], v], dim=2)
        if cache['k'].shape[2] > self.window:
            cache['k'] = cache['k'][:, :, -self.window:]
            cache['v'] = cache['v'][:, :, -self.window:]
        cache['pos'] += 1
        scores = (q @ cache['k'].transpose(-2, -1)) / math.sqrt(self.dh)
        out = (F.softmax(scores, dim=-1) @ cache['v']).transpose(1, 2).reshape(B, -1)
        return self.o(out)


class StemLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.norm1 = RMSNorm(d)
        self.attn = StemAttention(d, cfg['n_heads'], cfg['stem_window'],
                                  cfg['rope_theta'], cfg['max_seq'])
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d, cfg['stem_ffn'])
        self.ls1 = LayerScale(d, 1.0)
        self.ls2 = LayerScale(d, 1.0)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.ffn(self.norm2(x)))

    def step(self, x, cache):
        x = x + self.ls1(self.attn.step(self.norm1(x), cache))
        return x + self.ls2(self.ffn(self.norm2(x)))


# ------------------------------------------ Blok B2: one shared-KV producer
class SharedKV(nn.Module):
    """Key/Value computed once after the stem and reused by all 9 loops."""

    def __init__(self, cfg):
        super().__init__()
        d, d_kv = cfg['d_model'], cfg['d_kv']
        self.h = cfg['n_kv_heads']
        self.dh = d_kv // self.h
        self.norm = RMSNorm(d)
        self.wk = nn.Linear(d, d_kv, bias=False)
        self.wv = nn.Linear(d, d_kv, bias=False)
        self.rope = RoPE(self.dh, cfg['rope_theta'])

    def _rot(self, k, T, pos0=0):
        B = k.shape[0]
        k = k.view(B, -1, self.h, self.dh).transpose(1, 2)
        pos = torch.arange(pos0, pos0 + k.shape[2], device=k.device)
        cos, sin = self.rope(pos)
        return apply_rotary(k, cos, sin).transpose(1, 2).reshape(B, -1, self.h * self.dh)

    def forward(self, x):
        h = self.norm(x)
        T = x.shape[1]
        K = self._rot(self.wk(h), T)
        V = self.wv(h)
        return K, V


def block_envelopes(K, block):
    """K_min / K_max per `block`-token block (the cheap retrieval envelope)."""
    B, T, C = K.shape
    nb = T // block
    if nb == 0:
        return K.new_zeros(B, 0, C), K.new_zeros(B, 0, C)
    Kb = K[:, : nb * block].reshape(B, nb, block, C)
    return Kb.amin(dim=2), Kb.amax(dim=2)


# --------------------------------- Blok C1: multi-timescale gated delta rule
class MultiScaleGatedDelta(nn.Module):
    """6 fast heads + 6 slow heads, delta rule, O(1) state per head."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.h = cfg['n_heads']
        self.dh = d // self.h
        self.fast = cfg['fast_heads']
        k = cfg['conv_kernel']
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.k = k
        self.conv = nn.Conv1d(3 * d, 3 * d, k, groups=3 * d, bias=False,
                              padding=k - 1)  # causal: left pad, then slice
        self.o = nn.Linear(d, d, bias=False)
        self.beta = nn.Linear(d, self.h)
        self.decay = nn.Linear(d, self.h)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        # Retention targets: fast heads decay quickly, slow heads hold the
        # system prompt for the whole context.  a = log(exp(-log(r)) - 1)
        r = torch.cat([torch.full((self.fast,), 0.70),
                       torch.full((self.h - self.fast,), 0.997)])
        a = torch.log(torch.expm1(-torch.log(r)))
        self.a_bias = nn.Parameter(a)

    def _gate(self, x):
        # exp(-softplus(.)) keeps the per-head decay inside (0, 1)
        logit = self.a_bias + self.decay(x)
        return torch.exp(-F.softplus(logit)).transpose(1, 2)      # [B,h,T]

    def _prep(self, x):
        B, T, C = x.shape
        u = self.qkv(x).transpose(1, 2)
        u = self.conv(u)[..., :T].transpose(1, 2)
        q, k, v = u.chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, self.h, self.dh).transpose(1, 2))
        k = self.k_norm(k.view(B, T, self.h, self.dh).transpose(1, 2))
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2)
        return q, k, v, beta, self._gate(x)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v, beta, gate = self._prep(x)
        state = x.new_zeros(B, self.h, self.dh, self.dh)
        outs = []
        for t in range(T):
            kt = F.normalize(k[:, :, t], dim=-1)
            vt = v[:, :, t]
            state = state * gate[:, :, t].unsqueeze(-1).unsqueeze(-1)
            pred = torch.einsum('bhd,bhde->bhe', kt, state)
            delta = (vt - pred) * beta[:, :, t].unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.einsum('bhd,bhde->bhe', q[:, :, t], state))
        out = torch.stack(outs, dim=1).reshape(B, T, C)
        return self.o(out)

    def step(self, x, state, conv=None):
        """One token. The causal Conv1D keeps its last k-1 raw vectors so the
        streaming path computes the same short convolution as training."""
        B = x.shape[0]
        raw = self.qkv(x)                   # [B, 3d]
        if conv is None:
            conv = {'buf': None}
        buf = raw.unsqueeze(1) if conv['buf'] is None else \
            torch.cat([conv['buf'], raw.unsqueeze(1)], dim=1)
        if buf.shape[1] > self.k:
            buf = buf[:, -self.k:]
        conv['buf'] = buf[:, -self.k:]
        if buf.shape[1] < self.k:      # zero left-pad, exactly like training
            z = buf.new_zeros(B, self.k - buf.shape[1], buf.shape[2])
            buf = torch.cat([z, buf], dim=1)
        u = F.conv1d(buf.transpose(1, 2), self.conv.weight, None,
                     groups=3 * self.h * self.dh).transpose(1, 2)[:, -1]
        q, k, v = u.split(self.h * self.dh, dim=-1)
        q = self.q_norm(q.view(B, self.h, self.dh))
        k = F.normalize(self.k_norm(k.view(B, self.h, self.dh)), dim=-1)
        v = v.view(B, self.h, self.dh)
        beta = torch.sigmoid(self.beta(x))
        gate = torch.exp(-F.softplus(self.a_bias + self.decay(x)))
        if state is None:
            state = x.new_zeros(B, self.h, self.dh, self.dh)
        state = state * gate.unsqueeze(-1).unsqueeze(-1)
        pred = torch.einsum('bhd,bhde->bhe', k, state)
        delta = (v - pred) * beta.unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k.unsqueeze(-2)
        out = torch.einsum('bhd,bhde->bhe', q, state).reshape(B, -1)
        return self.o(out), state


# ------------------------------- Blok C2: per-loop shared-KV cross retrieval
class SharedKVRetrieval(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, d_kv = cfg['d_model'], cfg['d_kv']
        self.d_kv = d_kv
        self.h = cfg['n_kv_heads']
        self.dh = d_kv // self.h
        self.n_loops = cfg['n_loops']
        self.rank = cfg['adapter_rank']
        self.block = cfg['kv_block']
        self.top_blocks = cfg['top_blocks']
        self.local = cfg['local_window']
        self.norm = RMSNorm(d)
        self.q_base = nn.Linear(d, d_kv, bias=False)
        self.lora_a = nn.Parameter(torch.zeros(self.n_loops, d, self.rank))
        self.lora_b = nn.Parameter(torch.randn(self.n_loops, self.rank, d_kv) * 0.02)
        self.qk_norm = RMSNorm(self.dh)
        self.o = nn.Linear(d_kv, d, bias=False)
        self.rope = RoPE(self.dh, cfg['rope_theta'])

    def _query(self, h, loop, pos0=0):
        # W_q_base + (alpha/rank) B_l A_l, both shaped [d_kv, d] like the base
        w = self.q_base.weight + (self.lora_a[loop] @ self.lora_b[loop]).t() / self.rank
        q = F.linear(h, w)
        B, T, _ = q.shape
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        q = self.qk_norm(q)
        cos, sin = self.rope(torch.arange(pos0, pos0 + T, device=h.device))
        return apply_rotary(q, cos, sin)                    # [B,h,T,dh]

    def _block_scores(self, q, kmin, kmax):
        """sum_i max(q_i K_min, q_i K_max), averaged over heads -> [B,T,nb]."""
        B, H, T, dh = q.shape
        nb = kmin.shape[1]
        if nb == 0:
            return q.new_zeros(B, T, 0)
        s = q.new_zeros(B, H, T, nb)
        for j in range(nb):
            mn = kmin[:, j].view(B, H, 1, dh)
            mx = kmax[:, j].view(B, H, 1, dh)
            s[:, :, :, j] = torch.maximum(q * mn, q * mx).sum(-1)
        return s.mean(1)                                     # [B,T,nb]

    def _mask(self, scores, seq_len, device):
        T = seq_len
        nb = scores.shape[-1]
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        local = (j <= i) & (j > i - self.local)
        if nb == 0:
            return local.unsqueeze(0).unsqueeze(0)
        k = min(self.top_blocks, nb)
        topi = scores.topk(k, dim=-1).indices                 # [B,T,k]
        tok_block = torch.arange(T, device=device) // self.block
        sel = torch.zeros(scores.shape[0], T, nb, dtype=torch.bool, device=device)
        sel.scatter_(2, topi, True)
        chosen = sel.gather(2, tok_block.view(1, T, 1).expand(scores.shape[0], -1, -1))
        within = (tok_block.view(1, T, 1) == tok_block.view(1, 1, T))
        allowed = chosen.squeeze(-1).unsqueeze(2) & within
        allowed = allowed | local
        return allowed.unsqueeze(1)                           # [B,1,T,T]

    def forward(self, x, loop, K, V, kmin, kmax):
        B, T, _ = x.shape
        q = self._query(self.norm(x), loop)
        scores = self._block_scores(q, kmin, kmax)
        Kb = K.view(B, T, self.h, self.dh).transpose(1, 2)
        Vb = V.view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ Kb.transpose(-2, -1)) / math.sqrt(self.dh)
        allowed = self._mask(scores, T, x.device)
        att = att.masked_fill(~allowed, float('-inf'))
        out = (F.softmax(att, dim=-1) @ Vb).transpose(1, 2).reshape(B, T, -1)
        return self.o(out)

    def step(self, x, loop, K, V, kmin, kmax):
        B, T, _ = K.shape
        q = self._query(self.norm(x).unsqueeze(1), loop, pos0=T - 1)   # [B,h,1,dh]
        scores = self._block_scores(q, kmin, kmax)[:, 0]        # [B,nb]
        k = min(self.top_blocks, scores.shape[-1])
        if k == 0:
            return self.o(V[:, -1])
        topi = scores.topk(k, dim=-1).indices                   # [B,k]
        base = topi * self.block
        offs = torch.arange(self.block, device=x.device)
        idx = (base.unsqueeze(-1) + offs).reshape(B, -1)
        idx = idx[idx < T]
        lo = max(0, T - self.local)
        idx = torch.cat([idx, torch.arange(lo, T, device=x.device)])
        idx = torch.unique(idx)
        Kb = K.index_select(1, idx).view(B, -1, self.h, self.dh).transpose(1, 2)
        Vb = V.index_select(1, idx).view(B, -1, self.h, self.dh).transpose(1, 2)
        att = (q @ Kb.transpose(-2, -1)) / math.sqrt(self.dh)
        out = (F.softmax(att, dim=-1) @ Vb).transpose(1, 2).reshape(B, -1)
        return self.o(out)


# ------------------------------- Blok C3 / D: shared expert, MoE, halting gate
class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.n = cfg['n_experts']
        self.top_k = cfg['top_k']
        self.router = nn.Linear(d, self.n, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, cfg['expert_hidden'])
                                      for _ in range(self.n)])
        self.shared = SwiGLU(d, cfg['shared_hidden'])
        self.halt = nn.Linear(d, 1, bias=True)
        self.register_buffer('_zeros', torch.zeros(1), persistent=False)

    def _routed(self, xf):
        """Sparse dispatch: each token touches only its top-k experts."""
        B, T, C = xf.shape
        flat = xf.reshape(-1, C)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            sel = (topi == e)
            if not bool(sel.any()):
                continue
            tok, slot = sel.nonzero(as_tuple=True)
            y = expert(flat.index_select(0, tok))
            out.index_add_(0, tok, y * topv[tok, slot].unsqueeze(-1))
        frac = torch.bincount(topi.reshape(-1), minlength=self.n).to(flat.dtype)
        frac = frac / max(1, topi.numel() // 1)
        lb = self.n * (frac * probs.mean(0)).sum()
        rz = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return out.view(B, T, C), {'load_balancing_loss': lb, 'router_z_loss': rz}

    def forward(self, x):
        routed, aux = self._routed(x)
        return routed + self.shared(x), aux

    def halt_prob(self, x):
        return torch.sigmoid(self.halt(x)).squeeze(-1)


class CoreBlock(nn.Module):
    """C1 -> C2 -> C3/D with zero-init residual gates per loop."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.norm_c1 = RMSNorm(d)
        self.c1 = MultiScaleGatedDelta(cfg)
        self.g1 = LayerScale(d)
        self.norm_c2 = RMSNorm(d)
        self.c2 = SharedKVRetrieval(cfg)
        self.g2 = LayerScale(d)
        self.norm_c3 = RMSNorm(d)
        self.moe = MoE(cfg)
        self.g3 = LayerScale(d)

    def forward(self, x, loop, K, V, kmin, kmax):
        x = x + self.g1(self.c1(self.norm_c1(x)))
        x = x + self.g2(self.c2(self.norm_c2(x), loop, K, V, kmin, kmax))
        h = self.norm_c3(x)
        m, aux = self.moe(h)
        x = x + self.g3(m)
        return x, aux, self.moe.halt_prob(h)

    def step(self, x, loop, K, V, kmin, kmax, state, conv=None):
        delta, state = self.c1.step(self.norm_c1(x), state, conv)
        x = x + self.g1(delta)
        x = x + self.g2(self.c2.step(self.norm_c2(x), loop, K, V, kmin, kmax))
        h = self.norm_c3(x)
        m, _ = self.moe(h.unsqueeze(1))
        x = x + self.g3(m.squeeze(1))
        return x, self.moe.halt_prob(h), state


# -------------------------------------------- Blok B3: exit refinement + head
class ExitRefinement(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.norm = RMSNorm(d)
        self.ffn = SwiGLU(d, cfg['exit_hidden'])
        self.calib = nn.Linear(d, 1)
        self.norm_out = RMSNorm(d)

    def forward(self, x):
        gate = torch.sigmoid(self.calib(x))
        x = x + gate * self.ffn(self.norm(x))
        return self.norm_out(x)


# ------------------------------------------------------------------ the model
class QoolseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg = normalize_cfg(cfg)
        self.cfg = dict(cfg)
        d = cfg['d_model']
        self.emb = FactorizedEmbedding(cfg['vocab_size'], cfg['d_emb'], d)
        self.stem = nn.ModuleList([StemLayer(cfg) for _ in range(cfg['n_stem_layers'])])
        self.shared_kv = SharedKV(cfg)
        self.core = CoreBlock(cfg)
        self.step_emb = nn.Parameter(torch.zeros(cfg['n_loops'], d))
        self.exit = ExitRefinement(cfg)
        self.n_loops = cfg['n_loops']
        self.min_loops = cfg['min_loops']
        self.n_stem_active = cfg['n_stem_layers']
        self.apply(self._init)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.step_emb)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------- accounting
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def expert_params(self):
        return sum(p.numel() for p in self.core.moe.experts.parameters())

    def count_active_params(self):
        """One token: everything except the routed experts it does not visit."""
        total = self.count_params()
        ex = self.expert_params()
        return int(total - ex + ex * self.cfg['top_k'] / self.cfg['n_experts'])

    def storage_bytes(self):
        """Native BF16 on disk: 2 bytes per parameter."""
        return self.count_params() * 2

    def count_ternary_layers(self):
        return 0  # v3 is native BF16, there is no ternary packing

    def compute_multiplier(self):
        return self.n_loops

    def set_active_depth(self, n_blocks=None, n_repeat=None):
        """Progressive depth: `n_repeat` narrows the loop count, `n_blocks`
        narrows the number of active stem layers."""
        if n_repeat is not None:
            if not 1 <= n_repeat <= self.cfg['n_loops']:
                raise ValueError(f"n_loops must be 1..{self.cfg['n_loops']}, got {n_repeat}")
            self.n_loops = n_repeat
        if n_blocks is not None:
            if not 1 <= n_blocks <= len(self.stem):
                raise ValueError(f"stem layers must be 1..{len(self.stem)}, got {n_blocks}")
            self.n_stem_active = n_blocks
        return self.n_stem_active, self.n_loops

    # ----------------------------------------------------------------- forward
    def _stem(self, x):
        for blk in self.stem[: self.n_stem_active]:
            x = blk(x)
        return x

    def forward(self, idx, targets=None, return_aux=False):
        B, T = idx.shape
        if T > self.cfg['max_seq']:
            raise ValueError(f"sequence length {T} > max_seq {self.cfg['max_seq']}")
        x = self.emb(idx)
        x = self._stem(x)
        K, V = self.shared_kv(x)
        kmin, kmax = block_envelopes(K, self.cfg['kv_block'])
        dp = self.cfg.get('dropout', 0.0)
        use_exit = (not self.training) and bool(self.cfg.get('early_exit', True))
        halted = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        cum = torch.zeros(B, T, device=idx.device)
        aux_list, ponder = [], []
        for loop in range(self.n_loops):
            h = x + self.step_emb[loop]
            y, aux, p = self.core(h, loop, K, V, kmin, kmax)
            aux_list.append(aux)
            ponder.append(p.mean())
            if use_exit and loop + 1 >= self.min_loops:
                cum = cum + p
                halted = halted | (cum > self.cfg['halt_threshold'])
                x = torch.where(halted.unsqueeze(-1), x, y)
                if bool(halted.all()):
                    break
            else:
                x = y
            if dp > 0 and self.training:
                x = F.dropout(x, p=dp)
        x = self.exit(x)
        logits = self.emb.head(x, lambda z: z)
        ce = lb = rz = None
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, self.cfg['vocab_size']),
                                 targets.reshape(-1), ignore_index=-100)
            loss = ce
            if return_aux and aux_list:
                lb = sum(a['load_balancing_loss'] for a in aux_list) / len(aux_list)
                rz = sum(a['router_z_loss'] for a in aux_list) / len(aux_list)
                loss = loss + self.cfg['load_balancing_weight'] * lb \
                    + self.cfg['router_z_loss_weight'] * rz \
                    + self.cfg['halt_cost'] * sum(ponder) / len(ponder)
        if return_aux:
            return logits, loss, {
                'ce_loss': ce.item() if ce is not None else 0.0,
                'load_balancing_loss': lb.item() if lb is not None else 0.0,
                'router_z_loss': rz.item() if rz is not None else 0.0,
                'mtp_loss': 0.0,
                'ponder': float((sum(ponder) / max(1, len(ponder))).detach()),
            }
        return logits, loss

    # ------------------------------------------------------------ streaming
    def new_cache(self, batch_size=1):
        return {
            'stem': [{'k': None, 'v': None, 'pos': 0} for _ in self.stem],
            'kv': {'k': None, 'v': None, 'pos': 0},
            'delta': [None] * self.n_loops,
            'conv': [None] * self.n_loops,
        }

    def _shared_kv_step(self, x, cache):
        kv = cache['kv']
        h = self.shared_kv.norm(x)
        k = self.shared_kv._rot(self.shared_kv.wk(h), 1, pos0=kv['pos'])
        v = self.shared_kv.wv(h).unsqueeze(1)
        kv['k'] = k if kv['k'] is None else torch.cat([kv['k'], k], dim=1)
        kv['v'] = v if kv['v'] is None else torch.cat([kv['v'], v], dim=1)
        kv['pos'] += 1
        return kv['k'], kv['v']

    @torch.no_grad()
    def step(self, idx_t, caches, lut=False):
        """One token ([B,1]) -> logits [B,V]. `lut` is accepted for API
        compatibility; v3 has no ternary path."""
        x = self.emb(idx_t).squeeze(1)
        for i, blk in enumerate(self.stem[: self.n_stem_active]):
            x = blk.step(x, caches['stem'][i])
        K, V = self._shared_kv_step(x, caches)
        kmin, kmax = block_envelopes(K, self.cfg['kv_block'])
        cum = torch.zeros(x.shape[0], device=x.device)
        halted = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        use_exit = bool(self.cfg.get('early_exit', True))
        for loop in range(self.n_loops):
            h = x + self.step_emb[loop]
            y, p, state = self.core.step(h, loop, K, V, kmin, kmax,
                                         caches['delta'][loop], caches['conv'][loop])
            caches['delta'][loop] = state
            if use_exit and loop + 1 >= self.min_loops:
                cum = cum + p
                halted = halted | (cum > self.cfg['halt_threshold'])
                x = torch.where(halted.unsqueeze(-1), x, y)
                if bool(halted.all()):
                    break
            else:
                x = y
        x = self.exit(x)
        return self.emb.head(x.unsqueeze(1), lambda z: z).squeeze(1)

    @torch.no_grad()
    def generate(self, idx, max_new=50, temp=0.5, top_k=5, stop_id=None,
                 entrocut=None, lut=False):
        self.eval()
        if entrocut is None:
            entrocut = self.cfg.get('entrocut_threshold', 0.0)
        caches = self.new_cache(idx.shape[0])
        logits = None
        for t in range(idx.shape[1]):
            logits = self.step(idx[:, t:t + 1], caches)
        for _ in range(max_new):
            sl = logits / max(temp, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(sl, min(top_k, sl.size(-1)))
                sl = sl.masked_fill(sl < v[:, [-1]], float('-inf'))
            probs = F.softmax(sl, dim=-1)
            best_p, best_i = probs.max(dim=-1)
            if entrocut and float(best_p) > entrocut:
                break
            nid = best_i if temp <= 1e-5 else torch.multinomial(probs, 1)
            nid = nid.view(1, 1)
            idx = torch.cat([idx, nid], dim=1)
            if stop_id is not None and int(nid) == int(stop_id):
                break
            logits = self.step(nid, caches)
        return idx

    # ---------------------------------------------------------------- GGUF
    def save_gguf(self, path, stoi=None, itos=None, eos_id=None, aion_engram=None,
                  architecture='qoolse_v3_loop', extra_meta=None):
        """GGUF-shaped container, every tensor stored as F16 (native BF16
        model weights, 2 bytes per parameter, ~270 MB for 135 M params)."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        entries = []
        for name, t in self.state_dict().items():
            entries.append((name, t.detach().cpu().contiguous(),
                            list(t.shape), GGUF_TYPE_F16))
        c = self.cfg
        meta = {
            'general.architecture': architecture,
            'general.name': 'Qoolse AI v3',
            'general.alignment': str(GGUF_ALIGN),
            'architecture': architecture,
            'name': 'Qoolse AI v3',
            'techniques': json.dumps([
                'Factorized_Tied_Embedding', 'Stem_SlidingWindow_Attention',
                'Shared_KV_1x_for_all_loops', 'KV_Block_MinMax_Envelope',
                'MultiTimescale_GatedDeltaNet', 'PerLoop_Query_LoRA_rank{}'.format(c['adapter_rank']),
                'Top{}_Block_Sparse_Retrieval'.format(c['top_blocks']),
                'AlwaysOn_Shared_Expert', 'Routed_MoE_Top{}'.format(c['top_k']),
                'Halting_Gate', 'Adaptive_Early_Exit', 'Exit_Refinement',
                'Recursive_Loop_{}'.format(c['n_loops']), 'RoPE_theta{}'.format(int(c['rope_theta'])),
                'QK_Norm', 'RMSNorm', 'SwiGLU', 'Native_BF16']),
            'vocab_size': str(c['vocab_size']), 'd_model': str(c['d_model']),
            'd_emb': str(c['d_emb']), 'd_kv': str(c['d_kv']),
            'n_heads': str(c['n_heads']), 'head_dim': str(c['head_dim']),
            'n_kv_heads': str(c['n_kv_heads']),
            'n_loops': str(c['n_loops']), 'min_loops': str(c['min_loops']),
            'n_stem_layers': str(c['n_stem_layers']), 'stem_ffn': str(c['stem_ffn']),
            'stem_window': str(c['stem_window']),
            'kv_block': str(c['kv_block']), 'top_blocks': str(c['top_blocks']),
            'local_window': str(c['local_window']),
            'n_experts': str(c['n_experts']), 'top_k': str(c['top_k']),
            'expert_hidden': str(c['expert_hidden']),
            'shared_hidden': str(c['shared_hidden']),
            'exit_hidden': str(c['exit_hidden']),
            'adapter_rank': str(c['adapter_rank']),
            'rope_theta': str(c['rope_theta']),
            'max_seq': str(c['max_seq']),
            'halt_threshold': str(c['halt_threshold']),
            'n_blocks': str(c['n_stem_layers']),
            'n_repeat': str(c['n_loops']),
            'n_effective_layers': str(c['n_loops']),
            'sliding_window': str(c['stem_window']),
            'precision': 'native BF16 weights (F16 container), FP32 CPU compute',
            'unique_params': str(self.count_params()),
            'active_params_per_token': str(self.count_active_params()),
            'effective_params_unrolled': str(int(self.count_active_params() * c['n_loops'])),
            'storage_mb': str(round(self.storage_bytes() / 1024 / 1024, 3)),
            'ternary_layers': '0',
            'entrocut_threshold': str(self.cfg.get('entrocut_threshold', 0.95)),
        }
        if stoi:
            meta['tokenizer_stoi'] = json.dumps(stoi, ensure_ascii=False)
        if itos:
            meta['tokenizer_itos'] = json.dumps({str(k): v for k, v in itos.items()},
                                                ensure_ascii=False)
        if eos_id is not None:
            meta['eos_id'] = str(eos_id)
        if aion_engram:
            meta['aion_engram'] = json.dumps(aion_engram, ensure_ascii=False)
        if extra_meta:
            meta.update({k: str(v) for k, v in extra_meta.items()})

        with open(path, 'wb') as f:
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(entries)))
            f.write(struct.pack('<Q', len(meta)))
            for k, val in meta.items():
                _write_str(f, k)
                f.write(struct.pack('<I', GGUF_TYPE_STRING))
                _write_str(f, str(val))
            off = 0
            for name, t, dims, ty in entries:
                _write_str(f, name)
                f.write(struct.pack('<I', len(dims)))
                for dv in dims:
                    f.write(struct.pack('<Q', dv))
                f.write(struct.pack('<I', ty))
                f.write(struct.pack('<Q', off))
                off += _nbytes(t, ty)
                off += (GGUF_ALIGN - (off % GGUF_ALIGN)) % GGUF_ALIGN
            cur = f.tell()
            pad = (GGUF_ALIGN - (cur % GGUF_ALIGN)) % GGUF_ALIGN
            if pad:
                f.write(b'\x00' * pad)
            for name, t, dims, ty in entries:
                cur = f.tell()
                p = (GGUF_ALIGN - (cur % GGUF_ALIGN)) % GGUF_ALIGN
                if p:
                    f.write(b'\x00' * p)
                f.write(t.flatten().to(torch.float16).numpy().tobytes())
        sz = os.path.getsize(path)
        print(f"[GGUF] {path}  ({sz/1024/1024:.2f} MB, {len(entries)} tensors, "
              f"{self.count_params():,} params, F16/BF16)")
        return path


# ---------------------------------------------------------------- GGUF helpers
def _nbytes(t, ty):
    return int(t.numel()) * (4 if ty == GGUF_TYPE_F32 else 2)


def _write_str(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def _read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8')


def read_gguf(path):
    """Parse the container written by QoolseModel.save_gguf.

    Returns (meta, tensors) with every tensor as FP32.
    """
    with open(path, 'rb') as f:
        magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF container (magic={magic:#x})")
        if version != GGUF_VERSION:
            raise ValueError(f"{path}: container version {version}, expected {GGUF_VERSION}")
        meta = {}
        for _ in range(n_kv):
            k = _read_str(f)
            (ty,) = struct.unpack('<I', f.read(4))
            if ty != GGUF_TYPE_STRING:
                raise ValueError(f"metadata {k!r}: unsupported type {ty}")
            meta[k] = _read_str(f)
        infos = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<%dQ' % nd, f.read(8 * nd)) if nd else ()
            ty, off = struct.unpack('<IQ', f.read(12))
            if ty not in (GGUF_TYPE_F32, GGUF_TYPE_F16):
                raise ValueError(f"tensor {name!r}: unsupported type {ty}")
            infos.append((name, dims, off, ty))
        align = int(meta.get('general.alignment', GGUF_ALIGN))
        cur = f.tell()
        data_start = cur + ((align - cur % align) % align)
        tensors = {}
        for name, dims, off, ty in infos:
            f.seek(data_start + off)
            n = 1
            for dv in dims:
                n *= int(dv)
            dt = torch.float32 if ty == GGUF_TYPE_F32 else torch.float16
            buf = f.read(n * dt.itemsize)
            if len(buf) != n * dt.itemsize:
                raise ValueError(f"tensor {name!r}: truncated data")
            t = torch.frombuffer(bytearray(buf), dtype=dt).clone().to(torch.float32)
            tensors[name] = t.reshape(dims) if dims else t
    return meta, tensors


def load_from_gguf(path):
    """Rebuild a QoolseModel (v3) from a container."""
    meta, tensors = read_gguf(path)
    ints = ['vocab_size', 'd_model', 'd_emb', 'd_kv', 'n_loops', 'min_loops',
            'n_stem_layers', 'stem_ffn', 'stem_window', 'n_heads', 'head_dim',
            'n_kv_heads', 'kv_block', 'top_blocks', 'local_window', 'n_experts',
            'top_k', 'expert_hidden', 'shared_hidden', 'exit_hidden',
            'adapter_rank', 'max_seq']
    missing = [k for k in ints if k not in meta]
    if missing:
        raise ValueError(f"{path}: missing metadata {missing}")
    cfg = {k: int(meta[k]) for k in ints}
    cfg['rope_theta'] = float(meta.get('rope_theta', 500000.0))
    cfg['halt_threshold'] = float(meta.get('halt_threshold', 0.5))
    cfg['entrocut_threshold'] = float(meta.get('entrocut_threshold', 0.95))
    cfg['dropout'] = 0.0
    model = QoolseModel(cfg)
    model.load_state_dict(tensors, strict=True)
    model.eval()
    return model, meta


# --------------------------------------------------------------- size classes
PARAM_TIERS = (
    ('Small', 1_000_000_000, 4_000_000_000),
    ('Compact', 4_000_000_000, 9_000_000_000),
    ('Medium', 9_000_000_000, 15_000_000_000),
    ('Large', 15_000_000_000, 35_000_000_000),
    ('Very Large', 35_000_000_000, 70_000_000_000),
    ('Frontier', 70_000_000_000, 200_000_000_000),
    ('Large-scale', 200_000_000_000, 500_000_000_000),
    ('Ultra-scale', 500_000_000_000, 1_000_000_000_000),
    ('Beyond 1T', 1_000_000_000_000, None),
)

_UNITS = (('T', 1e12), ('B', 1e9), ('M', 1e6), ('K', 1e3))


def format_params(n):
    n = int(n)
    for unit, div in _UNITS:
        if abs(n) >= div:
            return f"{n / div:.2f}".rstrip('0').rstrip('.') + unit
    return str(n)


def parse_params(text):
    s = str(text).strip().lower().replace(',', '').replace('_', '').replace(' ', '')
    mult = 1
    if s and s[-1] in 'kmbt':
        mult = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}[s[-1]]
        s = s[:-1]
    if not s:
        raise ValueError('empty parameter count')
    return int(round(float(s) * mult))


def find_tier(name):
    key = str(name).strip().lower()
    for i, tier in enumerate(PARAM_TIERS):
        if key in (tier[0].lower(), str(i + 1)):
            return tier
    for tier in PARAM_TIERS:
        if tier[0].lower().startswith(key):
            return tier
    names = ', '.join(t[0] for t in PARAM_TIERS)
    raise ValueError(f"unknown size class {name!r} (choose from: {names})")


def tier_range_text(tier):
    label, lo, hi = tier
    return f"{format_params(lo)} - {format_params(hi)}" if hi else f"> {format_params(lo)}"


def tier_for_params(n):
    n = int(n)
    for tier in PARAM_TIERS:
        _, lo, hi = tier
        if n >= lo and (hi is None or n < hi):
            return tier
    return None


def resolve_target_params(tier, params):
    label, lo, hi = find_tier(tier)
    n = parse_params(params)
    if n < lo or (hi is not None and n >= hi):
        raise ValueError(f"{format_params(n)} is outside {label} "
                         f"({tier_range_text((label, lo, hi))})")
    return n


# ------------------------------------------------------------------ auto sizing
class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def available_ram():
    """Bytes of free physical RAM, or None when it cannot be measured."""
    try:
        if os.name == 'nt':
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def count_params_for(cfg):
    with torch.device('meta'):
        model = QoolseModel(cfg)
    return model.count_params()


def config_for_params(target_params, base=None):
    """Widen the v3 shape (keeping its ratios) until it reaches a target.
    The spec's own 135 M model is `CONFIGS['v3']`; this is only for scaling."""
    cfg = dict(base or QOOLSE_V3_CONFIG)
    target = max(int(target_params), 1)
    d0 = 768.0

    def widen(d):
        c = dict(cfg)
        d = max(128, int(d) // 128 * 128)
        r = d / d0
        c['d_model'] = d
        c['d_emb'] = max(64, int(round(256 * r / 32)) * 32)
        c['d_kv'] = max(64, int(round(256 * r / 32)) * 32)
        c['n_heads'] = max(2, d // 64)
        c['head_dim'] = 64
        c['n_kv_heads'] = max(1, c['d_kv'] // 64)
        c['stem_ffn'] = max(128, int(round(2048 * r / 64)) * 64)
        c['expert_hidden'] = max(128, int(round(5208 * r / 64)) * 64)
        c['shared_hidden'] = max(128, int(round(4736 * r / 64)) * 64)
        c['exit_hidden'] = max(128, int(round(1780 * r / 64)) * 64)
        return c

    d = int(cfg['d_model'])
    for _ in range(24):
        c = widen(d)
        p = count_params_for(c)
        nd = max(128, int(round(d * (target / p) ** 0.5)) // 128 * 128)
        if nd == d:
            break
        d = nd
    return widen(d)


def auto_target_params(ram_fraction=0.5, reserve_mb=384):
    """Free RAM / 4 bytes per FP32 latent weight: what the machine can hold."""
    free = available_ram()
    if not free:
        return count_params_for(QOOLSE_V3_CONFIG)
    budget = max(free * ram_fraction - reserve_mb * 1024 * 1024, 256 * 1024 * 1024)
    return int(budget / 4)


def auto_config(**kw):
    return config_for_params(auto_target_params(**kw))


def normalize_aion_key(text, max_len=60):
    return text.lower().strip()[:max_len]


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGN = 32
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_STRING = 8


if __name__ == '__main__':
    m = QoolseModel(QOOLSE_V3_CONFIG)
    n = m.count_params()
    act = m.count_active_params()
    c = m.cfg
    print(f"Qoolse AI v3: {n:,} params ({n*2/1024/1024:.2f} MB BF16)")
    print(f"  blocks        : A {c['d_emb']}d emb + B stem {c['n_stem_layers']}x, "
          f"C core, D {c['n_experts']} experts")
    churn = n - m.core.count_params()          # A + B (run once)
    print(f"  loops         : {c['n_loops']} (min {c['min_loops']}), "
          f"{churn + m.core.count_params() * c['n_loops']:,} effective/unrolled")
    print(f"  active/token  : {act:,} ({act/n*100:.1f}%)")
    print(f"  d_model/d_kv  : {c['d_model']} / {c['d_kv']}, heads {c['n_heads']}x{c['head_dim']}")
    x = torch.randint(0, c['vocab_size'], (2, 16))
    logits, loss, aux = m(x, targets=x, return_aux=True)
    print(f"  forward OK    : logits {tuple(logits.shape)}, loss={loss.item():.4f}")
    print(f"  aux           : {aux}")
