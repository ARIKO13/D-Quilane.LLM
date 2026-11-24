"""
D'QUILANE Chimera — minimal character-level language model for training.

Sandbox-scale: d_model=96, 4 blocks × 3 repeat = 12 eff. layers, 4 experts,
215,808 unique parameters (0.82 MB of FP32 weights).

Implements: LowRank GQA, FFN Bottleneck (SwiGLU), ReXMoE (dense top-2 over 4
experts -- all experts are computed, so there is NO sparse compute saving),
Weight Tying, LayerScale (init=1e-1), RMSNorm, RoPE, Load Balancing Loss
(Switch Transformer), Router Z-Loss (ST-MoE), EntroCut early stopping.

This file is a teaching / research sandbox, not a usable chat LLM: at 215K
parameters over a 115-token character vocabulary it can learn character
statistics and memorise snippets, but it cannot answer open questions.
The file format written by `save_gguf` mimics GGUF (same magic, same
header layout) but uses a custom `chimera_8b_effective` architecture, so
llama.cpp / ollama will NOT load it. Use `infer_dquilane.py`.
"""
import json
import math
import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F


CONFIGS = {
    'sandbox': {
        'd_model': 96, 'n_blocks': 4, 'n_repeat': 3,
        'n_experts': 4, 'attn_rank': 32, 'ffn_hidden': 48,
        # Must match the tokenizer built in build_dquilane.py:
        # 112 unique characters + <pad> + <eos>.
        'vocab_size': 114, 'max_seq': 80,
        'n_heads': 4, 'n_kv_heads': 1, 'sliding_window': 32,
        'entrocut_threshold': 0.95,
        'dropout': 0.1, 'weight_decay': 0.1,
        'load_balancing_weight': 0.01, 'router_z_loss_weight': 0.001,
    },
    # Target config. Never trained: no GPU run exists for it in this repo.
    'chimera_1gb': {
        'd_model': 2048, 'n_blocks': 6, 'n_repeat': 4,
        'n_experts': 6, 'attn_rank': 256, 'ffn_hidden': 1024,
        'vocab_size': 32000, 'max_seq': 4096,
        'n_heads': 16, 'n_kv_heads': 4, 'sliding_window': 1024,
        'entrocut_threshold': 0.95,
        'dropout': 0.1, 'weight_decay': 0.1,
        'load_balancing_weight': 0.01, 'router_z_loss_weight': 0.001,
        'gradient_checkpointing': True,
    },
}

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGN = 32
GGUF_TYPE_F32 = 0
GGUF_TYPE_STRING = 8

AION_KEY_MAX = 60


def normalize_aion_key(text, max_len=AION_KEY_MAX):
    """Single definition of the AION retrieval key, shared by build + inference."""
    return text.lower().strip()[:max_len]


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class LayerScale(nn.Module):
    def __init__(self, d, init=1e-1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d) * init)

    def forward(self, x):
        return x * self.scale


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq=4096, base=10000.0):
        super().__init__()
        self.d_head = d_head
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        cos, sin = self._build(max_seq)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def _build(self, T):
        t = torch.arange(T, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos(), freqs.sin()

    def forward(self, q, k, T):
        if T > self.cos.shape[0]:
            # Sequence longer than the precomputed table: extend instead of
            # crashing on a shape mismatch.
            self.cos, self.sin = self._build(T)
        return (apply_rotary(q, self.cos[:T], self.sin[:T]),
                apply_rotary(k, self.cos[:T], self.sin[:T]))


class LowRankAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, rank, window=None, max_seq=4096):
        super().__init__()
        self.h, self.n_kv = n_heads, n_kv_heads
        self.dh = d // n_heads
        self.rank = rank
        self.window = window
        self.kv_dim = self.dh * n_kv_heads
        self.q_down = nn.Linear(d, rank, bias=False)
        self.q_up = nn.Linear(rank, d, bias=False)
        self.k_down = nn.Linear(d, rank, bias=False)
        self.k_up = nn.Linear(rank, self.kv_dim, bias=False)
        self.v_down = nn.Linear(d, rank, bias=False)
        self.v_up = nn.Linear(rank, self.kv_dim, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = RoPE(self.dh, max_seq)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_up(self.q_down(x)).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k_up(self.k_down(x)).view(B, T, self.n_kv, self.dh).transpose(1, 2)
        v = self.v_up(self.v_down(x)).view(B, T, self.n_kv, self.dh).transpose(1, 2)
        q, k = self.rope(q, k, T)
        if self.n_kv < self.h:
            rep = self.h // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        idx_i = torch.arange(T, device=x.device).unsqueeze(1)
        idx_j = torch.arange(T, device=x.device).unsqueeze(0)
        mask = idx_j <= idx_i
        if self.window and self.window < T:
            mask = mask & (idx_j >= (idx_i - self.window + 1))
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o(out)


class FFNBottleneck(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.val = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.val(x))


class ExpertPool(nn.Module):
    """Top-2 router over N SwiGLU experts.

    Note: every expert is evaluated for every token (the gather is dense), so
    this costs N× a single FFN instead of 2×. It is a dense mixture, not a
    sparse MoE; the parameter count is honest but there is no speed-up.
    """

    def __init__(self, d, hidden, n_experts):
        super().__init__()
        self.n = n_experts
        self.experts = nn.ModuleList([FFNBottleneck(d, hidden) for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts, bias=False)

    def forward(self, x, return_aux=False):
        B, T, C = x.shape
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim=-1)
        k = min(2, self.n)
        topk_val, topk_idx = probs.topk(k, dim=-1)
        topk_val = topk_val / (topk_val.sum(-1, keepdim=True) + 1e-10)
        exp_outs = torch.stack([self.experts[e](x) for e in range(self.n)], dim=-2)
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        sel = torch.gather(exp_outs, dim=-2, index=idx)
        out = (sel * topk_val.unsqueeze(-1)).sum(dim=-2)
        if not return_aux:
            return out
        frac = probs.mean(dim=(0, 1))
        top1_mask = (topk_idx[..., 0:1] ==
                     torch.arange(self.n, device=x.device).view(1, 1, -1))
        top1_frac = top1_mask.float().mean(dim=(0, 1))
        lb = self.n * (frac * top1_frac).sum()
        rz = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return out, {'load_balancing_loss': lb, 'router_z_loss': rz}


class SharedBlock(nn.Module):
    """A block whose weights are reused n_repeat times (recurrent depth)."""

    def __init__(self, cfg, use_swa=False):
        super().__init__()
        d = cfg['d_model']
        w = cfg.get('sliding_window') if use_swa else None
        self.norm1 = RMSNorm(d)
        self.attn = LowRankAttention(d, cfg['n_heads'], cfg['n_kv_heads'],
                                     cfg['attn_rank'], w, cfg['max_seq'])
        self.norm2 = RMSNorm(d)
        self.ffn = FFNBottleneck(d, cfg['ffn_hidden'])
        self.ls1 = LayerScale(d)
        self.ls2 = LayerScale(d)

    def forward(self, x, expert_pool, return_aux=False, dropout_p=0.0):
        a = self.attn(self.norm1(x))
        if dropout_p > 0 and self.training:
            a = F.dropout(a, p=dropout_p)
        x = x + self.ls1(a)
        h = self.norm2(x)
        f = self.ffn(h)
        if return_aux:
            m, aux = expert_pool(h, return_aux=True)
        else:
            m = expert_pool(h, return_aux=False)
        s = f + m
        if dropout_p > 0 and self.training:
            s = F.dropout(s, p=dropout_p)
        x = x + self.ls2(s)
        if return_aux:
            return x, aux
        return x


class ChimeraModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = cfg['d_model']
        v = cfg['vocab_size']
        self.tok_emb = nn.Embedding(v, d)
        self.blocks = nn.ModuleList([
            SharedBlock(cfg, use_swa=(i % 2 == 1)) for i in range(cfg['n_blocks'])
        ])
        self.n_repeat = cfg['n_repeat']
        # Progressive stacking mutates this, never cfg / the ModuleList.
        self.n_blocks_active = cfg['n_blocks']
        self.expert_pool = ExpertPool(d, cfg['ffn_hidden'], cfg['n_experts'])
        self.norm_f = RMSNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)

    def set_active_depth(self, n_blocks=None, n_repeat=None):
        """Progressive stacking: enable a subset of blocks / repeat count.

        The weights of inactive blocks stay in the model (and in the file), so
        the parameter count never changes while training.
        """
        if n_blocks is not None:
            if not 1 <= n_blocks <= len(self.blocks):
                raise ValueError(f"n_blocks must be 1..{len(self.blocks)}, got {n_blocks}")
            self.n_blocks_active = n_blocks
        if n_repeat is not None:
            if n_repeat < 1:
                raise ValueError(f"n_repeat must be >= 1, got {n_repeat}")
            self.n_repeat = n_repeat
        return self.n_blocks_active, self.n_repeat

    def forward(self, idx, targets=None, return_aux=False):
        B, T = idx.shape
        if T > self.cfg['max_seq']:
            raise ValueError(f"sequence length {T} > max_seq {self.cfg['max_seq']}")
        x = self.tok_emb(idx)
        dp = self.cfg.get('dropout', 0.0)
        aux_list = []
        blocks = self.blocks[: self.n_blocks_active]
        for _ in range(self.n_repeat):
            for blk in blocks:
                if return_aux:
                    x, aux = blk(x, self.expert_pool, True, dp)
                    aux_list.append(aux)
                else:
                    x = blk(x, self.expert_pool, False, dp)
        x = self.norm_f(x)
        logits = F.linear(x, self.tok_emb.weight)  # weight tying
        ce = lb = rz = None
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, self.cfg['vocab_size']),
                                 targets.view(-1), ignore_index=-100)
            loss = ce
            if return_aux and aux_list:
                lb = sum(a['load_balancing_loss'] for a in aux_list) / len(aux_list)
                rz = sum(a['router_z_loss'] for a in aux_list) / len(aux_list)
                loss = ce + self.cfg.get('load_balancing_weight', 0.01) * lb \
                    + self.cfg.get('router_z_loss_weight', 0.001) * rz
        if return_aux:
            return logits, loss, {
                'ce_loss': ce.item() if ce is not None else 0.0,
                'load_balancing_loss': lb.item() if lb is not None else 0.0,
                'router_z_loss': rz.item() if rz is not None else 0.0,
            }
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def compute_multiplier(self):
        """Effective depth = blocks active × repeat. Do NOT call this "params":
        no model gets bigger by reusing weights."""
        return self.n_blocks_active * self.n_repeat

    def tokens_per_forward(self):
        """FLOPs-ish estimate for documentation, not a parameter count."""
        return self.count_params() * self.compute_multiplier()

    @torch.no_grad()
    def generate(self, idx, max_new=50, temp=0.5, top_k=5, stop_id=None,
                 entrocut=None):
        """Greedy-ish sampling with optional EntroCut early stop.

        EntroCut (this repo's own heuristic): stop when the sampled
        distribution's max probability exceeds `entrocut`, i.e. when the model
        is "confident" about the next character. It is a heuristic, not a
        calibrated uncertainty estimate.
        """
        self.eval()
        if entrocut is None:
            entrocut = self.cfg.get('entrocut_threshold', 0.0)
        for _ in range(max_new):
            x = idx if idx.size(1) <= self.cfg['max_seq'] else idx[:, -self.cfg['max_seq']:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / max(temp, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
            probs = F.softmax(logits, dim=-1)
            best_p, best_i = probs.max(dim=-1)
            if entrocut and float(best_p) > entrocut:
                break
            nid = best_i if temp <= 1e-5 else torch.multinomial(probs, 1)
            nid = nid.view(1, 1)
            idx = torch.cat([idx, nid], dim=1)
            if stop_id is not None and int(nid) == int(stop_id):
                break
        return idx

    # ------------------------------------------------------------------ GGUF
    def save_gguf(self, path, stoi=None, itos=None, eos_id=None, aion_engram=None,
                  architecture='chimera_8b_effective', extra_meta=None):
        """Write a GGUF-shaped container for the custom `chimera_*` arch.

        This is byte-compatible with the GGUF header layout (magic/version/
        kv/tensor-info + 32-byte aligned data), but llama.cpp only accepts
        architectures it knows, so these files load with `load_from_gguf`.
        """
        d, v = self.cfg['d_model'], self.cfg['vocab_size']
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        entries = self._tensor_entries()
        total = sum(int(t.numel()) for _, t, _ in entries)
        assert total == self.count_params(), (total, self.count_params())

        meta = {
            'general.architecture': architecture,
            'general.name': 'D-QUILANE',
            'general.alignment': str(GGUF_ALIGN),
            'architecture': architecture,
            'name': 'D-QUILANE',
            'techniques': json.dumps(['LowRank_Attention', 'FFN_Bottleneck', 'ReXMoE',
                                      'Weight_Tying', 'LayerScale', 'RMSNorm', 'AION_adaptive',
                                      'EntroCut', 'RoPE', 'GroupedQuery_Attention',
                                      'Load_Balancing_Loss', 'Router_Z_Loss']),
            'vocab_size': str(v), 'd_model': str(d),
            'n_heads': str(self.cfg['n_heads']),
            'n_kv_heads': str(self.cfg['n_kv_heads']),
            'd_head': str(d // self.cfg['n_heads']),
            'ffn_hidden': str(self.cfg['ffn_hidden']),
            'n_blocks': str(self.cfg['n_blocks']),
            'n_repeat': str(self.cfg['n_repeat']),
            'n_effective_layers': str(self.cfg['n_blocks'] * self.cfg['n_repeat']),
            'n_experts': str(self.cfg['n_experts']),
            'attn_rank': str(self.cfg['attn_rank']),
            'max_seq': str(self.cfg['max_seq']),
            'sliding_window': str(self.cfg.get('sliding_window', 0)),
            'entrocut_threshold': str(self.cfg.get('entrocut_threshold', 0.95)),
            'precision': 'FP32',
            'unique_params': str(self.count_params()),
            'size_mb': str(round(self.count_params() * 4 / 1024 / 1024, 2)),
            'effective_depth': str(self.cfg['n_blocks'] * self.cfg['n_repeat']),
            'effective_depth_note': 'blocks_active * n_repeat (weight reuse); '
                                    'not a parameter count and not an 8B model',
        }
        if stoi:
            meta['tokenizer_stoi'] = json.dumps(stoi, ensure_ascii=False)
        if itos:
            meta['tokenizer_itos'] = json.dumps({str(k): val for k, val in itos.items()},
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
            for name, t, dims in entries:
                _write_str(f, name)
                f.write(struct.pack('<I', len(dims)))
                for dv in dims:
                    f.write(struct.pack('<Q', dv))
                f.write(struct.pack('<I', GGUF_TYPE_F32))
                f.write(struct.pack('<Q', off))
                off += t.numel() * 4
                off += (GGUF_ALIGN - (off % GGUF_ALIGN)) % GGUF_ALIGN
            cur = f.tell()
            pad = (GGUF_ALIGN - (cur % GGUF_ALIGN)) % GGUF_ALIGN
            if pad:
                f.write(b'\x00' * pad)
            for _, t, _ in entries:
                cur = f.tell()
                p = (GGUF_ALIGN - (cur % GGUF_ALIGN)) % GGUF_ALIGN
                if p:
                    f.write(b'\x00' * p)
                f.write(t.flatten().to(torch.float32).numpy().tobytes())
        sz = os.path.getsize(path)
        print(f"[GGUF] {path}  ({sz/1024/1024:.2f} MB, {len(entries)} tensors, "
              f"{self.count_params():,} params)")
        return path

    def _tensor_entries(self):
        """(gguf_name, tensor, dims) for every parameter, from state_dict()."""
        return [(_state_to_gguf_name(n), t.detach().cpu().contiguous(),
                 list(t.shape)) for n, t in self.state_dict().items()]


# ---------------------------------------------------------------- GGUF helpers
def _write_str(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def _read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8')


def _state_to_gguf_name(n):
    if n.startswith('blocks.'):
        return 'block.' + n[len('blocks.'):]
    if n.startswith('expert_pool.experts.'):
        return 'expert.' + n[len('expert_pool.experts.'):]
    if n == 'expert_pool.router.weight':
        return 'router.weight'
    return n


def _gguf_to_state_name(n):
    if n.startswith('block.'):
        return 'blocks.' + n[len('block.'):]
    if n.startswith('expert.'):
        return 'expert_pool.experts.' + n[len('expert.'):]
    if n == 'router.weight':
        return 'expert_pool.router.weight'
    return n


def read_gguf(path):
    """Parse the container written by ChimeraModel.save_gguf. Returns (meta, tensors)."""
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
            if ty != GGUF_TYPE_F32:
                raise ValueError(f"tensor {name!r}: unsupported type {ty}")
            infos.append((name, dims, off))
        align = int(meta.get('general.alignment', GGUF_ALIGN))
        cur = f.tell()
        data_start = cur + ((align - cur % align) % align)
        tensors = {}
        for name, dims, off in infos:
            f.seek(data_start + off)
            n = 1
            for dv in dims:
                n *= dv
            buf = f.read(n * 4)
            if len(buf) != n * 4:
                raise ValueError(f"tensor {name!r}: truncated data")
            tensors[name] = torch.frombuffer(bytearray(buf), dtype=torch.float32) \
                .reshape(list(dims)).clone() if dims else \
                torch.frombuffer(bytearray(buf), dtype=torch.float32).clone()
    return meta, tensors


def load_from_gguf(path):
    """Rebuild a ChimeraModel from a file written by save_gguf."""
    meta, tensors = read_gguf(path)
    needed = ['vocab_size', 'd_model', 'n_blocks', 'n_repeat', 'n_experts',
              'attn_rank', 'ffn_hidden', 'max_seq', 'n_heads', 'n_kv_heads',
              'sliding_window']
    missing = [k for k in needed if k not in meta]
    if missing:
        raise ValueError(f"{path}: missing metadata {missing}")
    cfg = {k: int(meta[k]) for k in needed}
    cfg['entrocut_threshold'] = float(meta.get('entrocut_threshold', 0.95))
    cfg['dropout'] = 0.0
    model = ChimeraModel(cfg)
    sd = {_gguf_to_state_name(n): t for n, t in tensors.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, meta


if __name__ == '__main__':
    cfg = CONFIGS['sandbox'].copy()
    m = ChimeraModel(cfg)
    n = m.count_params()
    print(f"Chimera sandbox: {n:,} params ({n*4/1024/1024:.2f} MB FP32)")
    print(f"Eff. depth: {cfg['n_blocks']} blocks x {cfg['n_repeat']} repeat "
          f"= {cfg['n_blocks']*cfg['n_repeat']} passes")
    x = torch.randint(0, cfg['vocab_size'], (1, 16))
    logits, loss, aux = m(x, targets=x, return_aux=True)
    print(f"forward OK, loss={loss.item():.4f}, aux={aux}")
