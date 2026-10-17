"""
D'QUILANE Chimera-8B-Effective — minimal architecture for training.
Sandbox-scale: d_model=96, 4 blocks × 3 repeat = 12 eff. layers, 4 experts.
~218K params, 0.84 MB FP32.

Implements: LowRank GQA, FFN Bottleneck (SwiGLU), ReXMoE (shared experts),
Weight Tying, LayerScale (init=1e-1), RMSNorm, RoPE, Load Balancing Loss
(Switch Transformer), Router Z-Loss (ST-MoE).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import struct


CONFIGS = {
    'sandbox': {
        'd_model': 96, 'n_blocks': 4, 'n_repeat': 3,
        'n_experts': 4, 'attn_rank': 32, 'ffn_hidden': 48,
        'vocab_size': 144, 'max_seq': 80,
        'n_heads': 4, 'n_kv_heads': 1, 'sliding_window': 32,
        'entrocut_threshold': 0.95,
        'dropout': 0.1, 'weight_decay': 0.1,
        'load_balancing_weight': 0.01, 'router_z_loss_weight': 0.001,
    },
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
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', torch.cat([freqs, freqs], dim=-1))
        self.register_buffer('sin', torch.cat([freqs, freqs], dim=-1))
    def forward(self, q, k, T):
        return apply_rotary(q, self.cos[:T], self.sin[:T]), apply_rotary(k, self.cos[:T], self.sin[:T])


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
    def __init__(self, d, hidden, n_experts):
        super().__init__()
        self.n = n_experts
        self.experts = nn.ModuleList([FFNBottleneck(d, hidden) for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts, bias=False)
    def forward(self, x, return_aux=False):
        B, T, C = x.shape
        router_logits = self.router(x)
        probs = F.softmax(router_logits, dim=-1)
        top2_val, top2_idx = probs.topk(2, dim=-1)
        top2_val = top2_val / (top2_val.sum(-1, keepdim=True) + 1e-10)
        exp_outs = torch.stack([self.experts[e](x) for e in range(self.n)], dim=-2)
        idx = top2_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        sel = torch.gather(exp_outs, dim=-2, index=idx)
        out = (sel * top2_val.unsqueeze(-1)).sum(dim=-2)
        if not return_aux:
            return out
        frac = probs.mean(dim=(0, 1))
        top1_mask = (top2_idx[..., 0:1] == torch.arange(self.n, device=x.device).view(1, 1, -1))
        top1_frac = top1_mask.float().mean(dim=(0, 1))
        lb = self.n * (frac * top1_frac).sum()
        rz = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return out, {'load_balancing_loss': lb, 'router_z_loss': rz}


class SharedBlock(nn.Module):
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
        self.cfg = cfg
        d = cfg['d_model']
        v = cfg['vocab_size']
        self.tok_emb = nn.Embedding(v, d)
        self.blocks = nn.ModuleList([
            SharedBlock(cfg, use_swa=(i % 2 == 1)) for i in range(cfg['n_blocks'])
        ])
        self.n_repeat = cfg['n_repeat']
        self.expert_pool = ExpertPool(d, cfg['ffn_hidden'], cfg['n_experts'])
        self.norm_f = RMSNorm(d)
        self.apply(self._init)
        self.aion = {}
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0, std=0.02)
    def forward(self, idx, targets=None, return_aux=False):
        B, T = idx.shape
        x = self.tok_emb(idx)
        dp = self.cfg.get('dropout', 0.0)
        aux_list = []
        for _ in range(self.n_repeat):
            for blk in self.blocks:
                if return_aux:
                    x, aux = blk(x, self.expert_pool, True, dp)
                    aux_list.append(aux)
                else:
                    x = blk(x, self.expert_pool, False, dp)
        x = self.norm_f(x)
        logits = F.linear(x, self.tok_emb.weight)
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, self.cfg['vocab_size']),
                                  targets.view(-1), ignore_index=-100)
            if return_aux and aux_list:
                lb = sum(a['load_balancing_loss'] for a in aux_list) / len(aux_list)
                rz = sum(a['router_z_loss'] for a in aux_list) / len(aux_list)
                lbw = self.cfg.get('load_balancing_weight', 0.01)
                rzw = self.cfg.get('router_z_loss_weight', 0.001)
                loss = ce + lbw * lb + rzw * rz
            else:
                loss = ce
        if return_aux:
            return logits, loss, {
                'ce_loss': ce.item() if targets is not None else 0.0,
                'load_balancing_loss': lb.item() if (targets is not None and aux_list) else 0.0,
                'router_z_loss': rz.item() if (targets is not None and aux_list) else 0.0,
            }
        return logits, loss
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    def effective_params(self):
        return self.count_params() * self.n_repeat
    @torch.no_grad()
    def generate(self, idx, max_new=50, temp=0.5, top_k=5, stop_id=None):
        self.eval()
        for _ in range(max_new):
            x = idx if idx.size(1) <= self.cfg['max_seq'] else idx[:, -self.cfg['max_seq']:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / max(temp, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nid = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nid], dim=1)
            if stop_id is not None and nid.item() == stop_id:
                break
        return idx
    def save_gguf(self, path, stoi=None, itos=None, eos_id=None, aion_engram=None,
                   architecture='chimera_8b_effective'):
        GGUF_MAGIC = 0x46554747
        ALIGN = 32
        d, v = self.cfg['d_model'], self.cfg['vocab_size']
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        T = []
        T.append(('tok_emb.weight', self.tok_emb.weight.detach().cpu(), [v, d]))
        for i, blk in enumerate(self.blocks):
            p = f'block.{i}.'
            T.append((p+'norm1.weight', blk.norm1.weight.detach().cpu(), [d]))
            a = blk.attn
            T.append((p+'attn.q_down.weight', a.q_down.weight.detach().cpu(), [a.rank, d]))
            T.append((p+'attn.q_up.weight', a.q_up.weight.detach().cpu(), [d, a.rank]))
            T.append((p+'attn.k_down.weight', a.k_down.weight.detach().cpu(), [a.rank, d]))
            T.append((p+'attn.k_up.weight', a.k_up.weight.detach().cpu(), [a.kv_dim, a.rank]))
            T.append((p+'attn.v_down.weight', a.v_down.weight.detach().cpu(), [a.rank, d]))
            T.append((p+'attn.v_up.weight', a.v_up.weight.detach().cpu(), [a.kv_dim, a.rank]))
            T.append((p+'attn.o.weight', a.o.weight.detach().cpu(), [d, d]))
            T.append((p+'norm2.weight', blk.norm2.weight.detach().cpu(), [d]))
            T.append((p+'ffn.gate.weight', blk.ffn.gate.weight.detach().cpu(), [self.cfg['ffn_hidden'], d]))
            T.append((p+'ffn.val.weight', blk.ffn.val.weight.detach().cpu(), [self.cfg['ffn_hidden'], d]))
            T.append((p+'ffn.down.weight', blk.ffn.down.weight.detach().cpu(), [d, self.cfg['ffn_hidden']]))
            T.append((p+'ls1.scale', blk.ls1.scale.detach().cpu(), [d]))
            T.append((p+'ls2.scale', blk.ls2.scale.detach().cpu(), [d]))
        for i, e in enumerate(self.expert_pool.experts):
            T.append((f'expert.{i}.gate.weight', e.gate.weight.detach().cpu(), [self.cfg['ffn_hidden'], d]))
            T.append((f'expert.{i}.val.weight', e.val.weight.detach().cpu(), [self.cfg['ffn_hidden'], d]))
            T.append((f'expert.{i}.down.weight', e.down.weight.detach().cpu(), [d, self.cfg['ffn_hidden']]))
        T.append(('router.weight', self.expert_pool.router.weight.detach().cpu(), [self.cfg['n_experts'], d]))
        T.append(('norm_f.weight', self.norm_f.weight.detach().cpu(), [d]))
        meta = {
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
            'effective_params': str(self.effective_params()),
            'size_mb': str(round(self.count_params() * 4 / 1024 / 1024, 2)),
        }
        if stoi: meta['tokenizer_stoi'] = json.dumps(stoi, ensure_ascii=False)
        if itos: meta['tokenizer_itos'] = json.dumps({str(k): val for k, val in itos.items()}, ensure_ascii=False)
        if eos_id is not None: meta['eos_id'] = str(eos_id)
        if aion_engram: meta['aion_engram'] = json.dumps(aion_engram, ensure_ascii=False)
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', 1))
            f.write(struct.pack('<Q', len(T)))
            f.write(struct.pack('<Q', len(meta)))
            for k, val in meta.items():
                kb = k.encode('utf-8')
                f.write(struct.pack('<Q', len(kb))); f.write(kb)
                f.write(struct.pack('<I', 10))
                vb = str(val).encode('utf-8')
                f.write(struct.pack('<Q', len(vb))); f.write(vb)
            off = 0
            for n, t, dims in T:
                nb = n.encode('utf-8')
                f.write(struct.pack('<Q', len(nb))); f.write(nb)
                f.write(struct.pack('<I', len(dims)))
                for dv in dims: f.write(struct.pack('<Q', dv))
                f.write(struct.pack('<I', 0))
                f.write(struct.pack('<Q', off))
                off += len(t.flatten()) * 4
                off += (ALIGN - (off % ALIGN)) % ALIGN
            cur = f.tell()
            pad = (ALIGN - (cur % ALIGN)) % ALIGN
            if pad: f.write(b'\x00' * pad)
            for n, t, dims in T:
                cur = f.tell()
                p = (ALIGN - (cur % ALIGN)) % ALIGN
                if p: f.write(b'\x00' * p)
                arr = t.flatten().to(torch.float32).numpy()
                f.write(arr.tobytes())
        sz = os.path.getsize(path)
        print(f"[GGUF] {path}  ({sz/1024/1024:.2f} MB, {len(T)} tensors, "
              f"{self.count_params():,} params)")
        return path


if __name__ == '__main__':
    import sys
    cfg = CONFIGS['sandbox'].copy()
    m = ChimeraModel(cfg)
    n = m.count_params()
    print(f"Chimera sandbox: {n:,} params ({n*4/1024/1024:.2f} MB FP32)")
    print(f"Eff. layers: {cfg['n_blocks']*cfg['n_repeat']}")
    x = torch.randint(0, cfg['vocab_size'], (1, 16))
    logits, loss, aux = m(x, targets=x, return_aux=True)
    print(f"forward OK, loss={loss.item():.4f}, aux={aux}")
