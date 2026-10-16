"""
GGUF Inference for D'QUILANE Blueprint v2
==========================================
Loads GGUF file (F32, no quantization) and runs inference.

Usage:
    python gguf_infer.py model.gguf "Halo"
    python gguf_infer.py model.gguf "Satu tambah satu?" --interactive
    python gguf_infer.py model.gguf --interactive

Compatible with: CPU (no GPU needed for this small model)
GPU support: NOT included (would need custom CUDA kernels for custom arch)

For your RTX 3050 4GB:
    - This CPU inference works (model is tiny: ~3 MB)
    - GPU would need llama.cpp-compatible arch (not our custom hybrid)
    - This script is the simplest way to run
"""
import sys
import os
import json
import struct
import math
import time
from pathlib import Path

# ============================================================
# GGUF Reader
# ============================================================
class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()
    
    def _read_uint8(self):
        return struct.unpack('<B', self.f.read(1))[0]
    
    def _read_int8(self):
        return struct.unpack('<b', self.f.read(1))[0]
    
    def _read_uint16(self):
        return struct.unpack('<H', self.f.read(2))[0]
    
    def _read_int16(self):
        return struct.unpack('<h', self.f.read(2))[0]
    
    def _read_uint32(self):
        return struct.unpack('<I', self.f.read(4))[0]
    
    def _read_int32(self):
        return struct.unpack('<i', self.f.read(4))[0]
    
    def _read_uint64(self):
        return struct.unpack('<Q', self.f.read(8))[0]
    
    def _read_int64(self):
        return struct.unpack('<q', self.f.read(8))[0]
    
    def _read_float32(self):
        return struct.unpack('<f', self.f.read(4))[0]
    
    def _read_bool(self):
        return self._read_uint8() != 0
    
    def _read_string(self):
        length = self._read_uint64()
        return self.f.read(length).decode('utf-8')
    
    def _read_header(self):
        magic = self._read_uint32()
        if magic != 0x46554747:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
        self.version = self._read_uint32()
        self.tensor_count = self._read_uint64()
        self.metadata_count = self._read_uint64()
    
    def _read_metadata(self):
        self.metadata = {}
        for _ in range(self.metadata_count):
            key = self._read_string()
            vtype = self._read_uint32()
            if vtype == 0:  # UINT8
                self.metadata[key] = self._read_uint8()
            elif vtype == 1:  # INT8
                self.metadata[key] = self._read_int8()
            elif vtype == 2:  # UINT16
                self.metadata[key] = self._read_uint16()
            elif vtype == 3:  # INT16
                self.metadata[key] = self._read_int16()
            elif vtype == 4:  # UINT32
                self.metadata[key] = self._read_uint32()
            elif vtype == 5:  # INT32
                self.metadata[key] = self._read_int32()
            elif vtype == 6:  # UINT64
                self.metadata[key] = self._read_uint64()
            elif vtype == 7:  # INT64
                self.metadata[key] = self._read_int64()
            elif vtype == 8:  # FLOAT32
                self.metadata[key] = self._read_float32()
            elif vtype == 9:  # BOOL
                self.metadata[key] = self._read_bool()
            elif vtype == 10:  # STRING
                self.metadata[key] = self._read_string()
            else:
                raise ValueError(f"Unsupported metadata type: {vtype}")
    
    def _read_tensor_info(self):
        self.tensor_info = []  # [(name, dims, dtype, offset)]
        for _ in range(self.tensor_count):
            name = self._read_string()
            n_dims = self._read_uint32()
            dims = [self._read_uint64() for _ in range(n_dims)]
            dtype = self._read_uint32()
            offset = self._read_uint64()
            self.tensor_info.append((name, dims, dtype, offset))
        
        # Record data section start
        self.data_start = self.f.tell()
        # Align to 32 bytes
        padding = (32 - (self.data_start % 32)) % 32
        if padding > 0:
            self.f.read(padding)
            self.data_start += padding
    
    def read_tensor(self, name):
        """Read tensor data as flat list of floats."""
        info = None
        for n, dims, dtype, offset in self.tensor_info:
            if n == name:
                info = (n, dims, dtype, offset)
                break
        if info is None:
            raise KeyError(f"Tensor not found: {name}")
        
        _, dims, dtype, offset = info
        # Seek to position
        self.f.seek(self.data_start + offset)
        
        # Compute total elements
        total = 1
        for d in dims:
            total *= d
        
        if dtype == 0:  # F32
            import array
            arr = array.array('f')
            arr.fromfile(self.f, total)
            return list(arr), dims
        elif dtype == 1:  # F16
            # Read as uint16, convert to float32
            data = []
            for _ in range(total):
                h = struct.unpack('<H', self.f.read(2))[0]
                # Convert FP16 to FP32
                sign = (h >> 15) & 1
                exp = (h >> 10) & 0x1F
                mant = h & 0x3FF
                if exp == 0:
                    val = (-1)**sign * 2**(-14) * (mant / 1024)
                elif exp == 31:
                    val = float('inf') if mant == 0 else float('nan')
                else:
                    val = (-1)**sign * 2**(exp - 15) * (1 + mant / 1024)
                data.append(val)
            return data, dims
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    def close(self):
        self.f.close()


# ============================================================
# Model classes (vanilla Python, no PyTorch needed)
# ============================================================
def flatten(nested):
    """Flatten nested list."""
    if not isinstance(nested, list):
        return [nested]
    out = []
    for x in nested:
        if isinstance(x, list):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out


class DQuilaneGGUF:
    """D'QUILANE model loaded from GGUF file."""
    
    def __init__(self, gguf_path):
        print(f"Loading GGUF: {gguf_path}")
        start = time.time()
        
        self.reader = GGUFReader(gguf_path)
        self.metadata = self.reader.metadata
        
        # Parse config
        self.config = {
            'vocab_size': self.metadata.get('vocab_size', 0),
            'd_model': self.metadata.get('d_model', 96),
            'n_heads': self.metadata.get('n_heads', 6),
            'd_ff': self.metadata.get('d_ff', 384),
            'n_layers': self.metadata.get('n_layers', 12),
            'max_seq': self.metadata.get('max_seq', 80),
            'n_conv_blocks': self.metadata.get('n_conv_blocks', 0),
            'n_attn_blocks': self.metadata.get('n_attn_blocks', 0),
            'n_experts': self.metadata.get('n_experts', 0),
            'conv_kernel': self.metadata.get('conv_kernel', 4),
            'eos_id': self.metadata.get('eos_id', self.metadata.get('vocab_size', 0) - 1),
            'pad_id': self.metadata.get('pad_id', 0),
        }
        
        # Parse tokenizer
        self.stoi = json.loads(self.metadata.get('tokenizer_stoi', '{}'))
        self.itos = {int(k): v for k, v in json.loads(self.metadata.get('tokenizer_itos', '{}')).items()}
        
        # Parse AION engram
        self.aion_engram = json.loads(self.metadata.get('aion_engram', '{}'))
        
        # Architecture
        self.architecture = self.metadata.get('architecture', 'unknown')
        self.techniques = json.loads(self.metadata.get('techniques', '[]'))
        
        # Load all tensors into memory
        self.weights = {}
        for name, dims, dtype, offset in self.reader.tensor_info:
            data, _ = self.reader.read_tensor(name)
            self.weights[name] = data
        
        self.reader.close()
        
        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.2f}s")
        print(f"  Architecture: {self.architecture}")
        print(f"  Techniques: {', '.join(self.techniques)}")
        print(f"  Vocab: {self.config['vocab_size']}")
        print(f"  d_model: {self.config['d_model']}")
        print(f"  n_layers: {self.config['n_layers']}")
        print(f"  AION engram: {len(self.aion_engram)} Q&A pairs")
        
        # Parse tensor data into structured form
        self._parse_tensors()
    
    def _parse_tensors(self):
        """Convert flat tensor lists to structured form for inference."""
        d = self.config['d_model']
        n_experts = self.config['n_experts']
        n_conv = self.config['n_conv_blocks']
        n_attn = self.config['n_attn_blocks']
        n_heads = self.config['n_heads']
        d_head = d // n_heads
        n_kv = max(1, n_heads // 4)
        d_ff = self.config['d_ff']
        kernel = self.config['conv_kernel']
        
        self.tok_emb = self.weights.get('tok_emb.weight', [])
        self.pos_emb = self.weights.get('pos_emb.weight', [])
        self.norm_f = self.weights.get('norm_f.weight', [])
        
        if n_experts > 0:
            # Blueprint v2 architecture
            self.experts = []
            for i in range(n_experts):
                self.experts.append({
                    'gate': self.weights.get(f'experts.{i}.gate.weight', []),
                    'val': self.weights.get(f'experts.{i}.val.weight', []),
                    'down': self.weights.get(f'experts.{i}.down.weight', []),
                })
            self.router = self.weights.get('router.gate.weight', [])
            
            self.conv_blocks = []
            for i in range(n_conv):
                self.conv_blocks.append({
                    'norm1': self.weights.get(f'conv_blocks.{i}.norm1.weight', []),
                    'conv_weight': self.weights.get(f'conv_blocks.{i}.conv.weight', []),
                    'conv_bias': self.weights.get(f'conv_blocks.{i}.conv.bias', []),
                    'proj': self.weights.get(f'conv_blocks.{i}.proj.weight', []),
                    'norm2': self.weights.get(f'conv_blocks.{i}.norm2.weight', []),
                })
            
            self.attn_blocks = []
            for i in range(n_attn):
                self.attn_blocks.append({
                    'norm1': self.weights.get(f'attn_blocks.{i}.norm1.weight', []),
                    'q': self.weights.get(f'attn_blocks.{i}.q.weight', []),
                    'kv': self.weights.get(f'attn_blocks.{i}.kv.weight', []),
                    'out': self.weights.get(f'attn_blocks.{i}.out.weight', []),
                    'norm2': self.weights.get(f'attn_blocks.{i}.norm2.weight', []),
                })
    
    # Tokenizer
    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, ids):
        return ''.join(self.itos.get(int(i), '?') for i in ids if i != self.config['pad_id'])
    
    # AION bypass
    def aion_lookup(self, prompt, threshold=0.95):
        key = prompt.lower().strip()
        if key in self.aion_engram:
            return self.aion_engram[key], 1.0
        for k, v in self.aion_engram.items():
            if key == k or (len(key) > 5 and key in k):
                return v, 0.9
        return None, 0.0
    
    # Math ops
    def rmsNorm(self, x, weight, eps=1e-6):
        n = len(x)
        sumSq = sum(v*v for v in x)
        inv = 1.0 / math.sqrt(sumSq / n + eps)
        return [(x[i] * inv) * weight[i] for i in range(n)]
    
    def linear(self, x, W, inDim, outDim):
        out = [0.0] * outDim
        for o in range(outDim):
            base = o * inDim
            s = 0.0
            for i in range(inDim):
                s += x[i] * W[base + i]
            out[o] = s
        return out
    
    def silu(self, x):
        return [v / (1 + math.exp(-v)) for v in x]
    
    def softmax(self, x):
        n = len(x)
        maxVal = max(x)
        exps = [math.exp(v - maxVal) for v in x]
        s = sum(exps)
        return [e / s for e in exps]
    
    # Forward pass (Blueprint v2 architecture)
    def forward(self, ids):
        cfg = self.config
        d = cfg['d_model']
        n_heads = cfg['n_heads']
        d_head = d // n_heads
        n_kv = max(1, n_heads // 4)
        T = len(ids)
        
        # Embedding
        x = []
        for t in range(T):
            tokBase = ids[t] * d
            posBase = t * d
            x.append([self.tok_emb[tokBase + i] + self.pos_emb[posBase + i] for i in range(d)])
        
        # Conv blocks
        for blk in self.conv_blocks:
            x = self._conv_block_forward(x, T, d, blk, cfg['conv_kernel'])
        
        # Attention blocks
        for blk in self.attn_blocks:
            x = self._attn_block_forward(x, T, d, n_heads, d_head, n_kv, blk)
        
        # Final norm
        x_normed = [self.rmsNorm(x[t], self.norm_f) for t in range(T)]
        
        # Output head (tied to tok_emb)
        vocab = cfg['vocab_size']
        logits = []
        for t in range(T):
            row = []
            for v in range(vocab):
                wBase = v * d
                s = sum(x_normed[t][i] * self.tok_emb[wBase + i] for i in range(d))
                row.append(s)
            logits.append(row)
        
        return logits
    
    def _conv_block_forward(self, x, T, d, blk, kernel):
        # Pre-norm 1
        x_normed = [self.rmsNorm(x[t], blk['norm1']) for t in range(T)]
        
        # Gated short conv (causal)
        conv_w = blk['conv_weight']
        conv_b = blk.get('conv_bias') or [0.0] * (2 * d)
        out_2d = []
        for t in range(T):
            row = [0.0] * (2 * d)
            for c in range(2 * d):
                # Depthwise conv: 1 input channel, kernel
                s = conv_b[c] if conv_b else 0.0
                for k in range(kernel):
                    src_t = t - k
                    if src_t >= 0:
                        # For depthwise: input channel = c // 2 (since output 2*d, groups=d, each input gets 2 outputs)
                        in_ch = c // 2
                        s += x_normed[src_t][in_ch] * conv_w[c * kernel + k]
                row[c] = s
            out_2d.append(row)
        
        # Gating
        gated = []
        for t in range(T):
            gate = out_2d[t][:d]
            val = out_2d[t][d:]
            silu_gate = [g / (1 + math.exp(-g)) for g in gate]
            gated.append([silu_gate[i] * val[i] for i in range(d)])
        
        # Proj
        proj_out = [self.linear(gated[t], blk['proj'], d, d) for t in range(T)]
        
        # Residual
        x = [[x[t][i] + proj_out[t][i] for i in range(d)] for t in range(T)]
        
        # Pre-norm 2 + MoE FFN
        x_normed2 = [self.rmsNorm(x[t], blk['norm2']) for t in range(T)]
        
        # ReXMoE: router + experts
        moe_out = []
        for t in range(T):
            scores = self.linear(x_normed2[t], self.router, d, len(self.experts))
            probs = self.softmax(scores)
            expert_outs = []
            for i, exp in enumerate(self.experts):
                gate_out = self.linear(x_normed2[t], exp['gate'], d, self.config['d_ff'])
                val_out = self.linear(x_normed2[t], exp['val'], d, self.config['d_ff'])
                silu_gate = [g / (1 + math.exp(-g)) for g in gate_out]
                act = [silu_gate[j] * val_out[j] for j in range(self.config['d_ff'])]
                down_out = self.linear(act, exp['down'], self.config['d_ff'], d)
                expert_outs.append(down_out)
            # Weighted sum
            combined = [sum(probs[i] * expert_outs[i][j] for i in range(len(self.experts))) for j in range(d)]
            moe_out.append(combined)
        
        # Residual
        x = [[x[t][i] + moe_out[t][i] for i in range(d)] for t in range(T)]
        
        return x
    
    def _attn_block_forward(self, x, T, d, n_heads, d_head, n_kv, blk):
        # Pre-norm 1
        x_normed = [self.rmsNorm(x[t], blk['norm1']) for t in range(T)]
        
        # Q, K, V
        q_all = [self.linear(x_normed[t], blk['q'], d, d) for t in range(T)]
        kv_all = [self.linear(x_normed[t], blk['kv'], d, 2 * n_kv * d_head) for t in range(T)]
        
        # Reshape Q to [T, n_heads, d_head]
        q_heads = []
        for t in range(T):
            heads = []
            for h in range(n_heads):
                heads.append(q_all[t][h * d_head : (h + 1) * d_head])
            q_heads.append(heads)
        
        # Reshape K, V
        k_heads = []
        v_heads = []
        for h in range(n_kv):
            ks = []
            vs = []
            for t in range(T):
                k_t = kv_all[t][:n_kv * d_head]
                v_t = kv_all[t][n_kv * d_head:]
                ks.append(k_t[h * d_head : (h + 1) * d_head])
                vs.append(v_t[h * d_head : (h + 1) * d_head])
            k_heads.append(ks)
            v_heads.append(vs)
        
        # Repeat K, V for grouped query
        rep = n_heads // n_kv
        k_full = []
        v_full = []
        for h in range(n_heads):
            k_full.append(k_heads[h // rep])
            v_full.append(v_heads[h // rep])
        
        # Attention
        attn_out = []
        for t in range(T):
            row = [0.0] * d
            for h in range(n_heads):
                # Scores for this head
                scores = []
                for j in range(t + 1):
                    s = sum(q_heads[t][h][k] * k_full[h][j][k] for k in range(d_head))
                    scores.append(s / math.sqrt(d_head))
                # Softmax
                probs = self.softmax(scores)
                # Apply to V
                for k_idx in range(d_head):
                    s = sum(probs[j] * v_full[h][j][k_idx] for j in range(t + 1))
                    row[h * d_head + k_idx] = s
            attn_out.append(row)
        
        # Output projection
        proj_out = [self.linear(attn_out[t], blk['out'], d, d) for t in range(T)]
        
        # Residual
        x = [[x[t][i] + proj_out[t][i] for i in range(d)] for t in range(T)]
        
        # Pre-norm 2 + MoE FFN
        x_normed2 = [self.rmsNorm(x[t], blk['norm2']) for t in range(T)]
        
        moe_out = []
        for t in range(T):
            scores = self.linear(x_normed2[t], self.router, d, len(self.experts))
            probs = self.softmax(scores)
            expert_outs = []
            for i, exp in enumerate(self.experts):
                gate_out = self.linear(x_normed2[t], exp['gate'], d, self.config['d_ff'])
                val_out = self.linear(x_normed2[t], exp['val'], d, self.config['d_ff'])
                silu_gate = [g / (1 + math.exp(-g)) for g in gate_out]
                act = [silu_gate[j] * val_out[j] for j in range(self.config['d_ff'])]
                down_out = self.linear(act, exp['down'], self.config['d_ff'], d)
                expert_outs.append(down_out)
            combined = [sum(probs[i] * expert_outs[i][j] for i in range(len(self.experts))) for j in range(d)]
            moe_out.append(combined)
        
        x = [[x[t][i] + moe_out[t][i] for i in range(d)] for t in range(T)]
        
        return x
    
    # Generation
    def generate(self, prompt, max_new=50, temp=0.4, top_k=10, top_p=0.9, rep_penalty=1.2):
        # AION bypass first
        response, confidence = self.aion_lookup(prompt)
        if confidence > 0.95:
            return response, 'AION bypass (O(1) lookup, zero FLOPs)'
        
        # Fall through to transformer
        ids = self.encode(prompt)
        if not ids:
            return '', 'Transformer (empty prompt)'
        
        eos_id = self.config['eos_id']
        max_seq = self.config['max_seq']
        vocab = self.config['vocab_size']
        
        for step in range(max_new):
            ctx = ids[-max_seq:] if len(ids) > max_seq else ids
            logits = self.forward(ctx)
            last_logits = logits[-1]
            
            # Temperature
            scaled = [l / max(temp, 1e-5) for l in last_logits]
            
            # Repetition penalty
            if rep_penalty != 1.0:
                recent = ids[-20:]
                seen = set(recent)
                for t in seen:
                    if 0 <= t < vocab:
                        scaled[t] /= rep_penalty
            
            # Top-k
            if top_k > 0 and top_k < vocab:
                sorted_idx = sorted(range(vocab), key=lambda i: -scaled[i])
                threshold = scaled[sorted_idx[top_k - 1]]
                scaled = [s if s >= threshold else -float('inf') for s in scaled]
            
            # Top-p
            if top_p < 1.0:
                sorted_idx = sorted(range(vocab), key=lambda i: -scaled[i])
                maxVal = max(scaled)
                exps = [math.exp(s - maxVal) if s != -float('inf') else 0 for s in scaled]
                sumExp = sum(exps)
                cum = 0
                keep = [False] * vocab
                for idx in sorted_idx:
                    if exps[idx] == 0:
                        break
                    cum += exps[idx] / sumExp
                    keep[idx] = True
                    if cum >= top_p:
                        break
                scaled = [s if keep[i] else -float('inf') for i, s in enumerate(scaled)]
            
            # Softmax + sample
            probs = self.softmax(scaled)
            import random
            r = random.random()
            cum = 0
            next_id = 0
            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    next_id = i
                    break
            
            ids.append(next_id)
            if next_id == eos_id:
                break
        
        return self.decode(ids), 'Transformer (full forward pass)'


# ============================================================
# Main CLI
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python gguf_infer.py <model.gguf> [prompt] [--interactive]")
        print("       python gguf_infer.py model.gguf \"Halo\"")
        print("       python gguf_infer.py model.gguf --interactive")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    if not os.path.exists(gguf_path):
        print(f"Error: File not found: {gguf_path}")
        sys.exit(1)
    
    model = DQuilaneGGUF(gguf_path)
    
    if len(sys.argv) >= 3 and sys.argv[2] == '--interactive':
        # Interactive mode
        print("\n=== Interactive Mode ===")
        print("Ketik prompt lalu Enter. 'quit' untuk keluar.\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Bye!")
                break
            
            start = time.time()
            response, mode = model.generate(prompt, max_new=80)
            elapsed = time.time() - start
            print(f"\n[{mode}, {elapsed:.2f}s]")
            print(f"D'QUILANE: {response}\n")
    else:
        # Single prompt mode
        prompt = sys.argv[2] if len(sys.argv) >= 3 else "Halo"
        start = time.time()
        response, mode = model.generate(prompt, max_new=80)
        elapsed = time.time() - start
        print(f"\nPrompt: {prompt}")
        print(f"Mode: {mode}")
        print(f"Time: {elapsed:.2f}s")
        print(f"D'QUILANE: {response}")


if __name__ == '__main__':
    main()
