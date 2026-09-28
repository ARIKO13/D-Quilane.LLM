/**
 * D'QUILANE — JavaScript Inference Engine
 * Vanilla JS, no library. Reimplement transformer dari nol.
 * Compatible dengan model yang di-export dari dquilane_global.ipynb
 */

class DQuilane {
    constructor(modelData) {
        this.config = modelData.config;
        this.stoi = modelData.stoi;
        this.itos = modelData.itos;
        this.w = {};
        // Convert semua weight list ke Float32Array utk performance
        for (const [key, value] of Object.entries(modelData.weights)) {
            if (Array.isArray(value)) {
                this.w[key] = this._flatten(value);
            } else {
                this.w[key] = value;
            }
        }
        // Simpan shape untuk beberapa matrix
        const d = this.config.d_model;
        const v = this.config.vocab_size;
        const n = this.config.max_seq;
        const f = this.config.d_ff;
        this.shapes = {
            tok_emb_weight: [v, d],
            pos_emb_weight: [n, d],
            norm1_weight: [d],
            attn_qkv_weight: [3 * d, d],
            attn_out_weight: [d, d],
            norm2_weight: [d],
            ffn_gate_weight: [f, d],
            ffn_val_weight: [f, d],
            ffn_down_weight: [d, f],
            norm_f_weight: [d],
        };
    }

    // ========== Utils ==========
    _flatten(arr) {
        // Flatten nested array ke Float32Array
        if (typeof arr[0] === 'number') {
            return new Float32Array(arr);
        }
        const out = [];
        for (const sub of arr) {
            for (const x of sub) out.push(x);
        }
        return new Float32Array(out);
    }

    // Tokenizer
    encode(text) {
        const ids = [];
        for (const c of text) {
            if (c in this.stoi) ids.push(this.stoi[c]);
        }
        return ids;
    }

    decode(ids) {
        let out = '';
        for (const id of ids) {
            const idStr = String(id);
            if (idStr in this.itos) {
                const ch = this.itos[idStr];
                if (ch !== '<pad>') out += ch;
            }
        }
        return out;
    }

    // ========== Math Ops ==========
    // RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    rmsNorm(x, weight, eps = 1e-6) {
        const n = x.length;
        let sumSq = 0;
        for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
        const rms = Math.sqrt(sumSq / n + eps);
        const inv = 1 / rms;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = (x[i] * inv) * weight[i];
        return out;
    }

    // Linear: y = x @ W^T  (weight shape: [out, in])
    // x: Float32Array(length=in), weight: Float32Array(out*in)
    linear(x, weight, inDim, outDim) {
        const out = new Float32Array(outDim);
        for (let o = 0; o < outDim; o++) {
            let sum = 0;
            const base = o * inDim;
            for (let i = 0; i < inDim; i++) {
                sum += x[i] * weight[base + i];
            }
            out[o] = sum;
        }
        return out;
    }

    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu(x) {
        const out = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            out[i] = x[i] / (1 + Math.exp(-x[i]));
        }
        return out;
    }

    // Softmax (with masking option)
    softmax(x, mask = null) {
        const n = x.length;
        // find max for stability
        let maxVal = -Infinity;
        for (let i = 0; i < n; i++) {
            if (mask === null || mask[i]) {
                if (x[i] > maxVal) maxVal = x[i];
            }
        }
        let sum = 0;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            if (mask === null || mask[i]) {
                out[i] = Math.exp(x[i] - maxVal);
                sum += out[i];
            } else {
                out[i] = 0;
            }
        }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    // ========== Forward Pass ==========
    // Input: ids (array of token ids), output: logits[seq_len, vocab_size]
    forward(ids) {
        const cfg = this.config;
        const d = cfg.d_model;
        const nHeads = cfg.n_heads;
        const dHead = d / nHeads;
        const T = ids.length;

        // 1. Embedding: x[T, d] = tok_emb[ids] + pos_emb[0..T]
        let x = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const tokenId = ids[t];
            const tokBase = tokenId * d;
            const posBase = t * d;
            const xBase = t * d;
            for (let i = 0; i < d; i++) {
                x[xBase + i] = this.w.tok_emb_weight[tokBase + i] + this.w.pos_emb_weight[posBase + i];
            }
        }

        // 2. Shared Block (loop n_layers times)
        for (let layer = 0; layer < cfg.n_layers; layer++) {
            x = this.blockForward(x, T, d, nHeads, dHead);
        }

        // 3. Final norm
        x = this.applyRMSNorm(x, this.w.norm_f_weight, T, d);

        // 4. Output head (tied to tok_emb): logits[T, vocab]
        const vocab = cfg.vocab_size;
        const logits = new Float32Array(T * vocab);
        for (let t = 0; t < T; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            for (let v = 0; v < vocab; v++) {
                let sum = 0;
                const wBase = v * d;
                for (let i = 0; i < d; i++) {
                    sum += xRow[i] * this.w.tok_emb_weight[wBase + i];
                }
                logits[t * vocab + v] = sum;
            }
        }
        return logits;
    }

    applyRMSNorm(x, weight, T, d) {
        const out = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            const normed = this.rmsNorm(xRow, weight);
            out.set(normed, t * d);
        }
        return out;
    }

    // One transformer block forward
    blockForward(x, T, d, nHeads, dHead) {
        // ---- Pre-norm 1 + Attention ----
        let xNormed = this.applyRMSNorm(x, this.w.norm1_weight, T, d);

        // QKV: xNormed[T, d] @ qkv_weight^T([3d, d]) => [T, 3d]
        const qkv = new Float32Array(T * 3 * d);
        for (let t = 0; t < T; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            const qkvRow = this.linear(xRow, this.w.attn_qkv_weight, d, 3 * d);
            qkv.set(qkvRow, t * 3 * d);
        }

        // Reshape to [T, 3, nHeads, dHead] -> permute -> q,k,v each [nHeads, T, dHead]
        // Apply attention per head
        const attnOut = new Float32Array(T * d); // [T, nHeads, dHead] flattened then merge

        for (let h = 0; h < nHeads; h++) {
            // Extract Q, K, V for this head
            // qkv layout: [T, 3, nHeads, dHead] — for each token t, the segment is qkv[t*3*d : (t+1)*3*d]
            // Within that segment: [Q(nHeads*dHead), K(nHeads*dHead), V(nHeads*dHead)]
            const qHead = new Float32Array(T * dHead);
            const kHead = new Float32Array(T * dHead);
            const vHead = new Float32Array(T * dHead);
            for (let t = 0; t < T; t++) {
                const base = t * 3 * d;
                // Q for this head starts at base + h * dHead
                for (let i = 0; i < dHead; i++) {
                    qHead[t * dHead + i] = qkv[base + h * dHead + i];
                    kHead[t * dHead + i] = qkv[base + d + h * dHead + i];
                    vHead[t * dHead + i] = qkv[base + 2 * d + h * dHead + i];
                }
            }

            // Attention scores: [T, T] = qHead @ kHead^T / sqrt(dHead)
            const scores = new Float32Array(T * T);
            const scale = 1 / Math.sqrt(dHead);
            for (let i = 0; i < T; i++) {
                for (let j = 0; j < T; j++) {
                    let sum = 0;
                    for (let k = 0; k < dHead; k++) {
                        sum += qHead[i * dHead + k] * kHead[j * dHead + k];
                    }
                    scores[i * T + j] = sum * scale;
                }
            }

            // Causal mask + softmax
            for (let i = 0; i < T; i++) {
                const row = scores.subarray(i * T, (i + 1) * T);
                // Build mask: j <= i (causal)
                const mask = new Uint8Array(T);
                for (let j = 0; j <= i; j++) mask[j] = 1;
                const attn = this.softmaxWithMask(row, mask);
                // Save back to row (overwrites scores, that's fine)
                row.set(attn);
            }

            // attnOut[i, h*dHead : (h+1)*dHead] = sum_j attn[i,j] * vHead[j]
            for (let i = 0; i < T; i++) {
                for (let k = 0; k < dHead; k++) {
                    let sum = 0;
                    for (let j = 0; j < T; j++) {
                        sum += scores[i * T + j] * vHead[j * dHead + k];
                    }
                    attnOut[i * d + h * dHead + k] = sum;
                }
            }
        }

        // Output projection: attnOut[T, d] @ out_weight^T([d, d]) => [T, d]
        let attnFinal = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const xRow = attnOut.subarray(t * d, (t + 1) * d);
            const outRow = this.linear(xRow, this.w.attn_out_weight, d, d);
            attnFinal.set(outRow, t * d);
        }

        // Residual 1
        for (let i = 0; i < T * d; i++) x[i] += attnFinal[i];

        // ---- Pre-norm 2 + FFN (SwiGLU) ----
        xNormed = this.applyRMSNorm(x, this.w.norm2_weight, T, d);
        const ffnDim = this.config.d_ff;

        const ffnOut = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            // gate, val: [d] -> [ffn]
            const gate = this.linear(xRow, this.w.ffn_gate_weight, d, ffnDim);
            const val = this.linear(xRow, this.w.ffn_val_weight, d, ffnDim);
            // silu(gate) * val
            const act = new Float32Array(ffnDim);
            for (let i = 0; i < ffnDim; i++) {
                act[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * val[i];
            }
            // down: [ffn] -> [d]
            const down = this.linear(act, this.w.ffn_down_weight, ffnDim, d);
            ffnOut.set(down, t * d);
        }

        // Residual 2
        for (let i = 0; i < T * d; i++) x[i] += ffnOut[i];

        return x;
    }

    // Softmax with mask (mask: Uint8Array where 1 = keep)
    softmaxWithMask(x, mask) {
        const n = x.length;
        let maxVal = -Infinity;
        for (let i = 0; i < n; i++) {
            if (mask[i] && x[i] > maxVal) maxVal = x[i];
        }
        let sum = 0;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            if (mask[i]) {
                out[i] = Math.exp(x[i] - maxVal);
                sum += out[i];
            } else {
                out[i] = 0;
            }
        }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    // ========== Generation ==========
    async generate(prompt, maxNew = 80, temp = 0.7, topK = 10, onToken = null) {
        let ids = this.encode(prompt);
        if (ids.length === 0) return prompt;
        const vocab = this.config.vocab_size;
        const maxSeq = this.config.max_seq;

        for (let step = 0; step < maxNew; step++) {
            // Crop context to max_seq (ambil dari belakang)
            const ctx = ids.length > maxSeq ? ids.slice(ids.length - maxSeq) : ids;
            const logits = this.forward(ctx);
            const T = ctx.length;
            // Ambil logits token terakhir
            const lastLogits = logits.subarray((T - 1) * vocab, T * vocab);

            // Apply temperature
            const scaled = new Float32Array(vocab);
            for (let i = 0; i < vocab; i++) scaled[i] = lastLogits[i] / Math.max(temp, 1e-5);

            // Top-k: ambil k nilai tertinggi, lainnya jadi -inf
            if (topK > 0 && topK < vocab) {
                const indices = Array.from({ length: vocab }, (_, i) => i);
                indices.sort((a, b) => scaled[b] - scaled[a]);
                const threshold = scaled[indices[topK - 1]];
                for (let i = 0; i < vocab; i++) {
                    if (scaled[i] < threshold) scaled[i] = -Infinity;
                }
            }

            // Softmax
            const probs = this.softmax(scaled);

            // Sample
            const r = Math.random();
            let cum = 0;
            let nextId = 0;
            for (let i = 0; i < vocab; i++) {
                cum += probs[i];
                if (r < cum) { nextId = i; break; }
            }

            ids.push(nextId);
            if (onToken) {
                await onToken(this.itos[String(nextId)] || '?', step + 1);
            }
        }
        return this.decode(ids);
    }
}

// Export untuk di-load di browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DQuilane };
}
if (typeof window !== 'undefined') {
    window.DQuilane = DQuilane;
}
