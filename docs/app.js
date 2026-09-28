/**
 * D'QUILANE v0.5 — JavaScript Inference Engine
 * Vanilla JS, no library. Reimplement transformer dari nol.
 *
 * Fitur baru v0.5:
 * - Top-p (nucleus) sampling
 * - Repetition penalty
 * - In-context learning (custom context)
 * - Performance optimizations (Float32Array for all weights)
 */

class DQuilane {
    constructor(modelData) {
        this.config = modelData.config;
        this.stoi = modelData.stoi;
        this.itos = modelData.itos;
        this.w = {};
        // Convert semua weight list ke Float32Array utk performance
        for (const [key, value] of Object.entries(modelData.weights)) {
            this.w[key] = this._flatten(value);
        }
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

    _flatten(arr) {
        if (typeof arr[0] === 'number') return new Float32Array(arr);
        const out = [];
        for (const sub of arr) for (const x of sub) out.push(x);
        return new Float32Array(out);
    }

    // Tokenizer
    encode(text) {
        const ids = [];
        for (const c of text) if (c in this.stoi) ids.push(this.stoi[c]);
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

    // Math ops
    rmsNorm(x, weight, eps = 1e-6) {
        const n = x.length;
        let sumSq = 0;
        for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
        const inv = 1 / Math.sqrt(sumSq / n + eps);
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = (x[i] * inv) * weight[i];
        return out;
    }

    linear(x, weight, inDim, outDim) {
        const out = new Float32Array(outDim);
        for (let o = 0; o < outDim; o++) {
            let sum = 0;
            const base = o * inDim;
            for (let i = 0; i < inDim; i++) sum += x[i] * weight[base + i];
            out[o] = sum;
        }
        return out;
    }

    silu(x) {
        const out = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) out[i] = x[i] / (1 + Math.exp(-x[i]));
        return out;
    }

    softmax(x, mask = null) {
        const n = x.length;
        let maxVal = -Infinity;
        for (let i = 0; i < n; i++) {
            if (mask === null || mask[i]) if (x[i] > maxVal) maxVal = x[i];
        }
        let sum = 0;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            if (mask === null || mask[i]) { out[i] = Math.exp(x[i] - maxVal); sum += out[i]; }
            else out[i] = 0;
        }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    softmaxWithMask(x, mask) {
        const n = x.length;
        let maxVal = -Infinity;
        for (let i = 0; i < n; i++) {
            if (mask[i] && x[i] > maxVal) maxVal = x[i];
        }
        let sum = 0;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            if (mask[i]) { out[i] = Math.exp(x[i] - maxVal); sum += out[i]; }
            else out[i] = 0;
        }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    // Forward pass: returns logits[T, vocab]
    forward(ids) {
        const cfg = this.config;
        const d = cfg.d_model;
        const nHeads = cfg.n_heads;
        const dHead = d / nHeads;
        const T = ids.length;

        // 1. Embedding
        let x = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const tokBase = ids[t] * d;
            const posBase = t * d;
            const xBase = t * d;
            for (let i = 0; i < d; i++) {
                x[xBase + i] = this.w.tok_emb_weight[tokBase + i] + this.w.pos_emb_weight[posBase + i];
            }
        }

        // 2. Shared block (n_layers times)
        for (let layer = 0; layer < cfg.n_layers; layer++) {
            x = this.blockForward(x, T, d, nHeads, dHead);
        }

        // 3. Final norm
        x = this.applyRMSNorm(x, this.w.norm_f_weight, T, d);

        // 4. Output head (tied to tok_emb)
        const vocab = cfg.vocab_size;
        const logits = new Float32Array(T * vocab);
        for (let t = 0; t < T; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            for (let v = 0; v < vocab; v++) {
                let sum = 0;
                const wBase = v * d;
                for (let i = 0; i < d; i++) sum += xRow[i] * this.w.tok_emb_weight[wBase + i];
                logits[t * vocab + v] = sum;
            }
        }
        return logits;
    }

    applyRMSNorm(x, weight, T, d) {
        const out = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            out.set(this.rmsNorm(xRow, weight), t * d);
        }
        return out;
    }

    blockForward(x, T, d, nHeads, dHead) {
        // Pre-norm 1 + Attention
        let xNormed = this.applyRMSNorm(x, this.w.norm1_weight, T, d);

        const qkv = new Float32Array(T * 3 * d);
        for (let t = 0; t < T; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            qkv.set(this.linear(xRow, this.w.attn_qkv_weight, d, 3 * d), t * 3 * d);
        }

        const attnOut = new Float32Array(T * d);

        for (let h = 0; h < nHeads; h++) {
            const qHead = new Float32Array(T * dHead);
            const kHead = new Float32Array(T * dHead);
            const vHead = new Float32Array(T * dHead);
            for (let t = 0; t < T; t++) {
                const base = t * 3 * d;
                for (let i = 0; i < dHead; i++) {
                    qHead[t * dHead + i] = qkv[base + h * dHead + i];
                    kHead[t * dHead + i] = qkv[base + d + h * dHead + i];
                    vHead[t * dHead + i] = qkv[base + 2 * d + h * dHead + i];
                }
            }

            const scores = new Float32Array(T * T);
            const scale = 1 / Math.sqrt(dHead);
            for (let i = 0; i < T; i++) {
                for (let j = 0; j < T; j++) {
                    let sum = 0;
                    for (let k = 0; k < dHead; k++) sum += qHead[i * dHead + k] * kHead[j * dHead + k];
                    scores[i * T + j] = sum * scale;
                }
            }

            for (let i = 0; i < T; i++) {
                const row = scores.subarray(i * T, (i + 1) * T);
                const mask = new Uint8Array(T);
                for (let j = 0; j <= i; j++) mask[j] = 1;
                row.set(this.softmaxWithMask(row, mask));
            }

            for (let i = 0; i < T; i++) {
                for (let k = 0; k < dHead; k++) {
                    let sum = 0;
                    for (let j = 0; j < T; j++) sum += scores[i * T + j] * vHead[j * dHead + k];
                    attnOut[i * d + h * dHead + k] = sum;
                }
            }
        }

        let attnFinal = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const xRow = attnOut.subarray(t * d, (t + 1) * d);
            attnFinal.set(this.linear(xRow, this.w.attn_out_weight, d, d), t * d);
        }

        for (let i = 0; i < T * d; i++) x[i] += attnFinal[i];

        // Pre-norm 2 + FFN (SwiGLU)
        xNormed = this.applyRMSNorm(x, this.w.norm2_weight, T, d);
        const ffnDim = this.config.d_ff;

        const ffnOut = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            const gate = this.linear(xRow, this.w.ffn_gate_weight, d, ffnDim);
            const val = this.linear(xRow, this.w.ffn_val_weight, d, ffnDim);
            const act = new Float32Array(ffnDim);
            for (let i = 0; i < ffnDim; i++) act[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * val[i];
            ffnOut.set(this.linear(act, this.w.ffn_down_weight, ffnDim, d), t * d);
        }
        for (let i = 0; i < T * d; i++) x[i] += ffnOut[i];

        return x;
    }

    // ========== Generation dengan top-p + repetition penalty ==========
    async generate(prompt, maxNew = 200, temp = 0.8, topK = 40, topP = 0.95, repPenalty = 1.1, onToken = null) {
        let ids = this.encode(prompt);
        if (ids.length === 0) return prompt;
        const vocab = this.config.vocab_size;
        const maxSeq = this.config.max_seq;

        for (let step = 0; step < maxNew; step++) {
            const ctx = ids.length > maxSeq ? ids.slice(ids.length - maxSeq) : ids;
            const logits = this.forward(ctx);
            const T = ctx.length;
            const lastLogits = new Float32Array(vocab);
            for (let i = 0; i < vocab; i++) lastLogits[i] = logits[(T - 1) * vocab + i];

            // Temperature
            const scaled = new Float32Array(vocab);
            for (let i = 0; i < vocab; i++) scaled[i] = lastLogits[i] / Math.max(temp, 1e-5);

            // Repetition penalty
            if (repPenalty > 1.0) {
                const recent = ids.slice(-Math.min(20, ids.length));
                const seen = new Set(recent);
                for (const t of seen) {
                    if (t >= 0 && t < vocab) scaled[t] /= repPenalty;
                }
            }

            // Top-k
            if (topK > 0 && topK < vocab) {
                const indices = Array.from({ length: vocab }, (_, i) => i);
                indices.sort((a, b) => scaled[b] - scaled[a]);
                const threshold = scaled[indices[topK - 1]];
                for (let i = 0; i < vocab; i++) {
                    if (scaled[i] < threshold) scaled[i] = -Infinity;
                }
            }

            // Top-p (nucleus) sampling
            if (topP < 1.0) {
                // Urutkan descending
                const idx = Array.from({ length: vocab }, (_, i) => i);
                idx.sort((a, b) => scaled[b] - scaled[a]);
                // Hitung softmax dulu
                let maxVal = -Infinity;
                for (let i = 0; i < vocab; i++) if (scaled[i] > maxVal) maxVal = scaled[i];
                let sumExp = 0;
                const exps = new Float32Array(vocab);
                for (let i = 0; i < vocab; i++) {
                    if (scaled[i] !== -Infinity) {
                        exps[i] = Math.exp(scaled[i] - maxVal);
                        sumExp += exps[i];
                    }
                }
                // cari cutoff: kumulatif prob > top_p
                let cum = 0;
                const keep = new Uint8Array(vocab);
                for (let i = 0; i < idx.length; i++) {
                    const t = idx[i];
                    if (exps[t] === 0) break;
                    cum += exps[t] / sumExp;
                    keep[t] = 1;
                    if (cum >= topP) break;
                }
                for (let i = 0; i < vocab; i++) {
                    if (!keep[i]) scaled[i] = -Infinity;
                }
            }

            // Softmax final
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
                const token = this.itos[String(nextId)] || '?';
                await onToken(token, step + 1);
            }
        }
        return this.decode(ids);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) module.exports = { DQuilane };
if (typeof window !== 'undefined') window.DQuilane = DQuilane;
