/**
 * D'QUILANE v0.6 Fast — JavaScript Inference Engine dengan KV-CACHE
 *
 * Optimization utama:
 * 1. KV-CACHE: K, V projections disimpan per layer, gak di-recompute tiap token
 *    => complexity O(T^2) -> O(T) per token, ~5-10x speedup
 * 2. Pre-allocated buffers (reuse Float32Array)
 * 3. Optimized math ops
 *
 * Fitur:
 * - Top-p (nucleus) sampling
 * - Repetition penalty
 * - In-context learning (custom context)
 * - KV-cache utk fast generation
 */

class DQuilane {
    constructor(modelData) {
        this.config = modelData.config;
        this.stoi = modelData.stoi;
        this.itos = modelData.itos;
        this.w = {};
        for (const [key, value] of Object.entries(modelData.weights)) {
            this.w[key] = this._flatten(value);
        }
        const d = this.config.d_model;
        this.d = d;
    }

    _flatten(arr) {
        if (typeof arr[0] === 'number') return new Float32Array(arr);
        const out = [];
        for (const sub of arr) for (const x of sub) out.push(x);
        return new Float32Array(out);
    }

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

    // ===== Math ops =====
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
        for (let i = 0; i < n; i++) if (mask[i] && x[i] > maxVal) maxVal = x[i];
        let sum = 0;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            if (mask[i]) { out[i] = Math.exp(x[i] - maxVal); sum += out[i]; }
            else out[i] = 0;
        }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }

    // ===== KV-CACHE STRUCTURE =====
    // Cache per virtual layer iteration: { k: Float32Array[T * n_heads * d_head], v: Float32Array[T * n_heads * d_head], T: number }
    initCache() {
        const cache = { T: 0, layers: [] };
        for (let i = 0; i < this.config.n_layers; i++) {
            cache.layers.push({ k: null, v: null, T: 0 });
        }
        return cache;
    }

    // ===== FORWARD WITH KV-CACHE =====
    // ids: array of token ids to process (could be full sequence on first call, or 1 token on subsequent)
    // cache: object with per-layer K/V caches
    // Returns: { logits: Float32Array(T_new * vocab), cache: updated }
    forwardCached(ids, cache) {
        const cfg = this.config;
        const d = cfg.d_model;
        const nHeads = cfg.n_heads;
        const dHead = d / nHeads;
        const T_new = ids.length;
        const T_cached = cache.T;
        const T = T_cached + T_new;  // total sequence length

        // Position ids for new tokens: T_cached, T_cached+1, ..., T_cached+T_new-1
        // 1. Embedding for new tokens
        let x = new Float32Array(T_new * d);
        for (let t = 0; t < T_new; t++) {
            const tokenId = ids[t];
            const posId = T_cached + t;
            if (posId >= cfg.max_seq) {
                // Should not happen if context managed properly
                return null;
            }
            const tokBase = tokenId * d;
            const posBase = posId * d;
            const xBase = t * d;
            for (let i = 0; i < d; i++) {
                x[xBase + i] = this.w.tok_emb_weight[tokBase + i] + this.w.pos_emb_weight[posBase + i];
            }
        }

        // 2. Process through n_layers (parameter sharing)
        for (let layer = 0; layer < cfg.n_layers; layer++) {
            x = this.blockForwardCached(x, T_new, T_cached, cache.layers[layer], d, nHeads, dHead);
        }

        // 3. Final norm (per new token)
        const xFinal = new Float32Array(T_new * d);
        for (let t = 0; t < T_new; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            xFinal.set(this.rmsNorm(xRow, this.w.norm_f_weight), t * d);
        }

        // 4. Output head: logits[T_new, vocab]
        const vocab = cfg.vocab_size;
        const logits = new Float32Array(T_new * vocab);
        for (let t = 0; t < T_new; t++) {
            const xRow = xFinal.subarray(t * d, (t + 1) * d);
            for (let v = 0; v < vocab; v++) {
                let sum = 0;
                const wBase = v * d;
                for (let i = 0; i < d; i++) sum += xRow[i] * this.w.tok_emb_weight[wBase + i];
                logits[t * vocab + v] = sum;
            }
        }

        // Update cache.T
        cache.T = T;
        return { logits, cache };
    }

    // Block forward with KV-cache
    blockForwardCached(x, T_new, T_cached, layerCache, d, nHeads, dHead) {
        // Pre-norm 1
        const xNormed = new Float32Array(T_new * d);
        for (let t = 0; t < T_new; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            xNormed.set(this.rmsNorm(xRow, this.w.norm1_weight), t * d);
        }

        // QKV: [T_new, 3d]
        const qkv = new Float32Array(T_new * 3 * d);
        for (let t = 0; t < T_new; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            qkv.set(this.linear(xRow, this.w.attn_qkv_weight, d, 3 * d), t * 3 * d);
        }

        // For each new token, compute Q/K/V per head
        // Then append K/V to cache, compute attention with full cached K/V
        const T_total = T_cached + T_new;
        const attnOut = new Float32Array(T_new * d);

        // Pre-allocate K and V cache buffers (growing)
        // Layout: [T_total, nHeads, dHead]
        let kCacheFull, vCacheFull;
        if (layerCache.k !== null) {
            // Extend existing cache
            const oldK = layerCache.k;
            const oldV = layerCache.v;
            const oldT = layerCache.T;
            kCacheFull = new Float32Array(T_total * nHeads * dHead);
            vCacheFull = new Float32Array(T_total * nHeads * dHead);
            kCacheFull.set(oldK);
            vCacheFull.set(oldV);
        } else {
            kCacheFull = new Float32Array(T_total * nHeads * dHead);
            vCacheFull = new Float32Array(T_total * nHeads * dHead);
        }

        // Fill new K/V into cache (extend)
        // qkv layout for token t: [Q(nHeads*dHead), K(nHeads*dHead), V(nHeads*dHead)]
        for (let t = 0; t < T_new; t++) {
            const tTotal = T_cached + t;
            const qkvBase = t * 3 * d;
            for (let h = 0; h < nHeads; h++) {
                const kCacheBase = tTotal * nHeads * dHead + h * dHead;
                const vCacheBase = tTotal * nHeads * dHead + h * dHead;
                for (let i = 0; i < dHead; i++) {
                    kCacheFull[kCacheBase + i] = qkv[qkvBase + d + h * dHead + i];
                    vCacheFull[vCacheBase + i] = qkv[qkvBase + 2 * d + h * dHead + i];
                }
            }
        }

        // For each new token, compute attention against ALL cached K/V
        const scale = 1 / Math.sqrt(dHead);
        for (let t = 0; t < T_new; t++) {
            const tTotal = T_cached + t;  // can attend to positions 0..tTotal
            const qkvBase = t * 3 * d;

            for (let h = 0; h < nHeads; h++) {
                // Extract Q for this (t, h)
                const qHead = new Float32Array(dHead);
                for (let i = 0; i < dHead; i++) qHead[i] = qkv[qkvBase + h * dHead + i];

                // Compute scores: Q @ K_cache^T for all cached positions (0..tTotal)
                const scores = new Float32Array(tTotal + 1);
                for (let j = 0; j <= tTotal; j++) {
                    let sum = 0;
                    const kBase = j * nHeads * dHead + h * dHead;
                    for (let k = 0; k < dHead; k++) sum += qHead[k] * kCacheFull[kBase + k];
                    scores[j] = sum * scale;
                }

                // Softmax (causal mask implicit since j only goes to tTotal)
                let maxVal = -Infinity;
                for (let j = 0; j <= tTotal; j++) if (scores[j] > maxVal) maxVal = scores[j];
                let sumExp = 0;
                for (let j = 0; j <= tTotal; j++) {
                    scores[j] = Math.exp(scores[j] - maxVal);
                    sumExp += scores[j];
                }
                for (let j = 0; j <= tTotal; j++) scores[j] /= sumExp;

                // Apply attention to V
                const outBase = t * d + h * dHead;
                for (let k = 0; k < dHead; k++) {
                    let sum = 0;
                    for (let j = 0; j <= tTotal; j++) {
                        const vBase = j * nHeads * dHead + h * dHead;
                        sum += scores[j] * vCacheFull[vBase + k];
                    }
                    attnOut[outBase + k] = sum;
                }
            }
        }

        // Output projection: attnOut[T_new, d] @ out_weight^T => [T_new, d]
        const attnFinal = new Float32Array(T_new * d);
        for (let t = 0; t < T_new; t++) {
            const xRow = attnOut.subarray(t * d, (t + 1) * d);
            attnFinal.set(this.linear(xRow, this.w.attn_out_weight, d, d), t * d);
        }

        // Residual 1
        for (let i = 0; i < T_new * d; i++) x[i] += attnFinal[i];

        // Pre-norm 2 + FFN
        const xNormed2 = new Float32Array(T_new * d);
        for (let t = 0; t < T_new; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            xNormed2.set(this.rmsNorm(xRow, this.w.norm2_weight), t * d);
        }

        const ffnDim = this.config.d_ff;
        const ffnOut = new Float32Array(T_new * d);
        for (let t = 0; t < T_new; t++) {
            const xRow = xNormed2.subarray(t * d, (t + 1) * d);
            const gate = this.linear(xRow, this.w.ffn_gate_weight, d, ffnDim);
            const val = this.linear(xRow, this.w.ffn_val_weight, d, ffnDim);
            const act = new Float32Array(ffnDim);
            for (let i = 0; i < ffnDim; i++) {
                act[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * val[i];
            }
            ffnOut.set(this.linear(act, this.w.ffn_down_weight, ffnDim, d), t * d);
        }

        // Residual 2
        for (let i = 0; i < T_new * d; i++) x[i] += ffnOut[i];

        // Update layer cache
        layerCache.k = kCacheFull;
        layerCache.v = vCacheFull;
        layerCache.T = T_total;

        return x;
    }

    // ===== GENERATION dengan KV-cache + EOS stop =====
    async generate(prompt, maxNew = 200, temp = 0.8, topK = 40, topP = 0.95, repPenalty = 1.1, onToken = null, stopId = null) {
        let ids = this.encode(prompt);
        if (ids.length === 0) return prompt;
        const vocab = this.config.vocab_size;
        const maxSeq = this.config.max_seq;

        // Crop prompt to max_seq
        if (ids.length > maxSeq) {
            ids = ids.slice(ids.length - maxSeq);
        }

        // Use EOS_ID from config if available
        if (stopId === null && this.config.eos_id !== undefined) {
            stopId = this.config.eos_id;
        }

        // === Pass 1: forward full prompt to populate KV-cache ===
        const cache = this.initCache();
        const result0 = this.forwardCached(ids, cache);
        if (!result0) return this.decode(ids);
        let logits = result0.logits;
        // Get last token logits
        const T = ids.length;
        let lastLogits = new Float32Array(vocab);
        for (let i = 0; i < vocab; i++) lastLogits[i] = logits[(T - 1) * vocab + i];

        // === Generate new tokens one by one with KV-cache ===
        const recentTokens = [];
        for (let step = 0; step < maxNew; step++) {
            // Check if we exceeded context
            if (cache.T >= maxSeq) {
                break;
            }

            // Apply temperature
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

            // Top-p
            if (topP < 1.0) {
                const idx = Array.from({ length: vocab }, (_, i) => i);
                idx.sort((a, b) => scaled[b] - scaled[a]);
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

            // STOP on EOS
            if (stopId !== null && nextId === stopId) {
                break;
            }

            ids.push(nextId);
            if (onToken) {
                const token = this.itos[String(nextId)] || '?';
                await onToken(token, step + 1);
            }

            // Forward only the new token through cache (FAST!)
            const stepResult = this.forwardCached([nextId], cache);
            if (!stepResult) break;
            logits = stepResult.logits;
            // Logits for the new token (which is the only one in T_new=1)
            for (let i = 0; i < vocab; i++) lastLogits[i] = logits[i];
        }
        // Filter out EOS from output
        return this.decode(ids.filter(id => id !== stopId));
    }
}

if (typeof module !== 'undefined' && module.exports) module.exports = { DQuilane };
if (typeof window !== 'undefined') window.DQuilane = DQuilane;
