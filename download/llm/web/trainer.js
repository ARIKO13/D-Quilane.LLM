/**
 * D'QUILANE Trainer (vanilla JS) — runs in browser
 * =================================================
 * Lightweight training implementation that works in browser.
 * Uses vanilla JS matmul (CPU) with optional WebGPU acceleration.
 *
 * Limitations:
 * - Slower than Python/PyTorch (no native GPU)
 * - Suitable for SFT fine-tuning (~30-60s for 50 epochs)
 * - Pre-training 1000+ epochs too slow in vanilla JS (~minutes)
 */

class DQuilaneTrainer {
    constructor(modelData) {
        // Reuse DQuilane class from app.js for inference
        // But for training we need gradients, so we'll implement separately
        this.config = modelData.config;
        this.stoi = modelData.stoi;
        this.itos = modelData.itos;
        this.eos_id = modelData.eos_id || this.config.vocab_size - 1;
        
        // Convert weights to Float32Array (mutable for training)
        this.w = {};
        for (const [key, value] of Object.entries(modelData.weights)) {
            this.w[key] = new Float32Array(value.flat(Infinity));
        }
        
        // Track original shape info
        this.shapes = {};
        for (const [key, value] of Object.entries(modelData.weights)) {
            if (Array.isArray(value) && Array.isArray(value[0])) {
                this.shapes[key] = [value.length, value[0].length];
            } else if (Array.isArray(value)) {
                this.shapes[key] = [value.length];
            }
        }
        
        // Gradients (same shape as weights)
        this.grads = {};
        for (const [key, arr] of Object.entries(this.w)) {
            this.grads[key] = new Float32Array(arr.length);
        }
        
        // Optimizer state (Adam: m, v)
        this.adamState = {};
        for (const [key, arr] of Object.entries(this.w)) {
            this.adamState[key] = {
                m: new Float32Array(arr.length),
                v: new Float32Array(arr.length),
                t: 0
            };
        }
        
        // Detect WebGPU
        this.useGPU = false;
        this._detectGPU();
    }
    
    async _detectGPU() {
        try {
            if ('gpu' in navigator) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    this.useGPU = true;
                    this.gpuAdapter = adapter;
                    this.device = await adapter.requestDevice();
                    console.log('✅ WebGPU detected:', adapter.info?.description || 'GPU');
                }
            }
        } catch (e) {
            console.log('WebGPU not available, using CPU');
        }
    }
    
    // ===== Math ops (with gradients) =====
    
    // Linear: y = x @ W^T (forward + backward)
    // Forward: y[o] = sum_i(x[i] * W[o*inDim + i])
    // Backward: 
    //   dW[o*inDim + i] += x[i] * dy[o]
    //   dx[i] += sum_o(dy[o] * W[o*inDim + i])
    linearForward(x, W, inDim, outDim) {
        const out = new Float32Array(outDim);
        for (let o = 0; o < outDim; o++) {
            let sum = 0;
            const base = o * inDim;
            for (let i = 0; i < inDim; i++) sum += x[i] * W[base + i];
            out[o] = sum;
        }
        return out;
    }
    
    // RMSNorm forward + backward
    rmsNormForward(x, weight, eps = 1e-6) {
        const n = x.length;
        let sumSq = 0;
        for (let i = 0; i < n; i++) sumSq += x[i] * x[i];
        const rms = Math.sqrt(sumSq / n + eps);
        const inv = 1 / rms;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = (x[i] * inv) * weight[i];
        return { out, rms, inv, sumSq };
    }
    
    // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    siluForward(x) {
        const out = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            out[i] = x[i] / (1 + Math.exp(-x[i]));
        }
        return out;
    }
    
    // Softmax
    softmaxForward(x) {
        const n = x.length;
        let maxVal = -Infinity;
        for (let i = 0; i < n; i++) if (x[i] > maxVal) maxVal = x[i];
        let sum = 0;
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) { out[i] = Math.exp(x[i] - maxVal); sum += out[i]; }
        for (let i = 0; i < n; i++) out[i] /= sum;
        return out;
    }
    
    // ===== FORWARD PASS (returns logits + cache for backward) =====
    forward(ids) {
        const cfg = this.config;
        const d = cfg.d_model;
        const nHeads = cfg.n_heads;
        const dHead = d / nHeads;
        const T = ids.length;
        const cache = { ids, T, d, nHeads, dHead };
        
        // 1. Embedding
        let x = new Float32Array(T * d);
        cache.xEmb = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const tokBase = ids[t] * d;
            const posBase = t * d;
            const xBase = t * d;
            for (let i = 0; i < d; i++) {
                const val = this.w.tok_emb_weight[tokBase + i] + this.w.pos_emb_weight[posBase + i];
                x[xBase + i] = val;
                cache.xEmb[xBase + i] = val;
            }
        }
        
        // Loop through layers (parameter sharing)
        cache.layers = [];
        for (let layer = 0; layer < cfg.n_layers; layer++) {
            const layerCache = {};
            const result = this.blockForward(x, T, d, nHeads, dHead, layerCache);
            x = result.x;
            cache.layers.push(result.cache);
        }
        
        // Final norm
        const xNormed = new Float32Array(T * d);
        cache.normFCache = [];
        for (let t = 0; t < T; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            const { out, rms, inv, sumSq } = this.rmsNormForward(xRow, this.w.norm_f_weight);
            xNormed.set(out, t * d);
            cache.normFCache.push({ rms, inv, sumSq, xRow: new Float32Array(xRow) });
        }
        
        // Output head (tied to tok_emb)
        const vocab = cfg.vocab_size;
        const logits = new Float32Array(T * vocab);
        for (let t = 0; t < T; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            for (let v = 0; v < vocab; v++) {
                let sum = 0;
                const wBase = v * d;
                for (let i = 0; i < d; i++) sum += xRow[i] * this.w.tok_emb_weight[wBase + i];
                logits[t * vocab + v] = sum;
            }
        }
        
        cache.x = x;  // pre-norm
        cache.xNormed = xNormed;
        cache.logits = logits;
        cache.vocab = vocab;
        return { logits, cache };
    }
    
    blockForward(x, T, d, nHeads, dHead, cache) {
        // Pre-norm 1
        const xNormed = new Float32Array(T * d);
        cache.norm1Cache = [];
        for (let t = 0; t < T; t++) {
            const xRow = x.subarray(t * d, (t + 1) * d);
            const { out, rms, inv, sumSq } = this.rmsNormForward(xRow, this.w.norm1_weight);
            xNormed.set(out, t * d);
            cache.norm1Cache.push({ rms, inv, sumSq, xRow: new Float32Array(xRow) });
        }
        cache.xNormed1 = xNormed;
        
        // QKV
        const qkv = new Float32Array(T * 3 * d);
        cache.qkv = qkv;
        for (let t = 0; t < T; t++) {
            const xRow = xNormed.subarray(t * d, (t + 1) * d);
            const out = this.linearForward(xRow, this.w.attn_qkv_weight, d, 3 * d);
            qkv.set(out, t * 3 * d);
        }
        
        // Attention per head
        const attnOut = new Float32Array(T * d);
        cache.attn = { scores: [], attnProbs: [], vHeads: [], qHeads: [], kHeads: [] };
        
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
            cache.attn.qHeads.push(qHead);
            cache.attn.kHeads.push(kHead);
            cache.attn.vHeads.push(vHead);
            
            const scores = new Float32Array(T * T);
            const scale = 1 / Math.sqrt(dHead);
            for (let i = 0; i < T; i++) {
                for (let j = 0; j < T; j++) {
                    let sum = 0;
                    for (let k = 0; k < dHead; k++) sum += qHead[i * dHead + k] * kHead[j * dHead + k];
                    scores[i * T + j] = sum * scale;
                }
            }
            
            // Causal mask + softmax
            const attnProbs = new Float32Array(T * T);
            for (let i = 0; i < T; i++) {
                let maxVal = -Infinity;
                for (let j = 0; j <= i; j++) if (scores[i * T + j] > maxVal) maxVal = scores[i * T + j];
                let sum = 0;
                for (let j = 0; j <= i; j++) { attnProbs[i * T + j] = Math.exp(scores[i * T + j] - maxVal); sum += attnProbs[i * T + j]; }
                for (let j = 0; j <= i; j++) attnProbs[i * T + j] /= sum;
            }
            cache.attn.scores.push(scores);
            cache.attn.attnProbs.push(attnProbs);
            
            // Apply to V
            for (let i = 0; i < T; i++) {
                for (let k = 0; k < dHead; k++) {
                    let sum = 0;
                    for (let j = 0; j <= i; j++) sum += attnProbs[i * T + j] * vHead[j * dHead + k];
                    attnOut[i * d + h * dHead + k] = sum;
                }
            }
        }
        
        // Output projection
        const attnFinal = new Float32Array(T * d);
        cache.attnOut = attnOut;
        for (let t = 0; t < T; t++) {
            const xRow = attnOut.subarray(t * d, (t + 1) * d);
            const out = this.linearForward(xRow, this.w.attn_out_weight, d, d);
            attnFinal.set(out, t * d);
        }
        cache.attnFinal = attnFinal;
        
        // Residual 1
        const xRes1 = new Float32Array(T * d);
        for (let i = 0; i < T * d; i++) xRes1[i] = x[i] + attnFinal[i];
        cache.xRes1 = xRes1;
        
        // Pre-norm 2 + FFN
        const xNormed2 = new Float32Array(T * d);
        cache.norm2Cache = [];
        for (let t = 0; t < T; t++) {
            const xRow = xRes1.subarray(t * d, (t + 1) * d);
            const { out, rms, inv, sumSq } = this.rmsNormForward(xRow, this.w.norm2_weight);
            xNormed2.set(out, t * d);
            cache.norm2Cache.push({ rms, inv, sumSq, xRow: new Float32Array(xRow) });
        }
        cache.xNormed2 = xNormed2;
        
        const ffnDim = this.config.d_ff;
        const ffnOut = new Float32Array(T * d);
        cache.ffn = { gate: [], val: [], act: [] };
        for (let t = 0; t < T; t++) {
            const xRow = xNormed2.subarray(t * d, (t + 1) * d);
            const gate = this.linearForward(xRow, this.w.ffn_gate_weight, d, ffnDim);
            const val = this.linearForward(xRow, this.w.ffn_val_weight, d, ffnDim);
            const act = new Float32Array(ffnDim);
            for (let i = 0; i < ffnDim; i++) {
                act[i] = (gate[i] / (1 + Math.exp(-gate[i]))) * val[i];
            }
            cache.ffn.gate.push(gate);
            cache.ffn.val.push(val);
            cache.ffn.act.push(act);
            const down = this.linearForward(act, this.w.ffn_down_weight, ffnDim, d);
            ffnOut.set(down, t * d);
        }
        cache.ffnOut = ffnOut;
        
        // Residual 2
        const xFinal = new Float32Array(T * d);
        for (let i = 0; i < T * d; i++) xFinal[i] = xRes1[i] + ffnOut[i];
        
        return { x: xFinal, cache };
    }
    
    // ===== BACKWARD PASS =====
    backward(cache, targets, ignoreIndex = -100) {
        // Reset grads
        for (const key in this.grads) {
            this.grads[key].fill(0);
        }
        
        const { logits, vocab, T, d, xNormed, normFCache, layers } = cache;
        
        // Cross-entropy loss + grad
        let totalLoss = 0;
        const dLogits = new Float32Array(T * vocab);
        let nTargets = 0;
        for (let t = 0; t < T; t++) {
            const targetId = targets[t];
            if (targetId === ignoreIndex) continue;
            nTargets++;
            const rowBase = t * vocab;
            
            // Softmax
            let maxVal = -Infinity;
            for (let v = 0; v < vocab; v++) if (logits[rowBase + v] > maxVal) maxVal = logits[rowBase + v];
            let sum = 0;
            const probs = new Float32Array(vocab);
            for (let v = 0; v < vocab; v++) { probs[v] = Math.exp(logits[rowBase + v] - maxVal); sum += probs[v]; }
            for (let v = 0; v < vocab; v++) probs[v] /= sum;
            
            // Loss
            totalLoss += -Math.log(probs[targetId] + 1e-10);
            
            // dLogits = probs - one_hot(targetId)
            for (let v = 0; v < vocab; v++) {
                dLogits[rowBase + v] = probs[v] - (v === targetId ? 1 : 0);
            }
        }
        const loss = totalLoss / Math.max(nTargets, 1);
        
        // Backward through output head (tied to tok_emb)
        // dLogits[T, vocab] @ tok_emb[vocab, d] => d_xNormed[T, d]
        const dXNormed = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const rowBase = t * vocab;
            for (let v = 0; v < vocab; v++) {
                const dl = dLogits[rowBase + v];
                if (dl === 0) continue;
                const wBase = v * d;
                for (let i = 0; i < d; i++) {
                    dXNormed[t * d + i] += dl * this.w.tok_emb_weight[wBase + i];
                    this.grads.tok_emb_weight[wBase + i] += dl * xNormed[t * d + i];
                }
            }
        }
        
        // Backward through RMSNorm final
        const dX = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const { rms, inv, sumSq, xRow } = normFCache[t];
            const dXNormedRow = dXNormed.subarray(t * d, (t + 1) * d);
            // dX = (dXNormed * weight) * (1/rms - x*x/(n*rms^3))
            // dWeight = sum(dXNormed * xNormed)
            const n = d;
            for (let i = 0; i < n; i++) {
                this.grads.norm_f_weight[i] += dXNormedRow[i] * xNormed[t * d + i];
            }
            // Compute dX
            // dXNormed[i] = x[i] * inv * weight[i]
            // dX[i] = dXNormed[i] * weight[i] * (inv - x[i]^2 * inv^3 / n)
            // Actually need to handle RMS norm backward more carefully
            // dX[i] = (1/rms) * (dXNormed[i] * weight[i] - x[i] * sum(x * dXNormed * weight) / (n * rms^2))
            let sumXDX = 0;
            for (let i = 0; i < n; i++) {
                sumXDX += xRow[i] * dXNormedRow[i] * this.w.norm_f_weight[i];
            }
            for (let i = 0; i < n; i++) {
                dX[t * d + i] = (dXNormedRow[i] * this.w.norm_f_weight[i] / rms) -
                                (xRow[i] * sumXDX / (n * rms * rms * rms));
            }
        }
        
        // Backward through layers (reverse order)
        for (let l = layers.length - 1; l >= 0; l--) {
            this.blockBackward(dX, layers[l], T, d, this.config.n_heads, d / this.config.n_heads);
        }
        
        // Backward through embedding
        const ids = cache.ids;
        for (let t = 0; t < T; t++) {
            const tokBase = ids[t] * d;
            const posBase = t * d;
            const dXRow = dX.subarray(t * d, (t + 1) * d);
            for (let i = 0; i < d; i++) {
                this.grads.tok_emb_weight[tokBase + i] += dXRow[i];
                this.grads.pos_emb_weight[posBase + i] += dXRow[i];
            }
        }
        
        return { loss, dLogits };
    }
    
    blockBackward(dX, cache, T, d, nHeads, dHead) {
        // Residual 2: xFinal = xRes1 + ffnOut
        // dXRes1 = dX, dFfnOut = dX
        const dXRes1 = new Float32Array(dX);
        const dFfnOut = new Float32Array(dX);
        
        // Backward through ffn_down
        const dAct = new Float32Array(T * this.config.d_ff);
        const dFfnDownWeight = this.grads.ffn_down_weight;
        const dXNormed2 = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const dOutRow = dFfnOut.subarray(t * d, (t + 1) * d);
            const act = cache.ffn.act[t];
            // dAct = dOut @ ffn_down_weight
            for (let o = 0; o < this.config.d_ff; o++) {
                let sum = 0;
                const wBase = o * d;
                for (let i = 0; i < d; i++) sum += dOutRow[i] * this.w.ffn_down_weight[wBase + i];
                dAct[t * this.config.d_ff + o] = sum;
                // grad weight
                for (let i = 0; i < d; i++) dFfnDownWeight[wBase + i] += dOutRow[i] * act[o];
            }
        }
        
        // Backward through silu(gate) * val
        const dGate = new Float32Array(T * this.config.d_ff);
        const dVal = new Float32Array(T * this.config.d_ff);
        for (let t = 0; t < T; t++) {
            const gate = cache.ffn.gate[t];
            const val = cache.ffn.val[t];
            for (let i = 0; i < this.config.d_ff; i++) {
                const sigmoid = 1 / (1 + Math.exp(-gate[i]));
                const silu = gate[i] * sigmoid;
                // d/d gate: dAct * (silu'(gate)) = dAct * (sigmoid + gate * sigmoid * (1-sigmoid))
                dGate[t * this.config.d_ff + i] = dAct[t * this.config.d_ff + i] * (sigmoid + gate[i] * sigmoid * (1 - sigmoid)) * val[i];
                // d/d val: dAct * silu
                dVal[t * this.config.d_ff + i] = dAct[t * this.config.d_ff + i] * silu;
            }
        }
        
        // Backward through gate and val linear
        const dXNormed2Row = new Float32Array(d);
        for (let t = 0; t < T; t++) {
            const xRow = cache.xNormed2.subarray(t * d, (t + 1) * d);
            const gateRow = dGate.subarray(t * this.config.d_ff, (t + 1) * this.config.d_ff);
            const valRow = dVal.subarray(t * this.config.d_ff, (t + 1) * this.config.d_ff);
            const dXRow = new Float32Array(d);
            // gate = xNormed2 @ ffn_gate_weight^T
            for (let o = 0; o < this.config.d_ff; o++) {
                const wBase = o * d;
                for (let i = 0; i < d; i++) {
                    dXRow[i] += gateRow[o] * this.w.ffn_gate_weight[wBase + i];
                    this.grads.ffn_gate_weight[wBase + i] += gateRow[o] * xRow[i];
                }
            }
            for (let o = 0; o < this.config.d_ff; o++) {
                const wBase = o * d;
                for (let i = 0; i < d; i++) {
                    dXRow[i] += valRow[o] * this.w.ffn_val_weight[wBase + i];
                    this.grads.ffn_val_weight[wBase + i] += valRow[o] * xRow[i];
                }
            }
            dXNormed2.set(dXRow, t * d);
        }
        
        // Backward through RMSNorm 2
        for (let t = 0; t < T; t++) {
            const { rms, inv, sumSq, xRow } = cache.norm2Cache[t];
            const dXNormed2Row = dXNormed2.subarray(t * d, (t + 1) * d);
            for (let i = 0; i < d; i++) {
                this.grads.norm2_weight[i] += dXNormed2Row[i] * xRow[i];
            }
            let sumXDX = 0;
            for (let i = 0; i < d; i++) {
                sumXDX += xRow[i] * dXNormed2Row[i] * this.w.norm2_weight[i];
            }
            const dXRes1Row = dXRes1.subarray(t * d, (t + 1) * d);
            for (let i = 0; i < d; i++) {
                dXRes1Row[i] += (dXNormed2Row[i] * this.w.norm2_weight[i] / rms) -
                                (xRow[i] * sumXDX / (d * rms * rms * rms));
            }
        }
        
        // Backward through attn output projection
        const dAttnOut = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const dOutRow = dXRes1.subarray(t * d, (t + 1) * d);
            const xRow = cache.attnOut.subarray(t * d, (t + 1) * d);
            for (let o = 0; o < d; o++) {
                const wBase = o * d;
                for (let i = 0; i < d; i++) {
                    dAttnOut[t * d + i] += dOutRow[o] * this.w.attn_out_weight[wBase + i];
                    this.grads.attn_out_weight[wBase + i] += dOutRow[o] * xRow[i];
                }
            }
        }
        
        // Backward through attention (per head)
        const dQKV = new Float32Array(T * 3 * d);
        for (let h = 0; h < nHeads; h++) {
            const qHead = cache.attn.qHeads[h];
            const kHead = cache.attn.kHeads[h];
            const vHead = cache.attn.vHeads[h];
            const attnProbs = cache.attn.attnProbs[h];
            
            // dV
            const dVHead = new Float32Array(T * dHead);
            const dAttnProbs = new Float32Array(T * T);
            for (let i = 0; i < T; i++) {
                for (let k = 0; k < dHead; k++) {
                    let sum = 0;
                    for (let j = 0; j <= i; j++) sum += attnProbs[i * T + j] * vHead[j * dHead + k];
                    const dA = dAttnOut[i * d + h * dHead + k];
                    for (let j = 0; j <= i; j++) {
                        dVHead[j * dHead + k] += dA * attnProbs[i * T + j];
                        dAttnProbs[i * T + j] += dA * vHead[j * dHead + k];
                    }
                }
            }
            
            // dAttnProbs -> dScores (softmax backward)
            const dScores = new Float32Array(T * T);
            for (let i = 0; i < T; i++) {
                let sum = 0;
                for (let j = 0; j <= i; j++) sum += dAttnProbs[i * T + j];
                for (let j = 0; j <= i; j++) {
                    dScores[i * T + j] = attnProbs[i * T + j] * (dAttnProbs[i * T + j] - sum);
                }
            }
            
            // dScores = scale * (Q @ K^T) ... so dQ = dScores @ K, dK = dScores^T @ Q
            const scale = 1 / Math.sqrt(dHead);
            const dQHead = new Float32Array(T * dHead);
            const dKHead = new Float32Array(T * dHead);
            for (let i = 0; i < T; i++) {
                for (let j = 0; j <= i; j++) {
                    const ds = dScores[i * T + j] * scale;
                    for (let k = 0; k < dHead; k++) {
                        dQHead[i * dHead + k] += ds * kHead[j * dHead + k];
                        dKHead[j * dHead + k] += ds * qHead[i * dHead + k];
                    }
                }
            }
            
            // dQKV
            for (let t = 0; t < T; t++) {
                const base = t * 3 * d;
                for (let i = 0; i < dHead; i++) {
                    dQKV[base + h * dHead + i] = dQHead[t * dHead + i];
                    dQKV[base + d + h * dHead + i] = dKHead[t * dHead + i];
                    dQKV[base + 2 * d + h * dHead + i] = dVHead[t * dHead + i];
                }
            }
        }
        
        // Backward through QKV linear
        const dXNormed1 = new Float32Array(T * d);
        for (let t = 0; t < T; t++) {
            const dQKVRow = dQKV.subarray(t * 3 * d, (t + 1) * 3 * d);
            const xRow = cache.xNormed1.subarray(t * d, (t + 1) * d);
            for (let o = 0; o < 3 * d; o++) {
                const wBase = o * d;
                for (let i = 0; i < d; i++) {
                    dXNormed1[t * d + i] += dQKVRow[o] * this.w.attn_qkv_weight[wBase + i];
                    this.grads.attn_qkv_weight[wBase + i] += dQKVRow[o] * xRow[i];
                }
            }
        }
        
        // Backward through RMSNorm 1
        const dXInput = new Float32Array(dX.length); // actually dXRes1 minus the residual contribution
        for (let t = 0; t < T; t++) {
            const { rms, inv, sumSq, xRow } = cache.norm1Cache[t];
            const dXNormed1Row = dXNormed1.subarray(t * d, (t + 1) * d);
            for (let i = 0; i < d; i++) {
                this.grads.norm1_weight[i] += dXNormed1Row[i] * xRow[i];
            }
            let sumXDX = 0;
            for (let i = 0; i < d; i++) {
                sumXDX += xRow[i] * dXNormed1Row[i] * this.w.norm1_weight[i];
            }
            for (let i = 0; i < d; i++) {
                dX[t * d + i] += (dXNormed1Row[i] * this.w.norm1_weight[i] / rms) -
                                (xRow[i] * sumXDX / (d * rms * rms * rms));
            }
        }
        
        // dX now contains gradient flowing to input of block
        // Need to write back to dX (which was the original input)
        // Actually, dX has been updated to contain the full gradient
    }
    
    // ===== ADAM OPTIMIZER =====
    adamStep(lr, weightDecay = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, gradClip = 1.0) {
        // Clip grads
        let gradNorm = 0;
        for (const key in this.grads) {
            for (let i = 0; i < this.grads[key].length; i++) {
                gradNorm += this.grads[key][i] * this.grads[key][i];
            }
        }
        gradNorm = Math.sqrt(gradNorm);
        if (gradNorm > gradClip) {
            const scale = gradClip / gradNorm;
            for (const key in this.grads) {
                for (let i = 0; i < this.grads[key].length; i++) {
                    this.grads[key][i] *= scale;
                }
            }
        }
        
        // Adam update
        for (const key in this.w) {
            const w = this.w[key];
            const g = this.grads[key];
            const state = this.adamState[key];
            state.t += 1;
            for (let i = 0; i < w.length; i++) {
                // Decoupled weight decay
                w[i] -= lr * weightDecay * w[i];
                // Adam
                state.m[i] = beta1 * state.m[i] + (1 - beta1) * g[i];
                state.v[i] = beta2 * state.v[i] + (1 - beta2) * g[i] * g[i];
                const mHat = state.m[i] / (1 - Math.pow(beta1, state.t));
                const vHat = state.v[i] / (1 - Math.pow(beta2, state.t));
                w[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
        }
    }
    
    // ===== TRAIN STEP =====
    trainStep(ids, targets, lr) {
        const { logits, cache } = this.forward(ids);
        const { loss } = this.backward(cache, targets);
        this.adamStep(lr);
        return loss;
    }
    
    // ===== SAVE =====
    exportModel() {
        const weights = {};
        for (const [key, arr] of Object.entries(this.w)) {
            // Convert Float32Array back to nested list
            const shape = this.shapes[key];
            if (shape.length === 2) {
                const [rows, cols] = shape;
                const out = [];
                for (let r = 0; r < rows; r++) {
                    out.push(Array.from(arr.subarray(r * cols, (r + 1) * cols)));
                }
                weights[key] = out;
            } else {
                weights[key] = Array.from(arr);
            }
        }
        return {
            config: this.config,
            stoi: this.stoi,
            itos: this.itos,
            eos_id: this.eos_id,
            weights: weights
        };
    }
}

// SFT training data (subset for browser)
const SFT_EXAMPLES_BROWSER = [
    ['Halo', 'Halo juga! Senang bertemu denganmu.'],
    ['halo', 'Halo juga! Senang bertemu denganmu.'],
    ['Siapa kamu?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'],
    ['siapa kamu', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'],
    ['Satu tambah satu?', 'Satu tambah satu sama dengan dua.'],
    ['satu tambah satu', 'Satu tambah satu sama dengan dua.'],
    ['Dua dikali dua?', 'Dua dikali dua sama dengan empat.'],
    ['dua dikali dua', 'Dua dikali dua sama dengan empat.'],
    ['Lima dikali lima?', 'Lima dikali lima sama dengan dua puluh lima.'],
    ['Apa ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'],
    ['ibu kota jepang apa', 'Ibu kota Jepang adalah Tokyo.'],
    ['Apa ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'],
    ['tokyo itu apa', 'Tokyo adalah ibu kota Jepang.'],
    ['Siapa presiden pertama Indonesia?', 'Presiden pertama Indonesia adalah Soekarno.'],
    ['siapa presiden pertama indonesia', 'Presiden pertama Indonesia adalah Soekarno.'],
    ['Apa warna langit?', 'Warna langit adalah biru.'],
    ['Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'],
    ['Translate halo to English', 'Halo in English is Hello.'],
    ['Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'],
    ['terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'],
    ['Halo apa kabar?', 'Halo! Aku baik-baik saja, terima kasih.'],
    ['Apa suara kucing?', 'Kucing bersuara meong.'],
    ['Apa suara sapi?', 'Sapi bersuara moo.'],
    ['Apa suara anjing?', 'Anjing bersuara guk guk.'],
    ['Apa warna rumput?', 'Warna rumput adalah hijau.'],
    ['Apa warna darah?', 'Warna darah adalah merah.'],
    ['Berapa jari di tangan?', 'Ada lima jari di setiap tangan.'],
    ['Matahari terbit di?', 'Matahari terbit di timur.'],
    ['Apa fungsi akar?', 'Akar menyerap air dari tanah.'],
    ['Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'],
    ['test', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'],
    ['lagi apa', 'Aku siap membantu. Ada yang bisa aku bantu?'],
    ['kamu bisa apa', 'Aku bisa matematika dasar, terjemahan, dan menjawab pertanyaan sederhana.'],
    ['jepang di mana', 'Jepang adalah negara di Asia Timur.'],
];

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DQuilaneTrainer, SFT_EXAMPLES_BROWSER };
}
if (typeof window !== 'undefined') {
    window.DQuilaneTrainer = DQuilaneTrainer;
    window.SFT_EXAMPLES_BROWSER = SFT_EXAMPLES_BROWSER;
}
