"""
Qoolse AI Playground — local web UI to try the trained containers.

    python playground.py            # http://127.0.0.1:8000
    python playground.py --port 8080

Stdlib HTTP server + torch only (no extra dependencies). Each model is loaded
lazily and cached. Responses are labelled with where they came from:
`calculator` = evaluated by calculator.py (arithmetic, no dataset),
`AION` = retrieved from the prompt->answer table, `network` = sampled from the
Qoolse AI v3 model (135M BF16, 9 recursive loops, shared-KV, MoE top-2).
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from infer_qoolse import Session
from qoolse_model import (QOOLSE_V3_CONFIG, PARAM_TIERS, find_tier,
                            format_params, resolve_target_params, tier_range_text)

INFER_LOCK = threading.Lock()


class Engines:
    def __init__(self, folder):
        self.folder = folder
        self.cache = {}
        self.lock = threading.Lock()

    def paths(self):
        found = sorted(glob.glob(os.path.join(self.folder, '*.gguf')),
                       key=lambda p: ('trainer' not in os.path.basename(p),
                                      os.path.basename(p)))
        return {os.path.basename(p): p for p in found}

    def get(self, name):
        paths = self.paths()
        if name not in paths:
            raise KeyError(name)
        with self.lock:
            if name not in self.cache:
                self.cache[name] = Session(paths[name])
            return self.cache[name]

    def summary(self):
        out = []
        for name, path in self.paths().items():
            s = self.get(name)
            m = s.meta
            out.append({
                'name': name,
                'params': s.n_params,
                'active_params': s.model.count_active_params(),
                'experts': s.model.cfg['n_experts'],
                'shared': 1,
                'top_k': s.model.cfg['top_k'],
                'ternary': False,
                'vocab': s.model.cfg['vocab_size'],
                'blocks': s.model.cfg['n_stem_layers'],
                'repeat': s.model.cfg['n_loops'],
                'd_model': s.model.cfg['d_model'],
                'd_kv': s.model.cfg['d_kv'],
                'max_seq': s.model.cfg['max_seq'],
                'size_mb': round(os.path.getsize(path) / 1024 / 1024, 2),
                'aion_pairs': len(s.engram),
                'trained': m.get('trained', 'unknown'),
                'val_ppl': m.get('val_perplexity'),
                'val_loss': m.get('val_loss'),
                'train_ppl': m.get('train_perplexity'),
                'epochs': m.get('epochs'),
                'best_epoch': m.get('best_epoch'),
                'train_pairs': m.get('train_pairs'),
                'val_pairs': m.get('val_pairs'),
                'steps': m.get('train_steps'),
                'arch': m.get('architecture'),
            })
        return out


ENGINES = None

PAGE = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qoolse AI Playground</title>
<style>
  :root{
    --bg:#0c0f13; --panel:#141a21; --panel2:#1a222b; --line:#26313d;
    --text:#e7eef6; --muted:#93a4b5; --accent:#ff7a45; --accent2:#4cc2ff;
    --ok:#54d68a; --warn:#ffcf5c;
  }
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 80% -10%,#1b2530 0%,var(--bg) 55%);
    color:var(--text);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  header{display:flex;align-items:center;gap:14px;padding:18px 26px;border-bottom:1px solid var(--line)}
  .logo{width:34px;height:34px;border-radius:10px;background:linear-gradient(140deg,var(--accent),#b03a0f);
    display:grid;place-items:center;font-weight:800;color:#150a06}
  h1{font-size:17px;margin:0;letter-spacing:.2px}
  header .sub{color:var(--muted);font-size:12.5px}
  main{display:grid;grid-template-columns:320px 1fr;gap:18px;padding:18px 26px;max-width:1240px;margin:0 auto}
  @media (max-width:900px){main{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
    margin:0 0 12px;font-weight:600}
  select,input[type=text],textarea{width:100%;background:var(--panel2);color:var(--text);
    border:1px solid var(--line);border-radius:9px;padding:9px 11px;font:inherit;outline:none}
  select:focus,textarea:focus{border-color:var(--accent2)}
  .stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}
  .stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:9px 10px}
  .stat .k{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
  .stat .v{font-size:15px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
  .stat .v small{font-size:11px;color:var(--muted);font-weight:400}
  .row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:13px 0}
  .row label{font-size:13px;color:var(--muted)}
  .row .val{font-variant-numeric:tabular-nums;color:var(--text);font-size:13px}
  input[type=range]{-webkit-appearance:none;width:100%;height:4px;border-radius:4px;
    background:#2a3542;outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;
    background:var(--accent);cursor:pointer;border:2px solid #0c0f13}
  .switch{display:flex;align-items:center;gap:9px;cursor:pointer;user-select:none;font-size:13px}
  .switch input{display:none}
  .track{width:38px;height:21px;border-radius:20px;background:#2a3542;position:relative;transition:.18s}
  .track::after{content:"";position:absolute;top:3px;left:3px;width:15px;height:15px;border-radius:50%;
    background:#8ea1b4;transition:.18s}
  .switch input:checked + .track{background:rgba(255,122,69,.35)}
  .switch input:checked + .track::after{transform:translateX(17px);background:var(--accent)}
  .note{margin-top:14px;font-size:12px;color:var(--muted);border-left:2px solid var(--warn);
    padding:6px 0 6px 10px}
  .chat{display:flex;flex-direction:column;min-height:560px;grid-column:2;grid-row:2}
  .log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:14px;padding:6px 4px 14px;
    max-height:60vh}
  .msg{display:flex;gap:10px;align-items:flex-start}
  .bubble{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:11px 13px;
    max-width:78%;white-space:pre-wrap;word-break:break-word}
  .msg.user{flex-direction:row-reverse}
  .msg.user .bubble{background:#22303c;border-color:#31414f}
  .avatar{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:11px;
    font-weight:700;flex:0 0 auto;margin-top:2px}
  .msg.bot .avatar{background:rgba(76,194,255,.16);color:var(--accent2)}
  .msg.user .avatar{background:rgba(255,122,69,.16);color:var(--accent)}
  .tag{display:inline-block;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
    padding:2px 7px;border-radius:20px;margin-bottom:7px;font-weight:600}
  .tag.aion{background:rgba(84,214,138,.14);color:var(--ok);border:1px solid rgba(84,214,138,.3)}
  .tag.net{background:rgba(255,207,92,.12);color:var(--warn);border:1px solid rgba(255,207,92,.3)}
  .tag.calc{background:rgba(76,194,255,.14);color:var(--accent2);border:1px solid rgba(76,194,255,.32)}
  .meta{font-size:11px;color:var(--muted);margin-top:6px}
  .composer{display:flex;gap:10px;border-top:1px solid var(--line);padding-top:14px;margin-top:auto}
  textarea{resize:none;height:56px;line-height:1.4}
  button{background:linear-gradient(140deg,var(--accent),#d3531f);color:#180b05;border:0;
    border-radius:10px;padding:0 20px;font:inherit;font-weight:700;cursor:pointer;min-width:96px}
  button:disabled{opacity:.55;cursor:default}
  .chips{display:flex;flex-wrap:wrap;gap:7px;margin:2px 0 12px}
  .chip{background:var(--panel2);border:1px solid var(--line);color:var(--muted);border-radius:20px;
    padding:5px 11px;font-size:12px;cursor:pointer}
  .chip:hover{border-color:var(--accent2);color:var(--text)}
  .empty{color:var(--muted);font-size:13px;text-align:center;padding:50px 10px}
  .empty b{color:var(--text)}
  .tiers{border:1px solid var(--line);border-radius:10px;overflow:hidden;
    display:grid;grid-template-columns:repeat(3,1fr)}
  .tier:not(:nth-child(3n)){border-right:1px solid var(--line)}
  .tier:nth-last-child(-n+3){border-bottom:0}
  .target-panel{grid-column:1/-1;grid-row:1;align-self:start}
  .target-panel .tgt-foot{display:flex;gap:14px;align-items:center;margin-top:12px;flex-wrap:wrap}
  .target-panel .tgt-foot input{flex:0 0 240px}
  .target-panel .tgt-foot .note{margin:0;flex:1;min-width:240px}
  .chat{grid-column:2;grid-row:2}
  @media (max-width:900px){.target-panel,.chat{grid-column:auto;grid-row:auto}
    .tiers{grid-template-columns:1fr}.tier{border-right:0 !important}}
  .tier{display:grid;grid-template-columns:1fr auto;align-items:center;gap:10px;width:100%;
    padding:8px 11px;background:transparent;border:0;border-bottom:1px solid var(--line);
    color:var(--muted);font:inherit;font-size:13px;font-weight:500;text-align:left;
    cursor:pointer;border-radius:0;min-width:0}
  .tier:last-child{border-bottom:0}
  .tier:hover{background:var(--panel2);color:var(--text)}
  .tier.on{background:rgba(255,122,69,.12);color:var(--text)}
  .tier b{font-variant-numeric:tabular-nums;font-weight:600;font-size:12.5px}
  .target{margin-top:10px;font-variant-numeric:tabular-nums}
</style>
</head>
<body>
<header>
  <div class="logo">DQ</div>
  <div>
    <h1>Qoolse AI Playground</h1>
    <div class="sub">v3 native BF16 &middot; 135M &times; 9 loop &middot; shared-KV &middot; 8 routed experts (top-2) &middot; kalkulator (algoritma) + retrieval (tabel) + model (sampling)</div>
  </div>
</header>
<main>
  <section class="panel target-panel">
    <h2>Target parameter</h2>
    <div class="tiers" id="tiers"></div>
    <div class="tgt-foot">
      <input type="text" id="pval" placeholder="2. Params di dalam range, mis. 2.5B" inputmode="decimal">
      <div class="note" id="sizeout">Pilih golongan, lalu isi jumlah params di dalam range-nya.</div>
    </div>
  </section>

  <section class="panel">
    <h2>Model</h2>
    <select id="model"></select>
    <div class="stats" id="stats"></div>
    <div class="row"><label>Temperature</label><span class="val" id="tv">0.70</span></div>
    <input type="range" id="temp" min="0.1" max="1.5" step="0.05" value="0.7">
    <div class="row"><label>Top-k</label><span class="val" id="kv">10</span></div>
    <input type="range" id="topk" min="1" max="40" step="1" value="10">
    <div class="row"><label>Panjang jawaban</label><span class="val" id="nv">120</span></div>
    <input type="range" id="maxnew" min="20" max="300" step="10" value="120">
    <div class="switch" style="margin-top:14px">
      <label class="switch"><input type="checkbox" id="calc" checked><span class="track"></span>
        Pakai kalkulator (algoritma)</label>
    </div>
    <div class="switch" style="margin-top:10px">
      <label class="switch"><input type="checkbox" id="aion" checked><span class="track"></span>
        Pakai tabel retrieval (AION)</label>
    </div>
    <div class="note">
      <b>kalkulator (algoritma)</b>: ekspresi aritmatika dihitung oleh
      <code>calculator.py</code> &mdash; tanpa dataset dan tanpa bobot model.
      <b>retrieval (tabel)</b>: jawaban dibaca dari tabel question&rarr;answer, cocok
      hanya jika pertanyaannya sama (tanpa peduli tanda baca/urutan kata).
      <b>model (sampling)</b>: karakter disampel oleh Qoolse AI v3
      <span id="mp">—</span> parameter
      (embedding terfaktorisasi + head tied; stem 2 lapisan local-attention;
      1 shared-KV dipakai seluruh loop; 1 shared expert selalu aktif + 8 routed
      expert top-2; gated delta multi-skala 6 cepat / 6 lambat; inti rekursif
      diputar hingga 9 kali dengan halting gate) &mdash;
      berupa statistik karakter, bukan pemahaman, jadi sering terlihat ngawur.
      Matikan tabel untuk memaksa jalur model.
    </div>
  </section>

  <section class="panel chat">
    <div class="chips" id="chips"></div>
    <div class="log" id="log">
      <div class="empty" id="empty">
        <b>Belum ada percakapan.</b><br>Tanyakan sesuatu, atau klik salah satu contoh di atas.<br>
        Hitungan matematika dijawab kalkulator (algoritma), pertanyaan yang sama
        dengan dataset lewat retrieval (tabel), sisanya disampel model (sampling)
        yang hasilnya statistik karakter, bukan jawaban.
      </div>
    </div>
    <div class="composer">
      <textarea id="prompt" placeholder="Tulis pertanyaan... (Enter untuk kirim, Shift+Enter baris baru)"></textarea>
      <button id="send">Kirim</button>
    </div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let MODELS = [];

const fmt = (n) => n === null || n === undefined ? '—' : Number(n).toLocaleString('en-US');

function stats(m) {
  const pv = Number(m.val_ppl);
  const ppl = Number.isFinite(pv) ? pv.toFixed(2) : '—';
  $('mp').textContent = fmt(m.params);
  $('stats').innerHTML = [
    ['params', fmt(m.params)],
    ['aktif/token', fmt(m.active_params)],
    ['experts', `${m.shared} shared + ${m.experts} routed · top-${m.top_k}`],
    ['depth', `${m.blocks} stem × ${m.repeat} loop`],
    ['shared KV', `${m.d_model}d → ${m.d_kv}d · 1×`],
    ['val ppl', ppl],
    ['ukuran file', m.size_mb + ' MB'],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  $('aion').disabled = m.aion_pairs === 0;
  if (m.aion_pairs === 0) $('aion').checked = false;
}

let TIERS = [];
let TIER = null;

const formatShort = (n) => n >= 1e12 ? (n / 1e12) + 'T' : n >= 1e9 ? (n / 1e9) + 'B' : (n / 1e6) + 'M';

async function loadTiers() {
  const d = await (await fetch('/api/tiers')).json();
  TIERS = d.tiers;
  $('tiers').innerHTML = TIERS.map(t =>
    `<button type="button" class="tier" data-label="${t.label}">` +
    `<span>${t.label}</span><b>${t.range}</b></button>`).join('');
  const pick = (label) => {
    TIER = TIERS.find(t => t.label === label);
    document.querySelectorAll('.tier').forEach(el =>
      el.classList.toggle('on', el.dataset.label === label));
    $('pval').placeholder = `2. Params di dalam ${TIER.range}, mis. ${TIER.label === 'Beyond 1T' ? '2T' : formatShort(TIER.lo)}`;
    $('sizeout').textContent = `pilih jumlah params di dalam ${TIER.range}`;
  };
  document.querySelectorAll('.tier').forEach(el =>
    el.onclick = () => { $('pval').value = ''; pick(el.dataset.label); $('pval').focus(); });
  pick(TIERS[0].label);
  $('pval').oninput = updateSize;
}

async function updateSize() {
  const raw = $('pval').value.trim();
  if (!raw || !TIER) return;
  const url = `/api/size?tier=${encodeURIComponent(TIER.label)}&params=${encodeURIComponent(raw)}`;
  const d = await (await fetch(url)).json();
  $('sizeout').textContent = d.error ? d.error
    : `target ${d.target_text} (${d.tier}) → d_model ${fmt(d.d_model)} · file ~${d.file_text} · latent FP32 ${d.ram_text}`;
}

async function loadModels() {
  const res = await fetch('/api/models');
  MODELS = await res.json();
  $('model').innerHTML = MODELS.map(m =>
    `<option value="${m.name}">${m.name}${m.trained === 'true' ? '' : ' (untrained)'}</option>`).join('');
  $('model').onchange = () => stats(MODELS.find(m => m.name === $('model').value));
  stats(MODELS[0]);
  $('chips').innerHTML = ['4 - 4 + 7 = ?', 'lima tambah lima berapa', '12 kali 3',
    'apa itu fotosintesis?', 'ibukota jawa barat', 'jelaskan siklus air']
    .map(t => `<span class="chip">${t}</span>`).join('');
  document.querySelectorAll('.chip').forEach(c =>
    c.onclick = () => { $('prompt').value = c.textContent; $('prompt').focus(); });
  loadTiers();
}

function bubble(role, text, source, ms) {
  const el = document.createElement('div');
  el.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  const tag = source === 'aion'
    ? '<div class="tag aion">retrieval (tabel)</div>'
    : (source === 'calculator'
      ? '<div class="tag calc">kalkulator (algoritma)</div>'
      : (source === 'network' ? '<div class="tag net">model (sampling)</div>' : ''));
  const meta = ms !== undefined ? `<div class="meta">${(ms / 1000).toFixed(2)} s</div>` : '';
  el.innerHTML = `<div class="avatar">${role === 'user' ? 'YOU' : 'DQ'}</div>
    <div class="bubble">${tag}<span class="txt"></span>${meta}</div>`;
  el.querySelector('.txt').textContent = text;
  $('empty')?.remove();
  $('log').appendChild(el);
  $('log').scrollTop = $('log').scrollHeight;
  return el;
}

async function send() {
  const prompt = $('prompt').value.trim();
  if (!prompt) return;
  const model = $('model').value;
  const useAion = $('aion').checked;
  const useCalc = $('calc').checked;
  $('prompt').value = '';
  bubble('user', prompt);
  const pending = document.createElement('div');
  pending.className = 'msg bot';
  pending.innerHTML = '<div class="avatar">DQ</div><div class="bubble"><span class="txt">…</span></div>';
  $('log').appendChild(pending);
  $('log').scrollTop = $('log').scrollHeight;
  $('send').disabled = true;
  try {
    const t0 = performance.now();
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model, prompt, use_aion: useAion, use_calc: useCalc,
        temp: parseFloat($('temp').value),
        top_k: parseInt($('topk').value, 10),
        max_new: parseInt($('maxnew').value, 10),
      }),
    });
    const data = await res.json();
    pending.remove();
    if (!res.ok) { bubble('bot', 'Error: ' + (data.error || res.status)); }
    else { bubble('bot', data.reply || '(kosong)', data.source, performance.now() - t0); }
  } catch (e) {
    pending.remove();
    bubble('bot', 'Error: ' + e);
  } finally {
    $('send').disabled = false;
    $('prompt').focus();
  }
}

$('send').onclick = send;
$('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('temp').oninput = () => $('tv').textContent = parseFloat($('temp').value).toFixed(2);
$('topk').oninput = () => $('kv').textContent = $('topk').value;
$('maxnew').oninput = () => $('nv').textContent = $('maxnew').value;
loadModels();
</script>
</body>
</html>
"""


def _size_text(mb):
    if mb >= 1024 * 1024:
        return f'{mb / 1024 / 1024:.2f} TB'
    if mb >= 1024:
        return f'{mb / 1024:.2f} GB'
    return f'{mb:.0f} MB'


class Handler(BaseHTTPRequestHandler):
    server_version = 'QOOLSE/1.0'

    def log_message(self, fmt, *args):
        sys.stderr.write("[http] " + fmt % args + "\n")

    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, PAGE, 'text/html')
        elif self.path == '/api/models':
            self._send(200, json.dumps(ENGINES.summary()))
        elif self.path.startswith('/api/tiers'):
            self._send(200, json.dumps({'tiers': [
                {'label': t[0], 'lo': t[1], 'hi': t[2], 'range': tier_range_text(t)}
                for t in PARAM_TIERS]}))
        elif self.path.startswith('/api/size'):
            q = parse_qs(urlparse(self.path).query)
            tier = q.get('tier', ['Small'])[0]
            raw = q.get('params', [''])[0]
            try:
                n = resolve_target_params(tier, raw)
            except (ValueError, TypeError) as e:
                self._send(200, json.dumps({'error': str(e)}))
                return
            base_params = 135000806      # v3 blueprint: 135.0 M at d_model 768
            base_d = QOOLSE_V3_CONFIG['d_model']
            d = int(round((n / (base_params / base_d ** 2)) ** 0.5 / 128) * 128)
            file_mb = n * 2 / 1024 / 1024
            ram_gb = n * 4 / 1024 ** 3
            self._send(200, json.dumps({
                'tier': find_tier(tier)[0], 'range': tier_range_text(find_tier(tier)),
                'target': n, 'target_text': format_params(n),
                'd_model': max(128, d),
                'file_mb': round(file_mb, 1), 'file_text': _size_text(file_mb),
                'ram_gb': round(ram_gb, 2), 'ram_text': _size_text(ram_gb * 1024),
            }))
        elif self.path == '/favicon.ico':
            self._send(204, b'', 'image/x-icon')
        else:
            self._send(404, json.dumps({'error': 'not found'}))

    def do_POST(self):
        if self.path != '/api/chat':
            self._send(404, json.dumps({'error': 'not found'}))
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            payload = json.loads(self.rfile.read(length) or b'{}')
            prompt = str(payload.get('prompt', ''))[:1000]
            if not prompt.strip():
                self._send(400, json.dumps({'error': 'empty prompt'}))
                return
            session = ENGINES.get(payload.get('model', ''))
            t0 = time.time()
            with INFER_LOCK:
                text, source, detail = session.answer(
                    prompt,
                    max_new=max(1, min(int(payload.get('max_new', 120)), 400)),
                    temp=max(0.05, min(float(payload.get('temp', 0.7)), 2.0)),
                    top_k=max(1, min(int(payload.get('top_k', 10)), 100)),
                    use_aion=bool(payload.get('use_aion', True)),
                    use_calc=bool(payload.get('use_calc', True)),
                )
            self._send(200, json.dumps({
                'reply': text, 'source': source,
                'seconds': round(time.time() - t0, 3), 'detail': detail,
            }))
        except KeyError as e:
            self._send(404, json.dumps({'error': f'model not found: {e}'}))
        except Exception as e:  # keep the UI alive, report the message
            self._send(500, json.dumps({'error': f'{type(e).__name__}: {e}'}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--dir', default=HERE)
    args = ap.parse_args()

    threads = max(1, min(8, (os.cpu_count() or 4) // 2))
    torch.set_num_threads(threads)
    print(f"[playground] torch threads: {threads}")

    global ENGINES
    ENGINES = Engines(args.dir)
    if not ENGINES.paths():
        print(f"ERROR: no .gguf files in {args.dir}")
        sys.exit(1)
    print(f"[playground] models: {', '.join(ENGINES.paths())}")
    print(f"[playground] serving http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
