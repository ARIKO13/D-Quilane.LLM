"""
D'QUILANE Playground — local web UI to try the trained containers.

    python playground.py            # http://127.0.0.1:8000
    python playground.py --port 8080

Stdlib HTTP server + torch only (no extra dependencies). Each model is loaded
lazily and cached. Responses are labelled with where they came from:
`AION` = retrieved from the prompt->answer table, `network` = sampled from the
215K-parameter character model.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from infer_dquilane import Session

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
                'vocab': s.model.cfg['vocab_size'],
                'blocks': s.model.cfg['n_blocks'],
                'repeat': s.model.cfg['n_repeat'],
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
<title>D'QUILANE Playground</title>
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
  .chat{display:flex;flex-direction:column;min-height:560px}
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
</style>
</head>
<body>
<header>
  <div class="logo">DQ</div>
  <div>
    <h1>D'QUILANE Playground</h1>
    <div class="sub">215K-parameter character model &middot; AION retrieval + network sampling</div>
  </div>
</header>
<main>
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
      <label class="switch"><input type="checkbox" id="aion" checked><span class="track"></span>
        Pakai tabel AION (retrieval)</label>
    </div>
    <div class="note">
      Jawaban berlabel <b>AION</b> diambil dari tabel question&rarr;answer, bukan dari model.
      Matikan AION untuk melihat kemampuan jaringan sebenarnya.
    </div>
  </section>

  <section class="panel chat">
    <div class="chips" id="chips"></div>
    <div class="log" id="log">
      <div class="empty" id="empty">
        <b>Belum ada percakapan.</b><br>Tanyakan sesuatu, atau klik salah satu contoh di atas.<br>
        Soal dari dataset akan dijawab lewat AION; pertanyaan bebas akan disampel oleh model.
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
  const ppl = m.val_ppl ? Number(m.val_ppl).toFixed(2) : '—';
  $('stats').innerHTML = [
    ['params', fmt(m.params)],
    ['vocab', m.vocab],
    ['depth', `${m.blocks}×${m.repeat}`],
    ['val ppl', ppl],
    ['AION pairs', fmt(m.aion_pairs)],
    ['size', m.size_mb + ' MB'],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  $('aion').disabled = m.aion_pairs === 0;
  if (m.aion_pairs === 0) $('aion').checked = false;
}

async function loadModels() {
  const res = await fetch('/api/models');
  MODELS = await res.json();
  $('model').innerHTML = MODELS.map(m =>
    `<option value="${m.name}">${m.name}${m.trained === 'true' ? '' : ' (untrained)'}</option>`).join('');
  $('model').onchange = () => stats(MODELS.find(m => m.name === $('model').value));
  stats(MODELS[0]);
  $('chips').innerHTML = ['4 - 4 + 7 = ?', 'apa itu fotosintesis?', 'ibukota jawa barat',
    'sebutkan alat musik tradisional', 'jelaskan siklus air']
    .map(t => `<span class="chip">${t}</span>`).join('');
  document.querySelectorAll('.chip').forEach(c =>
    c.onclick = () => { $('prompt').value = c.textContent; $('prompt').focus(); });
}

function bubble(role, text, source, ms) {
  const el = document.createElement('div');
  el.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  const tag = source === 'aion'
    ? '<div class="tag aion">AION retrieval</div>'
    : (source === 'network' ? '<div class="tag net">network sample</div>' : '');
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
        model, prompt, use_aion: useAion,
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


class Handler(BaseHTTPRequestHandler):
    server_version = 'DQUILANE/1.0'

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
