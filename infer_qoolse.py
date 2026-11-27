"""
Minimal CPU inference for Qoolse AI containers (qoolse_v3_loop).

These files are GGUF-shaped but use a custom architecture, so llama.cpp and
ollama will refuse them. Load them here:

    python infer_qoolse.py --model QoolseAI-trainer.gguf --prompt "apa itu fotosintesis"
    python infer_qoolse.py --model QoolseAI-trainer.gguf            # interactive

Three paths, tried in this order:
  1. Calculator       -> the prompt is a math expression; `calculator.py`
     evaluates it exactly. No dataset, no weights, no sampling.
  2. AION engram hit  -> the answer is read from the stored question->answer
     table (this is retrieval, not generation). The prompt must match a stored
     question exactly after normalisation, or with the same set of words; a
     shared prefix is not enough, so free questions fall through to path 3.
  3. AION miss        -> the v3 model (135M BF16, 9 recursive loops) continues
     the text with a streaming cache. Expect character statistics, not answers.
"""
import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import calculator
from qoolse_model import load_from_gguf, normalize_aion_key


_TOKEN_RE = re.compile(r'[0-9a-z]+')


def aion_tokens(key):
    """Order-insensitive word tokens, used to match punctuation variants only."""
    return tuple(sorted(_TOKEN_RE.findall(key)))


def build_aion_index(engram):
    """Map each stored question's token set to its key (first key wins)."""
    index = {}
    for k in engram:
        toks = aion_tokens(k)
        if toks:
            index.setdefault(toks, k)
    return index


def aion_lookup(prompt, engram, index=None):
    """Return (answer, matched_key, match_kind) or (None, None, None).

    A hit requires the same normalised key, or exactly the same set of words
    (so 'apa itu fotosintesis' matches 'apa itu fotosintesis?'). Merely sharing
    a prefix, a word or a suffix is no longer a hit: that used to hand free
    questions an unrelated stored answer.
    """
    key = normalize_aion_key(prompt)
    if key in engram:
        return engram[key], key, 'exact'
    toks = aion_tokens(key)
    if not toks:
        return None, None, None
    if index is None:
        index = build_aion_index(engram)
    hit = index.get(toks)
    if hit is not None:
        return engram[hit], hit, 'token_set'
    return None, None, None


class Session:
    """A loaded container plus its tokenizer and retrieval table."""

    def __init__(self, path):
        self.path = path
        self.model, self.meta = load_from_gguf(path)
        self.stoi = json.loads(self.meta.get('tokenizer_stoi', '{}'))
        self.itos = {int(k): v for k, v in
                     json.loads(self.meta.get('tokenizer_itos', '{}')).items()}
        self.eos_id = int(self.meta.get('eos_id', -1))
        self.engram = json.loads(self.meta.get('aion_engram', '{}'))
        self.aion_index = build_aion_index(self.engram)
        self.pad_id = self.stoi.get('<pad>')

    @property
    def n_params(self):
        return self.model.count_params()

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id or i == self.pad_id:
                break
            out.append(self.itos.get(i, ''))
        return ''.join(out)

    def answer(self, prompt, max_new=120, temp=0.7, top_k=10,
               use_aion=True, entrocut=0.0, use_calc=True):
        """Returns (text, source, detail); source is 'calculator', 'aion'
        (retrieval) or 'network' (sampled characters)."""
        if use_calc:
            res = calculator.solve(prompt)
            if res is not None:
                if 'error' in res:
                    return res['error'], 'calculator', {'expr': res['expr']}
                return res['text'], 'calculator', {'expr': res['expr']}
        if self.engram and use_aion:
            hit, key, match = aion_lookup(prompt, self.engram, self.aion_index)
            if hit is not None:
                return hit, 'aion', {'key': key, 'match': match}
        ids = self.encode(prompt) or self.encode('a')
        if len(ids) >= self.model.cfg['max_seq']:
            ids = ids[-(self.model.cfg['max_seq'] - max_new - 1):]
        gen = self.model.generate(torch.tensor([ids], dtype=torch.long),
                                  max_new=max_new, temp=temp, top_k=top_k,
                                  stop_id=self.eos_id if self.eos_id >= 0 else None,
                                  entrocut=entrocut)
        return self.decode(gen[0, len(ids):].tolist()), 'network', {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(HERE, 'model', 'untrained',
                                                    'QoolseAI-base.gguf'))
    ap.add_argument('--prompt', default=None)
    ap.add_argument('--max-new', type=int, default=120)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--entrocut', type=float, default=0.0,
                    help='stop when max next-char probability > this (0 = off)')
    ap.add_argument('--no-aion', action='store_true',
                    help='skip the retrieval table and sample from the network')
    ap.add_argument('--no-calc', action='store_true',
                    help='skip the arithmetic tool, even for math prompts')
    args = ap.parse_args()

    s = Session(args.model)
    c = s.model.cfg
    print(f"[load] {os.path.basename(args.model)}: {s.n_params:,} params "
          f"({s.model.count_active_params():,} active/token, native BF16), "
          f"vocab {c['vocab_size']}, "
          f"1 shared + {c['n_experts']} routed experts top-{c['top_k']}, "
          f"{c['n_loops']} recursive loops (min {c['min_loops']}), "
          f"d_model {c['d_model']} / d_kv {c['d_kv']}, "
          f"AION pairs {len(s.engram):,}, "
          f"val ppl {s.meta.get('val_perplexity', 'n/a')}")

    def ask(prompt):
        text, source, detail = s.answer(
            prompt, max_new=args.max_new, temp=args.temp, top_k=args.top_k,
            use_aion=not args.no_aion, entrocut=args.entrocut,
            use_calc=not args.no_calc)
        if source == 'calculator':
            print(f"[calc] evaluated by calculator.py (expr={detail.get('expr')!r}) - "
                  f"not model output")
        elif source == 'aion':
            print(f"[aion] retrieved (key={detail.get('key')!r}) - "
                  f"not generated by the network")
        else:
            print(f"[net] sampled from the Qoolse AI v3 char model "
                  f"({s.model.count_active_params():,} active params/token)")
        return text

    if args.prompt:
        print((ask(args.prompt) or '(empty)').strip())
        return

    print("Interactive. Ctrl+C to quit.")
    while True:
        try:
            prompt = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt:
            print((ask(prompt) or '(empty)').strip())


if __name__ == '__main__':
    main()
