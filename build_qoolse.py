"""
Build QoolseAI-base.gguf (untrained) and QoolseAI-trainer.gguf (trained)
for the v3 (1B-Loop Native BF16) architecture.

Pipeline:
  1. Load every parquet file in dataset/, keep one row per normalised question.
  2. Deterministic 95/5 train/validation split.
  3. Train full epochs (default 5) at full depth, with a short depth-1 warmup,
     cosine LR decay, and validation perplexity after every epoch. Weights are
     native BF16 (no ternary packing); Muon on 2-D matrices, AdamW elsewhere.
  4. Keep the best-validation weights, build the AION table from TRAIN ONLY,
     and write both containers with the measured numbers in metadata.

    python build_qoolse.py --base-only      # v3 blueprint, 135M, random init
    python build_qoolse.py --epochs 5 --batch 16

Requires: torch (CPU is enough), pyarrow.
"""
import argparse
import copy
import glob
import hashlib
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from qoolse_model import (CONFIGS, PARAM_TIERS, QOOLSE_V3_CONFIG, QoolseModel,
                            auto_target_params, available_ram, config_for_params,
                            find_tier, format_params, normalize_aion_key,
                            resolve_target_params, tier_for_params, tier_range_text)

OUT_DIR = HERE
DATASET_DIR = os.path.join(HERE, 'dataset')
SEED = 42
VAL_FRACTION = 0.05
WARMUP_STEPS = 100

# Char vocabulary (must match CONFIGS['sandbox']['vocab_size'])
vocab_chars = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-_/()[]{}@#$%&*+=<>~`\n"
    "éèêëàâçïôùûüñáéíóúàèìòù"
)
stoi = {c: i for i, c in enumerate(vocab_chars)}
stoi['<pad>'] = len(stoi)
stoi['<eos>'] = len(stoi)
# Native tool-call tokens (v2): the model can learn to emit these at pre-training
# time so tool routing does not depend on prompt parsing afterwards.
for tok in ('<|call|>', '<|calc|>', '<|code|>', '<|search|>'):
    stoi[tok] = len(stoi)
itos = {i: c for c, i in stoi.items()}

pad_id = stoi['<pad>']
eos_id = stoi['<eos>']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=4e-3)
    ap.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--params', type=float, default=0,
                    help='step 2: target parameter count inside the chosen class, '
                         'e.g. 2.5e9 (0 = ask, or auto when not interactive)')
    ap.add_argument('--tier', default=None,
                    help='step 1: size class, e.g. Small, Compact, Medium, Large, '
                         'Very Large, Frontier, Large-scale, Ultra-scale, "Beyond 1T"')
    ap.add_argument('--auto', action='store_true',
                    help='skip the size-class picker and use the largest count that '
                         'fits free RAM')
    ap.add_argument('--v3', action='store_true',
                    help='force the locked v3 blueprint model (135M) without asking')
    ap.add_argument('--list-tiers', action='store_true',
                    help='print the size classes and exit')
    ap.add_argument('--base-only', action='store_true',
                    help='write QoolseAI-base.gguf (random init) and stop, no training')
    return ap.parse_args()


def find_col(columns, options):
    for o in options:
        if o in columns:
            return o
    return None


def load_dedup_pairs():
    """One row per normalised question, first occurrence wins."""
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, '*.parquet')))
    if not paths:
        print(f"ERROR: no parquet files in {DATASET_DIR}")
        sys.exit(1)
    rows = []
    files = 0
    for path in paths:
        table = pq.read_table(path)
        cols = table.column_names
        c_in = find_col(cols, ('input', 'question', 'prompt', 'instruction'))
        c_out = find_col(cols, ('output', 'answer', 'response', 'completion'))
        if c_in is None or c_out is None:
            print(f"  [skip] {os.path.basename(path)}: columns {cols}")
            continue
        d = table.select([c_in, c_out]).to_pydict()
        rows.extend(zip(d[c_in], d[c_out]))
        files += 1
    print(f"  Read {files} parquet files, {len(rows):,} rows")

    seen = set()
    pairs = []
    for inp, out in rows:
        if not inp or not out:
            continue
        key = normalize_aion_key(str(inp), max_len=10_000)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((str(inp), str(out)))
    print(f"  Deduplicated: {len(rows) - len(pairs):,} rows removed, "
          f"{len(pairs):,} unique questions left")
    return pairs


def encode_pair(inp, out, max_seq):
    """(inputs, targets); -100 on prompt tokens so only answers are scored."""
    prompt_ids = [stoi[c] for c in inp if c in stoi]
    answer_ids = [stoi[c] for c in (' ' + out) if c in stoi]
    ids = (prompt_ids + answer_ids + [eos_id])[:max_seq]
    if len(ids) < 2:
        return None
    if ids[-1] != eos_id:
        ids[-1] = eos_id
    xs = ids[:-1]
    targets = [-100 if j < len(prompt_ids) else ids[j] for j in range(1, len(ids))]
    return xs, targets


def make_batch(samples):
    L = max(len(xs) for xs, _ in samples)
    xb = torch.full((len(samples), L), pad_id, dtype=torch.long)
    yb = torch.full((len(samples), L), -100, dtype=torch.long)
    for i, (xs, tg) in enumerate(samples):
        xb[i, :len(xs)] = torch.tensor(xs, dtype=torch.long)
        yb[i, :len(tg)] = torch.tensor(tg, dtype=torch.long)
    return xb, yb


@torch.no_grad()
def evaluate(model, data, batch_size=32):
    """Token-weighted CE over answer tokens -> (loss, perplexity, tokens)."""
    model.eval()
    total_ce = 0.0
    total_tok = 0
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        xb, yb = make_batch(chunk)
        logits, _ = model(xb)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1),
                             ignore_index=-100, reduction='sum')
        n = int((yb != -100).sum())
        total_ce += float(ce)
        total_tok += n
    loss = total_ce / max(total_tok, 1)
    return loss, math.exp(min(loss, 20)), total_tok


def lr_at(step, total_steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total_steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Muon(torch.optim.Optimizer):
    """Momentum Orthogonalized by Newton-Schulz, for matrix-shaped weights.

    Orthogonalising the update before the ternary quantisation keeps the row
    distributions even, which is what keeps {-1, 0, 1} rounding from collapsing
    whole rows. Embeddings, routers and norms stay on AdamW.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                                            nesterov=nesterov, ns_steps=ns_steps))

    @staticmethod
    def _orthogonalize(g, steps):
        a, b, c = 3.4445, -4.7750, 2.0315
        x = g.float()
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.mT
        x = x / (x.norm() + 1e-7)
        for _ in range(steps):
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
        if transposed:
            x = x.mT
        return x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                shape = p.shape
                g = p.grad if p.ndim == 2 else p.grad.reshape(-1, shape[-1])
                buf = self.state[p].setdefault('momentum_buffer', torch.zeros_like(g))
                buf.mul_(group['momentum']).add_(g)
                upd = g.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                upd = self._orthogonalize(upd, group['ns_steps'])
                scale = max(1.0, shape[-2] / shape[-1]) ** 0.5
                p.add_(upd.reshape(shape).to(p.dtype),
                       alpha=-group['lr'] * scale)
        return loss


def show_tiers():
    print("\n  Size classes (pick one, then the exact target inside its range):")
    for i, tier in enumerate(PARAM_TIERS, 1):
        print(f"    {i}. {tier[0]:<12} {tier_range_text(tier)}")


def ask_tier(default=None):
    show_tiers()
    while True:
        hint = f" [Enter = {default[0]}]" if default else ''
        raw = input(f"\n  class (name or 1-{len(PARAM_TIERS)}){hint}: ").strip()
        if not raw and default:
            return default
        try:
            return find_tier(raw)
        except ValueError as e:
            print(f"  {e}")


def ask_params(tier):
    label, lo, hi = tier
    example = format_params(lo)
    while True:
        raw = input(f"  target params in {label} "
                    f"({tier_range_text(tier)}) [e.g. {example}]: ").strip()
        if not raw:
            print("  a target is required")
            continue
        try:
            return resolve_target_params(label, raw)
        except ValueError as e:
            print(f"  {e}")


def choose_target(args):
    """Two-step target selection: size class first, then the count in range."""
    tier = find_tier(args.tier) if args.tier else None
    if args.params > 0:
        target = int(args.params)
        if tier and not (target >= tier[1] and (tier[2] is None or target < tier[2])):
            raise SystemExit(f"  {format_params(target)} is outside {tier[0]} "
                             f"({tier_range_text(tier)})")
        label = tier[0] if tier else (tier_for_params(target) or (None,))[0]
        return target, f"from --params: {format_params(target)}" + (
            f" in {label}" if label else '')

    if args.auto or not sys.stdin.isatty():
        target = auto_target_params()
        cls = tier_for_params(target)
        return target, ('auto: half of free RAM, minus 384 MB reserve' +
                        (f" -> {cls[0]}" if cls else ' -> below Small'))

    print("\n" + "=" * 64)
    print("Target size")
    print("=" * 64)
    tier = ask_tier(tier)
    target = ask_params(tier)
    return target, f"{tier[0]} / {format_params(target)} (picked interactively)"


def main():
    args = parse_args()
    if args.list_tiers:
        print("Qoolse AI size classes")
        show_tiers()
        return
    if args.threads:
        torch.set_num_threads(args.threads)

    free = available_ram()
    if args.params > 0 or args.tier:
        target, how = choose_target(args)
        cfg = config_for_params(target)
    elif args.auto:
        target = auto_target_params()
        cfg = config_for_params(target)
        how = 'auto: half of free RAM, minus 384 MB reserve'
    elif args.v3 or not sys.stdin.isatty():
        # The blueprint's own model: 135.0 M physical, 9 loops, 1.015 B unrolled.
        target = 0
        cfg = dict(CONFIGS['v3'])
        how = 'v3 blueprint (locked 135M)'
    else:
        # Interactive: pick the size class first, then the count inside its range.
        target, how = choose_target(args)
        cfg = config_for_params(target)

    print("=" * 64)
    print("Qoolse AI build" + (" (base only)" if args.base_only else " + training"))
    print("=" * 64)
    print(f"  size target      {how}: {target:,} params" if target else
          f"  size target      {how}")
    print(f"  free RAM         {(free or 0)/1024/1024/1024:.1f} GB")
    need_gb = target * 4 / 1024 ** 3
    if target and free and need_gb > free / 1024 ** 3:
        print(f"  RAM warning      {need_gb:.1f} GB latent FP32 needed, "
              f"{free/1024**3:.1f} GB free -- this build will likely be killed")
    print(f"  d_model          {cfg['d_model']} (heads {cfg['n_heads']}x{cfg['head_dim']}, "
          f"kv {cfg['n_kv_heads']}x{cfg['d_kv'] // cfg['n_kv_heads']})")
    print(f"  embedding        factorized {cfg['d_emb']}d, tied LM head (V {cfg['vocab_size']})")
    print(f"  loops            {cfg['n_loops']} recursive passes (min {cfg['min_loops']}), "
          f"halting gate threshold {cfg['halt_threshold']}")
    print(f"  stem             {cfg['n_stem_layers']} local-attn layers, "
          f"window {cfg['stem_window']}, ffn {cfg['stem_ffn']}")
    print(f"  shared KV        d_kv {cfg['d_kv']}, {cfg['kv_block']}-token blocks, "
          f"top-{cfg['top_blocks']} + {cfg['local_window']} local")
    print(f"  experts          shared {cfg['shared_hidden']} hidden + "
          f"{cfg['n_experts']} routed x {cfg['expert_hidden']}, top-{cfg['top_k']}")
    print(f"  weights          native BF16 (FP32 compute on CPU), no ternary packing")
    print(f"  vocab            {cfg['vocab_size']} BPE slots (local char tokenizer uses {len(stoi)})")
    print(f"  max_seq          {cfg['max_seq']}")
    if not args.base_only:
        print(f"  epochs           {args.epochs}, batch {args.batch}, lr {args.lr}")
    print(f"  threads          {torch.get_num_threads()}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = random.Random(SEED)

    model = QoolseModel(cfg)
    n_params = model.count_params()
    n_active = model.count_active_params()
    print(f"\n[Model] {n_params:,} params ({model.storage_bytes()/1024/1024:.2f} MB BF16)")
    print(f"        {n_active:,} active per token ({n_active/n_params*100:.1f}%), "
          f"{cfg['n_loops']} loops = {n_active*cfg['n_loops']:,} effective/token")

    ORIG_BLOCKS, ORIG_REPEAT = cfg['n_stem_layers'], cfg['n_loops']

    # ---------------------------------------------------------------- data
    print("\n" + "=" * 64)
    print("Dataset")
    print("=" * 64)
    pairs = load_dedup_pairs()
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"  Split: {len(train_pairs):,} train / {len(val_pairs):,} validation "
          f"({args.val_fraction*100:.0f}%)")

    max_seq = cfg['max_seq']
    train_data = [d for d in (encode_pair(i, o, max_seq) for i, o in train_pairs) if d]
    val_data = [d for d in (encode_pair(i, o, max_seq) for i, o in val_pairs) if d]
    print(f"  Encoded: {len(train_data):,} train / {len(val_data):,} validation samples")
    lens = [len(x) for x, _ in train_data]
    print(f"  Length: mean {sum(lens)/len(lens):.1f}, max {max(lens)} chars")

    steps_per_epoch = math.ceil(len(train_data) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps: {steps_per_epoch:,}/epoch x {args.epochs} = {total_steps:,} "
          f"({total_steps*args.batch/len(train_data):.2f} epochs worth of samples)")

    # ---------------------------------------------------------------- base
    base_aion = {}
    base_path = os.path.join(OUT_DIR, 'QoolseAI-base.gguf')
    print("\n[Save] QoolseAI-base.gguf (random init, empty AION)")
    model.save_gguf(base_path, stoi=stoi, itos=itos, eos_id=eos_id,
                    aion_engram=base_aion, architecture='qoolse_v3_loop',
                    extra_meta={'trained': 'false', 'val_perplexity': 'n/a',
                                'size_target': how})
    if args.base_only:
        print("\nDone (base only, no training run).")
        return

    # ------------------------------------------------------------- training
    print("\n" + "=" * 64)
    print(f"Training {args.epochs} epochs")
    print("=" * 64)
    # Muon for matrix weights (2-D, and 3-D expert stacks flattened to a
    # matrix); AdamW for embeddings, routers, norms and other vectors.
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        # 2-D matrices only; embeddings, routers, LoRAs, convs and vectors on AdamW.
        if p.ndim == 2 and not any(k in name for k in ('tok', 'router', 'lora', 'conv')):
            muon_params.append(p)
        else:
            adam_params.append(p)
    muon_lr_mult = 5.0
    muon = Muon(muon_params, lr=args.lr * muon_lr_mult)
    opt = torch.optim.AdamW(adam_params, lr=args.lr,
                            weight_decay=cfg['weight_decay'])
    print(f"  optimizers       Muon {len(muon_params)} matrices (lr x{muon_lr_mult}), "
          f"AdamW {len(adam_params)} tensors")
    t0 = time.time()
    history = []
    best = {'ppl': float('inf'), 'epoch': 0, 'state': None}
    step = 0

    val_loss0, val_ppl0, val_tok = evaluate(model, val_data)
    print(f"[Epoch 0] validation before training: loss {val_loss0:.4f}, "
          f"ppl {val_ppl0:.2f} ({val_tok:,} tokens)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0
        epoch_t0 = time.time()
        for start in range(0, len(order), args.batch):
            idxs = order[start:start + args.batch]
            samples = [train_data[i] for i in idxs]
            xb, yb = make_batch(samples)
            lr = lr_at(step, total_steps, args.lr, WARMUP_STEPS)
            for g in opt.param_groups:
                g['lr'] = lr
            for g in muon.param_groups:
                g['lr'] = lr * muon_lr_mult
            # Short warmup at depth 1, then the full 4x3 depth.
            model.set_active_depth(1, 1) if step < WARMUP_STEPS else \
                model.set_active_depth(ORIG_BLOCKS, ORIG_REPEAT)
            muon.zero_grad()
            opt.zero_grad()
            _, loss, _ = model(xb, yb, return_aux=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step()
            opt.step()
            run_loss += loss.item()
            run_n += 1
            step += 1
            if step % 200 == 0 or step == total_steps:
                done = (step - (epoch - 1) * steps_per_epoch) / steps_per_epoch
                print(f"  epoch {epoch} {done*100:5.1f}%  step {step:,}/{total_steps:,}  "
                      f"loss {run_loss/run_n:.4f}  lr {lr:.2e}  "
                      f"{time.time()-epoch_t0:.0f}s")
                run_loss, run_n = 0.0, 0

        train_loss, train_ppl, train_tok = evaluate(model, train_data)
        val_loss, val_ppl, val_tok = evaluate(model, val_data)
        history.append((epoch, train_loss, train_ppl, val_loss, val_ppl))
        print(f"[Epoch {epoch}] train loss {train_loss:.4f} ppl {train_ppl:.2f} | "
              f"val loss {val_loss:.4f} ppl {val_ppl:.2f} | {time.time()-epoch_t0:.0f}s")
        if val_ppl < best['ppl']:
            best = {'ppl': val_ppl, 'epoch': epoch,
                    'state': copy.deepcopy(model.state_dict())}
            print(f"          new best validation ppl {val_ppl:.2f}")

    train_time = time.time() - t0
    model.load_state_dict(best['state'])
    model.eval()
    train_loss, train_ppl, train_tok = evaluate(model, train_data)
    val_loss, val_ppl, val_tok = evaluate(model, val_data)
    print(f"\n[Training done in {train_time/60:.1f} min, {total_steps:,} steps]")
    print(f"[Best epoch {best['epoch']}] train ppl {train_ppl:.2f} | "
          f"val ppl {val_ppl:.2f}")

    # -------------------------------------------------------------- AION
    print(f"\n[Build AION engram from {len(train_pairs):,} train questions only]")
    trainer_aion = {}
    for inp, out in train_pairs:
        key = normalize_aion_key(inp)
        if key and key not in trainer_aion:
            trainer_aion[key] = out
    print(f"  AION keys: {len(trainer_aion):,} (validation questions excluded)")

    assert model.count_params() == n_params
    trainer_path = os.path.join(OUT_DIR, 'QoolseAI-trainer.gguf')
    print("\n[Save] QoolseAI-trainer.gguf")
    model.save_gguf(
        trainer_path, stoi=stoi, itos=itos, eos_id=eos_id,
        aion_engram=trainer_aion, architecture='qoolse_v3_loop',
        extra_meta={
            'trained': 'true',
            'train_pairs': str(len(train_pairs)),
            'val_pairs': str(len(val_pairs)),
            'epochs': str(args.epochs),
            'batch_size': str(args.batch),
            'train_steps': str(total_steps),
            'best_epoch': str(best['epoch']),
            'active_params_per_token': str(n_active),
            'train_loss': f'{train_loss:.4f}',
            'train_perplexity': f'{train_ppl:.2f}',
            'val_loss': f'{val_loss:.4f}',
            'val_perplexity': f'{val_ppl:.2f}',
            'val_tokens': str(val_tok),
            'train_minutes': f'{train_time/60:.1f}',
        })

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  epoch  train_loss  train_ppl  val_loss  val_ppl")
    for e, tl, tp, vl, vp in history:
        print(f"  {e:5d}  {tl:10.4f}  {tp:9.2f}  {vl:8.4f}  {vp:7.2f}")
    for label, p, aion in (('base', base_path, base_aion),
                           ('trainer', trainer_path, trainer_aion)):
        print(f"  {label:8s} {os.path.getsize(p):,} bytes  AION {len(aion):,}  "
              f"sha256 {file_hash(p)[:16]}")
    print("\nDone.")


if __name__ == '__main__':
    main()
