"""
Mini BPE Tokenizer — implementasi sederhana dari nol.
Tidak pakai library eksternal selain regex dan collections.
Cocok untuk dataset kecil, vocab 256-1024 token.
"""
import re
import json
from collections import Counter, defaultdict
from pathlib import Path


PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# fallback regex yang lebih simple (tidak butuh \p{L} unicode property)
PAT_SIMPLE = r"\s+|\w+|\S"


def _get_pairs(word):
    """Return set of symbol pairs in a word (tuple of symbols)."""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class MiniBPETokenizer:
    def __init__(self):
        self.merges = {}      # pair -> new_id
        self.vocab = {}       # id -> bytes (decoded)
        self._id_to_token = {}
        self._token_to_id = {}

    # ---------- training ----------
    def train(self, text: str, target_vocab: int = 512, verbose: bool = True):
        # pre-tokenize pakai regex simple (split whitespace + word boundary)
        words_raw = re.findall(PAT_SIMPLE, text)
        # represent each word sebagai list of byte-ids (0..255)
        word_freqs = Counter(words_raw)
        words = {w: list(w.encode("utf-8")) for w in word_freqs}

        # init vocab: 0..255 = byte literal
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        num_merges = target_vocab - 256

        for i in range(num_merges):
            # hitung frekuensi pair
            pair_counts = Counter()
            for w, syms in words.items():
                for p in _get_pairs(syms):
                    pair_counts[p] += word_freqs[w]
            if not pair_counts:
                break
            best, _ = pair_counts.most_common(1)[0]
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # apply merge ke semua word
            new_words = {}
            for w, syms in words.items():
                new_words[w] = self._merge_word(syms, best, new_id)
            words = new_words

            if verbose and (i % 50 == 0 or i == num_merges - 1):
                print(f"  [bpe] merge {i+1}/{num_merges} -> {best} (id={new_id})")

        # build lookup
        self._id_to_token = {i: b.decode("utf-8", errors="replace") for i, b in self.vocab.items()}
        self._token_to_id = {t: i for i, t in self._id_to_token.items()}
        # special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        # override 0,1,2 reserved (jarang dipakai byte control)
        self._token_to_id["<pad>"] = self.pad_id
        self._token_to_id["<bos>"] = self.bos_id
        self._token_to_id["<eos>"] = self.eos_id
        self.vocab_size = max(self.vocab.keys()) + 1
        print(f"[bpe] training done. vocab_size={self.vocab_size}")

    @staticmethod
    def _merge_word(syms, pair, new_id):
        new = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and (syms[i], syms[i+1]) == pair:
                new.append(new_id)
                i += 2
            else:
                new.append(syms[i])
                i += 1
        return new

    # ---------- encoding / decoding ----------
    def encode(self, text: str):
        words_raw = re.findall(PAT_SIMPLE, text)
        ids = [self.bos_id]
        for w in words_raw:
            syms = list(w.encode("utf-8"))
            # iteratively apply merges
            while len(syms) >= 2:
                pairs = _get_pairs(syms)
                # cari pair dengan rank terendah (paling awal ditrain)
                best = None
                best_rank = None
                for p in pairs:
                    if p in self.merges:
                        rank = list(self.merges.keys()).index(p)
                        if best_rank is None or rank < best_rank:
                            best = p
                            best_rank = rank
                if best is None:
                    break
                syms = self._merge_word(syms, best, self.merges[best])
            ids.extend(syms)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        out_bytes = b""
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if i in self.vocab:
                out_bytes += self.vocab[i]
        return out_bytes.decode("utf-8", errors="replace")

    # ---------- persistence ----------
    def save(self, path: str):
        serial = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
        }
        Path(path).write_text(json.dumps(serial, ensure_ascii=False))

    @classmethod
    def load(cls, path: str):
        data = json.loads(Path(path).read_text())
        tok = cls()
        tok.merges = {tuple(p): v for p, v in data["merges"]}
        # rebuild vocab
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in sorted(tok.merges.items(), key=lambda x: x[1]):
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        tok.vocab_size = data["vocab_size"]
        tok.pad_id = data["pad_id"]
        tok.bos_id = data["bos_id"]
        tok.eos_id = data["eos_id"]
        tok._id_to_token = {i: b.decode("utf-8", errors="replace") for i, b in tok.vocab.items()}
        tok._token_to_id = {t: i for i, t in tok._id_to_token.items()}
        return tok


if __name__ == "__main__":
    import sys
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus.txt").read_text()
    tok = MiniBPETokenizer()
    tok.train(corpus, target_vocab=384)
    tok.save("tokenizer.json")
    # sanity test
    sample = "Indonesia adalah negara"
    enc = tok.encode(sample)
    dec = tok.decode(enc)
    print(f"sample: {sample}")
    print(f"encoded: {enc}")
    print(f"decoded: {dec}")
