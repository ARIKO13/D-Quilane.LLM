"""
infer.py — Inference & generation script.
Bisa di-import atau di-run langsung:
    python3 infer.py "Indonesia adalah"
"""
import sys
from pathlib import Path

import torch

# import local
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from tokenizer import MiniBPETokenizer
from mini_gpt_llm import MiniGPT

LLM_DIR = Path("/home/z/my-project/download/llm")
MODEL_PATH = LLM_DIR / "mini_llm.pt"
TOK_PATH = LLM_DIR / "tokenizer.json"


def load_model_and_tokenizer():
    tok = MiniBPETokenizer.load(str(TOK_PATH))
    model = MiniGPT.load(str(MODEL_PATH))
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int = 50,
            temperature: float = 0.8, top_k: int | None = 20):
    """Generate text dari prompt. Mendukung KV-cache utk fast sampling."""
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    out_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tok.eos_id,
    )
    out_ids = out_ids[0].tolist()
    # strip BOS/EOS
    return tok.decode(out_ids)


def main():
    if len(sys.argv) < 2:
        prompt = "Indonesia"
    else:
        prompt = sys.argv[1]

    model, tok = load_model_and_tokenizer()
    print(f"Prompt: {prompt!r}")
    print(f"Model info:")
    model.info()
    print()
    print("--- generating ---")
    out = generate(model, tok, prompt, max_new_tokens=40, temperature=0.8, top_k=20)
    print(out)


if __name__ == "__main__":
    main()
