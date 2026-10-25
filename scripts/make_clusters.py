"""Assign every GSM8K train problem to a semantic cluster, once, for methods that need it (BV-Blend).

    pip install sentence-transformers scikit-learn
    python scripts/make_clusters.py                       # writes data/gsm8k_train_clusters_k16.json

The BV-Blend paper (arXiv 2606.28707, section 3.2.1) embeds each prompt with "a frozen encoder" and
assigns it to the nearest centroid of a k-means codebook trained offline, but names neither the encoder
nor k. Our choices (docs/implementation-notes.md): the raw question text only (no chat template, no
answer), sentence-transformers/all-MiniLM-L6-v2 at the Hub commit of the day (recorded), L2-normalised, k-means with
k = 16, n_init 10, max_iter 300, random_state 0, fit on all 7,473 train questions. The file records all
of that, plus a hash of the questions, so a run can check it matches its data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def questions_sha1(questions: list[str]) -> str:
    return hashlib.sha1("\n".join(questions).encode("utf-8")).hexdigest()


def encoder_revision() -> str:
    """The commit of the encoder on the Hub right now. It is loaded at exactly that commit and written to
    the output file, so the clusters can be rebuilt bit for bit later."""
    from huggingface_hub import model_info
    return model_info(ENCODER).sha


def embed_minilm(questions: list[str], revision: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER, revision=revision)
    return model.encode(questions, batch_size=128, normalize_embeddings=True, show_progress_bar=True)


def make_clusters(questions: list[str], k: int = 16, seed: int = 0, embed=None) -> dict:
    """embed(questions) -> [n, d] array. Default: all-MiniLM-L6-v2 at its current Hub commit."""
    import numpy as np
    from sklearn.cluster import KMeans
    revision = None
    if embed is None:
        revision = encoder_revision()
        x = embed_minilm(questions, revision)
    else:
        x = embed(questions)
    x = np.asarray(x, dtype=np.float32)
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed).fit(x)
    ids = km.labels_.astype(int).tolist()
    return {
        "cluster_ids": ids, "k": k, "seed": seed, "sizes": np.bincount(ids, minlength=k).tolist(),
        "encoder": ENCODER if embed is None else getattr(embed, "__name__", "custom"),
        "encoder_revision": revision,
        "text": "raw GSM8K train question, no template", "n": len(questions),
        "questions_sha1": questions_sha1(questions),
    }


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--local-train", default=None, help="a GSM8K train .jsonl instead of the Hub")
    args = p.parse_args(argv)
    from src.data import load_gsm8k
    data = load_gsm8k(local={"train": args.local_train} if args.local_train else None)
    questions = [r["question"] for r in data["train"]]
    info = make_clusters(questions, k=args.k, seed=args.seed)
    out = Path(args.out) if args.out else ROOT / "data" / f"gsm8k_train_clusters_k{args.k}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info), encoding="utf-8")
    print(f"{len(questions)} questions -> {args.k} clusters, sizes {info['sizes']}  -> {out}")
    return out


if __name__ == "__main__":
    main()
