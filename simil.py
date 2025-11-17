# similarity.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from embed import midi_gist_features, pool_features

def compute_embeddings(paths, key_invariant=False):
    pooled = []
    keep = []
    for p in paths:
        try:
            X = midi_gist_features(str(p), key_invariant=key_invariant)
            if X is None:
                print(f"skip (no notes): {p}")
                continue
            pooled.append(pool_features(X))
            keep.append(p)
        except:
            print(f"[SKIP] {p}")
            continue

    if not pooled:
        raise SystemExit("No embeddings produced.")
    E = np.vstack(pooled)  # (N x D)
    return keep, E
# simil.py (add)

from embed import midi_mood_features

def compute_mood_embeddings(paths, key_invariant=False):
    vecs=[]; kept=[]
    roots=[]; modes=[]
    for p in paths:
        try:
            out = midi_mood_features(str(p), key_invariant=key_invariant)
            if out is None: 
                print(f"skip {p.name}: no features")
                continue
            v, r, m = out
            vecs.append(v); kept.append(p); roots.append(r); modes.append(m)
        except Exception as e:
            print(f"skip {p.name}: {e}")
    if not vecs: raise SystemExit("No mood embeddings produced.")
    E = np.vstack(vecs)
    return kept, E, roots, modes

def mood_transform(E):
    # z-score across set, PCA optional, then L2
    mu = E.mean(axis=0, keepdims=True)
    sd = E.std(axis=0, keepdims=True) + 1e-8
    Z = (E - mu)/sd
    # (optional) PCA for 32 comps
    U,S,Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(32, Vt.shape[0])
    Y = Z @ Vt[:k].T
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True)+1e-8)
    return Y

def fit_whitener(E, n_components=48):
    # z-score
    mean = E.mean(axis=0, keepdims=True)
    std  = E.std(axis=0, keepdims=True) + 1e-8
    Z = (E - mean) / std
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    W = Vt[:k]            # projection
    scale = (S[:k] / (Z.shape[0] - 1 + 1e-8))**0  # set to 0 for plain PCA, or −1 for whitening
    return {"mean": mean, "std": std, "W": W, "scale": scale}

def transform(E, whitener):
    Z = (E - whitener["mean"]) / whitener["std"]
    Y = Z @ whitener["W"].T
    Y = Y * (whitener["scale"] + 1.0)  # keep as PCA (no whitening) for stability
    # L2 normalize rows for cosine
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return Y

def cosine_matrix(Y):
    return Y @ Y.T  # since rows are unit-norm

# -------- CLI (optional) --------
def _glob(patterns):
    out = []
    for pat in patterns:
        out.extend(Path().glob(pat) if any(c in pat for c in "*?[]") else [Path(pat)])
    # dedup
    seen, uniq = set(), []
    for p in out:
        r = p.resolve()
        if r not in seen and p.is_file():
            seen.add(r); uniq.append(p)
    return uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="MIDI files or globs")
    ap.add_argument("--key-invariant", action="store_true")
    ap.add_argument("--components", type=int, default=48)
    ap.add_argument("--query", help="filename to list nearest neighbors for")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    paths = _glob(args.inputs)
    paths, E = compute_embeddings(paths, key_invariant=args.key_invariant)
    W = fit_whitener(E, n_components=args.components)
    Y = transform(E, W)
    S = cosine_matrix(Y)

    # If query provided, print top-k
    if args.query:
        try:
            idx = [p.name for p in paths].index(Path(args.query).name)
        except ValueError:
            raise SystemExit("query not found among inputs")
        sims = [(i, S[idx, i]) for i in range(len(paths)) if i != idx]
        sims.sort(key=lambda t: t[1], reverse=True)
        for i, s in sims[:args.topk]:
            print(f"{s:6.3f}  {paths[i].name}")
    else:
        # print small matrix summary
        print("mean off-diag similarity:", float((S - np.eye(len(paths))).sum()/(len(paths)*(len(paths)-1))))
def _short(name: str, max_len: int = 24) -> str:
    if len(name) <= max_len:
        return name
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return name[:left] + "…" + name[-right:]

def print_similarity_matrix(paths, S, precision=3, width=7, markdown=False):
    names = [p.name for p in paths]
    fmt = f"{{:>{width}.{precision}f}}"

    if markdown:
        # Markdown table
        header = [""] + [f"`{_short(n)}`" for n in names]
        print("| " + " | ".join(header) + " |")
        print("|" + " --- |" * (len(header)) )
        for i, row_name in enumerate(names):
            cells = [f"`{_short(row_name)}`"]
            for j in range(len(names)):
                val = 1.0 if i == j else S[i, j]
                cells.append(f"{val:.{precision}f}")
            print("| " + " | ".join(cells) + " |")
    else:
        # Plain monospace table
        name_w = max(8, min(28, max(len(n) for n in names)))
        # header row
        print(" " * (name_w + 1), end="")
        for n in names:
            print(_short(n, name_w).rjust(width + 1), end="")
        print()
        # rows
        for i, row_name in enumerate(names):
            print(_short(row_name, name_w).ljust(name_w), end=" ")
            for j in range(len(names)):
                val = 1.0 if i == j else S[i, j]
                print(fmt.format(val), end="")
            print()
if __name__ == "__main__":
    main()
