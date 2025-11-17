import argparse
from pathlib import Path
from typing import List
from simil import compute_embeddings, fit_whitener, transform, cosine_matrix, compute_mood_embeddings, mood_transform
from qualimidi import analyze_midi, format_report

MIDI_EXTS = {".mid", ".midi"}

def _expand_inputs(args: list[str]) -> list[Path]:
    out, seen = [], set()
    for a in args:
        p = Path(a)
        if any(c in a for c in "*?[]"):
            it = Path().glob(a)
        elif p.is_dir():
            it = p.rglob("*")
        else:
            it = [p]
        for f in it:
            if f.is_file() and f.suffix.lower() in MIDI_EXTS:
                r = f.resolve()
                if r not in seen:
                    seen.add(r); out.append(r)
    return out

def _centroid(vectors: "np.ndarray") -> "np.ndarray":
    import numpy as np
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("empty reference set")
    c = vectors.mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-8)  # unit-norm
# ---------- NEW: corruption check / cleaner ----------
def _is_bad_midi(path: Path) -> (bool, str):
    """Return (bad?, reason)."""
    import pretty_midi as pm
    try:
        m = pm.PrettyMIDI(str(path))
    except Exception as e:
        return True, f"load-failed: {e}"

    # Empty file / no timing?
    end = m.get_end_time()
    if not (end > 0):
        return True, "no-duration"

    # Try a couple of typical operations that often raise on broken files
    try:
        # Some files raise on global tempo; that’s fine—don’t mark bad just for that.
        # But if it raises *and* there are <2 notes total, mark as bad.
        total_notes = sum(len(ins.notes) for ins in m.instruments)
        try:
            _ = m.estimate_tempo()
        except Exception as te:
            if total_notes < 2:
                return True, f"tempo/notes: {te}"
    except Exception as e:
        return True, f"misc-op: {e}"

    # Sanity checks: absurdly large ticks sometimes signal corruption (pretty_midi already throws).
    # Also guard against all-drum empty content.
    if total_notes == 0:
        return True, "no-notes"

    return False, "ok"

def _clean_bad_midis(inputs: List[str], yes: bool = False, verbose: bool = True):
    bad: List[Path] = []
    files = _expand_inputs(inputs)
    if not files:
        print("No MIDI files found.", file=sys.stderr)
        return

    for p in files:
        bad_flag, reason = _is_bad_midi(p)
        if bad_flag:
            bad.append(p)
            print(f"[BAD] {p}  — {reason}")
        elif verbose:
            print(f"[OK ] {p}")

    if not bad:
        print("No bad MIDIs detected.")
        return

    if yes:
        for p in bad:
            try:
                p.unlink()
                print(f"🗑️  deleted: {p}")
            except Exception as e:
                print(f"⚠️  failed to delete {p}: {e}")
    else:
        print(f"\nDry run: {len(bad)} files flagged. Re-run with --yes to delete.")

# ---------- QS (qualitative-only) helpers ----------
def _qual_vector_from_plus(s) -> "np.ndarray":
    import numpy as np
    cad = 1.0 if getattr(s, "cadence_on_last_bar", False) else 0.0
    v = np.array([
        float(s.scale_adherence),
        float(s.tonality_strength),

        float(s.chord_changes_per_bar),
        float(s.mean_chord_dur_bars),
        float(s.chord_change_variability),

        float(s.dom_to_tonic_rate),
        float(s.plagal_rate),
        float(s.leading_tone_to_tonic_rate),
        cad,

        float(s.lead_mean_interval),
        float(s.lead_step_ratio),
        float(s.lead_leap_ratio),
        float(s.lead_contour_entropy),
        float(s.arpeggiation_index),
    ], dtype=float)
    v[~np.isfinite(v)] = 0.0
    return v

def _qual_embeddings(paths):
    import numpy as np
    from qualimidi_plus import analyze_plus
    vecs, kept = [], []
    for p in paths:
        try:
            s = analyze_plus(p)      # inside, _bars may raise; that's ok
            v = _qual_vector_from_plus(s)
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
            continue
        vecs.append(v); kept.append(p)
    if not vecs:
        raise SystemExit("No qualitative vectors produced.")
    return kept, np.vstack(vecs)

def _zscore_l2(E: "np.ndarray") -> "np.ndarray":
    import numpy as np
    mu = E.mean(axis=0, keepdims=True)
    sd = E.std(axis=0, keepdims=True) + 1e-8
    Z = (E - mu) / sd
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    return Z

def _cosine_matrix(Y: "np.ndarray") -> "np.ndarray":
    return Y @ Y.T

def _intersect_order(ref: Path, A_paths: List[Path], B_paths: List[Path]):
    """Return unified path list (ref-first) present in BOTH lists, preserving A_paths order after ref."""
    aset = {p.resolve() for p in A_paths}
    bset = {p.resolve() for p in B_paths}
    common = [p for p in A_paths if p.resolve() in bset]
    # ensure ref at front
    R = ref.resolve()
    common = [p for p in common if p.resolve() != R]
    final = [ref] + common
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["s","q","qq","qs","qh","c", "sm"])
    ap.add_argument("input", nargs="*", help="Candidates (files/dirs/globs).")
    ap.add_argument("--ref", action="append", default=[],
                    help="Reference files/dirs/globs (repeatable). If omitted in s/qs/qh, the first positional is used as single reference.")
    ap.add_argument("--topk", type=int, default=9999)
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--components", type=int, default=48)
    ap.add_argument("--key-invariant", action="store_true")
    ap.add_argument("--w-embed", type=float, default=0.5)
    ap.add_argument("--w-qual", type=float, default=0.5)

    # NEW: cleaner flags
    ap.add_argument("--yes", action="store_true", help="Actually delete bad files in 'c' mode")
    ap.add_argument("--quiet", action="store_true", help="Suppress [OK] prints in 'c' mode")
    args = ap.parse_args()
    if args.mode in {"s","qs","qh"}:
        R = _expand_inputs(args.ref) if args.ref else []
        C = _expand_inputs(args.input)

        # Back-compat: if no --ref given, use the *first* positional as ref, rest as candidates
        if not R:
            if not args.input:
                raise SystemExit("Provide at least one MIDI (or use --ref).")
            first = _expand_inputs([args.input[0]])
            if not first:
                raise SystemExit(f"Reference '{args.input[0]}' not found.")
            R = first
            C = _expand_inputs(args.input[1:])

        # De-dup between R and C; keep anything not in R as candidates
        R_set = {p.resolve() for p in R}
        C = [p for p in C if p.resolve() not in R_set]

        if not C:
            raise SystemExit("No candidate MIDIs found to compare.")
    # ---------------- existing modes ----------------
    if args.mode == "s":
        all_paths = R + C
        e_paths, eE = compute_embeddings(all_paths, key_invariant=args.key_invariant)
        W = fit_whitener(eE, n_components=args.components)
        eY = transform(eE, W)  # each row unit-norm

        # find rows for R and C in e_paths
        idx_map = {p.resolve(): i for i, p in enumerate(e_paths)}
        r_idx = [idx_map[p.resolve()] for p in R if p.resolve() in idx_map]
        c_idx = [idx_map[p.resolve()] for p in C if p.resolve() in idx_map]
        if not r_idx or not c_idx:
            raise SystemExit("Reference or candidates missing after embedding.")

        import numpy as np
        ref_centroid = _centroid(eY[r_idx])
        sims = [(e_paths[i], float(np.dot(ref_centroid, eY[i]))) for i in c_idx]
        sims.sort(key=lambda t: t[1], reverse=True)

        print(f"\nEmbedding similarity to REF({len(r_idx)} files):")
        for p, s in sims[: min(args.topk, len(sims))]:
            print(f"{s:6.3f}  {p.name}")
        if args.matrix:
            # optional: show only candidates vs centroid is 1-D; instead, build full matrix for R∪C
            from simil import print_similarity_matrix
            S = cosine_matrix(eY)
            print("\n— Embedding matrix (R∪C) —")
            print_similarity_matrix(e_paths, S, precision=3, width=7, markdown=False)
        return


    if args.mode == "q":
        summ = analyze_midi(args.input[0])
        print(summ.key, summ.mode, summ.density_notes_per_s)
        return

    if args.mode == "qq":
        from qualimidi_plus import analyze_plus, format_plus
        s = analyze_plus(args.input[0])
        print(format_plus(s))
        print(s.dom_to_tonic_rate, getattr(s, "arpeggiation_index", None))
        return

    if args.mode == "qs":
        all_paths = R + C
        q_paths, qE = _qual_embeddings(all_paths)   # (N x D)
        qY = _zscore_l2(qE)                         # (z-score on R∪C) → L2 rows
        idx_map = {p.resolve(): i for i, p in enumerate(q_paths)}
        r_idx = [idx_map[p.resolve()] for p in R if p.resolve() in idx_map]
        c_idx = [idx_map[p.resolve()] for p in C if p.resolve() in idx_map]
        if not r_idx or not c_idx:
            raise SystemExit("Reference or candidates missing after qualitative analysis.")

        import numpy as np
        ref_centroid = _centroid(qY[r_idx])
        sims = [(q_paths[i], float(np.dot(ref_centroid, qY[i]))) for i in c_idx]
        sims.sort(key=lambda t: t[1], reverse=True)

        print(f"\nQualitative similarity to REF({len(r_idx)} files):")
        for p, s in sims[: min(args.topk, len(sims))]:
            print(f"{s:6.3f}  {p.name}")
        if args.matrix:
            from simil import print_similarity_matrix
            Sq = _cosine_matrix(qY)
            print("\n— Qualitative matrix (R∪C) —")
            print_similarity_matrix(q_paths, Sq, precision=3, width=7, markdown=False)
        return

    if args.mode == "c":
        _clean_bad_midis(args.input, yes=args.yes, verbose=not args.quiet)
        return


        # ---------------- HYBRID mode (qh) ----------------
    if args.mode == "qh":
        all_paths = R + C

        # Embedding branch
        e_paths, eE = compute_embeddings(all_paths, key_invariant=args.key_invariant)
        W = fit_whitener(eE, n_components=args.components)
        eY = transform(eE, W)
        emap = {p.resolve(): i for i, p in enumerate(e_paths)}

        # Qualitative branch
        q_paths, qE = _qual_embeddings(all_paths)
        qY = _zscore_l2(qE)
        qmap = {p.resolve(): i for i, p in enumerate(q_paths)}

        # Align (present in both)
        common = [p for p in all_paths if p.resolve() in emap and p.resolve() in qmap]
        if len(common) < 2:
            raise SystemExit("Not enough overlap between embedding and qualitative sets.")

        import numpy as np
        eM = np.vstack([eY[emap[p.resolve()]] for p in common])
        qM = np.vstack([qY[qmap[p.resolve()]] for p in common])

        # split back into refs/cands
        R_res = {p.resolve() for p in R}
        r_mask = np.array([p.resolve() in R_res for p in common], dtype=bool)
        c_mask = ~r_mask
        if r_mask.sum() == 0 or c_mask.sum() == 0:
            raise SystemExit("Reference or candidates empty after alignment.")

        e_ref = _centroid(eM[r_mask])
        q_ref = _centroid(qM[r_mask])

        w_e = float(args.w_embed); w_q = float(args.w_qual)
        denom = (w_e + w_q) or 1.0

        sims = []
        for p, e_vec, q_vec, is_cand in zip(common, eM, qM, c_mask):
            if not is_cand:  # skip reference members in ranking
                continue
            s = (w_e * float(np.dot(e_ref, e_vec)) + w_q * float(np.dot(q_ref, q_vec))) / denom
            sims.append((p, s))
        sims.sort(key=lambda t: t[1], reverse=True)

        print(f"\nHybrid similarity to REF({r_mask.sum()} files) (w_embed={w_e}, w_qual={w_q}):")
        for p, s in sims[: min(args.topk, len(sims))]:
            print(f"{s:6.3f}  {p.name}")

        if args.matrix:
            from simil import print_similarity_matrix
            # show candidate-only matrices in common order if you want; otherwise skip (centroid mode is 1→N)
            print("\n— Note — matrices are less meaningful with centroid queries; ranking above is the key output.")
        return
    if args.mode == "sm":
        import numpy as np
        # resolve R (refs) and C (cands) like you already do for centroid modes
        R = _expand_inputs(args.ref) if args.ref else _expand_inputs([args.input[0]])
        C = _expand_inputs(args.input[1:] if not args.ref else args.input)
        Rset = {p.resolve() for p in R}
        C = [p for p in C if p.resolve() not in Rset]
        all_paths = R + C

        m_paths, mE, _, _ = compute_mood_embeddings(all_paths, key_invariant=False)
        idx = {p.resolve(): i for i,p in enumerate(m_paths)}
        r_idx = [idx[p.resolve()] for p in R if p.resolve() in idx]
        c_idx = [idx[p.resolve()] for p in C if p.resolve() in idx]
        if not r_idx or not c_idx: raise SystemExit("Missing refs or cands after mood embed.")

        Y = mood_transform(mE)
        ref_centroid = (Y[r_idx].mean(axis=0)); ref_centroid /= (np.linalg.norm(ref_centroid)+1e-8)

        sims = [(m_paths[i], float(np.dot(ref_centroid, Y[i]))) for i in c_idx]
        sims.sort(key=lambda t: t[1], reverse=True)
        print(f"\nMood-forward similarity to REF({len(r_idx)} files):")
        for p, s in sims[:min(args.topk, len(sims))]:
            print(f"{s:6.3f}  {p.name}")
        return

if __name__ == "__main__":
    main()
