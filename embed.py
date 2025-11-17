# embed.py
import numpy as np, pretty_midi as pm

def _bars(pm_obj, default_tsig=(4,4)):

    tempo = pm_obj.estimate_tempo()
    beats_per_bar = default_tsig[0]
    spb = 60.0 / max(tempo, 1e-6) * beats_per_bar
    end = pm_obj.get_end_time()
    return np.arange(0.0, end + 1e-6, spb)

def _rotate_to_c(chroma):
    idx = np.argmax(chroma)
    return np.roll(chroma, -idx)

def midi_gist_features(midi_path, onset_bins=16, ioi_range=(0.05,1.0,12),
                       key_invariant=False):
    m = pm.PrettyMIDI(midi_path)
    bars = _bars(m)
    if len(bars) < 2: return None

    notes = []
    for inst in m.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity))
    if not notes: return None
    notes = np.array(notes, float)
    starts, ends, pitches = notes[:,0], notes[:,1], notes[:,2]
    on_edges = np.linspace(0, 1, onset_bins+1)
    ioi_edges = np.geomspace(ioi_range[0], ioi_range[1], ioi_range[2]+1)

    feats = []
    for b0, b1 in zip(bars[:-1], bars[1:]):
        L = b1 - b0
        if L < 0.1: continue

        # CHROMA (bar-summed)
        C = np.zeros(12)
        mask = (starts < b1) & (ends > b0)
        if np.any(mask):
            # weight by overlap duration
            for s,e,p,_ in notes[mask]:
                dur = max(0.0, min(e, b1) - max(s, b0))
                if dur > 0: C[int(p) % 12] += dur
        C = C / (C.sum()+1e-8)
        if key_invariant:
            C = _rotate_to_c(C)

        # ONSET HIST (16ths within bar)
        s_in = starts[(starts>=b0) & (starts<b1)]
        oh = np.zeros(onset_bins)
        if s_in.size:
            idx = np.clip(((s_in - b0)/L * onset_bins).astype(int), 0, onset_bins-1)
            np.add.at(oh, idx, 1)
            oh = oh / (oh.sum()+1e-8)

        # IOI HIST (within bar, tempo-robust via log bins)
        if s_in.size >= 2:
            ioi = np.diff(np.sort(s_in))
            ih, _ = np.histogram(ioi, bins=ioi_edges)
            ih = ih / (ih.sum()+1e-8)
        else:
            ih = np.zeros(len(ioi_edges)-1)

        # DENSITY / REGISTER / ENTROPY
        dens = s_in.size / L
        reg_mean = pitches[mask].mean() if np.any(mask) else 60.0
        reg_std  = pitches[mask].std()  if np.any(mask) else 0.0
        # rhythmic entropy of onset pattern
        p = oh + 1e-12
        ent = -np.sum(p*np.log(p))

        feats.append(np.concatenate([C, oh, ih, [dens, reg_mean, reg_std, ent]]))

    if not feats: return None
    X = np.vstack(feats)  # (bars x D)
    return X  # we return per-bar matrix; pooling happens later

def pool_features(X):
    """Robust pool: mean + IQR (keeps some variability)."""
    mu = X.mean(axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    return np.concatenate([mu, iqr])
_PN = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def _tempo_and_bars(m: pm.PrettyMIDI, default_tsig=(4,4)):
    tempo = m.estimate_tempo() or 120.0
    beats_per_bar = (m.time_signature_changes[0].numerator
                     if m.time_signature_changes else default_tsig[0])
    spb = 60.0/max(tempo,1e-6) * beats_per_bar
    end = m.get_end_time()
    bars = np.arange(0.0, end + 1e-6, spb)
    return tempo, bars

def _chroma_time(m: pm.PrettyMIDI, hop=0.02):
    end = m.get_end_time()
    t = np.arange(0, end + 1e-9, hop)
    C = np.zeros((12, len(t)))
    active = []  # per-frame active pitches for polyphony / consonance
    for i, ti in enumerate(t):
        act = []
        for inst in m.instruments:
            if inst.is_drum: continue
            for n in inst.notes:
                if n.start <= ti < n.end:
                    C[n.pitch%12, i] += 1.0
                    act.append(n.pitch)
        active.append(sorted(act))
    # light smooth
    if C.size:
        C = np.convolve(C.flatten(), np.ones(3)/3, mode="same").reshape(12, -1)
    return C, np.array(active, dtype=object), t

def _key_mode(C):
    if C.size == 0 or C.sum() == 0: return 0, "major", 0.0
    avg = C.mean(axis=1)
    best, root, mode = -1e9, 0, "major"
    for r in range(12):
        smaj = float(np.dot(avg, np.roll(_KS_MAJOR, r)))
        smin = float(np.dot(avg, np.roll(_KS_MINOR, r)))
        if smaj > best: best, root, mode = smaj, r, "major"
        if smin > best: best, root, mode = smin, r, "minor"
    # tonality strength ~ cosine to chosen profile
    prof = np.roll(_KS_MAJOR if mode=="major" else _KS_MINOR, root)
    a = avg/np.linalg.norm(avg); b = prof/np.linalg.norm(prof)
    ton_str = float(np.dot(a,b))
    return root, mode, ton_str

def _consonance_rate(active_pitches):
    # percent of simultaneous pairs that are consonant (unison, m/M3, P5, m/M6, octave)
    cons_ic = {0,3,4,7,8,9}
    total=0; cons=0
    for frame in active_pitches:
        k = len(frame)
        if k < 2: continue
        for i in range(k):
            for j in range(i+1,k):
                ic = abs(frame[j]-frame[i]) % 12
                total += 1
                if ic in cons_ic: cons += 1
    if total==0: return 1.0
    return cons/total

def _lead_monoline(m: pm.PrettyMIDI):
    cand = [ins for ins in m.instruments if not ins.is_drum and ins.notes]
    if not cand: return []
    lead = max(cand, key=lambda ins: np.mean([n.pitch for n in ins.notes]))
    notes = sorted(lead.notes, key=lambda n:(n.start, -n.pitch))
    mono=[]
    for n in notes:
        if not mono or n.start >= mono[-1].start + 1e-4: mono.append(n)
        elif n.pitch > mono[-1].pitch: mono[-1] = n
    return mono

def _lead_smoothness(notes):
    if len(notes)<2: return 0.0, 0.0, 0.0
    itv = np.abs(np.diff([n.pitch for n in notes]))
    mean_int = float(itv.mean())
    steps = np.count_nonzero(np.isin(itv, [1,2])); leaps = np.count_nonzero(itv>=3)
    step_ratio = steps/max(1, steps+leaps)
    # contour entropy
    bins = np.clip(itv, 0, 12)
    hist = np.bincount(bins, minlength=13).astype(float)
    p = hist/hist.sum()
    ent = float(-(p[p>0]*np.log2(p[p>0])).sum())
    return mean_int, step_ratio, ent

def midi_mood_features(midi_path: str, key_invariant: bool=False):
    """Return per-bar feature matrix (focus on mood/valence over texture)."""
    m = pm.PrettyMIDI(midi_path)
    tempo, bars = _tempo_and_bars(m)
    C, active, t = _chroma_time(m, hop=0.02)
    if C.size == 0 or len(bars)<2: return None

    key_root, key_mode, ton_str = _key_mode(C)
    # per-bar features with density-normalized chroma and section cues
    feats=[]
    for b0,b1 in zip(bars[:-1], bars[1:]):
        i0 = int(b0/0.02); i1 = int(b1/0.02)
        if i1 <= i0: continue
        c = C[:, i0:i1].sum(axis=1)
        c = c / (c.sum()+1e-8)              # density-normalize → valence over volume
        if key_invariant:                   # optional rotation invariance
            c = np.roll(c, -key_root)
        # register warmth: proportion of notes in 48–72 (C3–C5)
        pitches = [p for fr in active[i0:i1] for p in fr]
        if pitches:
            warmth = np.mean([(48 <= p <= 72) for p in pitches])
            poly = np.mean([len(fr) for fr in active[i0:i1]])
        else:
            warmth, poly = 0.0, 0.0
        # consonance (bar)
        cons = _consonance_rate(active[i0:i1])
        feats.append(np.concatenate([c, [warmth, poly, cons]]))
    if not feats: return None
    X = np.vstack(feats)  # bars x (12 + 3)

    # lead smoothness global (added as 3 scalars later)
    lead = _lead_monoline(m)
    mean_int, step_ratio, ent = _lead_smoothness(lead)

    # simple section-aware pooling: weight bars by (tonality strength proxy * 1/(1+poly_var))
    poly_col = X[:, -2]  # polyphony per bar
    poly_var = (poly_col - poly_col.mean())**2
    # tonality strength per bar via cosine to global key profile
    prof = (np.roll(_KS_MAJOR if key_mode=="major" else _KS_MINOR, key_root))
    prof = prof/np.linalg.norm(prof)
    cos_bar = (X[:,:12] @ (prof/ (np.linalg.norm(prof)+1e-8)))
    w = cos_bar / (1.0 + poly_var + 1e-6)
    w = np.clip(w, 0, None); w = w / (w.sum()+1e-8)

    pooled = (w[:,None] * X).sum(axis=0)               # weighted mean of bars
    spread = np.percentile(X, 75, axis=0)-np.percentile(X,25,axis=0)  # IQR
    mood_vec = np.concatenate([pooled, spread, [ton_str, mean_int, step_ratio, ent]])
    # L2 at caller; return also (key_root, key_mode) for rotation if needed
    return mood_vec, key_root, key_mode
