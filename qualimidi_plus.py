# qualimidi_plus.py
# pip install pretty_midi numpy
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pretty_midi as pm

PN = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

@dataclass
class PlusSummary:
    path: Path
    # core
    key_root: int
    key_name: str
    key_mode: str
    tempo_est: float
    duration: float
    # chords / harmonic rhythm
    chords_by_bar: List[str]
    chord_roots_by_bar: List[int]  # -1 for N
    chord_changes_per_bar: float
    mean_chord_dur_bars: float
    chord_change_variability: float
    # resolution metrics
    dom_to_tonic_rate: float
    plagal_rate: float
    cadence_on_last_bar: bool
    leading_tone_to_tonic_rate: float
    scale_adherence: float
    tonality_strength: float
    # melody “arp vs singable”
    lead_mean_interval: float
    lead_step_ratio: float
    lead_leap_ratio: float
    lead_contour_entropy: float
    arpeggiation_index: float  # proportion of melodic notes that are local chord tones

# ---------- helpers ----------
def _bars(m: pm.PrettyMIDI) -> np.ndarray:
    tempo = m.estimate_tempo() or 120.0
    ts = m.time_signature_changes[0] if m.time_signature_changes else None
    beats_per_bar = ts.numerator if ts else 4
    spb = 60.0/(tempo if tempo>1e-6 else 120.0)*beats_per_bar
    end = m.get_end_time()
    return np.arange(0.0, end+1e-6, spb)

def _chroma_time(m: pm.PrettyMIDI, hop=0.02):
    end = m.get_end_time()
    t = np.arange(0, end+1e-9, hop)
    C = np.zeros((12, len(t)))
    for inst in m.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            i0 = int(max(0, np.floor(n.start / hop)))
            i1 = int(min(len(t), np.ceil(n.end / hop)))
            if i1>i0: C[n.pitch%12, i0:i1] += 1.0
    if C.size:
        C = np.convolve(C.flatten(), np.ones(3)/3, mode="same").reshape(12, -1)
    return C, t

def _key_from_chroma(C: np.ndarray) -> Tuple[int,str]:
    if C.size==0 or C.sum()==0: return 0, "major"
    avg = C.mean(axis=1)
    best, root, mode = -1e9, 0, "major"
    for r in range(12):
        smaj = float(np.dot(avg, np.roll(KS_MAJOR, r)))
        smin = float(np.dot(avg, np.roll(KS_MINOR, r)))
        if smaj>best: best, root, mode = smaj, r, "major"
        if smin>best: best, root, mode = smin, r, "minor"
    return root, mode

def _scale_pcs(root:int, mode:str)->set:
    # major: 0,2,4,5,7,9,11 ; natural minor: 0,2,3,5,7,8,10
    steps = [0,2,4,5,7,9,11] if mode=="major" else [0,2,3,5,7,8,10]
    return {(root+s)%12 for s in steps}

def _rough_chords_per_bar(m: pm.PrettyMIDI, bars: np.ndarray) -> Tuple[List[str], List[int]]:
    # simple triad guess per bar by chroma mass; returns names and roots (-1 for N)
    C, _ = _chroma_time(m, hop=0.02)
    names, roots = [], []
    for b0,b1 in zip(bars[:-1], bars[1:]):
        if b1-b0<0.1: continue
        i0 = int(b0/0.02); i1 = int(b1/0.02)
        c = C[:, i0:i1].sum(axis=1)
        if c.sum()==0:
            names.append("N"); roots.append(-1); continue
        r = int(np.argmax(c))
        # choose maj/min by which triad collects more mass
        maj = c[[r,(r+4)%12,(r+7)%12]].sum()
        mino= c[[r,(r+3)%12,(r+7)%12]].sum()
        nm = PN[r] if maj>=mino else PN[r]+"m"
        names.append(nm); roots.append(r)
    return names, roots

def _lead_melody(m: pm.PrettyMIDI):
    # pick the highest-register non-drum instrument as lead, flatten to monophonic by selecting highest note per onset slice
    cand = [inst for inst in m.instruments if not inst.is_drum and inst.notes]
    if not cand: return []
    inst = max(cand, key=lambda ins: np.mean([n.pitch for n in ins.notes]))
    notes = sorted(inst.notes, key=lambda n:(n.start, -n.pitch))
    # greedy monophonic: keep highest pitch when overlaps start at same time
    lead = []
    last_end = -1.0
    for n in notes:
        if not lead or n.start >= lead[-1].start + 1e-4:
            lead.append(n)
        else:
            # same onset block: replace with higher if needed
            if n.pitch > lead[-1].pitch:
                lead[-1] = n
    return lead

def _intervals(notes):
    if len(notes)<2: return np.array([])
    return np.diff([n.pitch for n in notes])

def _is_chord_tone(pitch:int, root:int, name:str)->bool:
    if root<0: return False
    is_minor = name.endswith("m")
    third = (root+3)%12 if is_minor else (root+4)%12
    fifth = (root+7)%12
    return (pitch%12) in {root, third, fifth}

# ---------- main analysis ----------
def analyze_plus(path:str|Path) -> PlusSummary:
    p = Path(path)
    m = pm.PrettyMIDI(str(p))
    dur = m.get_end_time() or 0.0
    tempo = float(m.estimate_tempo() or 120.0)
    bars = _bars(m)
    C, _ = _chroma_time(m)
    key_root, key_mode = _key_from_chroma(C)
    key_name = f"{PN[key_root]} {key_mode}"

    # chords & harmonic rhythm
    ch_names, ch_roots = _rough_chords_per_bar(m, bars)
    # chord-change stats
    changes = [i for i in range(1,len(ch_roots)) if ch_roots[i]!=ch_roots[i-1]]
    chords_per_bar = len(changes)/max(1,(len(bars)-1))
    if changes:
        gaps = np.diff([0]+changes+[len(ch_roots)-1])  # bars between changes
        mean_dur = float(np.mean(gaps))
        var_dur  = float(np.std(gaps))
    else:
        mean_dur, var_dur = float(len(ch_roots)), 0.0

    # resolution metrics
    dom = (key_root+7)%12  # V
    tonic = key_root
    iv = (key_root+5)%12
    dom2tonic = 0; plagal = 0; transitions = 0
    for i in range(1,len(ch_roots)):
        a,b = ch_roots[i-1], ch_roots[i]
        if a<0 or b<0: continue
        transitions += 1
        if a==dom and b==tonic: dom2tonic += 1
        if a==iv and b==tonic: plagal += 1
    dom_rate = dom2tonic/max(1,transitions)
    plagal_rate = plagal/max(1,transitions)
    last_is_tonic = (ch_roots[-1]==tonic) if ch_roots else False

    # melody: leading-tone → tonic & interval stats
    lead = _lead_melody(m)
    lt = (tonic+11)%12  # leading tone below tonic in major; fine as heuristic in minor too
    lt2t = 0; mel_steps = 0; mel_leaps = 0
    itv = _intervals(lead)
    for n_prev, n_next in zip(lead[:-1], lead[1:]):
        # step vs leap
        d = abs(n_next.pitch - n_prev.pitch)
        if d in (1,2): mel_steps += 1
        elif d>=3: mel_leaps += 1
        # leading tone to tonic
        if n_prev.pitch%12==lt and n_next.pitch%12==tonic:
            lt2t += 1
    total_pairs = max(1, len(lead)-1)
    leading_rate = lt2t/total_pairs
    mean_interval = float(np.mean(np.abs(itv))) if itv.size else 0.0
    step_ratio = mel_steps/max(1, mel_steps+mel_leaps)
    leap_ratio = mel_leaps/max(1, mel_steps+mel_leaps)

    # contour entropy (higher = more erratic)
    if itv.size:
        bins = np.clip(np.abs(itv), 0, 12)
        hist = np.bincount(bins, minlength=13).astype(float)
        p_hist = hist/hist.sum()
        contour_entropy = float(-(p_hist[p_hist>0]*np.log2(p_hist[p_hist>0])).sum())
    else:
        contour_entropy = 0.0

    # arpeggiation index: fraction of lead notes that are chord tones of current bar chord
    arp_hits = 0; total_lead = 0
    for n in lead:
        # find bar index
        bi = np.searchsorted(bars, n.start, side="right")-1
        bi = int(np.clip(bi, 0, len(ch_names)-1)) if ch_names else 0
        if len(ch_names)==0: continue
        if _is_chord_tone(n.pitch, ch_roots[bi], ch_names[bi]): arp_hits += 1
        total_lead += 1
    arp_index = arp_hits/max(1,total_lead)

    # scale adherence (notes in key)
    scale = _scale_pcs(key_root, key_mode)
    all_notes = [n for inst in m.instruments if not inst.is_drum for n in inst.notes]
    if all_notes:
        in_scale = sum(1 for n in all_notes if (n.pitch%12) in scale)
        scale_adherence = in_scale/len(all_notes)
    else:
        scale_adherence = 1.0

    # tonality strength ~ cosine between mean chroma and chosen key profile
    profile = np.roll(KS_MAJOR if key_mode=="major" else KS_MINOR, key_root)
    mean_chroma = C.mean(axis=1) if C.size else np.zeros(12)
    if mean_chroma.sum()==0: ton_strength = 0.0
    else:
        a = mean_chroma/np.linalg.norm(mean_chroma)
        b = profile/np.linalg.norm(profile)
        ton_strength = float(np.dot(a,b))

    return PlusSummary(
        path=p,
        key_root=key_root, key_name=f"{PN[key_root]}", key_mode=key_mode,
        tempo_est=tempo, duration=dur,
        chords_by_bar=ch_names, chord_roots_by_bar=ch_roots,
        chord_changes_per_bar=float(chords_per_bar),
        mean_chord_dur_bars=mean_dur, chord_change_variability=var_dur,
        dom_to_tonic_rate=dom_rate, plagal_rate=plagal_rate,
        cadence_on_last_bar=bool(last_is_tonic),
        leading_tone_to_tonic_rate=leading_rate,
        scale_adherence=float(scale_adherence),
        tonality_strength=ton_strength,
        lead_mean_interval=mean_interval,
        lead_step_ratio=step_ratio, lead_leap_ratio=leap_ratio,
        lead_contour_entropy=contour_entropy,
        arpeggiation_index=arp_index,
    )

def format_plus(s: PlusSummary) -> str:
    cc = f"{s.chord_changes_per_bar:.2f}/bar (mean dur {s.mean_chord_dur_bars:.2f} bars; σ={s.chord_change_variability:.2f})"
    cad = f"V→I {s.dom_to_tonic_rate:.2f}, IV→I {s.plagal_rate:.2f}, LT→I {s.leading_tone_to_tonic_rate:.2f}, final tonic: {s.cadence_on_last_bar}"
    lead = (f"interval {s.lead_mean_interval:.2f} st; steps {s.lead_step_ratio:.2f}, "
            f"leaps {s.lead_leap_ratio:.2f}; contour H={s.lead_contour_entropy:.2f}; arp-index {s.arpeggiation_index:.2f}")
    tonal = f"Key: {s.key_name} {s.key_mode} | scale-adherence {s.scale_adherence:.2f} | tonality-strength {s.tonality_strength:.2f}"
    return (
        f"File: {s.path.name}\n"
        f"{tonal}\n"
        f"Harmonic rhythm: {cc}\n"
        f"Resolution: {cad}\n"
        f"Lead profile: {lead}\n"
        f"Chords(first 8 bars): {' '.join(s.chords_by_bar[:8])}{' …' if len(s.chords_by_bar)>8 else ''}\n"
    )

# -------- CLI --------
def _glob(patterns: List[str]) -> List[Path]:
    out=[]
    for pat in patterns:
        out.extend(Path().glob(pat) if any(c in pat for c in "*?[]") else [Path(pat)])
    # dedup files only
    seen, uniq=set(), []
    for p in out:
        r=p.resolve()
        if p.is_file() and r not in seen:
            seen.add(r); uniq.append(p)
    return uniq

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Advanced qualitative MIDI analysis")
    ap.add_argument("inputs", nargs="+", help="MIDI files or globs")
    args = ap.parse_args()
    for path in _glob(args.inputs):
        try:
            s = analyze_plus(path)
            print(format_plus(s))
        except Exception as e:
            print(f"[{Path(path).name}] error: {e}")
