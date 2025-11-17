

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pretty_midi as pm

# --- Krumhansl-Schmuckler key profiles (major/minor) ---
_KS_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

PITCH_NAME = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

@dataclass
class MidiSummary:
    path: Path
    duration: float
    tempo_est: float
    time_sig: str
    total_notes: int
    instruments: List[str]
    pitch_range: Tuple[int,int]
    mean_register: float
    density_notes_per_s: float
    syncopation: float
    repetition: float
    key: str
    mode: str
    chords_by_bar: List[str]
    sections: List[Tuple[float,float,str]]  # (start,end,label)

def _bars(m: pm.PrettyMIDI) -> np.ndarray:
    # crude bar grid using estimated tempo & first time signature (fallback 4/4)
    tempo = m.estimate_tempo() or 120.0
    ts = m.time_signature_changes[0] if m.time_signature_changes else None
    beats_per_bar = ts.numerator if ts else 4
    spb = 60.0 / (tempo if tempo > 1e-6 else 120.0) * beats_per_bar
    end = m.get_end_time()
    return np.arange(0.0, end + 1e-6, spb)

def _chroma_from_notes(m: pm.PrettyMIDI, hop=0.02) -> Tuple[np.ndarray,np.ndarray]:
    end = m.get_end_time()
    t = np.arange(0, end + 1e-9, hop)
    C = np.zeros((12, len(t)), dtype=float)
    for inst in m.instruments:
        if inst.is_drum: continue
        for n in inst.notes:
            i0 = int(max(0, np.floor(n.start / hop)))
            i1 = int(min(len(t), np.ceil(n.end / hop)))
            if i1 > i0:
                C[n.pitch % 12, i0:i1] += 1.0
    # small smooth
    if C.size:
        C = np.convolve(C.flatten(), np.ones(3)/3, mode="same").reshape(12, -1)
    return C, t

def _key_from_chroma(C: np.ndarray) -> Tuple[str,str]:
    if C.size == 0 or C.sum() == 0:
        return "C","major"
    avg = C.mean(axis=1)
    best_score, best_key, best_mode = -1e9, 0, "major"
    for r in range(12):
        maj = np.roll(_KS_MAJOR, r); mino = np.roll(_KS_MINOR, r)
        smaj = float(np.dot(avg, maj))
        smin = float(np.dot(avg, mino))
        if smaj > best_score:
            best_score, best_key, best_mode = smaj, r, "major"
        if smin > best_score:
            best_score, best_key, best_mode = smin, r, "minor"
    return PITCH_NAME[best_key], best_mode

def _rough_chords_per_bar(m: pm.PrettyMIDI, bars: np.ndarray) -> List[str]:
    # super-simple triad matching by bar-wise chroma peaks
    triads = {
        # root: [(name, pc set), ...]
        r: [(PITCH_NAME[r], {r,(r+4)%12,(r+7)%12}), (PITCH_NAME[r]+"m", {r,(r+3)%12,(r+7)%12})]
        for r in range(12)
    }
    C, _ = _chroma_from_notes(m, hop=0.02)
    out = []
    for b0, b1 in zip(bars[:-1], bars[1:]):
        if b1 - b0 < 0.1: continue
        i0 = int(b0/0.02); i1 = int(b1/0.02)
        c = C[:, i0:i1].sum(axis=1)
        if c.sum() == 0:
            out.append("N")
            continue
        # guess bass (lowest pitch active)
        bass_root = int(np.argmax(c))
        best, best_score = "N", -1
        for r in [bass_root, (bass_root+7)%12, (bass_root+5)%12, (bass_root+2)%12]:
            for name, pcs in triads[r]:
                score = float(c[list(pcs)].sum())
                if score > best_score: best, best_score = name, score
        out.append(best)
    return out

def _syncopation(onset_hist: np.ndarray) -> float:
    # weight offbeats higher (simple 16th grid: strong beats at 0,4,8,12)
    weights = np.array([0.5,1.0,1.2,1.0, 0.3,1.0,1.2,1.0, 0.4,1.0,1.2,1.0, 0.3,1.0,1.2,1.0])
    p = onset_hist / (onset_hist.sum()+1e-9)
    return float((p * weights).sum())

def _repetition(starts: np.ndarray) -> float:
    # IOI periodicity score: normalized energy at dominant lag
    if starts.size < 4: return 0.0
    s = np.diff(np.sort(starts))
    if s.size < 2: return 0.0
    # histogram IOIs up to 2s
    hist, edges = np.histogram(s, bins=np.linspace(0.05, 2.0, 40))
    hist = hist.astype(float)
    if hist.sum() == 0: return 0.0
    hist /= hist.sum()
    return float(hist.max())  # higher = stronger repeated spacing

def analyze_midi(path: str | Path) -> MidiSummary:
    p = Path(path)
    m = pm.PrettyMIDI(str(p))
    dur = m.get_end_time() or 0.0
    tempo = float(m.estimate_tempo() or 120.0)
    ts = m.time_signature_changes[0] if m.time_signature_changes else None
    ts_str = f"{ts.numerator}/{ts.denominator}" if ts else "4/4"

    # gather notes
    notes = []
    names = []
    for inst in m.instruments:
        if inst.is_drum: continue
        names.append(pm.program_to_instrument_name(inst.program))
        for n in inst.notes:
            notes.append((n.start, n.end, n.pitch, n.velocity))
    notes = np.array(notes, float) if notes else np.zeros((0,4))
    total = int(len(notes))
    pr_lo = int(notes[:,2].min()) if total else 60
    pr_hi = int(notes[:,2].max()) if total else 60
    mean_reg = float(notes[:,2].mean()) if total else 60.0
    dens = float(total / max(dur, 1e-9))

    # rhythm stats
    bars = _bars(m)
    onset_bins = 16
    onset_hist = np.zeros(onset_bins)
    starts = notes[:,0] if total else np.array([])
    # fill onset histogram on a 16th grid inside bars
    for b0, b1 in zip(bars[:-1], bars[1:]):
        L = b1 - b0
        if L <= 0: continue
        idx = ((starts[(starts>=b0)&(starts<b1)] - b0) / L * onset_bins).astype(int)
        idx = idx[(idx>=0)&(idx<onset_bins)]
        np.add.at(onset_hist, idx, 1)
    sync = _syncopation(onset_hist)
    rep = _repetition(starts)

    # key / mode
    C, _ = _chroma_from_notes(m)
    key, mode = _key_from_chroma(C)

    # chords per bar (rough)
    chords = _rough_chords_per_bar(m, bars)

    # sections via novelty over bar-chroma
    # compute bar-chroma and detect novelty peaks
    bar_chroma = []
    for b0, b1 in zip(bars[:-1], bars[1:]):
        i0 = int(b0/0.02); i1 = int(b1/0.02)
        ch = C[:, i0:i1].sum(axis=1)
        bar_chroma.append(ch / (ch.sum()+1e-9))
    B = np.vstack(bar_chroma) if bar_chroma else np.zeros((0,12))
    # simple novelty: L2 distance between consecutive bar chroma
    nov = np.linalg.norm(np.diff(B, axis=0), axis=1) if len(B) > 1 else np.zeros(0)
    # threshold at mean+std to mark boundaries
    cuts = [0]
    if nov.size:
        th = nov.mean() + nov.std()
        cuts += [i+1 for i,v in enumerate(nov) if v > th]
    cuts = sorted(set(cuts + [len(bars)-1]))
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    secs = []
    for i,(a,b) in enumerate(zip(cuts[:-1], cuts[1:])):
        secs.append((float(bars[a]), float(bars[b]), labels[i % len(labels)]))

    return MidiSummary(
        path=p, duration=dur, tempo_est=tempo, time_sig=ts_str,
        total_notes=total, instruments=names,
        pitch_range=(pr_lo, pr_hi), mean_register=mean_reg,
        density_notes_per_s=dens, syncopation=sync, repetition=rep,
        key=key, mode=mode, chords_by_bar=chords, sections=secs
    )

def format_report(s: MidiSummary) -> str:
    pr = f"{PITCH_NAME[s.pitch_range[0]%12]}{s.pitch_range[0]//12-1}–{PITCH_NAME[s.pitch_range[1]%12]}{s.pitch_range[1]//12-1}"
    insts = ", ".join(sorted(set(s.instruments))) if s.instruments else "—"
    secs = ", ".join([f"{lab}@{start:.1f}s–{end:.1f}s" for (start,end,lab) in s.sections]) if s.sections else "—"
    # show first ~8 chords
