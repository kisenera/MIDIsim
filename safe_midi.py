# safe_midi.py (or top of your script)
from pathlib import Path
import pretty_midi as pm

def safe_pretty_midi(path: str | Path):
    try:
        return pm.PrettyMIDI(str(path))
    except Exception as e:
        print(f"skip {Path(path).name}: cannot parse ({e})")
        return None

def safe_estimate_tempo(m: pm.PrettyMIDI, fallback: float = 120.0) -> float:
    try:
        t = float(m.estimate_tempo())
        return t if t > 1e-6 else fallback
    except Exception:
        return fallback
