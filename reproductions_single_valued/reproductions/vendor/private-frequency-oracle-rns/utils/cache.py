import json
from pathlib import Path
from typing import Optional
from filelock import FileLock

# Directory and file setup
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODULI_CACHE_PATH = RESULTS_DIR / "mss_moduli_cache.json"
LOCK_PATH = MODULI_CACHE_PATH.with_suffix(".lock")  # e.g., mss_moduli_cache.lock

def load_mod_cache():
    if MODULI_CACHE_PATH.exists():
        with FileLock(str(LOCK_PATH)):
            try:
                with MODULI_CACHE_PATH.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                # A killed experiment can leave the upstream non-atomic cache
                # partially written. Treat it as a cache miss.
                return {}
    return {}

def save_mod_cache(cache_dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    with FileLock(str(LOCK_PATH)):
        with MODULI_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=0)

def get_mod_cache_key(
    k: int,
    epsilon: Optional[float] = None,
    l_max: Optional[int] = None,
    seed: Optional[int] = None,
    prefix: str = "opt"
) -> str:
    parts = [f"k={k}"]
    if epsilon is not None:
        parts.append(f"eps={epsilon:.4f}")
    if l_max is not None:
        parts.append(f"lmax={l_max}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return f"{prefix}:" + ":".join(parts)
