import numpy as np

def _sample_at_or_before(sample_seconds, sample_values, seconds):
    if sample_seconds is None or len(sample_seconds) == 0:
        return np.nan
    target_second = float(np.floor(seconds))
    index = int(np.searchsorted(sample_seconds, target_second, side="right")) - 1
    if index < 0:
        return np.nan
    return float(sample_values[index])

secs = np.array([0., 1., 2., 3., 4.])
alts = np.array([25.1, 25.4, 25.3, 26.0, 26.2])

for t in [0.0, 0.9, 2.0, 2.7, 2.999, 3.0, 4.5, -1.0,5]:
    print(f"frame at {t:6} -> {_sample_at_or_before(secs, alts, t)}")