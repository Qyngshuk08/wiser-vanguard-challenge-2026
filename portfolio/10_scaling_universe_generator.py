"""
Scalable Asset Universe Generator — for scalability testing only.
Extends the 12-asset Vanguard-anchored universe with additional synthetic
assets sampled within the same per-class return/vol ranges, so we can test
QUBO/QAOA behavior at n = 16, 20, 24, ... assets.

NOT a replacement for asset_universe.md -- that remains the credible,
Vanguard-anchored 12-asset baseline used for the actual portfolio
recommendation. This file exists solely to generate larger *synthetic* test
problems for the scalability curve required by the judging criteria.
"""
import numpy as np
import pandas as pd

RNG_SEED = 123
rng = np.random.default_rng(RNG_SEED)

# Per-class (return_range, vol_range) drawn from the same Vanguard-anchored
# baseline in asset_universe.md -- extra assets are sampled within these
# same realistic bounds, not arbitrary noise.
CLASS_RANGES = {
    "Equity":       {"return": (0.03, 0.07), "vol": (0.15, 0.24)},
    "FixedIncome":  {"return": (0.038, 0.056), "vol": (0.05, 0.11)},
    "Commodities":  {"return": (0.04, 0.055), "vol": (0.16, 0.20)},
    "Alternatives": {"return": (0.05, 0.07), "vol": (0.18, 0.22)},
    "Cash":         {"return": (0.03, 0.04), "vol": (0.005, 0.02)},
}
CLASS_WEIGHTS = {"Equity": 0.42, "FixedIncome": 0.33, "Commodities": 0.08, "Alternatives": 0.08, "Cash": 0.09}


def generate_universe(n_assets, seed=RNG_SEED):
    r = np.random.default_rng(seed)
    classes = r.choice(list(CLASS_WEIGHTS.keys()), size=n_assets, p=list(CLASS_WEIGHTS.values()))
    mu = np.array([r.uniform(*CLASS_RANGES[c]["return"]) for c in classes])
    vol = np.array([r.uniform(*CLASS_RANGES[c]["vol"]) for c in classes])
    asset_ids = [f"A{i+1:02d}" for i in range(n_assets)]

    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if classes[i] == classes[j]:
                val = r.uniform(0.5, 0.8) if classes[i] == "Equity" else r.uniform(0.35, 0.6)
            elif {classes[i], classes[j]} == {"Equity", "FixedIncome"}:
                val = r.uniform(-0.1, 0.2)
            elif "Cash" in (classes[i], classes[j]):
                val = r.uniform(-0.02, 0.02)
            else:
                val = r.uniform(0.0, 0.3)
            corr[i, j] = corr[j, i] = val

    # project to nearest PSD (same method as the 12-asset generator)
    eigval, eigvec = np.linalg.eigh(corr)
    eigval = np.clip(eigval, 1e-6, None)
    corr_psd = eigvec @ np.diag(eigval) @ eigvec.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    np.fill_diagonal(corr_psd, 1.0)

    cov = np.diag(vol) @ corr_psd @ np.diag(vol)
    return asset_ids, mu, cov


if __name__ == "__main__":
    for n in [12, 16, 20, 24, 28]:
        ids, mu, cov = generate_universe(n)
        eigmin = np.linalg.eigvalsh(cov).min()
        print(f"n={n}: PSD check (min eigenvalue >= 0): {eigmin:.8f}")
