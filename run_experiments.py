"""
Run the evaluation sweeps (like the paper's Fig. 5 and Fig. 6).

    python run_experiments.py            # full run (~8 min on 8 cores)
    python run_experiments.py --quick    # smoke test

Outputs: results/raw.json, results/summary.csv, results/figures/*.png
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

from obqr.clustering import ALL_CLUSTERING  # noqa: E402
from obqr.experiment import Scenario, run_scenario  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SWEEPS = {
    "speed": ("speed_kmh", [18, 36, 54, 72, 90, 108], dict(n_vehicles=100, tx_range=250)),
    "range": ("tx_range", [100, 150, 200, 250, 300], dict(n_vehicles=100, speed_kmh=54)),
    "density": ("n_vehicles", [25, 50, 100, 150, 200], dict(speed_kmh=54, tx_range=250)),
}


def _job(args):
    sweep, key, val, base, seed, sim_time = args
    sc = Scenario(**{**base, key: val}, seed=seed, sim_time=sim_time)
    r = run_scenario(sc, clustering=ALL_CLUSTERING)
    r["sweep"], r["x"] = sweep, val
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time", type=float, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    seeds = 1 if a.quick else a.seeds
    T = 60 if a.quick else a.time
    out = os.path.join(OUT, "quick") if a.quick else OUT
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    jobs = [(sw, key, v, base, s, T) for sw, (key, vals, base) in SWEEPS.items()
            for v in vals for s in range(seeds)]
    print(f"{len(jobs)} runs x {T:.0f}s ...", flush=True)
    raw = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            raw.append(r)
            if i % 8 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    with open(os.path.join(out, "raw.json"), "w") as f:
        json.dump(raw, f, indent=1, default=float)

    from plots import make_all
    make_all(out)
    print("done ->", out)


if __name__ == "__main__":
    main()
