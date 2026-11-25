"""Figures for the slides (matplotlib, static PNG)."""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical palette (fixed slot order) — colour follows the entity.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

CLUS = [("obqr", "OBQR", SLOT[0], "o"),
        ("position", "Position-based", SLOT[2], "s"),
        ("speed", "Speed-based", SLOT[3], "^"),
        ("lowest_id", "Lowest-ID", SLOT[4], "v")]
ROUT = [("obqr", "OBQR", SLOT[0], "o"),
        ("aodv", "AODV (min-hop)", SLOT[2], "s"),
        ("greedy", "Greedy geographic", SLOT[3], "^")]
XLAB = {"speed": "Mean desired speed (km/h)", "range": "Transmission range (m)",
        "density": "Number of vehicles"}

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "savefig.dpi": 200, "figure.facecolor": "white",
})


def collect(raw, sweep, family, key, metric):
    xs = sorted({r["x"] for r in raw if r["sweep"] == sweep})
    mean, std = [], []
    for x in xs:
        vals = [r[family][key][metric] for r in raw if r["sweep"] == sweep and r["x"] == x]
        mean.append(np.mean(vals))
        std.append(np.std(vals))
    return np.array(xs), np.array(mean), np.array(std)


def panel(ax, raw, sweep, family, series, metric, title, ylab, scale=1.0):
    for key, label, color, mk in series:
        x, m, s = collect(raw, sweep, family, key, metric)
        m, s = m * scale, s * scale
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0)
        ax.plot(x, m, color=color, lw=2, marker=mk, ms=6, label=label,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_title(title, loc="left")
    ax.set_xlabel(XLAB[sweep])
    ax.set_ylabel(ylab)
    ax.set_xticks(x)


def fig_family(raw, sweep, family, out):
    if family == "clustering":
        series = CLUS
        specs = [("cluster_lifetime", "Cluster-head lifetime", "seconds  (higher = better)", 1),
                 ("transitions_per_vehicle", "Cluster transitions", "per vehicle  (lower = better)", 1),
                 ("discovery_msgs", "Clustering control messages", "total  (lower = better)", 1),
                 ("avg_cluster_size", "Average cluster size", "vehicles / cluster", 1)]
    else:
        series = ROUT
        specs = [("pdr", "Packet delivery ratio", "%  (higher = better)", 100),
                 ("throughput_kbps", "Throughput", "kbit/s  (higher = better)", 1),
                 ("avg_delay_ms", "Average end-to-end delay", "ms  (lower = better)", 1),
                 ("routing_overhead", "Normalised routing load", "ctrl msgs / delivered pkt", 1)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (m, t, y, sc) in zip(axes, specs):
        panel(ax, raw, sweep, family, series, m, t, y, sc)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, 1.04),
               fontsize=11, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{family}_{sweep}.png"), bbox_inches="tight")
    plt.close(fig)


def write_summary(raw, path):
    rows = []
    for sweep in ("speed", "range", "density"):
        for fam, series in (("clustering", CLUS),
                            ("routing", ROUT)):
            metrics = list(raw[0][fam][series[0][0]].keys())
            for key, label, *_ in series:
                for m in metrics:
                    x, mean, std = collect(raw, sweep, fam, key, m)
                    for xi, mi, si in zip(x, mean, std):
                        rows.append([sweep, xi, fam, label, m, round(float(mi), 4), round(float(si), 4)])
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sweep", "x", "family", "protocol", "metric", "mean", "std"])
        wr.writerows(rows)


def make_all(out_dir):
    with open(os.path.join(out_dir, "raw.json")) as f:
        raw = json.load(f)
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    for sweep in ("speed", "range", "density"):
        fig_family(raw, sweep, "clustering", figs)
        fig_family(raw, sweep, "routing", figs)
    write_summary(raw, os.path.join(out_dir, "summary.csv"))


if __name__ == "__main__":
    make_all(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
