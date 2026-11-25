# OBQR: Orientation-Based Source QoS Routing in VANETs (re-implementation)

A Python re-implementation of *Nahar, Das & Das, "OBQR: Orientation-Based Source QoS Routing in VANETs", ACM MSWiM 2020*, with a live demo in the browser.

OBQR works in two phases:

1. **Clustering (Algorithm 1):** vehicles driving in the same direction are grouped using cosine similarity of their heading. The vehicle with the lowest cosine distance and the most neighbours becomes the cluster head.
2. **QoS routing (Algorithm 2):** each link gets one weight from link-availability time, cost and delay (scalarisation). The source finds the paths with the lowest total weight, uses the best one and keeps the others as backups.

## Quick start

```bash
pip install -r requirements.txt
python ui/server.py              # live demo -> http://127.0.0.1:5050
python run_experiments.py        # evaluation sweeps (~8 min on 8 cores), or --quick
```

## What is implemented

| Paper | Code |
|---|---|
| Car-following model (Eq. 1–3), safe turning with b_safe = 4 m/s² (Eq. 4–5) | `obqr/mobility.py` |
| Cosine similarity, cosine distance, threshold ρ̂, angular distance (Eq. 6–10), Algorithm 1 | `obqr/clustering.py` (`OBQRClustering`) |
| Euclidean distance (Eq. 11), link-availability time (Eq. 12), Gaussian link model (Eq. 13–14), scalarisation (Eq. 15–16) | `obqr/qos.py` |
| Algorithm 2: route discovery, k lowest-weight paths, switching to a backup path | `obqr/routing.py` (`OBQRRouter`) |
| Comparison methods | Lowest-ID, position-based and speed-based clustering; AODV (min-hop) and greedy geographic routing |

Implementation choices where the paper leaves details open:

* α = 0.4, β = 0.3, γ = 0.3 for the scalarised weight.
* Link-availability time is computed from the two vehicles' positions and velocity vectors.
* Paths with the lowest total weight are chosen, so longer-lasting links get a lower weight.
* The source keeps k = 3 paths (best + two backups).
* The similarity threshold is capped at 45° so vehicles on crossing streets are not grouped.

## Layout

```
obqr/                core library: mobility, clustering, qos, routing, experiment
ui/server.py         Flask demo server;  ui/static/  front end (no external JS)
run_experiments.py   sweeps over speed / transmission range / density
plots.py             figures -> results/figures/*.png, results/summary.csv
results/             raw.json, summary.csv, figures/
```

## Simulation setup

* Python simulator on a 1.6 km × 1.6 km road grid (the paper uses ns-3.28 + SUMO on 10 km × 10 km).
* 25–200 vehicles, 18–108 km/h, 100–300 m transmission range, 300 s runs, 5 seeds.
* Every method runs on exactly the same vehicle movements, so comparisons are fair.
* Lane changing, traffic lights and roadside units are not modelled.
