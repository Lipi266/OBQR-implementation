"""Run one scenario: a single mobility trace replayed for every protocol."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .clustering import ALL_CLUSTERING, OBQRClustering, Snapshot
from .mobility import Mobility, MobilityConfig
from .routing import ALL_ROUTERS, OBQRRouter, bfs_component


@dataclass
class Scenario:
    n_vehicles: int = 100
    speed_kmh: float = 54.0
    tx_range: float = 250.0
    sim_time: float = 300.0
    warmup: float = 20.0
    n_flows: int = 10
    pkts_per_sec: int = 4
    seed: int = 0


def pick_flows(s: Snapshot, n_flows: int, rng) -> list[tuple[int, int]]:
    """Random source/destination pairs that are multi-hop connected at start."""
    flows, tries = [], 0
    while len(flows) < n_flows and tries < 2000:
        tries += 1
        a, b = (int(x) for x in rng.choice(s.n, 2, replace=False))
        seen, _ = bfs_component(s.A, a)
        if seen[b] and not s.A[a, b] or tries > 1500:
            flows.append((a, b))
    return flows


def run_scenario(sc: Scenario, clustering=None, routers=None) -> dict:
    """routers: list of Router classes, or dict name -> factory(n, rng)."""
    clustering = clustering if clustering is not None else ALL_CLUSTERING
    routers = routers if routers is not None else ALL_ROUTERS
    mob = Mobility(MobilityConfig(n_vehicles=sc.n_vehicles, mean_speed_kmh=sc.speed_kmh, seed=sc.seed))
    rng = np.random.default_rng(1000 + sc.seed)
    clus = {c.name: c(sc.n_vehicles) for c in clustering}
    if "obqr" not in clus:
        clus["obqr"] = OBQRClustering(sc.n_vehicles)
    if not isinstance(routers, dict):
        routers = {r.name: r for r in routers}
    routs = {k: f(sc.n_vehicles, np.random.default_rng(7 + sc.seed)) for k, f in routers.items()}

    steps_per_sec = int(round(1 / mob.cfg.dt))
    flows = None
    speeds = []
    n_sec = int(sc.sim_time)
    for sec in range(n_sec):
        for _ in range(steps_per_sec):
            mob.step()
        s = Snapshot(mob.positions, mob.headings, mob.v.copy(), sc.tx_range, t=float(sec))
        speeds.append(mob.v.mean())
        for c in clus.values():
            c.update(s)
        if sec < sc.warmup:
            continue
        if flows is None:
            flows = pick_flows(s, sc.n_flows, rng)
        for r in routs.values():
            r.step(s, clus["obqr"], flows, sc.pkts_per_sec)

    active = sc.sim_time - sc.warmup
    out = {"scenario": asdict(sc), "mean_speed_kmh_measured": float(np.mean(speeds) * 3.6),
           "clustering": {k: c.summary() for k, c in clus.items() if k in {x.name for x in clustering}},
           "routing": {k: r.summary(active) for k, r in routs.items()}}
    return out
