"""
Source-based QoS routing (paper Section 3.2, Algorithm 2) and baselines.

Packets of each CBR flow use the cached source route while every hop is still
inside transmission range. When the primary route breaks, OBQR falls back to
its next pre-computed alternative path (multipath) before re-flooding.
"""
from __future__ import annotations

from collections import deque
from itertools import islice

import networkx as nx
import numpy as np

from . import qos
from .clustering import ClusteringProtocol, Snapshot

PROC_MS = 10.0   # per-hop processing of RREQ/RREP during discovery


def path_valid(path, s: Snapshot) -> bool:
    return all(s.A[a, b] for a, b in zip(path[:-1], path[1:]))


def bfs_component(adj_mask: np.ndarray, src: int, allowed: np.ndarray | None = None):
    n = adj_mask.shape[0]
    seen = np.zeros(n, bool)
    seen[src] = True
    q = deque([src])
    parent = {src: -1}
    while q:
        u = q.popleft()
        for v in np.where(adj_mask[u] & ~seen)[0]:
            if allowed is not None and not allowed[v]:
                continue
            seen[v] = True
            parent[int(v)] = u
            q.append(int(v))
    return seen, parent


class Router:
    name = label = "base"
    multipath = False

    def __init__(self, n: int, rng: np.random.Generator):
        self.n = n
        self.rng = rng
        self.routes: dict[int, list[list[int]]] = {}
        self.sent = self.delivered = self.ctrl = self.discoveries = self.breaks = 0
        self.delay_sum = 0.0
        self.hops_sum = 0
        self.route_ages: list[float] = []
        self.route_born: dict[int, float] = {}

    # ------------------------------------------------------------------ API
    def discover(self, s: Snapshot, clus: ClusteringProtocol, src: int, dst: int):
        """Return (list_of_paths, control_messages)."""
        raise NotImplementedError

    def step(self, s: Snapshot, clus: ClusteringProtocol, flows, pkts_per_flow: int):
        for f, (src, dst) in enumerate(flows):
            paths = self.routes.get(f, [])
            path = next((p for p in paths if path_valid(p, s)), None)
            extra = 0.0
            if path is None:
                if paths:  # the cached route broke -> RERR back to the source
                    self.breaks += 1
                    self.ctrl += len(paths[0]) - 1
                    if f in self.route_born:
                        self.route_ages.append(s.t - self.route_born[f])
                new, msgs = self.discover(s, clus, src, dst)
                self.ctrl += msgs
                self.discoveries += 1
                if not new:
                    self.routes.pop(f, None)
                    self.route_born.pop(f, None)
                    self.sent += pkts_per_flow
                    continue
                self.routes[f] = new
                self.route_born[f] = s.t
                path = new[0]
                extra = 2 * (len(path) - 1) * PROC_MS
            elif path is not paths[0]:
                # switch to a backup path: one notification, no flood
                self.breaks += 1
                self.ctrl += 1
                self.routes[f] = [p for p in paths if p is path] + [p for p in paths if p is not path]
            self._send(s, path, pkts_per_flow, extra)

    def _send(self, s: Snapshot, path, k: int, extra_ms: float):
        a, b = np.array(path[:-1]), np.array(path[1:])
        d = s.D[a, b]
        p_ok = float(np.prod(qos.hop_success(d, s.R)))
        delay = float((qos.link_cost(d, s.R) * (qos.PKT_BITS / qos.RATE_BPS * 1e3 + 0.1
                                               + 0.25 * s.degree[b])).sum())
        ok = self.rng.random(k) < p_ok
        self.sent += k
        self.delivered += int(ok.sum())
        if ok.any():
            self.delay_sum += delay * ok.sum() + extra_ms
            self.hops_sum += (len(path) - 1) * int(ok.sum())

    def summary(self, sim_time: float) -> dict:
        d = max(self.delivered, 1)
        return {
            "pdr": self.delivered / max(self.sent, 1),
            "throughput_kbps": self.delivered * qos.PKT_BITS / sim_time / 1e3,
            "avg_delay_ms": self.delay_sum / d,
            "routing_overhead": self.ctrl / d,         # normalised routing load
            "ctrl_msgs": self.ctrl,
            "discoveries": self.discoveries,
            "avg_hops": self.hops_sum / d,
            "route_breaks": self.breaks,
        }


# ======================================================================= OBQR
class OBQRRouter(Router):
    """Algorithm 2.

    1. Route requests are flooded only over the cluster backbone (cluster heads
       + gateways), so discovery costs |backbone| messages instead of |V|.
    2. Each candidate link gets the single-objective weight of Eq. (16):
       link-availability time (Eq. 12), cost ct (Gaussian link model,
       Eq. 13-14) and delay dl.
    3. A heuristic (Yen's k-shortest simple paths on W) returns k routes
       {P1..Pk}; the cheapest is used, the rest are kept as backups.
    """
    name, label = "obqr", "OBQR"
    multipath = True

    def __init__(self, n, rng, k=3, alpha=0.4, beta=0.3, gamma=0.3, use_backbone=True):
        super().__init__(n, rng)
        self.k, self.alpha, self.beta, self.gamma = k, alpha, beta, gamma
        self.use_backbone = use_backbone

    def link_weights(self, s: Snapshot, I, J):
        let = s.let[I, J]
        ct = qos.link_cost(s.D[I, J], s.R)
        dl_ij = ct * (qos.PKT_BITS / qos.RATE_BPS * 1e3 + 0.1 + 0.25 * s.degree[J])
        dl_ji = ct * (qos.PKT_BITS / qos.RATE_BPS * 1e3 + 0.1 + 0.25 * s.degree[I])
        dl = (dl_ij + dl_ji) / 2
        W = qos.scalarized_weight(let, ct, dl, self.alpha, self.beta, self.gamma)
        return W, let, ct, dl

    def candidate_paths(self, s: Snapshot, clus: ClusteringProtocol, src: int, dst: int, k=None):
        allowed = np.ones(self.n, bool)
        if self.use_backbone:
            allowed = clus.is_ch | clus.gateways(s)
            allowed[src] = allowed[dst] = True
        seen, _ = bfs_component(s.A, src, allowed)
        msgs = int(seen.sum())                 # one RREQ rebroadcast per backbone node
        if not seen[dst]:
            return [], msgs, None
        sub = np.where(seen)[0]
        Asub = np.triu(s.A[np.ix_(sub, sub)])
        ii, jj = np.where(Asub)
        I, J = sub[ii], sub[jj]
        W, let, ct, dl = self.link_weights(s, I, J)
        G = nx.Graph()
        for a, b, w, l, c, d in zip(I.tolist(), J.tolist(), W, let, ct, dl):
            G.add_edge(a, b, w=float(w) + 1e-6, let=float(l), ct=float(c), dl=float(d))
        paths = list(islice(nx.shortest_simple_paths(G, src, dst, weight="w"), k or self.k))
        return paths, msgs, G

    def discover(self, s, clus, src, dst):
        paths, msgs, _ = self.candidate_paths(s, clus, src, dst)
        return paths, msgs + sum(len(p) - 1 for p in paths)   # + RREPs


# ================================================================= baselines
class AODVRouter(Router):
    """Reactive min-hop routing: RREQ flooded by every reachable vehicle,
    single path, rediscover on every break."""
    name, label = "aodv", "AODV (min-hop)"

    def discover(self, s, clus, src, dst):
        seen, parent = bfs_component(s.A, src)
        msgs = int(seen.sum())
        if not seen[dst]:
            return [], msgs
        path, v = [], dst
        while v != -1:
            path.append(int(v))
            v = parent[v]
        path = path[::-1]
        return [path], msgs + len(path) - 1


class GreedyGeoRouter(Router):
    """Position-based greedy forwarding. No route
    discovery; each hop hands the packet to the neighbour closest to the
    destination, failing at a local maximum."""
    name, label = "greedy", "Greedy geographic"

    def step(self, s, clus, flows, pkts_per_flow):
        for src, dst in flows:
            path, cur, seen = [src], src, {src}
            while cur != dst and len(path) < 50:
                nb = np.where(s.A[cur])[0]
                if s.A[cur, dst]:
                    cur = dst
                else:
                    if len(nb) == 0:
                        cur = -1
                        break
                    dn = s.D[nb, dst]
                    j = nb[np.argmin(dn)]
                    if dn.min() >= s.D[cur, dst] or j in seen:
                        cur = -1
                        break
                    cur = int(j)
                seen.add(cur)
                path.append(cur)
            if cur != dst:
                self.sent += pkts_per_flow
                continue
            self._send(s, path, pkts_per_flow, 0.0)


ALL_ROUTERS = [AODVRouter, GreedyGeoRouter, OBQRRouter]
