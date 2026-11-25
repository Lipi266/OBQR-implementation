"""
Urban mobility model for OBQR (paper Section 2.1, Fig. 2).

* Manhattan grid road network, one lane per direction (right-hand traffic).
* Car-following model (Eq. 1-3): the Intelligent Driver Model (IDM) gives the
  acceleration a_m(G_x, v_a, dv_a) as a function of the gap G_x to the leader,
  own speed and the speed difference to the leader.
* Turning-decision model (Fig. 2b, Eq. 4): at each intersection a vehicle picks
  straight / left / right; before a turn it slows to a safe turning speed and
  deceleration is bounded by the safe-braking limit b_safe = 4 m/s^2 (Eq. 4-5).
* Orientation in [0, 2*pi) is the heading of the lane the vehicle is on.

The model is deliberately small (pure numpy) so the same trace can be replayed
for every protocol, which makes protocol comparisons fair (common random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

B_SAFE = 4.0  # m/s^2, paper Eq. (4)-(5)


@dataclass
class MobilityConfig:
    n_vehicles: int = 100
    mean_speed_kmh: float = 54.0
    area: float = 1600.0          # metres (square)
    block: float = 400.0          # distance between intersections
    lane_offset: float = 4.0      # lateral offset of a lane from road centre
    veh_len: float = 4.5          # L_alpha in Eq. (1)
    turn_speed: float = 8.0       # m/s, safe speed through an intersection
    p_straight: float = 0.6       # turning-decision probabilities
    dt: float = 0.5
    seed: int = 0
    # IDM parameters
    a_max: float = 2.0
    b_comf: float = 2.0
    T_headway: float = 1.2
    s0: float = 2.0


class RoadNetwork:
    """Grid of intersections; every road segment has two directed lanes."""

    def __init__(self, area: float, block: float, lane_offset: float):
        n = int(round(area / block)) + 1
        self.n = n
        self.nodes = np.array([(i * block, j * block) for j in range(n) for i in range(n)], float)
        lanes = []
        for j in range(n):
            for i in range(n):
                u = j * n + i
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        lanes.append((u, jj * n + ii))
        self.lanes = np.array(lanes, int)
        p0 = self.nodes[self.lanes[:, 0]]
        p1 = self.nodes[self.lanes[:, 1]]
        d = p1 - p0
        self.length = np.linalg.norm(d, axis=1)
        self.heading = d / self.length[:, None]
        # right-hand normal (x right, y up)
        self.normal = np.stack([self.heading[:, 1], -self.heading[:, 0]], axis=1)
        self.lane_offset = lane_offset
        self.out_lanes: dict[int, list[int]] = {}
        for k, (u, _) in enumerate(self.lanes):
            self.out_lanes.setdefault(int(u), []).append(k)

    def turn_kind(self, lane_in: int, lane_out: int) -> str:
        h1, h2 = self.heading[lane_in], self.heading[lane_out]
        dot = float(h1 @ h2)
        if dot > 0.5:
            return "straight"
        if dot < -0.5:
            return "uturn"
        cross = h1[0] * h2[1] - h1[1] * h2[0]
        return "left" if cross > 0 else "right"

    def segments(self):
        """Road centre-lines for drawing (undirected, deduplicated)."""
        seen, out = set(), []
        for u, v in self.lanes:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                out.append([*self.nodes[u].tolist(), *self.nodes[v].tolist()])
        return out


class Mobility:
    def __init__(self, cfg: MobilityConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.net = RoadNetwork(cfg.area, cfg.block, cfg.lane_offset)
        self.t = 0.0
        n = cfg.n_vehicles
        vmean = cfg.mean_speed_kmh / 3.6
        self.v0 = np.clip(self.rng.normal(vmean, 0.15 * vmean, n), 0.5 * vmean, 1.5 * vmean)
        self.lane = np.zeros(n, int)
        self.s = np.zeros(n)
        self._place()
        self.v = 0.8 * self.v0
        self.next_lane = np.array([self._choose_next(int(l)) for l in self.lane])

    # ------------------------------------------------------------------ setup
    def _place(self):
        n, L = self.cfg.n_vehicles, len(self.net.lanes)
        placed: dict[int, list[float]] = {}
        for i in range(n):
            for _ in range(1000):
                l = int(self.rng.integers(L))
                s = float(self.rng.uniform(0, self.net.length[l] - 10))
                if all(abs(s - o) > 15 for o in placed.get(l, [])):
                    break
            placed.setdefault(l, []).append(s)
            self.lane[i], self.s[i] = l, s

    def _choose_next(self, lane: int) -> int:
        v_node = int(self.net.lanes[lane, 1])
        opts = [k for k in self.net.out_lanes[v_node] if self.net.turn_kind(lane, k) != "uturn"]
        if not opts:  # dead end -> U-turn
            return self.net.out_lanes[v_node][0]
        straight = [k for k in opts if self.net.turn_kind(lane, k) == "straight"]
        turns = [k for k in opts if k not in straight]
        if straight and (not turns or self.rng.random() < self.cfg.p_straight):
            return straight[0]
        return int(self.rng.choice(turns))

    # ------------------------------------------------------------------ state
    @property
    def positions(self) -> np.ndarray:
        net = self.net
        base = net.nodes[net.lanes[self.lane, 0]]
        return base + self.s[:, None] * net.heading[self.lane] + net.lane_offset * net.normal[self.lane]

    @property
    def headings(self) -> np.ndarray:
        """Unit orientation vector of every vehicle."""
        return self.net.heading[self.lane]

    @property
    def velocities(self) -> np.ndarray:
        return self.v[:, None] * self.headings

    @property
    def angles(self) -> np.ndarray:
        h = self.headings
        return np.mod(np.arctan2(h[:, 1], h[:, 0]), 2 * np.pi)

    # ------------------------------------------------------------------- step
    def step(self):
        c, net = self.cfg, self.net
        n = c.n_vehicles
        order = np.lexsort((self.s, self.lane))
        ls = self.lane[order]
        leader = -np.ones(n, int)
        same = ls[:-1] == ls[1:]
        leader[order[:-1][same]] = order[1:][same]
        gap = np.full(n, 1e4)
        has = leader >= 0
        gap[has] = self.s[leader[has]] - self.s[has] - c.veh_len  # Eq. (1)
        # front-most vehicle of a lane looks at the tail of its next lane
        uniq, first_idx = np.unique(ls, return_index=True)
        tail = dict(zip(uniq.tolist(), order[first_idx].tolist()))
        for i in np.where(~has)[0]:
            j = tail.get(int(self.next_lane[i]))
            if j is not None and j != i:
                leader[i] = j
                gap[i] = net.length[self.lane[i]] - self.s[i] + self.s[j] - c.veh_len
        gap = np.maximum(gap, 0.1)
        dv = np.where(leader >= 0, self.v - self.v[np.maximum(leader, 0)], 0.0)

        # turning-decision model: slow down before a turn
        v0 = self.v0.copy()
        turning = np.array([net.turn_kind(int(a), int(b)) != "straight"
                            for a, b in zip(self.lane, self.next_lane)])
        to_end = net.length[self.lane] - self.s
        brake_dist = np.maximum(self.v ** 2 - c.turn_speed ** 2, 0) / (2 * 0.75 * B_SAFE) + 10
        slow = turning & (to_end < brake_dist)
        v0[slow] = c.turn_speed

        # IDM acceleration a_m(G_x, v, dv)  (Eq. 3)
        s_star = c.s0 + np.maximum(0, self.v * c.T_headway + self.v * dv / (2 * np.sqrt(c.a_max * c.b_comf)))
        acc = c.a_max * (1 - (self.v / v0) ** 4 - (s_star / gap) ** 2)
        acc = np.clip(acc, -2 * B_SAFE, c.a_max)  # emergency bound; normal braking stays above -b_safe

        dt = c.dt
        v_new = np.maximum(self.v + acc * dt, 0.0)
        ds = np.maximum((self.v + v_new) / 2 * dt, 0.0)
        ds = np.minimum(ds, np.maximum(gap - 0.5, 0) + 0.0)  # never drive into the leader
        self.s = self.s + ds
        self.v = v_new

        # lane transitions at intersections
        for i in np.where(self.s >= net.length[self.lane])[0]:
            over = self.s[i] - net.length[self.lane[i]]
            self.lane[i] = self.next_lane[i]
            self.s[i] = min(over, net.length[self.lane[i]] - 0.1)
            self.next_lane[i] = self._choose_next(int(self.lane[i]))
        self.t += dt
