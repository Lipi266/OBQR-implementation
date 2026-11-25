"""
Cluster formation (paper Section 3.1, Eq. 6-10, Algorithm 1) and baselines.

All schemes share the *same* election/maintenance framework (greedy election in
priority order, sitting cluster heads keep priority, members stay with their
head while it is still eligible). They differ only in

  * ``affinity``  - which neighbours may share a cluster, and
  * ``ch_score``  - who is preferred as cluster head.

So any difference in the results is caused by the similarity metric, which is
exactly the claim the paper makes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import qos


# --------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Everything a vehicle learns from one round of beacons."""
    P: np.ndarray        # positions  (N,2)
    H: np.ndarray        # unit orientation vectors (N,2)
    speed: np.ndarray    # (N,)
    R: float             # transmission range
    t: float = 0.0

    @property
    def n(self):
        return len(self.P)

    @property
    def V(self):
        return self.speed[:, None] * self.H

    @cached_property
    def D(self):
        return qos.eq_dis(self.P)

    @cached_property
    def A(self):
        """neiMat: adjacency (within transmission range)."""
        A = self.D <= self.R
        np.fill_diagonal(A, False)
        return A

    @cached_property
    def degree(self):
        return self.A.sum(1)

    @cached_property
    def cos_sim(self):
        """Eq. (6): cosine similarity of the orientation vectors."""
        Hn = self.H / np.linalg.norm(self.H, axis=1, keepdims=True)
        return np.clip(Hn @ Hn.T, -1, 1)

    @cached_property
    def let(self):
        return qos.let_vector(self.P, self.V, self.R)


def obqr_terms(snap: Snapshot, i: int, cap: float):
    """Per-vehicle quantities from Algorithm 1 (used by the UI explainer)."""
    nb = np.where(snap.A[i])[0]
    eps = snap.cos_sim[i, nb]
    k_hat = 1 - eps                                   # Eq. (7)
    k_theta = np.arccos(np.clip(eps, -1, 1)) / np.pi  # Eq. (9)
    eps_theta = 1 - k_theta                           # Eq. (10)
    rho = float(k_hat.mean()) if len(nb) else 0.0     # Eq. (8)
    return nb, eps, k_hat, k_theta, eps_theta, rho


# --------------------------------------------------------------------------
@dataclass
class ClusterStats:
    tenures: list = field(default_factory=list)
    transitions: int = 0
    discovery_msgs: int = 0
    n_clusters: list = field(default_factory=list)
    singleton_frac: list = field(default_factory=list)
    ch_changes: int = 0


class ClusteringProtocol:
    name = "base"
    label = "base"

    def __init__(self, n: int):
        self.n = n
        self.is_ch = np.zeros(n, bool)
        self.head = -np.ones(n, int)       # cluster head of every vehicle
        self.ch_since = np.full(n, np.nan)
        self.stats = ClusterStats()
        self.last_t = 0.0

    # --- to override --------------------------------------------------------
    def affinity(self, s: Snapshot) -> tuple[np.ndarray, np.ndarray]:
        """Return (ok, dist): ok[i,j] - may share a cluster; dist[i,j] - how
        different they are (lower = better match)."""
        raise NotImplementedError

    def ch_score(self, s: Snapshot, ok: np.ndarray, dist: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- shared election + maintenance -------------------------------------
    def update(self, s: Snapshot):
        ok, dist = self.affinity(s)
        ok = ok & s.A
        score = self.ch_score(s, ok, dist)
        prio = score + 1e6 * self.is_ch          # sitting heads go first
        order = np.argsort(-prio, kind="stable")
        new_ch = np.zeros(self.n, bool)
        for i in order:
            if not (new_ch & ok[i]).any():
                new_ch[i] = True
        new_head = np.where(new_ch, np.arange(self.n), -1)
        for i in np.where(~new_ch)[0]:
            cand = np.where(new_ch & ok[i])[0]
            if self.head[i] in cand:                    # stay if possible
                new_head[i] = self.head[i]
            else:
                new_head[i] = cand[np.argmin(dist[i, cand])]
        self._book_keep(new_ch, new_head, s.t)
        self.is_ch, self.head = new_ch, new_head

    def _book_keep(self, new_ch, new_head, t):
        st = self.stats
        first = np.all(self.head < 0)
        if not first:
            changed = (new_head != self.head)
            st.transitions += int(changed.sum())
            st.discovery_msgs += 2 * int(changed.sum())     # JOIN_REQ + JOIN_ACK
        elected = new_ch & ~self.is_ch
        resigned = self.is_ch & ~new_ch
        st.discovery_msgs += int(elected.sum()) + int(resigned.sum())  # CH_ANNOUNCE / CH_RESIGN
        if not first:
            st.ch_changes += int(elected.sum() + resigned.sum())
        for i in np.where(resigned)[0]:
            st.tenures.append(t - self.ch_since[i])
            self.ch_since[i] = np.nan
        self.ch_since[elected] = t
        sizes = np.bincount(new_head, minlength=self.n)[new_ch]
        st.n_clusters.append(int(new_ch.sum()))
        st.singleton_frac.append(float((sizes == 1).mean()) if len(sizes) else 0.0)
        self.last_t = t

    def gateways(self, s: Snapshot) -> np.ndarray:
        """Members with a neighbour in another cluster."""
        other = self.head[:, None] != self.head[None, :]
        return (~self.is_ch) & (s.A & other).any(1)

    def summary(self) -> dict:
        st = self.stats
        ongoing = [self.last_t - x for x in self.ch_since[~np.isnan(self.ch_since)]]
        # only count heads with members as "clusters" for lifetime
        ten = st.tenures + ongoing
        return {
            "cluster_lifetime": float(np.mean(ten)) if ten else 0.0,
            "transitions_per_vehicle": st.transitions / self.n,
            "discovery_msgs": st.discovery_msgs,
            "avg_clusters": float(np.mean(st.n_clusters)),
            "singleton_frac": float(np.mean(st.singleton_frac)),
            "avg_cluster_size": self.n / max(float(np.mean(st.n_clusters)), 1e-9),
        }


# ============================================================ OBQR (paper)
class OBQRClustering(ClusteringProtocol):
    """Orientation-based clustering, Algorithm 1.

    * epsilon_ij = cosine similarity of orientation vectors       (Eq. 6)
    * k_ij = 1 - epsilon_ij  (cosine distance)                     (Eq. 7)
    * rho_i = mean k over i's neighbours (threshold distance)      (Eq. 8)
    * j is in i's cluster neighbourhood if k_ij <= rho_i (adaptive) and
      k_ij <= cap (implementation choice: never group > ~45 deg apart)
    * cluster head = most similar neighbours, lowest mean cosine distance
    """
    name, label = "obqr", "OBQR (direction)"

    def __init__(self, n, cap=1 - np.cos(np.deg2rad(45))):
        super().__init__(n)
        self.cap = cap

    def _dist(self, s: Snapshot):
        return 1 - s.cos_sim

    def affinity(self, s):
        k = self._dist(s)
        Af = s.A.astype(float)
        deg = np.maximum(Af.sum(1), 1)
        rho = (k * Af).sum(1) / deg                        # Eq. (8)
        thr = np.minimum(np.maximum(rho, 1e-6), self.cap)
        ok = (k <= thr[:, None] + 1e-9) | (k <= thr[None, :] + 1e-9)
        return ok, k

    def ch_score(self, s, ok, dist):
        deg = ok.sum(1)
        mean_k = np.where(deg > 0, (dist * ok).sum(1) / np.maximum(deg, 1), 1.0)
        mean_d = np.where(deg > 0, (s.D * ok).sum(1) / np.maximum(deg, 1), s.R)
        # max adjacency, then lowest cosine distance, then centrality
        return deg + (1 - mean_k / 2) + 0.1 * (1 - mean_d / s.R)


# ============================================================ baselines
class LowestIDClustering(ClusteringProtocol):
    """Classic Lowest-ID clustering: any neighbour, smallest ID is head."""
    name, label = "lowest_id", "Lowest-ID"

    def affinity(self, s):
        return s.A.copy(), s.D / s.R

    def ch_score(self, s, ok, dist):
        return -np.arange(self.n) / self.n


class PositionClustering(ClusteringProtocol):
    """Position-only clustering: any neighbour may join; the most
    central vehicle (many neighbours, small mean distance) becomes head.
    Ignores direction entirely."""
    name, label = "position", "Position-based"

    def affinity(self, s):
        return s.A.copy(), s.D / s.R

    def ch_score(self, s, ok, dist):
        deg = ok.sum(1)
        mean_d = np.where(deg > 0, (s.D * ok).sum(1) / np.maximum(deg, 1), s.R)
        return deg + (1 - mean_d / s.R)


class SpeedClustering(ClusteringProtocol):
    """Speed-based mobility clustering: neighbours with a
    similar *speed* (|dv| <= 5 m/s) share a cluster; head = lowest mean
    relative speed. Uses magnitude, not orientation - exactly the contrast the
    OBQR abstract draws."""
    name, label = "speed", "Speed-based"

    def __init__(self, n, dv_max=5.0):
        super().__init__(n)
        self.dv_max = dv_max

    def affinity(self, s):
        dv = np.abs(s.speed[:, None] - s.speed[None, :])
        return dv <= self.dv_max, dv / self.dv_max

    def ch_score(self, s, ok, dist):
        deg = ok.sum(1)
        mean_dv = np.where(deg > 0, (dist * ok).sum(1) / np.maximum(deg, 1), 1.0)
        return deg + (1 - mean_dv)


ALL_CLUSTERING = [LowestIDClustering, PositionClustering, SpeedClustering,
                  OBQRClustering]
