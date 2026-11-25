"""
QoS link metrics used by OBQR (paper Section 3.2, Eq. 11-16).

Every function works on whole N x N matrices so a snapshot of the network is
evaluated in one shot.
"""
from __future__ import annotations

import numpy as np

LET_CAP = 120.0          # seconds; a link that "never breaks" is capped here
PKT_BITS = 512 * 8       # Table 1: packet size 512 bytes
RATE_BPS = 6e6           # DSRC default data rate
MAC_RETRIES = 3


# ----------------------------------------------------------------- distances
def eq_dis(P: np.ndarray) -> np.ndarray:
    """Eq. (11): Euclidean distance between every pair of vehicles."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# ------------------------------------------------------ link availability time
def let_vector(P: np.ndarray, V: np.ndarray, R: float) -> np.ndarray:
    """Link-availability time, Eq. (12), computed with velocity vectors.

    Solves |(p_i - p_j) + (v_i - v_j) t| = R for the positive root t: the time
    until the two vehicles move out of each other's radio range.
    """
    b = P[:, None, 0] - P[None, :, 0]
    d = P[:, None, 1] - P[None, :, 1]
    a = V[:, None, 0] - V[None, :, 0]
    c = V[:, None, 1] - V[None, :, 1]
    aa = a * a + c * c
    disc = aa * R * R - (a * d - b * c) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        let = (-(a * b + c * d) + np.sqrt(np.maximum(disc, 0))) / aa
    let[(aa < 1e-9)] = LET_CAP
    let[~np.isfinite(let)] = LET_CAP
    return np.clip(let, 0, LET_CAP)


# --------------------------------------------------------------- cost / delay
def link_reliability(D: np.ndarray, R: float) -> np.ndarray:
    """Gaussian link-quality model, Eq. (13) with a=1, b=0, c=0.7R.

    Probability that a single transmission over distance d succeeds.
    """
    sigma = 0.7 * R
    return np.exp(-(D ** 2) / (2 * sigma ** 2))


def link_cost(D: np.ndarray, R: float) -> np.ndarray:
    """ct: expected transmissions (ETX = 1/reliability), capped at 10."""
    return np.minimum(1.0 / np.maximum(link_reliability(D, R), 1e-3), 10.0)


def hop_success(D: np.ndarray, R: float) -> np.ndarray:
    """Delivery probability of one hop with MAC retransmissions."""
    r = link_reliability(D, R)
    return 1 - (1 - r) ** (MAC_RETRIES + 1)


def link_delay(D: np.ndarray, degree: np.ndarray, R: float) -> np.ndarray:
    """dl (ms): ETX x (airtime + contention).  Contention grows with the
    receiving node's neighbour count (busier channel -> longer back-off)."""
    airtime_ms = PKT_BITS / RATE_BPS * 1e3 + 0.1
    contention_ms = 0.25 * degree[None, :]
    return link_cost(D, R) * (airtime_ms + contention_ms)


# ---------------------------------------------------------------- scalarise
def scalarized_weight(let, ct, dl, alpha=0.4, beta=0.3, gamma=0.3):
    """Scalarisation, Eq. (15-16): one weight per link from link-availability
    time, cost and delay, each normalised by its maximum (alpha+beta+gamma=1).

    Paths with the lowest total weight are chosen, so the lifetime term is
    written as alpha*(1 - LN/LNmax): a link that lasts longer gets a lower weight.
    """
    lmax = max(float(np.max(let)), 1e-9)
    cmax = max(float(np.max(ct)), 1e-9)
    dmax = max(float(np.max(dl)), 1e-9)
    let_term = 1 - let / lmax
    return alpha * let_term + beta * ct / cmax + gamma * dl / dmax
