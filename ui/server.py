"""
OBQR live demo server.

    python ui/server.py            ->  http://127.0.0.1:5050

Wraps the same `obqr` package used by the experiments: the browser only draws.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from obqr.clustering import ALL_CLUSTERING, Snapshot, obqr_terms  # noqa: E402
from obqr.mobility import Mobility, MobilityConfig  # noqa: E402
from obqr.routing import AODVRouter, OBQRRouter, path_valid  # noqa: E402

app = Flask(__name__, static_folder="static", static_url_path="/static")
RESULTS = os.path.join(ROOT, "results")


class LiveSim:
    def __init__(self, n=80, speed=54.0, tx_range=250.0, seed=1):
        self.cfg = dict(n=n, speed=speed, tx_range=tx_range, seed=seed)
        self.mob = Mobility(MobilityConfig(n_vehicles=n, mean_speed_kmh=speed, seed=seed))
        self.R = tx_range
        self.clus = {c.name: c(n) for c in ALL_CLUSTERING}
        self.substep = 0
        self.tracked: dict[str, dict] = {}
        self.snap = self._snapshot()
        for c in self.clus.values():
            c.update(self.snap)

    def _snapshot(self):
        m = self.mob
        return Snapshot(m.positions, m.headings, m.v.copy(), self.R, t=m.t)

    def advance(self, k=1):
        for _ in range(k):
            self.mob.step()
            self.substep += 1
            if self.substep % 2 == 0:            # beacons/clustering every 1 s
                self.snap = self._snapshot()
                for c in self.clus.values():
                    c.update(self.snap)
                for tr in self.tracked.values():
                    if not tr["alive"]:
                        continue
                    if not path_valid(tr["path"], self.snap):
                        # OBQR fails over to a still-valid backup path (multipath)
                        alt = next((p for p in tr.get("backups", []) if path_valid(p, self.snap)), None)
                        if alt is None:
                            tr["alive"] = False
                            continue
                        tr["backups"].remove(alt)
                        tr["path"] = alt
                        tr["switches"] += 1
                    tr["age"] = self.mob.t - tr["born"]
        if self.substep % 2:
            self.snap = self._snapshot()

    # ------------------------------------------------------------ serialise
    def state(self, colour_by="obqr"):
        s, c = self.snap, self.clus[colour_by]
        gw = c.gateways(s)
        P, ang = self.mob.positions, self.mob.angles
        vehicles = [{"id": i, "x": round(float(P[i, 0]), 1), "y": round(float(P[i, 1]), 1),
                     "a": round(float(ang[i]), 3), "v": round(float(self.mob.v[i] * 3.6), 1),
                     "head": int(c.head[i]), "ch": bool(c.is_ch[i]), "gw": bool(gw[i])}
                    for i in range(len(P))]
        live = {}
        for k, cl in self.clus.items():
            sm = cl.summary()
            live[k] = {"label": cl.label, "clusters": int(cl.is_ch.sum()),
                       "transitions": cl.stats.transitions, "lifetime": round(sm["cluster_lifetime"], 1),
                       "msgs": cl.stats.discovery_msgs, "singleton": round(sm["singleton_frac"] * 100)}
        return {"t": round(self.mob.t, 1), "vehicles": vehicles, "live": live,
                "tracked": {k: {"path": v["path"], "alive": v["alive"], "age": round(v["age"], 1),
                                "label": v["label"], "switches": v["switches"]}
                            for k, v in self.tracked.items()},
                "R": self.R, "area": self.mob.cfg.area}

    def explain(self, i, colour_by="obqr"):
        s = self.snap
        nb, eps, k_hat, k_th, eps_th, rho = obqr_terms(s, i, cap=self.clus["obqr"].cap)
        ok, dist = self.clus["obqr"].affinity(s)
        ok &= s.A
        score = self.clus["obqr"].ch_score(s, ok, dist)
        let_v = s.let[i]
        rows = [{"j": int(j), "dist": round(float(s.D[i, j]), 1), "eps": round(float(e), 3),
                 "k": round(float(k), 3), "k_theta": round(float(kt), 3), "eps_theta": round(float(et), 3),
                 "similar": bool(ok[i, j]), "let_vec": round(float(let_v[j]), 1),
                 "dv": round(float(abs(s.speed[i] - s.speed[j]) * 3.6), 1)}
                for j, e, k, kt, et in zip(nb, eps, k_hat, k_th, eps_th)]
        rows.sort(key=lambda r: r["k"])
        c = self.clus["obqr"]
        return {"id": i, "speed": round(float(s.speed[i] * 3.6), 1),
                "angle_deg": round(float(np.degrees(self.mob.angles[i])), 0),
                "rho": round(rho, 3), "threshold": round(min(max(rho, 1e-6), c.cap), 3),
                "cap": round(float(c.cap), 3), "score": round(float(score[i]), 3),
                "is_ch": bool(c.is_ch[i]), "head": int(c.head[i]),
                "n_neighbours": int(len(nb)), "n_similar": int(ok[i].sum()), "rows": rows}

    def route(self, src, dst):
        s, cl = self.snap, self.clus["obqr"]
        rng = np.random.default_rng(0)
        r = OBQRRouter(s.n, rng)
        paths, msgs, G = r.candidate_paths(s, cl, src, dst)
        out = {"src": src, "dst": dst, "obqr": [], "obqr_msgs": msgs}
        for p in paths:
            e = [G.edges[a, b] for a, b in zip(p[:-1], p[1:])]
            out["obqr"].append({"path": p, "W": round(sum(x["w"] for x in e), 3),
                                "min_let": round(min(x["let"] for x in e), 1),
                                "cost": round(sum(x["ct"] for x in e), 2),
                                "delay": round(sum(x["dl"] for x in e), 2), "hops": len(p) - 1})
        aodv, amsgs = AODVRouter(s.n, rng).discover(s, cl, src, dst)
        out["aodv_msgs"] = amsgs
        for key, p in (("aodv", aodv[0] if aodv else None),):
            if p:
                lets = s.let[np.array(p[:-1]), np.array(p[1:])]
                out[key] = {"path": p, "hops": len(p) - 1, "min_let": round(float(lets.min()), 1)}
        # track the primary routes so the UI can show which breaks first
        self.tracked = {}
        for key, label, p in (("obqr", "OBQR", paths[0] if paths else None),
                              ("aodv", "AODV min-hop", aodv[0] if aodv else None)):
            if p:
                self.tracked[key] = {"path": p, "alive": True, "born": self.mob.t, "age": 0.0,
                                     "label": label, "switches": 0,
                                     "backups": [q for q in paths[1:]] if key == "obqr" else []}
        return out


SIM = LiveSim()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/reset")
def reset():
    global SIM
    b = request.get_json(force=True)
    SIM = LiveSim(int(b.get("n", 80)), float(b.get("speed", 54)), float(b.get("range", 250)),
                  int(b.get("seed", 1)))
    return jsonify(SIM.state(b.get("colour", "obqr")))


@app.get("/api/state")
def state():
    k = int(request.args.get("k", 0))
    if k:
        SIM.advance(min(k, 20))
    return jsonify(SIM.state(request.args.get("colour", "obqr")))


@app.get("/api/roads")
def roads():
    return jsonify({"segments": SIM.mob.net.segments(), "area": SIM.mob.cfg.area,
                    "offset": SIM.mob.cfg.lane_offset})


@app.get("/api/vehicle/<int:i>")
def vehicle(i):
    return jsonify(SIM.explain(i))


@app.post("/api/route")
def route():
    b = request.get_json(force=True)
    return jsonify(SIM.route(int(b["src"]), int(b["dst"])))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=False)
