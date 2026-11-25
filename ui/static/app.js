// OBQR demo front end — draws whatever the Python simulator returns.
const $ = (s) => document.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");
const COL = { blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", ink: "#0b0b0b", road: "#d9d7d0" };

let roads = null, st = null, playing = false, busy = false;
let selected = null, pickSrc = null, route = null;
const MARGIN = 40;

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  return r.json();
}
function W2C(x, y) {
  const a = st.area, s = (cv.width - 2 * MARGIN) / a;
  return [MARGIN + x * s, cv.height - MARGIN - y * s];
}
function scale() { return (cv.width - 2 * MARGIN) / st.area; }
function hue(id) { return `hsl(${(id * 137.508) % 360}, 62%, 52%)`; }
const colour = () => $("#colour").value;

// ------------------------------------------------------------------ drawing
function draw() {
  if (!st || !roads) return;
  ctx.clearRect(0, 0, cv.width, cv.height);
  // roads
  ctx.lineCap = "round";
  for (const [x1, y1, x2, y2] of roads.segments) {
    const [a, b] = W2C(x1, y1), [c, d] = W2C(x2, y2);
    ctx.strokeStyle = COL.road; ctx.lineWidth = 22;
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.setLineDash([10, 10]);
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); ctx.setLineDash([]);
  }
  const V = st.vehicles;
  // selected vehicle: range circle + links
  if (selected !== null && V[selected]) {
    const v = V[selected], [x, y] = W2C(v.x, v.y);
    ctx.fillStyle = "rgba(42,120,214,0.06)"; ctx.strokeStyle = "rgba(42,120,214,0.5)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, st.R * scale(), 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    if (window._veh && window._veh.id === selected) {
      for (const r of window._veh.rows) {
        const w = V[r.j], [a, b] = W2C(w.x, w.y);
        ctx.strokeStyle = r.similar ? "rgba(0,131,0,0.8)" : "rgba(160,160,160,0.7)";
        ctx.lineWidth = r.similar ? 3 : 1.5; ctx.setLineDash(r.similar ? [] : [6, 6]);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(a, b); ctx.stroke(); ctx.setLineDash([]);
      }
    }
  }
  // member -> head spokes
  ctx.lineWidth = 1.2;
  for (const v of V) {
    if (v.ch || v.head < 0) continue;
    const h = V[v.head], [a, b] = W2C(v.x, v.y), [c, d] = W2C(h.x, h.y);
    ctx.strokeStyle = hue(v.head).replace("52%)", "52%, 0.45)").replace("hsl", "hsla");
    ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
  }
  // routes (tracked)
  const tr = st.tracked || {};
  const routeStyle = { aodv: [COL.aqua, 5, []], obqr: [COL.blue, 7, []] };
  if (route && route.obqr) {
    route.obqr.slice(1).forEach((p) => drawPath(p.path, COL.blue, 3, [4, 8], 0.55));
  }
  for (const k of ["aodv", "obqr"]) {
    if (!tr[k]) continue;
    const [c, w, dash] = routeStyle[k];
    drawPath(tr[k].path, c, w, tr[k].alive ? dash : [2, 10], tr[k].alive ? 0.95 : 0.35);
  }
  // vehicles
  for (const v of V) {
    const [x, y] = W2C(v.x, v.y), c = hue(v.head);
    ctx.save(); ctx.translate(x, y); ctx.rotate(-v.a);
    if (v.ch) {
      ctx.fillStyle = c; ctx.strokeStyle = COL.ink; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(0, 0, 11, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(-4, 5); ctx.lineTo(-4, -5); ctx.closePath(); ctx.fill();
    } else {
      ctx.beginPath(); ctx.moveTo(12, 0); ctx.lineTo(-8, 7); ctx.lineTo(-8, -7); ctx.closePath();
      if (v.gw) { ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.stroke(); }
      else { ctx.fillStyle = c; ctx.fill(); }
    }
    ctx.restore();
    if (v.id === selected || v.id === pickSrc || (route && (v.id === route.src || v.id === route.dst))) {
      ctx.strokeStyle = COL.ink; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(x, y, 17, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillStyle = COL.ink; ctx.font = "bold 22px -apple-system, sans-serif";
      const tag = route && v.id === route.src ? "S" : route && v.id === route.dst ? "D" : v.id === pickSrc ? "S?" : "";
      ctx.fillText(`${v.id}${tag ? " " + tag : ""}`, x + 18, y - 14);
    }
  }
}
function drawPath(p, color, w, dash, alpha) {
  const V = st.vehicles;
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.strokeStyle = "#fff"; ctx.lineWidth = w + 4; ctx.setLineDash([]);
  ctx.beginPath(); p.forEach((id, i) => { const [x, y] = W2C(V[id].x, V[id].y); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash);
  ctx.beginPath(); p.forEach((id, i) => { const [x, y] = W2C(V[id].x, V[id].y); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
  ctx.restore();
}

// ------------------------------------------------------------------ panels
const LABELS = { obqr: "OBQR (direction)", position: "Position-based", speed: "Speed-based", lowest_id: "Lowest-ID" };
function renderLive() {
  $("#tnow").textContent = st.t.toFixed(0);
  const rows = Object.entries(LABELS).filter(([k]) => st.live[k]).map(([k, label]) => {
    const s = st.live[k];
    return `<tr class="${k === colour() ? "hl" : ""}"><td>${label}</td><td>${s.clusters}</td><td>${s.transitions}</td><td>${s.lifetime}</td><td>${s.msgs}</td></tr>`;
  });
  $("#liveTbl tbody").innerHTML = rows.join("");
  renderTracked();
}
function renderTracked() {
  const tr = st.tracked || {};
  if (!Object.keys(tr).length) { $("#trackBody").innerHTML = ""; return; }
  const names = { obqr: ["OBQR", COL.blue], aodv: ["AODV", COL.aqua] };
  $("#trackBody").innerHTML = `<b style="font-size:12px;color:#52514e">ROUTE SURVIVAL — press Play and watch which breaks first</b>
  <table><thead><tr><th>Route</th><th>Hops</th><th>Backup switches</th><th>Status</th><th>Alive for</th></tr></thead><tbody>${
    ["obqr", "aodv"].filter((k) => tr[k]).map((k) =>
      `<tr><td><span style="color:${names[k][1]}">■</span> ${names[k][0]}</td><td>${tr[k].path.length - 1}</td><td>${k === "obqr" ? tr[k].switches : "–"}</td>
       <td>${tr[k].alive ? '<span class="ok">● up</span>' : '<span class="no">✕ broken</span>'}</td><td>${tr[k].age.toFixed(0)} s</td></tr>`).join("")
  }</tbody></table>`;
}
async function inspect(id) {
  selected = id;
  const d = await api(`/api/vehicle/${id}`);
  window._veh = d;
  $("#vehEmpty").style.display = "none"; $("#vehBody").style.display = "";
  $("#v_id").textContent = `#${d.id}`;
  $("#v_sp").textContent = `${d.speed} km/h · ${d.angle_deg}°`;
  $("#v_role").textContent = d.is_ch ? "Cluster head" : `Member of #${d.head}`;
  $("#v_rho").textContent = d.threshold;
  $("#v_nb").textContent = d.n_neighbours;
  $("#v_sim").textContent = d.n_similar;
  $("#vehRows").innerHTML = d.rows.map((r) => `<tr><td>${r.j}</td><td>${r.dist}</td><td>${r.eps}</td><td>${r.k}</td>
     <td>${r.let_vec >= 120 ? "120+" : r.let_vec}</td>
     <td>${r.similar ? '<span class="ok">✓</span>' : '<span class="no">✕</span>'}</td></tr>`).join("") || `<tr><td colspan=6 class="hint">No neighbours in range.</td></tr>`;
  showInsp("veh");
  draw();
}
async function doRoute(src, dst) {
  route = await api("/api/route", { src, dst });
  pickSrc = null;
  const r = route;
  $("#routeBody").classList.remove("hint");
  if (!r.obqr.length) {
    $("#routeBody").innerHTML = `<span class="no">No path from #${src} to #${dst}</span>: these two cars are not connected right now. Try another pair.`;
  } else {
    $("#routeBody").innerHTML = `
      <p style="margin:0 0 8px"><b>Source #${src} → destination #${dst}</b></p>
      <table><thead><tr><th>Path</th><th>Hops</th><th>Total weight W</th><th>Shortest link life (s)</th></tr></thead><tbody>
      ${r.obqr.map((p, i) => `<tr class="${i ? "" : "hl"}"><td>OBQR path ${i + 1}${i ? " (backup)" : " (used)"}</td><td>${p.hops}</td><td>${p.W}</td><td>${p.min_let}</td></tr>`).join("")}
      ${r.aodv ? `<tr><td>AODV (fewest hops)</td><td>${r.aodv.hops}</td><td>–</td><td>${r.aodv.min_let}</td></tr>` : ""}
      </tbody></table>
      <p class="hint" style="margin:6px 0 0">Lower W is better. Press Play to see which route breaks first.</p>`;
  }
  st = await api(`/api/state?colour=${colour()}`);
  renderLive(); showInsp("route"); draw();
}

// ------------------------------------------------------------------ loop
async function tick(k) {
  if (busy) return; busy = true;
  try {
    st = await api(`/api/state?k=${k}&colour=${colour()}`);
    renderLive();
    if (selected !== null && (st.t * 2) % 2 === 0) inspect(selected); else draw();
  } finally { busy = false; }
}
let last = 0;
function loop(ts) {
  if (playing && ts - last > 110) { last = ts; tick(+$("#rate").value); }
  requestAnimationFrame(loop);
}

async function reset() {
  const body = { n: +$("#n").value, speed: +$("#speed").value, range: +$("#range").value, seed: +$("#seed").value, colour: colour() };
  st = await api("/api/reset", body);
  roads = await api("/api/roads");
  selected = null; route = null; pickSrc = null; window._veh = null;
  $("#vehEmpty").style.display = ""; $("#vehBody").style.display = "none";
  $("#routeBody").innerHTML = "No route yet."; $("#routeBody").classList.add("hint"); $("#trackBody").innerHTML = "";
  renderLive(); draw();
}

// ------------------------------------------------------------------ events
for (const id of ["n", "speed", "range", "seed"]) $("#" + id).addEventListener("input", (e) => ($("#o_" + id).textContent = e.target.value));
$("#reset").onclick = reset;
$("#colour").onchange = () => tick(0);
$("#play").onclick = () => { playing = !playing; $("#play").textContent = playing ? "❚❚ Pause" : "▶ Play"; };
$("#stepBtn").onclick = () => tick(2);
$("#randRoute").onclick = () => {
  const n = st.vehicles.length; let a = Math.floor(Math.random() * n), b = a;
  while (b === a) b = Math.floor(Math.random() * n);
  doRoute(a, b);
};
cv.addEventListener("click", (e) => {
  const r = cv.getBoundingClientRect(), sx = cv.width / r.width;
  const mx = (e.clientX - r.left) * sx, my = (e.clientY - r.top) * sx;
  let best = null, bd = 30 * 30;
  for (const v of st.vehicles) { const [x, y] = W2C(v.x, v.y), d = (x - mx) ** 2 + (y - my) ** 2; if (d < bd) { bd = d; best = v.id; } }
  if (best === null) return;
  if (e.shiftKey) {
    if (pickSrc === null) { pickSrc = best; $("#routeHint").textContent = `source #${best} — shift-click the destination`; showInsp("route"); draw(); }
    else if (best !== pickSrc) { $("#routeHint").textContent = "or shift-click two vehicles"; doRoute(pickSrc, best); }
  } else inspect(best);
});
function showInsp(w) {
  document.querySelectorAll("[data-insp]").forEach((b) => b.classList.toggle("on", b.dataset.insp === w));
  $("#insp-veh").style.display = w === "veh" ? "" : "none";
  $("#insp-route").style.display = w === "route" ? "" : "none";
}
document.querySelectorAll("[data-insp]").forEach((b) => (b.onclick = () => showInsp(b.dataset.insp)));
document.querySelectorAll("nav button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("on", p.id === "page-" + b.dataset.page));
}));

reset().then(() => requestAnimationFrame(loop));
