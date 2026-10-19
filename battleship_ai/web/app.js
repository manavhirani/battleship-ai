// Battleship AI front end: fleet setup, human-vs-AI battle, AI duel and lab.
// Plain ES module, no build step. Talks to the Python server's JSON API.

const LETTERS = "ABCDEFGHIJKL";
const RESULT_LABEL = { miss: "Miss", hit: "Hit", sunk: "Sunk" };
const AGENT_COLOR = { random: "var(--agent-random)", mcts: "var(--agent-mcts)", neural: "var(--agent-neural)" };
const AGENT_LEVEL = { random: 1, mcts: 3, neural: 5 };
const ICONS = {
  random: '<svg viewBox="0 0 32 32"><rect x="5" y="5" width="22" height="22" rx="5"/><circle cx="11" cy="11" r="1.6" fill="currentColor"/><circle cx="21" cy="11" r="1.6" fill="currentColor"/><circle cx="16" cy="16" r="1.6" fill="currentColor"/><circle cx="11" cy="21" r="1.6" fill="currentColor"/><circle cx="21" cy="21" r="1.6" fill="currentColor"/></svg>',
  mcts: '<svg viewBox="0 0 32 32"><circle cx="16" cy="5.5" r="2.8"/><circle cx="7" cy="16" r="2.8"/><circle cx="25" cy="16" r="2.8"/><circle cx="3.8" cy="27" r="2.2"/><circle cx="12" cy="27" r="2.2"/><circle cx="25" cy="27" r="2.2"/><path d="M14.3 7.8 8.7 13.7M17.7 7.8l5.6 5.9M6 18.6l-1.4 6.2M8.2 18.6l2.8 6.3M25 18.8v6"/></svg>',
  neural: '<svg viewBox="0 0 32 32"><circle cx="5" cy="9" r="2.4"/><circle cx="5" cy="23" r="2.4"/><circle cx="16" cy="5" r="2.4"/><circle cx="16" cy="16" r="2.4"/><circle cx="16" cy="27" r="2.4"/><circle cx="27" cy="16" r="2.8"/><path d="M7.3 9.6l6.4-3.9M7.3 9.8l6.5 5.2M7.3 22.2l6.5-5.2M7.3 22.4l6.4 3.9M18.4 5.9l6.6 8.3M18.6 16H24M18.4 26.1l6.6-8.3M7.3 10.4l6.6 15"/></svg>',
};
const EXPLAIN = {
  random: ["Picks any cell it has not fired at, with equal chance.",
           "It does not use hits, misses or sunk ships.",
           "This is the baseline: it needs nearly the whole board to win."],
  mcts: ["Samples about 400 enemy layouts that fit every miss, hit and sunk ship seen so far.",
         "Runs thousands of UCT simulations. Each one plays out a short hunt-and-target sequence in one sampled layout.",
         "Scores each first shot by the discounted hits that follow, then fires the most visited shot."],
  neural: ["A 5-block residual CNN reads misses, hits, sunk ships and which ships are still afloat.",
           "It outputs a ship probability for every cell. It learned this from self-play positions where the true layout was the label.",
           "It fires at the cell with the highest probability. One forward pass takes a few milliseconds."],
};

const $ = (sel, root = document) => root.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const coord = (r, c) => `${LETTERS[c]}${r + 1}`;
const pct = (v) => `${Math.round(v * 100)}%`;
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid);
  return el;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem("bsai." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("bsai." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch("/api/" + path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(text, kind = "") {
  const el = h("div", { class: "toast " + kind }, text);
  $("#toasts").append(el);
  setTimeout(() => el.remove(), 2700);
}

// ------------------------------------------------------------------ sound
const Sound = {
  ctx: null, on: store.get("sound", true), noise: null,
  init() {
    if (this.ctx) return;
    try {
      this.ctx = new AudioContext();
      const len = this.ctx.sampleRate;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    } catch { this.ctx = null; }
  },
  burst(dur, freq, type, gain, q = 1) {
    const c = this.ctx, t = c.currentTime;
    const src = c.createBufferSource(); src.buffer = this.noise;
    const f = c.createBiquadFilter(); f.type = type; f.frequency.setValueAtTime(freq, t); f.Q.value = q;
    f.frequency.exponentialRampToValueAtTime(Math.max(40, freq / 6), t + dur);
    const g = c.createGain(); g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(f).connect(g).connect(c.destination); src.start(t); src.stop(t + dur);
  },
  tone(freq, dur, gain = 0.15, type = "sine", end = null, delay = 0) {
    const c = this.ctx, t = c.currentTime + delay;
    const o = c.createOscillator(); o.type = type; o.frequency.setValueAtTime(freq, t);
    if (end) o.frequency.exponentialRampToValueAtTime(end, t + dur);
    const g = c.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(c.destination); o.start(t); o.stop(t + dur + 0.02);
  },
  play(kind) {
    if (!this.on) return;
    this.init();
    if (!this.ctx) return;
    if (this.ctx.state === "suspended") this.ctx.resume();
    switch (kind) {
      case "fire": this.burst(0.18, 2400, "bandpass", 0.25, 2); break;
      case "miss": this.burst(0.5, 1800, "highpass", 0.18); this.tone(900, 0.25, 0.05, "sine", 500); break;
      case "hit": this.burst(0.7, 900, "lowpass", 0.9); this.tone(90, 0.5, 0.35, "sine", 40); break;
      case "sunk": this.burst(1.3, 700, "lowpass", 1); this.tone(120, 1.1, 0.4, "sawtooth", 30); break;
      case "ping": this.tone(1320, 0.5, 0.06, "sine", 1250); break;
      case "win": [523, 659, 784, 1046].forEach((f, i) => this.tone(f, 0.35, 0.12, "triangle", null, i * 0.12)); break;
      case "lose": [392, 330, 262, 196].forEach((f, i) => this.tone(f, 0.45, 0.12, "triangle", null, i * 0.16)); break;
    }
  },
};

// ------------------------------------------------------------- board view
function heatColor(v) {
  // transparent -> violet -> magenta -> orange -> yellow
  const stops = [[0, 40, 10, 90, 0], [0.25, 91, 31, 168, 0.45], [0.5, 194, 24, 155, 0.62], [0.75, 255, 106, 61, 0.75], [1, 255, 224, 102, 0.85]];
  v = Math.max(0, Math.min(1, v));
  let i = 0;
  while (i < stops.length - 2 && v > stops[i + 1][0]) i++;
  const [a, b] = [stops[i], stops[i + 1]];
  const t = (v - a[0]) / (b[0] - a[0]);
  const mix = (k) => a[k] + (b[k] - a[k]) * t;
  return `rgba(${mix(1) | 0},${mix(2) | 0},${mix(3) | 0},${mix(4).toFixed(3)})`;
}

class BoardView {
  constructor(host, size, opts = {}) {
    this.host = host; this.size = size; this.opts = opts;
    host.innerHTML = "";
    this.el = h("div", { class: "board" + (opts.interactive ? " interactive" : "") + (opts.setup ? " setup" : "") });
    this.el.style.setProperty("--n", size);
    const cols = h("div", { class: "labels cols" }, [...LETTERS.slice(0, size)].map((l) => h("span", {}, l)));
    const rows = h("div", { class: "labels rows" }, Array.from({ length: size }, (_, i) => h("span", {}, String(i + 1))));
    this.grid = h("div", { class: "grid" });
    this.shipLayer = h("div", { class: "layer ships" });
    this.cellLayer = h("div", { class: "cells" });
    this.heatLayer = h("div", { class: "layer heat off" });
    this.fxLayer = h("div", { class: "layer fx" });
    this.cells = [];
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const cell = h("div", { class: "cell c0" });
        cell.dataset.r = r; cell.dataset.c = c;
        this.cells.push(cell);
        this.cellLayer.append(cell);
      }
    }
    this.grid.append(this.shipLayer, this.cellLayer, this.heatLayer, this.fxLayer);
    this.el.append(h("div"), cols, rows, this.grid);
    host.append(this.el);
    this.codes = new Array(size * size).fill(0);

    const at = (e) => {
      const cell = e.target.closest(".cell");
      return cell ? [Number(cell.dataset.r), Number(cell.dataset.c)] : null;
    };
    this.cellLayer.addEventListener("click", (e) => { const p = at(e); if (p && opts.onClick) opts.onClick(...p); });
    this.cellLayer.addEventListener("mouseover", (e) => { const p = at(e); if (p && opts.onHover) opts.onHover(...p); });
    this.cellLayer.addEventListener("mouseleave", () => opts.onLeave && opts.onLeave());
    this.cellLayer.addEventListener("contextmenu", (e) => { if (opts.onContext) { e.preventDefault(); const p = at(e); opts.onContext(...(p || [])); } });

    this.resize = () => {
      const max = opts.maxCell || 42;
      const avail = host.clientWidth - 24;
      const cell = Math.max(20, Math.min(max, Math.floor(avail / size)));
      this.el.style.setProperty("--cell", cell + "px");
      this.el.style.setProperty("--label", Math.max(16, Math.round(cell * 0.6)) + "px");
    };
    this.ro = new ResizeObserver(this.resize);
    this.ro.observe(host);
    this.resize();
  }
  cell(r, c) { return this.cells[r * this.size + c]; }
  setGrid(grid) {
    grid.forEach((row, r) => row.forEach((v, c) => {
      const i = r * this.size + c;
      if (this.codes[i] !== v) { this.cells[i].className = "cell c" + v; this.codes[i] = v; }
    }));
  }
  setLast(r, c) {
    this.cellLayer.querySelector(".cell.last")?.classList.remove("last");
    if (r != null) this.cell(r, c).classList.add("last");
  }
  setLocked(locked) { this.el.classList.toggle("locked", locked); }
  setShips(ships, cls = "") {
    this.shipLayer.innerHTML = "";
    for (const s of ships) this.shipLayer.append(this.shipEl(s, cls + (s.sunk ? " sunk" : "")));
  }
  shipEl(s, cls = "") {
    const el = h("div", { class: `ship ${s.horizontal ? "h" : "v"} ${cls}` });
    el.style.setProperty("--r", s.row); el.style.setProperty("--c", s.col); el.style.setProperty("--len", s.length);
    return el;
  }
  addShip(s, cls) { const el = this.shipEl(s, cls); this.shipLayer.append(el); return el; }
  setHeat(insight, visible = true) {
    if (!insight) { this.heatLayer.classList.add("off"); return; }
    if (!this.heatCells) {
      this.heatCells = [];
      for (let i = 0; i < this.size * this.size; i++) { const el = h("i"); this.heatCells.push(el); this.heatLayer.append(el); }
    }
    insight.heatmap.forEach((row, r) => row.forEach((v, c) => {
      const el = this.heatCells[r * this.size + c];
      el.style.backgroundColor = heatColor(Math.pow(v, 1.6));  // gamma: only strong beliefs glow
      el.classList.remove("pick");
    }));
    if (insight.pick) this.heatCells[insight.pick[0] * this.size + insight.pick[1]].classList.add("pick");
    this.heatLayer.classList.toggle("off", !visible);
  }
  showHeat(visible) { this.heatLayer.classList.toggle("off", !visible || !this.heatCells); }
  fx(r, c, kind) {
    const el = h("div", { class: kind === "miss" ? "splash" : "boom" + (kind === "sunk" ? " big" : "") });
    el.style.setProperty("--r", r); el.style.setProperty("--c", c);
    if (kind !== "miss") for (let i = 0; i < 10; i++) { const s = h("span"); s.style.setProperty("--a", `${i * 36 + Math.random() * 20}deg`); el.append(s); }
    this.fxLayer.append(el);
    setTimeout(() => el.remove(), 1100);
    if (kind !== "miss" && this.opts.shake) { this.el.classList.remove("shake"); void this.el.offsetWidth; this.el.classList.add("shake"); }
  }
  destroy() { this.ro.disconnect(); }
}

function fleetStatus(host, side, known) {
  // known: true when every hit on this fleet is visible (your own fleet)
  host.innerHTML = "";
  const sunk = new Set(side.sunk.map((s) => s.name));
  for (const { name, length } of App.meta.fleet) {
    const ship = side.ships?.find((s) => s.name === name);
    let hits = 0;
    if (ship && known) {
      for (let i = 0; i < length; i++) {
        const r = ship.row + (ship.horizontal ? 0 : i), c = ship.col + (ship.horizontal ? i : 0);
        if (side.grid[r][c] >= 2) hits++;
      }
    }
    const segs = h("div", { class: "segs" }, Array.from({ length }, (_, i) => h("i", { class: i < hits ? "hit" : "" })));
    host.append(h("div", { class: "fleet-ship" + (sunk.has(name) ? " sunk" : known ? "" : " unknown") }, h("b", {}, name), segs));
  }
}

function avatar(key, small = false) {
  const el = h("div", { class: "avatar" + (small ? " small" : ""), html: ICONS[key] || "" });
  el.style.setProperty("--accent", AGENT_COLOR[key]);
  return el;
}
const agentInfo = (key) => App.meta.agents.find((a) => a.key === key);

function mindPanel(host, agentKey, insight) {
  const info = agentInfo(agentKey);
  host.style.setProperty("--accent", AGENT_COLOR[agentKey]);
  host.innerHTML = "";
  host.append(h("div", { class: "mind-head" }, avatar(agentKey, true),
    h("div", {}, h("b", {}, info.name), h("small", {}, insight ? insight.label : "Waiting for its first shot"))));
  if (!insight) {
    host.append(h("p", {}, info.description));
    return;
  }
  host.append(h("div", { class: "legend" }), h("div", { class: "legend-labels" }, h("span", {}, "unlikely"), h("span", {}, "likely")));
  const s = insight.stats || {};
  const rows = [];
  if (insight.pick) rows.push(["Target", coord(...insight.pick)]);
  if (s.hit_chance != null) rows.push(["Estimated hit chance", pct(s.hit_chance)]);
  if (s.iterations != null) rows.push(["Simulations", s.iterations.toLocaleString()]);
  if (s.worlds != null) rows.push(["Imagined fleets", s.worlds.toLocaleString()]);
  if (s.expected_ship_cells != null) rows.push(["Ship cells it expects", s.expected_ship_cells]);
  if (s.options != null) rows.push(["Open cells", s.options]);
  if (s.ms != null) rows.push(["Think time", `${s.ms} ms`]);
  host.append(h("div", { class: "kv" }, rows.flatMap(([k, v]) => [h("span", {}, k), h("b", {}, String(v))])));
}

// --------------------------------------------------------------------- app
const App = {
  meta: null,
  view: "battle",
  async init() {
    try {
      this.meta = await api("meta");
    } catch (e) {
      document.querySelector("main").innerHTML = `<div class="panel empty">Could not reach the game server: ${e.message}</div>`;
      return;
    }
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => this.show(t.dataset.view)));
    const snd = $("#sound-toggle");
    snd.classList.toggle("muted", !Sound.on);
    snd.addEventListener("click", () => { Sound.on = !Sound.on; store.set("sound", Sound.on); snd.classList.toggle("muted", !Sound.on); if (Sound.on) Sound.play("ping"); });
    Setup.init();
    Battle.init();
    Duel.init();
    Lab.render();
  },
  show(view) {
    this.view = view;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
    const id = view === "battle" ? (Battle.game ? "battle" : "setup") : view;
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + id));
    if (view === "lab") Lab.animate();
  },
};

// ------------------------------------------------------------------- setup
const Setup = {
  size: store.get("size", 10),
  opponent: store.get("opponent", "neural"),
  placed: {}, selected: null, horizontal: true, hover: null,
  init() {
    const cards = $("#opponent-cards");
    for (const a of App.meta.agents) {
      const b = a.benchmark;
      const card = h("button", { class: "card", "data-key": a.key },
        avatar(a.key),
        h("div", {},
          h("div", { class: "card-name" }, a.name),
          h("div", { class: "card-tag" }, a.tagline),
          h("div", { class: "card-meta" },
            h("span", { class: "pips" }, Array.from({ length: 5 }, (_, i) => h("i", { class: i < AGENT_LEVEL[a.key] ? "on" : "" }))),
            h("span", {}, b ? `~${b.mean} shots to win` : ""))));
      card.style.setProperty("--accent", AGENT_COLOR[a.key]);
      card.addEventListener("click", () => this.pickOpponent(a.key));
      cards.append(card);
    }
    const seg = $("#size-picker");
    for (let n = App.meta.sizes.min; n <= App.meta.sizes.max; n++) {
      seg.append(h("button", { "data-n": n, onclick: () => this.setSize(n) }, `${n}×${n}`));
    }
    $("#btn-rotate").addEventListener("click", () => this.rotate());
    $("#btn-random").addEventListener("click", () => this.randomize());
    $("#btn-clear").addEventListener("click", () => { this.placed = {}; this.select(App.meta.fleet[0].name); });
    $("#btn-engage").addEventListener("click", () => this.engage());
    document.addEventListener("keydown", (e) => {
      if (!$("#view-setup").classList.contains("active")) return;
      if (e.key === "r" || e.key === "R") this.rotate();
      if (e.key === "Escape") this.select(null);
    });
    this.pickOpponent(this.opponent);
    this.setSize(this.size);
  },
  pickOpponent(key) {
    if (!agentInfo(key)) key = "neural";
    this.opponent = key; store.set("opponent", key);
    document.querySelectorAll(".card").forEach((c) => c.classList.toggle("selected", c.dataset.key === key));
  },
  setSize(n) {
    this.size = n; store.set("size", n);
    document.querySelectorAll("#size-picker button").forEach((b) => b.classList.toggle("active", Number(b.dataset.n) === n));
    this.board?.destroy();
    this.board = new BoardView($("#setup-board"), n, {
      setup: true, maxCell: 44,
      onClick: (r, c) => this.click(r, c),
      onHover: (r, c) => { this.hover = [r, c]; this.draw(); },
      onLeave: () => { this.hover = null; this.draw(); },
      onContext: () => this.rotate(),
    });
    this.placed = {};
    this.select(App.meta.fleet[0].name);
  },
  ship(name, r, c, horizontal) {
    return { name, length: App.meta.fleet.find((s) => s.name === name).length, row: r, col: c, horizontal };
  },
  cellsOf(s) { return Array.from({ length: s.length }, (_, i) => [s.row + (s.horizontal ? 0 : i), s.col + (s.horizontal ? i : 0)]); },
  valid(s) {
    const taken = new Set(Object.values(this.placed).filter((p) => p.name !== s.name).flatMap((p) => this.cellsOf(p).map(([r, c]) => r * 100 + c)));
    return this.cellsOf(s).every(([r, c]) => r >= 0 && c >= 0 && r < this.size && c < this.size && !taken.has(r * 100 + c));
  },
  ghost() {
    if (!this.selected || !this.hover) return null;
    const s = this.ship(this.selected, ...this.hover, this.horizontal);
    // keep the ghost on the board near the edges
    if (s.horizontal) s.col = Math.min(s.col, this.size - s.length); else s.row = Math.min(s.row, this.size - s.length);
    return s;
  },
  click(r, c) {
    if (this.selected) {
      const s = this.ghost();
      if (s && this.valid(s)) {
        this.placed[s.name] = s;
        Sound.play("ping");
        const next = App.meta.fleet.find((f) => !this.placed[f.name]);
        this.select(next ? next.name : null);
      }
      return;
    }
    const hitShip = Object.values(this.placed).find((p) => this.cellsOf(p).some(([rr, cc]) => rr === r && cc === c));
    if (hitShip) {
      delete this.placed[hitShip.name];
      this.horizontal = hitShip.horizontal;
      this.select(hitShip.name);
    }
  },
  select(name) { this.selected = name; this.draw(); },
  rotate() { this.horizontal = !this.horizontal; this.draw(); },
  async randomize() {
    const { ships } = await api(`random-fleet?size=${this.size}`);
    this.placed = Object.fromEntries(ships.map((s) => [s.name, s]));
    this.select(null);
    Sound.play("ping");
  },
  draw() {
    const dock = $("#dock");
    dock.innerHTML = "";
    for (const { name, length } of App.meta.fleet) {
      const isPlaced = !!this.placed[name];
      const hull = h("div", { class: "mini-hull", style: `width:${length * 22}px` });
      dock.append(h("button", {
        class: "dock-ship" + (this.selected === name ? " selected" : "") + (isPlaced ? " placed" : ""),
        onclick: () => { if (isPlaced) delete this.placed[name]; this.select(name); },
      }, h("div", { class: "row" }, h("span", {}, name), h("em", {}, isPlaced ? "✓ placed" : `${length} cells`)), hull));
    }
    this.board.setShips(Object.values(this.placed));
    const g = this.ghost();
    if (g) this.board.addShip(g, "ghost" + (this.valid(g) ? "" : " bad"));
    $("#btn-engage").disabled = Object.keys(this.placed).length !== App.meta.fleet.length;
  },
  async engage() {
    Sound.init();
    $("#btn-engage").disabled = true;
    try {
      await Battle.start(this.opponent, this.size, Object.values(this.placed));
    } catch (e) {
      toast(e.message);
    } finally {
      this.draw();
    }
  },
};

// ------------------------------------------------------------------ battle
const Battle = {
  game: null, busy: false, insight: null, lastSetup: null,
  init() {
    $("#mind-toggle").addEventListener("change", (e) => this.yours?.showHeat(e.target.checked));
    $("#btn-abandon").addEventListener("click", () => this.leave());
    $("#modal-rematch").addEventListener("click", () => { this.closeModal(); this.start(...this.lastSetup); });
    $("#modal-setup").addEventListener("click", () => { this.closeModal(); this.leave(); });
    $("#modal-review").addEventListener("click", () => this.closeModal());
  },
  async start(opponent, size, ships) {
    this.lastSetup = [opponent, size, ships];
    const game = await api("play", { opponent, size, ships });
    this.game = game; this.insight = null; this.busy = false;
    this.enemy?.destroy(); this.yours?.destroy();
    this.enemy = new BoardView($("#enemy-board"), size, { interactive: true, onClick: (r, c) => this.fire(r, c) });
    this.yours = new BoardView($("#your-board"), size, { shake: true });
    this.yours.setShips(game.you.ships);
    $("#log").innerHTML = "";
    const color = AGENT_COLOR[opponent];
    document.querySelector(".switch").style.setProperty("--agent", color);
    $("#hud-enemy-name").textContent = game.opponent.name;
    $("#hud-enemy-name").style.color = color;
    App.show("battle");
    this.render();
    mindPanel($("#mind"), opponent, null);
  },
  leave() {
    this.game = null;
    App.show("battle");
  },
  status(text, color = "var(--cyan)", thinking = false) {
    const el = $("#status");
    el.style.setProperty("--sc", color);
    el.classList.toggle("thinking", thinking);
    $("#status-text").textContent = text;
  },
  render() {
    const g = this.game;
    this.enemy.setGrid(g.enemy.grid);
    this.yours.setGrid(g.you.grid);
    this.enemy.setLocked(g.turn !== "player" || this.busy);
    const stats = (side) => [h("span", {}, h("b", {}, side.shots), "shots"), h("span", {}, h("b", {}, side.hits), "hits"),
      h("span", {}, h("b", {}, side.shots ? pct(side.hits / side.shots) : "–"), "accuracy")];
    $("#hud-you").replaceChildren(...stats(g.enemy));
    $("#hud-enemy").replaceChildren(...stats(g.you));
    fleetStatus($("#enemy-fleet"), g.enemy, false);
    fleetStatus($("#your-fleet"), g.you, true);
    const panels = document.querySelectorAll("#view-battle .board-panel");
    panels[0].classList.toggle("turn", g.turn === "player");
    panels[1].classList.toggle("turn", g.turn === "ai");
    const color = AGENT_COLOR[g.opponent.key];
    if (g.turn === "player") this.status("Your move: pick a target in enemy waters");
    else if (g.turn === "ai") this.status(`${g.opponent.name} is computing a firing solution…`, color, true);
    else this.status(g.winner === "player" ? "Victory: enemy fleet destroyed" : "Defeat: your fleet is lost", g.winner === "player" ? "var(--gold)" : "var(--hit)");
  },
  log(who, isYou, shot) {
    const li = h("li", {}, h("span", { class: "who" + (isYou ? " you" : "") }, who),
      h("span", { class: "coord" }, coord(shot.row, shot.col) + (shot.sunk ? ` · ${shot.sunk.name}` : "")),
      h("span", { class: "res " + shot.result }, RESULT_LABEL[shot.result]));
    $("#log").prepend(li);
  },
  async fire(r, c) {
    const g = this.game;
    if (!g || this.busy || g.turn !== "player" || g.enemy.grid[r][c] !== 0) return;
    this.busy = true;
    this.enemy.setLocked(true);
    Sound.play("fire");
    try {
      const res = await api(`play/${g.id}/fire`, { row: r, col: c });
      this.game = res.state;
      this.impact(this.enemy, res.shot, true);
      this.log("You", true, res.shot);
      this.render();
      if (this.game.turn === "ai") {
        await sleep(420);
        const t0 = performance.now();
        const ai = await api(`play/${g.id}/ai-turn`, {});
        await sleep(Math.max(0, 550 - (performance.now() - t0)));
        Sound.play("fire");
        await sleep(160);
        this.game = ai.state;
        this.insight = ai.insight ? { ...ai.insight, pick: [ai.shot.row, ai.shot.col] } : null;
        this.yours.setHeat(this.insight, $("#mind-toggle").checked);
        mindPanel($("#mind"), this.game.opponent.key, this.insight);
        this.impact(this.yours, ai.shot, false);
        this.log(this.game.opponent.name, false, ai.shot);
      }
    } catch (e) {
      toast(e.message);
    } finally {
      this.busy = false;
      if (this.game) {
        this.render();
        if (this.game.turn === "over") this.finish();
      }
    }
  },
  impact(board, shot, mine) {
    board.fx(shot.row, shot.col, shot.result);
    board.setLast(shot.row, shot.col);
    setTimeout(() => Sound.play(shot.result), 60);
    if (shot.sunk) {
      if (mine) {
        const el = board.addShip(shot.sunk, "enemy reveal");
        el.classList.add("sunk");
        toast(`You sank their ${shot.sunk.name}!`, "good");
      } else {
        this.yours.setShips(this.game.you.ships);
        toast(`${this.game.opponent.name} sank your ${shot.sunk.name}`, "sunk");
      }
    }
  },
  async finish() {
    const g = this.game, won = g.winner === "player";
    this.enemy.setShips(g.enemy.ships.map((s) => ({ ...s })), "enemy");
    this.enemy.setLocked(true);
    await sleep(900);
    Sound.play(won ? "win" : "lose");
    const card = $("#modal .modal-card");
    card.className = "modal-card " + (won ? "win" : "lose");
    $("#modal-kicker").textContent = `vs ${g.opponent.name} · ${g.size}×${g.size}`;
    $("#modal-title").textContent = won ? "VICTORY" : "DEFEAT";
    $("#modal-text").textContent = won
      ? `You sank the whole enemy fleet in ${g.enemy.shots} shots. ${g.opponent.name} still needed ${g.you.remaining.length} more ship${g.you.remaining.length === 1 ? "" : "s"}.`
      : `${g.opponent.name} found and sank your fleet in ${g.you.shots} shots.`;
    const box = (v, l) => h("div", {}, h("b", {}, v), h("span", {}, l));
    $("#modal-stats").replaceChildren(box(g.enemy.shots, "your shots"), box(g.enemy.shots ? pct(g.enemy.hits / g.enemy.shots) : "–", "your accuracy"),
      box(g.you.shots ? pct(g.you.hits / g.you.shots) : "–", "AI accuracy"));
    $("#modal").hidden = false;
  },
  closeModal() { $("#modal").hidden = true; },
};

// -------------------------------------------------------------------- duel
const Duel = {
  a: store.get("duelA", "mcts"), b: store.get("duelB", "neural"), game: null, running: false, tally: [0, 0], token: 0,
  init() {
    for (const [side, host] of [["a", $("#duel-a")], ["b", $("#duel-b")]]) {
      for (const ag of App.meta.agents) {
        const chip = h("button", { class: "chip", "data-key": ag.key, onclick: () => this.pick(side, ag.key) }, h("span", { class: "dot" }), ag.name);
        chip.style.setProperty("--accent", AGENT_COLOR[ag.key]);
        host.append(chip);
      }
    }
    const sel = $("#duel-size");
    for (let n = App.meta.sizes.min; n <= App.meta.sizes.max; n++) sel.append(h("option", { value: n }, `${n}×${n}`));
    sel.value = App.meta.sizes.default;
    $("#duel-start").addEventListener("click", () => this.toggle());
    $("#duel-step").addEventListener("click", () => this.step());
    this.pick("a", this.a); this.pick("b", this.b);
    this.newGame(false);
  },
  pick(side, key) {
    this[side] = key; store.set(side === "a" ? "duelA" : "duelB", key);
    document.querySelectorAll(`#duel-${side} .chip`).forEach((c) => c.classList.toggle("selected", c.dataset.key === key));
    this.tally = [0, 0];
    if (this.game) this.newGame(false);
  },
  delay() { const v = Number($("#duel-speed").value); return Math.round(1200 * Math.pow(1 - v / 100, 2)); },
  async newGame(autostart) {
    const token = ++this.token;
    const size = Number($("#duel-size").value);
    const game = await api("duel", { a: this.a, b: this.b, size });
    if (token !== this.token) return;
    this.game = game;
    this.boards?.forEach((b) => b.destroy());
    this.boards = [0, 1].map((i) => new BoardView($(`#duel-board-${i}`), size, { maxCell: 40, shake: true }));
    game.sides.forEach((s, i) => this.boards[i].setShips(s.ships));
    this.render();
    if (autostart) this.loop();
  },
  async toggle() {
    Sound.init();
    if (this.running) { this.running = false; this.render(); return; }
    if (!this.game || this.game.winner !== null || Number($("#duel-size").value) !== this.game.size) await this.newGame(false);
    this.loop();
  },
  async loop() {
    this.running = true;
    this.render();
    while (this.running && this.game && this.game.winner === null) {
      await this.step(true);
      await sleep(this.delay());
    }
  },
  async step(fromLoop = false) {
    if (!this.game || this.game.winner !== null || (this.stepping)) return;
    if (!fromLoop && this.running) return;
    this.stepping = true;
    const token = this.token;
    try {
      const res = await api(`duel/${this.game.id}/step`, {});
      if (token !== this.token) return;
      this.game = res.state;
      const board = this.boards[res.shooter];
      const insight = res.insight ? { ...res.insight, pick: [res.shot.row, res.shot.col] } : null;
      board.setHeat(insight, true);
      board.fx(res.shot.row, res.shot.col, res.shot.result);
      board.setLast(res.shot.row, res.shot.col);
      if (res.shot.sunk) board.setShips(this.game.sides[res.shooter].ships);
      if (this.delay() > 120 || res.shot.sunk) Sound.play(res.shot.result === "miss" ? "ping" : res.shot.result);
      if (this.game.winner !== null) {
        this.tally[this.game.winner]++;
        this.running = false;
        Sound.play("win");
        toast(`${this.game.sides[this.game.winner].agent.name} wins in ${this.game.sides[this.game.winner].shots} shots`, "good");
      }
    } catch (e) {
      toast(e.message);
      this.running = false;
    } finally {
      this.stepping = false;
      this.render();
    }
  },
  render() {
    const g = this.game;
    if (!g) return;
    g.sides.forEach((s, i) => {
      this.boards[i].setGrid(s.grid);
      const color = AGENT_COLOR[s.agent.key];
      const title = $(`#duel-title-${i}`);
      title.innerHTML = "";
      title.append(`${s.agent.name} `, h("span", { class: "sub" }, `attacks fleet ${i ? "A" : "B"}`));
      title.style.color = color;
      $(`#duel-sub-${i}`).innerHTML = g.winner === i ? '<span class="crown">★ WINNER</span>' : `${s.shots} shots · ${s.hits} hits`;
      fleetStatus($(`#duel-fleet-${i}`), s, true);
      const panel = document.querySelector(`#view-duel .board-panel[data-side="${i}"]`);
      panel.classList.toggle("winner", g.winner === i);
      panel.classList.toggle("turn", g.winner === null && g.turn === i && this.running);
    });
    const [a, b] = g.sides.map((s) => s.agent);
    const score = $("#duel-score");
    score.replaceChildren(
      h("span", { class: "who", style: `color:${AGENT_COLOR[a.key]}` }, a.name),
      h("span", { class: "tally" }, `${this.tally[0]} : ${this.tally[1]}`),
      h("span", { class: "who", style: `color:${AGENT_COLOR[b.key]}` }, b.name));
    const done = g.winner !== null;
    $("#duel-start").textContent = this.running ? "Pause" : done ? "Next duel" : g.sides[0].shots + g.sides[1].shots ? "Resume" : "Start duel";
    $("#duel-step").disabled = this.running || done;
  },
};

// --------------------------------------------------------------------- lab
const Lab = {
  render() {
    const agents = App.meta.agents.filter((a) => a.benchmark);
    const bench = App.meta.benchmark;
    if (!agents.length) {
      for (const id of ["#lab-bars", "#lab-h2h", "#lab-hist"]) $(id).append(h("div", { class: "empty" }, "No benchmark yet. Run: python -m battleship_ai evaluate --save"));
    } else {
      $("#lab-bars-note").textContent = `Average shots to sink a random fleet on ${bench.size}×${bench.size} over ${agents[0].benchmark.games} games. Fewer is better.`;
      const max = Math.max(...agents.map((a) => a.benchmark.mean)) * 1.05;
      for (const a of agents) {
        const row = h("div", { class: "bar-row" }, h("span", { class: "name" }, a.name),
          h("div", { class: "bar-track" }, h("div", { class: "bar-fill", "data-w": (a.benchmark.mean / max) * 100 })),
          h("span", { class: "val" }, a.benchmark.mean.toFixed(1), h("small", {}, `± ${a.benchmark.stdev}`)));
        row.style.setProperty("--accent", AGENT_COLOR[a.key]);
        $("#lab-bars").append(row);
      }
      this.h2h(agents, bench.head_to_head);
      this.hist(agents);
    }
    this.training();
    for (const a of App.meta.agents) {
      const art = h("article", {}, avatar(a.key), h("h4", {}, a.name), h("p", {}, a.tagline), h("ul", {}, EXPLAIN[a.key].map((t) => h("li", {}, t))));
      art.style.setProperty("--accent", AGENT_COLOR[a.key]);
      $("#lab-explain").append(art);
    }
  },
  animate() {
    requestAnimationFrame(() => document.querySelectorAll(".bar-fill").forEach((b) => { b.style.width = b.dataset.w + "%"; }));
  },
  h2h(agents, table) {
    const t = h("table", { class: "h2h" });
    t.append(h("tr", {}, h("th"), agents.map((a) => h("th", { style: `color:${AGENT_COLOR[a.key]}` }, a.name))));
    for (const r of agents) {
      const tr = h("tr", {}, h("th", { class: "rowh", style: `--accent:${AGENT_COLOR[r.key]}` }, r.name));
      for (const c of agents) {
        if (r.key === c.key) { tr.append(h("td", { class: "self" }, "—")); continue; }
        const direct = table[`${r.key}_vs_${c.key}`], inverse = table[`${c.key}_vs_${r.key}`];
        const v = direct ? direct.a : inverse ? inverse.b : null;
        const td = h("td", {}, v == null ? "?" : pct(v));
        if (v != null) td.style.background = `color-mix(in srgb, ${v >= 0.5 ? "var(--green)" : "var(--hit)"} ${Math.round(Math.abs(v - 0.5) * 70 + 6)}%, transparent)`;
        tr.append(td);
      }
      t.append(tr);
    }
    $("#lab-h2h").append(t);
  },
  hist(agents) {
    const W = 560, H = 230, P = { l: 34, r: 10, t: 10, b: 30 };
    const all = agents.flatMap((a) => a.shots);
    const lo = Math.min(...all), hi = Math.max(...all), bin = 4;
    const start = Math.floor(lo / bin) * bin, nb = Math.ceil((hi - start + 1) / bin);
    const series = agents.map((a) => {
      const counts = new Array(nb).fill(0);
      a.shots.forEach((s) => counts[Math.floor((s - start) / bin)]++);
      return { a, counts: counts.map((c) => c / a.shots.length) };
    });
    const ymax = Math.max(...series.flatMap((s) => s.counts)) * 1.1;
    const x = (i) => P.l + (i / nb) * (W - P.l - P.r), y = (v) => H - P.b - (v / ymax) * (H - P.t - P.b);
    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Distribution of shots to win">`;
    for (let k = 0; k <= 4; k++) { const v = (ymax * k) / 4, yy = y(v); svg += `<line class="gridline" x1="${P.l}" x2="${W - P.r}" y1="${yy}" y2="${yy}"/><text x="${P.l - 6}" y="${yy + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`; }
    for (let i = 0; i <= nb; i += Math.ceil(nb / 8)) svg += `<text x="${x(i)}" y="${H - 10}" text-anchor="middle">${start + i * bin}</text>`;
    for (const s of series) {
      const pts = s.counts.map((v, i) => `${x(i + 0.5)},${y(v)}`);
      const color = AGENT_COLOR[s.a.key];
      svg += `<polygon points="${x(0.5)},${y(0)} ${pts.join(" ")} ${x(nb - 0.5)},${y(0)}" fill="${color}" opacity=".16"/>`;
      svg += `<polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linejoin="round"/>`;
    }
    svg += `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${y(0)}" y2="${y(0)}"/><text x="${W - P.r}" y="${H}" text-anchor="end">shots to win →</text></svg>`;
    $("#lab-hist").innerHTML = svg;
    $("#lab-hist").append(h("div", { class: "legend-labels", style: "justify-content:center;gap:18px;margin-top:6px" },
      agents.map((a) => h("span", { style: `color:${AGENT_COLOR[a.key]}` }, `● ${a.name}`))));
  },
  training() {
    const t = App.meta.training;
    if (!t || !t.history) { $("#lab-train").append(h("div", { class: "empty" }, "No training history found.")); return; }
    const hist = t.history;
    const params = t.channels ? `${t.blocks} residual blocks × ${t.channels} channels` : "";
    $("#lab-train-note").textContent = `Greedy shots-to-win on 10×10 after each self-play round (${params}, ${Math.round((t.train_seconds || 0) / 60)} min on CPU). Dashed lines: the other agents.`;
    const ref = App.meta.agents.filter((a) => a.benchmark && a.key !== "neural");
    const vals = [...hist.map((r) => r.shots_10x10), ...ref.map((a) => a.benchmark.mean)];
    const W = 560, H = 230, P = { l: 34, r: 70, t: 12, b: 30 };
    const lo = Math.floor(Math.min(...vals) / 5) * 5 - 2, hi = Math.ceil(Math.max(...vals) / 5) * 5 + 2;
    const x = (i) => P.l + (i / Math.max(1, hist.length - 1)) * (W - P.l - P.r), y = (v) => P.t + ((hi - v) / (hi - lo)) * (H - P.t - P.b);
    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Training curve">`;
    for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4; svg += `<line class="gridline" x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${P.l - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`; }
    for (const a of ref) svg += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(a.benchmark.mean)}" y2="${y(a.benchmark.mean)}" stroke="${AGENT_COLOR[a.key]}" stroke-dasharray="5 5" stroke-width="1.5"/><text x="${W - P.r + 6}" y="${y(a.benchmark.mean) + 4}" style="fill:${AGENT_COLOR[a.key]}">${a.name}</text>`;
    const pts = hist.map((r, i) => `${x(i)},${y(r.shots_10x10)}`).join(" ");
    svg += `<polyline points="${pts}" fill="none" stroke="var(--agent-neural)" stroke-width="2.5" stroke-linejoin="round" style="filter:drop-shadow(0 0 6px var(--agent-neural))"/>`;
    hist.forEach((r, i) => { svg += `<circle cx="${x(i)}" cy="${y(r.shots_10x10)}" r="3.2" fill="var(--agent-neural)"><title>round ${r.round}: ${r.shots_10x10} shots</title></circle>`; });
    svg += `<text x="${(W - P.r + P.l) / 2}" y="${H - 6}" text-anchor="middle">self-play round</text></svg>`;
    $("#lab-train").innerHTML = svg;
  },
};

App.init();
