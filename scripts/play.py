"""Human-play Flask harness for wanderbench.

Renders the EXACT same screenshots the model sees, accepts mouse + keyboard via the
SAME 6 action primitives, dispatches through the SAME WorldSim.step() path. Used
both for sanity-checking task feel and for generating human baselines.

Usage:
  pip install -e ".[play]"
  python scripts/play.py --task-id sf_pac_heights_01
  # open http://localhost:5000
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time as _time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, g, jsonify, request

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.sim import WorldSim  # noqa: E402
from core.tasks import load_tasks  # noqa: E402


app = Flask(__name__)

# --------------------------------------------------------------------------
# Per-session state (scalable / multi-user).
#
# The simulator state lives per browser session (cookie `lb_sid`), not in one
# global — so concurrent players don't clobber each other and the service can
# run many at once. Session sims are kept in-process under a lock with idle TTL
# eviction; behind an autoscaler use session affinity so a player's requests
# return to the instance holding their sim. Static config (difficulty toggles,
# tasks path) stays global. The `STATE` proxy routes session keys to the
# current request's session dict and config keys to the global dict, so all the
# existing `STATE[...]` call-sites keep working unchanged.
# --------------------------------------------------------------------------
_CONFIG = {
    "opts": {
        "show_compass": os.environ.get("LB_COMPASS", "1") == "1",
        "map_show_self": os.environ.get("LB_MAP_SELF", "1") == "1",
    },
    "tasks_path": None,
    "save_dir": None,
    "default_task": os.environ.get("LB_DEFAULT_TASK"),
}
_CONFIG_KEYS = set(_CONFIG.keys())

_SESSIONS: dict = {}
_SESS_LOCK = threading.Lock()
_SESS_TTL = int(os.environ.get("LB_SESSION_TTL", "1800"))   # idle eviction (s)
_SESS_MAX = int(os.environ.get("LB_MAX_SESSIONS", "2000"))  # hard cap


def _fresh_session() -> dict:
    return {"sim": None, "task_id": None, "history": [], "history_idx": -1,
            "actions": [], "_t_start": None, "_started_at": None, "_saved_for": None}


def _sid() -> str:
    sid = getattr(g, "_sid", None)
    if sid:
        return sid
    sid = request.cookies.get("lb_sid")
    g._sid_isnew = not sid
    if not sid:
        sid = uuid.uuid4().hex
    g._sid = sid
    return sid


def _session() -> dict:
    sid = _sid()
    now = _time.time()
    with _SESS_LOCK:
        # idle eviction + hard cap (drop oldest)
        if _SESSIONS:
            stale = [k for k, v in _SESSIONS.items() if now - v["last"] > _SESS_TTL]
            for k in stale:
                _SESSIONS.pop(k, None)
            if len(_SESSIONS) > _SESS_MAX:
                for k in sorted(_SESSIONS, key=lambda k: _SESSIONS[k]["last"])[:len(_SESSIONS) - _SESS_MAX]:
                    _SESSIONS.pop(k, None)
        s = _SESSIONS.get(sid)
        if s is None:
            s = {"data": _fresh_session(), "last": now}
            _SESSIONS[sid] = s
        s["last"] = now
        return s["data"]


class _StateProxy:
    """dict-like: config keys -> global; everything else -> per-session."""
    def __getitem__(self, k):
        return _CONFIG[k] if k in _CONFIG_KEYS else _session()[k]
    def __setitem__(self, k, v):
        if k in _CONFIG_KEYS:
            _CONFIG[k] = v
        else:
            _session()[k] = v
    def get(self, k, default=None):
        try:
            return self[k]
        except (KeyError, RuntimeError):
            return default


STATE = _StateProxy()


@app.after_request
def _persist_sid(resp):
    if getattr(g, "_sid_isnew", False) and getattr(g, "_sid", None):
        # On HTTPS (prod), SameSite=None;Secure so the cookie survives being
        # embedded cross-origin in the LostBench <iframe>. On plain HTTP (local
        # dev), fall back to Lax/no-Secure so it still works.
        secure = request.is_secure
        resp.set_cookie("lb_sid", g._sid, max_age=_SESS_TTL,
                        samesite="None" if secure else "Lax",
                        secure=secure, httponly=True)
    return resp


def _tasks_file():
    """Tasks jsonl to load (overridable via --tasks-path)."""
    return _CONFIG.get("tasks_path") or (REPO_ROOT / "data").joinpath("tasks.jsonl")


def _new_sim(task):
    """Construct a WorldSim with the active difficulty toggles."""
    return WorldSim(task=task, panos_dir=REPO_ROOT / "data" / "panos",
                    **STATE.get("opts", {}))


HTML = r"""<!doctype html>
<html><head><title>wanderbench</title>
<style>
  body { background: #111; color: #eee; font-family: system-ui, sans-serif; margin: 0; padding: 12px; }
  .row { display: flex; gap: 16px; }
  .panel { background: #222; padding: 12px; border-radius: 6px; }
  #view-wrap { position: relative; width: 1024px; height: 768px; overflow: hidden; border-radius: 4px; background: #000; }
  #view { width: 1024px; height: 768px; cursor: crosshair; user-select: none; -webkit-user-drag: none;
    will-change: transform; transition: none; display: block; }
  #view.dragging { cursor: grabbing; }
  .key { background: #333; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
  table { border-collapse: collapse; }
  td { padding: 2px 8px; }
  td.k { color: #888; }
  h2 { margin: 4px 0 8px; font-size: 14px; color: #888; font-weight: normal; text-transform: uppercase; letter-spacing: 0.06em; }
  #done-banner { display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: rgba(20, 200, 60, 0.95); color: white; padding: 20px 40px; border-radius: 6px;
    font-size: 24px; font-weight: bold; pointer-events: none; }
  #busy { position: absolute; top: 8px; left: 8px; background: rgba(255, 200, 0, 0.85); color: #000;
    padding: 4px 10px; border-radius: 3px; font-size: 12px; font-weight: bold; display: none; }
</style></head>
<body>
<div class="row">
  <div>
    <div id="view-wrap">
      <img id="view" draggable="false" />
      <div id="busy">working…</div>
      <div id="done-banner">GUESS SUBMITTED</div>
      <canvas id="debug-minimap" width="380" height="380" style="display:none;
              position:absolute;top:10px;right:10px;border:2px solid #fff;
              border-radius:4px;background:#fafafa;box-shadow:0 2px 8px rgba(0,0,0,0.4)"></canvas>
    </div>
    <div style="font-size:12px;color:#888;margin-top:6px">
      <b>Click on the street</b> to walk there (click near horizon = travel far). <b>Click-and-drag</b> = pan camera.
      <b>Scroll</b> = zoom. <span class="key">M</span> open map, <span class="key">N</span> close map,
      <span class="key">R</span> reset.
    </div>
  </div>
  <div class="panel" style="min-width:260px">
    <h2>Task</h2>
    <select id="task-picker" style="width:100%;padding:6px;background:#111;color:#eee;
            border:1px solid #444;border-radius:4px;margin-bottom:10px;font-family:monospace"></select>
    <table id="task-info"></table>
    <h2 style="margin-top:14px">State</h2>
    <table id="state-info"></table>
    <h2 style="margin-top:14px">Last action</h2>
    <div id="last-action" style="font-family:monospace;font-size:12px;color:#aaf;word-break:break-all"></div>
    <button id="submit-guess-btn" style="margin-top:14px;padding:10px 16px;
            background:#c91e1e;color:#fff;border:0;border-radius:4px;
            font-size:14px;font-weight:bold;cursor:pointer;width:100%">
      Submit guess (I'm here)
    </button>
    <button id="debug-toggle-btn" style="margin-top:8px;padding:6px 12px;
            background:#333;color:#eee;border:1px solid #555;border-radius:4px;
            font-size:12px;cursor:pointer;width:100%">
      Debug mode: OFF
    </button>
    <div id="debug-replay" style="display:none;margin-top:8px;
            background:#1c1c1c;border:1px solid #444;border-radius:4px;padding:8px;">
      <div style="font-size:11px;color:#aaa;margin-bottom:6px">Load rollout:</div>
      <select id="rollout-picker" style="width:100%;padding:4px;background:#111;
              color:#eee;border:1px solid #444;border-radius:3px;font-size:11px;
              font-family:monospace;margin-bottom:8px"></select>
      <div id="rollout-info" style="font-size:10px;color:#8ab;margin-bottom:8px;display:none"></div>
      <div style="font-size:11px;color:#aaa;margin-bottom:4px">Replay
        <span id="replay-pos" style="float:right;color:#fff;font-family:monospace">0/0</span>
      </div>
      <div style="display:flex;gap:4px;align-items:center">
        <button id="replay-back" style="flex:1;padding:4px;background:#333;color:#eee;
                border:1px solid #555;border-radius:3px;font-size:14px;cursor:pointer">⏪</button>
        <button id="replay-fwd" style="flex:1;padding:4px;background:#333;color:#eee;
                border:1px solid #555;border-radius:3px;font-size:14px;cursor:pointer">⏩</button>
        <button id="replay-live" style="flex:1;padding:4px;background:#333;color:#eee;
                border:1px solid #555;border-radius:3px;font-size:11px;cursor:pointer">Live</button>
      </div>
      <input id="replay-slider" type="range" min="0" max="0" value="0" step="1"
             style="width:100%;margin-top:6px">
      <div id="replay-label" style="font-size:10px;color:#888;font-family:monospace;
           margin-top:4px;word-break:break-all;min-height:14px"></div>
    </div>
    <div id="guess-result" style="margin-top:10px;font-size:13px;display:none"></div>
  </div>
</div>

<script>
const VIEW_W = 1024, VIEW_H = 768;
const CLICK_PX_THRESHOLD = 25;  // matches sim.py — generous, since trackpad clicks jitter
let cursorX = VIEW_W / 2, cursorY = VIEW_H / 2;  // mirror of WorldSim cursor
let viewEl = document.getElementById('view');
let busyEl = document.getElementById('busy');
let stateEl = document.getElementById('state-info');
let taskEl = document.getElementById('task-info');
let lastActionEl = document.getElementById('last-action');
let doneBanner = document.getElementById('done-banner');

// Drag-preview state
let dragActive = false;
let dragStartX = 0, dragStartY = 0;  // image-space
let dragCurX = 0, dragCurY = 0;

// Debug minimap state
let debugMode = false;
let lastState = null;
let cachedGraph = null;       // {task_id, bbox, nodes:[{pano_id,lat,lng,compass_angle,neighbors:[...]}], start_*, goal_*}
let cachedGraphTaskId = null;
const minimapCanvas = document.getElementById('debug-minimap');
const minimapCtx = minimapCanvas.getContext('2d');
const tileCache = new Map();   // url → HTMLImageElement (after load)
// Minimap interaction state (debug only)
let miniDragActive = false;
let miniDragStart = {x: 0, y: 0};
let miniDragCur = {x: 0, y: 0};
let miniDragShift = false;       // shift held when drag started → yaw mode
let miniDragPanStart = {x: 0, y: 0};  // pan offsets at drag start (for incremental pan)
let miniPanX = 0, miniPanY = 0;  // canvas-pixel offset from "centered on current cam"
let miniZoomOverride = null;     // integer zoom level set by scroll; null = default
const MINI_CLICK_PX = 6;        // distance ≤ this on release = click (teleport)
const MINI_HIT_RADIUS_PX = 14;  // teleport snaps to nearest dot within this radius
const MINI_DEFAULT_Z = 18;
const MINI_MIN_Z = 15, MINI_MAX_Z = 20;

function loadTile(url) {
  if (tileCache.has(url)) return tileCache.get(url);
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => drawMinimap();   // trigger re-render once tile is in
  img.onerror = () => {};
  img.src = url;
  tileCache.set(url, img);
  return img;
}

function latLngToWorldPx(lat, lng, z) {
  const n = Math.pow(2, z);
  const x = (lng + 180) / 360 * n * 256;
  const sl = Math.sin(lat * Math.PI / 180);
  const y = (0.5 - Math.log((1 + sl) / (1 - sl)) / (4 * Math.PI)) * n * 256;
  return {x, y};
}

// Scroll debounce
let scrollAccum = 0;
let scrollTimer = null;

// Network busy tracker
let pendingRequests = 0;
function setBusy(v) {
  pendingRequests = Math.max(0, pendingRequests + (v ? 1 : -1));
  busyEl.style.display = pendingRequests > 0 ? 'block' : 'none';
}

async function callBatch(actions) {
  if (!actions.length) return;
  setBusy(true);
  lastActionEl.textContent = actions.map(a => a.tool +
    (a.args && Object.keys(a.args).length ? ' ' + JSON.stringify(a.args) : '')).join(' → ');
  try {
    const r = await fetch('/action_batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({actions}),
    });
    const data = await r.json();
    applyResponse(data);
  } finally {
    setBusy(false);
  }
}

function applyResponse(data) {
  viewEl.src = 'data:image/jpeg;base64,' + data.image_b64;
  cursorX = data.cursor_x;
  cursorY = data.cursor_y;
  renderState(data.state);
  doneBanner.style.display = data.state.done ? 'block' : 'none';
  lastState = data.state;
  if (debugMode) {
    ensureGraphLoaded().then(drawMinimap);
    refreshHistoryUI();
  }
  if (data.state.guess_submitted) {
    const r = document.getElementById('guess-result');
    const err = data.state.guess_error_m;
    const init = data.state.initial_distance_m;
    const score = Math.max(0, Math.min(1, 1 - err / Math.max(1, init)));
    r.style.display = 'block';
    r.innerHTML = `<b>Guess submitted</b><br>Error: ${err.toFixed(1)} m<br>` +
                  `Reward: ${score.toFixed(3)}`;
    document.getElementById('submit-guess-btn').disabled = true;
  }
}

function renderState(s) {
  const rows = [
    ['turn', s.turn_count],
    ['steps', s.steps_taken],
    ['view', s.view_mode],
    ['mouse', s.mouse_is_down ? 'DOWN' : 'up'],
    ['cursor', `(${cursorX}, ${cursorY})`],
    ['yaw', s.yaw_deg.toFixed(1) + '°'],
    ['fov', s.fov_deg.toFixed(0) + '°'],
    ['pano', s.current_pano_id],
    ['dist_to_goal', s.distance_to_goal_m.toFixed(1) + ' m'],
    ['done', s.done ? 'YES' : 'no'],
  ];
  stateEl.innerHTML = rows.map(r => `<tr><td class="k">${r[0]}</td><td>${r[1]}</td></tr>`).join('');
}

function renderTask(t) {
  const rows = [
    ['task_id', t.task_id],
    ['city', t.city],
    ['optimal_steps', t.optimal_steps],
    ['straight_line_m', t.initial_distance_m.toFixed(0)],
    ['goal_radius_m', t.goal_radius_m],
  ];
  taskEl.innerHTML = rows.map(r => `<tr><td class="k">${r[0]}</td><td>${r[1]}</td></tr>`).join('');
}

function moveAction(fromX, fromY, toX, toY) {
  const dx = toX - fromX;
  const dy = toY - fromY;
  const distance = Math.round(Math.hypot(dx, dy));
  if (distance < 1) return null;
  const direction = Math.atan2(-dy, dx) * 180 / Math.PI;
  return {tool: 'move_cursor', args: {direction_deg: direction, distance_px: distance}};
}

function eventToImgCoords(e) {
  const rect = viewEl.getBoundingClientRect();
  return [
    Math.round((e.clientX - rect.left) * VIEW_W / rect.width),
    Math.round((e.clientY - rect.top) * VIEW_H / rect.height),
  ];
}

viewEl.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  e.preventDefault();
  const [x, y] = eventToImgCoords(e);
  dragActive = true;
  dragStartX = x; dragStartY = y;
  dragCurX = x; dragCurY = y;
  viewEl.classList.add('dragging');
});

viewEl.addEventListener('mousemove', (e) => {
  if (!dragActive) return;
  const [x, y] = eventToImgCoords(e);
  dragCurX = x; dragCurY = y;
  // Live CSS-translate preview of the drag (only meaningful in pano view).
  // Drag right → image shifts right (mirrors the eventual yaw rotation visually).
  const tx = x - dragStartX;
  const ty = y - dragStartY;
  viewEl.style.transform = `translate(${tx}px, ${ty}px)`;
});

window.addEventListener('mouseup', async (e) => {
  if (!dragActive) return;
  dragActive = false;
  viewEl.classList.remove('dragging');
  viewEl.style.transform = '';

  const dist = Math.hypot(dragCurX - dragStartX, dragCurY - dragStartY);
  const actions = [];
  if (dist < CLICK_PX_THRESHOLD) {
    // Click: move cursor to click point, then mouse_down + mouse_up
    const m = moveAction(cursorX, cursorY, dragCurX, dragCurY);
    if (m) actions.push(m);
    actions.push({tool: 'mouse_down'});
    actions.push({tool: 'mouse_up'});
  } else {
    // Drag: move to start, mouse_down, move to end (this fires the pan), mouse_up
    const m1 = moveAction(cursorX, cursorY, dragStartX, dragStartY);
    if (m1) actions.push(m1);
    actions.push({tool: 'mouse_down'});
    const m2 = moveAction(dragStartX, dragStartY, dragCurX, dragCurY);
    if (m2) actions.push(m2);
    actions.push({tool: 'mouse_up'});
  }
  await callBatch(actions);
});

viewEl.addEventListener('wheel', (e) => {
  e.preventDefault();
  // Debounce: accumulate wheel delta, fire once after 80ms of quiet
  scrollAccum += e.deltaY > 0 ? -1 : 1;
  if (scrollTimer) clearTimeout(scrollTimer);
  scrollTimer = setTimeout(async () => {
    const d = Math.max(-10, Math.min(10, scrollAccum * 2));
    scrollAccum = 0;
    scrollTimer = null;
    if (d !== 0) await callBatch([{tool: 'scroll_wheel', args: {delta_y: d}}]);
  }, 80);
}, {passive: false});

document.addEventListener('keydown', async (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  // If the task picker has focus, release it so M/N/R don't also jump to an
  // option starting with that letter.
  if (e.target.tagName === 'SELECT') e.target.blur();
  if (e.key === 'm' || e.key === 'M') await callBatch([{tool: 'open_map'}]);
  else if (e.key === 'n' || e.key === 'N') await callBatch([{tool: 'close_map'}]);
  else if (e.key === 'r' || e.key === 'R') {
    setBusy(true);
    try {
      const r = await fetch('/init');
      applyResponse(await r.json());
    } finally { setBusy(false); }
  }
});

document.getElementById('submit-guess-btn').addEventListener('click', async () => {
  await callBatch([{tool: 'submit_guess'}]);
});

// Debug mode toggle + minimap drawing
const debugBtn = document.getElementById('debug-toggle-btn');
const replayPanel = document.getElementById('debug-replay');
const replaySlider = document.getElementById('replay-slider');
const replayBack = document.getElementById('replay-back');
const replayFwd = document.getElementById('replay-fwd');
const replayLive = document.getElementById('replay-live');
const replayPos = document.getElementById('replay-pos');
const replayLabel = document.getElementById('replay-label');
let actionLabels = [];  // mirrors server's STATE["actions"]

debugBtn.addEventListener('click', async () => {
  debugMode = !debugMode;
  debugBtn.textContent = 'Debug mode: ' + (debugMode ? 'ON' : 'OFF');
  minimapCanvas.style.display = debugMode ? 'block' : 'none';
  replayPanel.style.display = debugMode ? 'block' : 'none';
  if (debugMode) {
    await ensureGraphLoaded();
    await refreshHistoryUI();
    await refreshRolloutPicker();
    drawMinimap();
  }
});

// Rollout picker: list saved benchmark runs from /rollouts and let the user
// load one into the sim history (replaces the live history with the rollout)
const rolloutPicker = document.getElementById('rollout-picker');
const rolloutInfo = document.getElementById('rollout-info');
rolloutPicker.addEventListener('wheel', (e) => e.preventDefault(), {passive:false});
async function refreshRolloutPicker() {
  const r = await fetch('/rollouts');
  if (!r.ok) return;
  const d = await r.json();
  rolloutPicker.innerHTML = '<option value="">(none — live)</option>';
  for (const ro of (d.rollouts || [])) {
    const opt = document.createElement('option');
    opt.value = ro.filename;
    const rew = (ro.reward !== null && ro.reward !== undefined) ? ro.reward.toFixed(2) : '?';
    const err = (ro.guess_error_m !== null && ro.guess_error_m !== undefined) ? ro.guess_error_m.toFixed(0)+'m' : 'no submit';
    opt.textContent = `${ro.model}  t=${ro.n_turns}  R=${rew}  ${err}  [${ro.task_id}]`;
    rolloutPicker.appendChild(opt);
  }
}
rolloutPicker.addEventListener('change', async () => {
  rolloutPicker.blur();
  const fname = rolloutPicker.value;
  if (!fname) { rolloutInfo.style.display = 'none'; return; }
  setBusy(true);
  try {
    const r = await fetch('/load_rollout', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({filename: fname}),
    });
    const data = await r.json();
    if (data.error) { rolloutInfo.textContent = data.error; rolloutInfo.style.display = 'block'; return; }
    if (data.rollout_loaded) {
      const ri = data.rollout_loaded;
      rolloutInfo.textContent = `${ri.model} • ${ri.n_turns} turns • R=${(ri.reward||0).toFixed(2)} • err=${(ri.guess_error_m||0).toFixed(0)}m`;
      rolloutInfo.style.display = 'block';
    }
    // Refresh task display + minimap cache since the task may have changed
    cachedGraph = null; cachedGraphTaskId = null;
    applyResponse(data);
    await ensureGraphLoaded();
    await refreshHistoryUI();
    drawMinimap();
  } finally { setBusy(false); }
});

async function refreshHistoryUI() {
  const r = await fetch('/history');
  if (!r.ok) return;
  const d = await r.json();
  actionLabels = d.actions || [];
  const total = d.len;
  const idx = d.idx === -1 ? total - 1 : d.idx;
  replaySlider.max = Math.max(0, total - 1);
  replaySlider.value = idx;
  replayPos.textContent = `${idx + 1}/${total}`;
  // Label entries: actions[i] produced history[i+1]. So when at history idx,
  // the action that just ran is actions[idx-1] (or "(initial)" at idx=0).
  replayLabel.textContent = idx === 0 ? '(initial)' : (actionLabels[idx - 1] || '');
  replayBack.disabled = idx <= 0;
  replayFwd.disabled = idx >= total - 1;
  replayLive.disabled = d.idx === -1;
}

async function seekHistory(idx) {
  setBusy(true);
  try {
    const r = await fetch('/history_seek', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({idx}),
    });
    applyResponse(await r.json());
    await refreshHistoryUI();
  } finally { setBusy(false); }
}

replayBack.addEventListener('click', () => {
  const cur = parseInt(replaySlider.value, 10);
  if (cur > 0) seekHistory(cur - 1);
});
replayFwd.addEventListener('click', () => {
  const cur = parseInt(replaySlider.value, 10);
  if (cur < parseInt(replaySlider.max, 10)) seekHistory(cur + 1);
});
replayLive.addEventListener('click', () => {
  seekHistory(parseInt(replaySlider.max, 10));
});
replaySlider.addEventListener('input', () => {
  seekHistory(parseInt(replaySlider.value, 10));
});

// Keyboard: [/] (and ,/.) step backward/forward in history while debug is on
document.addEventListener('keydown', (e) => {
  if (!debugMode) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === '[' || e.key === ',') { replayBack.click(); }
  else if (e.key === ']' || e.key === '.') { replayFwd.click(); }
});

async function ensureGraphLoaded() {
  const taskId = lastState && lastState.current_pano_id
    ? (document.getElementById('task-picker').value || null) : null;
  if (!taskId) return;
  if (cachedGraph && cachedGraphTaskId === taskId) return;
  const r = await fetch('/graph?task_id=' + encodeURIComponent(taskId));
  if (!r.ok) return;
  cachedGraph = await r.json();
  cachedGraphTaskId = taskId;
}

// Minimap zoom: Web Mercator integer zoom level. Default 18 (~150m/tile at
// SF lat → ~190m canvas-wide). Scroll wheel adjusts to MINI_MIN_Z..MINI_MAX_Z.
// Tiles render the OSM basemap (street names, parks, etc.) behind the
// overlay (road waypoints + current-position marker + heading arrow).

// Camera position of a node (image_lat/lng if set, else centerline)
function camPosOf(n) {
  if (!n) return {lat: 0, lng: 0};
  return {
    lat: n.image_lat ? n.image_lat : n.lat,
    lng: n.image_lng ? n.image_lng : n.lng,
  };
}

// Compute the same projection drawMinimap uses, so click/drag handlers can
// project lat/lng → canvas px (for hit-testing) and canvas px → lat/lng
// (for click-to-teleport queries).
function getMinimapProjection() {
  if (!cachedGraph || !lastState) return null;
  const W = minimapCanvas.width, H = minimapCanvas.height;
  const nodeById = {};
  for (const n of cachedGraph.nodes) nodeById[n.pano_id] = n;
  const cur = nodeById[lastState.current_pano_id];
  const curCam = cur ? camPosOf(cur) : null;
  const centerLat = curCam ? curCam.lat : (cachedGraph.bbox[1] + cachedGraph.bbox[3]) / 2;
  const centerLng = curCam ? curCam.lng : (cachedGraph.bbox[0] + cachedGraph.bbox[2]) / 2;
  const z = miniZoomOverride != null ? miniZoomOverride : MINI_DEFAULT_Z;
  const center = latLngToWorldPx(centerLat, centerLng, z);
  // miniPanX/Y is an offset in canvas px (drag pans the viewport); subtract
  // from world-px top-left so panning right moves the map left.
  const tlx = center.x - W / 2 - miniPanX, tly = center.y - H / 2 - miniPanY;
  return {
    W, H, z, tlx, tly, cur, curCam, nodeById,
    proj: (lat, lng) => {
      const wp = latLngToWorldPx(lat, lng, z);
      return {x: wp.x - tlx, y: wp.y - tly};
    },
  };
}

function drawMinimap() {
  if (!debugMode || !cachedGraph || !lastState) return;
  const ctx = minimapCtx;
  const P = getMinimapProjection();
  if (!P) return;
  const {W, H, z, tlx, tly, cur, curCam, nodeById, proj} = P;
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, W, H);
  const inFrame = (p) => p.x >= -10 && p.x <= W+10 && p.y >= -10 && p.y <= H+10;

  // OSM tile basemap underneath everything
  const txLo = Math.floor(tlx / 256), txHi = Math.floor((tlx + W) / 256);
  const tyLo = Math.floor(tly / 256), tyHi = Math.floor((tly + H) / 256);
  for (let tx = txLo; tx <= txHi; tx++) {
    for (let ty = tyLo; ty <= tyHi; ty++) {
      const url = `https://a.tile.openstreetmap.org/${z}/${tx}/${ty}.png`;
      const img = loadTile(url);
      if (img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, tx*256 - tlx, ty*256 - tly);
      }
    }
  }

  // Bbox border, if visible
  const bbox = cachedGraph.bbox;
  if (bbox && bbox.length === 4) {
    const bp1 = proj(bbox[1], bbox[0]);
    const bp2 = proj(bbox[3], bbox[2]);
    ctx.strokeStyle = '#d62728'; ctx.lineWidth = 1.5;
    ctx.strokeRect(bp1.x, bp2.y, bp2.x - bp1.x, bp1.y - bp2.y);
  }

  // Road edges (only those with at least one endpoint in frame)
  ctx.strokeStyle = '#5483c9'; ctx.lineWidth = 2;
  ctx.beginPath();
  const drawn = new Set();
  for (const n of cachedGraph.nodes) {
    const a = proj(n.lat, n.lng);
    if (!inFrame(a)) continue;
    for (const nbId of n.neighbors) {
      const key = n.pano_id < nbId ? n.pano_id+'|'+nbId : nbId+'|'+n.pano_id;
      if (drawn.has(key)) continue;
      drawn.add(key);
      const nb = nodeById[nbId]; if (!nb) continue;
      const b = proj(nb.lat, nb.lng);
      ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    }
  }
  ctx.stroke();

  // Node dots — sized so individual waypoints are clickable/readable at this zoom
  const NODE_R = 4;
  for (const n of cachedGraph.nodes) {
    const p = proj(n.lat, n.lng);
    if (!inFrame(p)) continue;
    ctx.fillStyle = '#1f4d8f';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.arc(p.x, p.y, NODE_R, 0, 2*Math.PI);
    ctx.fill(); ctx.stroke();
  }

  // Start (green) and goal (red) — slightly larger so they stand out from the
  // regular waypoint dots, but same general shape
  const sp = proj(cachedGraph.start_lat, cachedGraph.start_lng);
  if (inFrame(sp)) {
    ctx.fillStyle = '#1ec900'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(sp.x, sp.y, NODE_R + 2, 0, 2*Math.PI); ctx.fill(); ctx.stroke();
  }
  const gp = proj(cachedGraph.goal_lat, cachedGraph.goal_lng);
  if (inFrame(gp)) {
    ctx.fillStyle = '#c91e1e'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(gp.x, gp.y, NODE_R + 2, 0, 2*Math.PI); ctx.fill(); ctx.stroke();
  }

  // Current node: same waypoint dot but recolored + glow ring + direction arrow.
  // Drawn at the CAMERA position (where the photo was actually taken), which
  // can be 2-5m off the centerline since cars drive in lanes. The user dot
  // therefore lands where what they see in the pano was shot from.
  if (cur) {
    const cp = proj(curCam.lat, curCam.lng);
    // Outer glow ring
    ctx.strokeStyle = '#ffeb3b'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cp.x, cp.y, NODE_R + 4, 0, 2*Math.PI); ctx.stroke();
    // Recolored waypoint dot
    ctx.fillStyle = '#ffeb3b'; ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cp.x, cp.y, NODE_R, 0, 2*Math.PI); ctx.fill(); ctx.stroke();

    // Direction arrow
    const worldLook = ((cur.compass_angle || 0) + (lastState.yaw_deg || 0)) % 360;
    const rad = worldLook * Math.PI / 180;
    const ax = Math.sin(rad), ay = -Math.cos(rad);
    const len = 22;
    const tipX = cp.x + ax * len, tipY = cp.y + ay * len;
    ctx.strokeStyle = '#000'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cp.x, cp.y); ctx.lineTo(tipX, tipY); ctx.stroke();
    const perpX = -ay, perpY = ax;
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - ax*8 + perpX*5, tipY - ay*8 + perpY*5);
    ctx.lineTo(tipX - ax*8 - perpX*5, tipY - ay*8 - perpY*5);
    ctx.closePath(); ctx.fill();
  }

  // Drag preview: dashed line from current dot toward cursor — shows the
  // direction the camera will face if released here.
  if (miniDragActive && cur) {
    const cp = proj(curCam.lat, curCam.lng);
    const dx = miniDragCur.x - cp.x, dy = miniDragCur.y - cp.y;
    if (Math.hypot(dx, dy) >= MINI_CLICK_PX) {
      ctx.save();
      ctx.strokeStyle = '#ff8800'; ctx.lineWidth = 3;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(cp.x, cp.y);
      ctx.lineTo(miniDragCur.x, miniDragCur.y); ctx.stroke();
      ctx.restore();
      // Show the resulting bearing in the HUD
    }
  }

  // HUD: task id + look bearing + zoom
  ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fillRect(0, 0, W, 18);
  ctx.fillStyle = '#fff'; ctx.font = '11px ui-monospace,monospace';
  const look = cur ? (((cur.compass_angle || 0) + (lastState.yaw_deg || 0)) % 360).toFixed(0) : '?';
  let hudText = `z${z}  look=${look}°  click=tp · drag=pan · shift+drag=yaw · wheel=zoom`;
  if (miniDragActive && miniDragShift && curCam) {
    const cp = proj(curCam.lat, curCam.lng);
    const dx = miniDragCur.x - cp.x, dy = miniDragCur.y - cp.y;
    if (Math.hypot(dx, dy) >= MINI_CLICK_PX) {
      const dragBearing = ((Math.atan2(dx, -dy) * 180 / Math.PI) + 360) % 360;
      hudText = `set yaw → ${dragBearing.toFixed(0)}°   (release to apply)`;
    }
  } else if (miniDragActive) {
    hudText = `panning   (z${z})`;
  }
  ctx.fillText(hudText, 4, 13);
}

// Minimap interaction handlers
//   • click (no drag) → teleport to nearest dot
//   • drag without shift → PAN the map
//   • drag with shift → set camera yaw to bearing-from-current-dot-to-release
//   • mouse wheel → zoom in/out (Mercator integer zoom 15..20)
function minimapCanvasCoords(e) {
  const rect = minimapCanvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (minimapCanvas.width / rect.width),
    y: (e.clientY - rect.top) * (minimapCanvas.height / rect.height),
  };
}
minimapCanvas.addEventListener('mousedown', (e) => {
  if (!debugMode) return;
  e.preventDefault();
  const c = minimapCanvasCoords(e);
  miniDragActive = true;
  miniDragShift = e.shiftKey;
  miniDragStart = c;
  miniDragCur = c;
  miniDragPanStart = {x: miniPanX, y: miniPanY};
});
minimapCanvas.addEventListener('mousemove', (e) => {
  if (!miniDragActive) return;
  const c = minimapCanvasCoords(e);
  miniDragCur = c;
  if (!miniDragShift) {
    // Pan: update offset incrementally from drag start
    miniPanX = miniDragPanStart.x + (c.x - miniDragStart.x);
    miniPanY = miniDragPanStart.y + (c.y - miniDragStart.y);
  }
  drawMinimap();
});
window.addEventListener('mouseup', async (e) => {
  if (!miniDragActive) return;
  miniDragActive = false;
  const c = minimapCanvasCoords(e);
  const moved = Math.hypot(c.x - miniDragStart.x, c.y - miniDragStart.y);
  const wasShift = miniDragShift;
  miniDragShift = false;

  if (moved < MINI_CLICK_PX) {
    // CLICK → teleport to nearest node within MINI_HIT_RADIUS_PX
    const P = getMinimapProjection();
    if (!P) { drawMinimap(); return; }
    let best = null, bestDist = MINI_HIT_RADIUS_PX;
    for (const n of cachedGraph.nodes) {
      const c2 = camPosOf(n);
      const p = P.proj(c2.lat, c2.lng);
      const d = Math.hypot(p.x - c.x, p.y - c.y);
      if (d < bestDist) { bestDist = d; best = n; }
    }
    if (best && best.pano_id !== lastState.current_pano_id) {
      setBusy(true);
      try {
        const r = await fetch('/debug_set_pose', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({pano_id: best.pano_id}),
        });
        // Re-center on the new pano after teleport
        miniPanX = 0; miniPanY = 0;
        applyResponse(await r.json());
      } finally { setBusy(false); }
    }
    return;
  }

  if (wasShift) {
    // SHIFT+DRAG → set camera yaw
    const P = getMinimapProjection();
    if (P && P.curCam) {
      const cp = P.proj(P.curCam.lat, P.curCam.lng);
      const dx = c.x - cp.x, dy = c.y - cp.y;
      const worldLook = ((Math.atan2(dx, -dy) * 180 / Math.PI) + 360) % 360;
      setBusy(true);
      try {
        const r = await fetch('/debug_set_pose', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({world_look_deg: worldLook}),
        });
        applyResponse(await r.json());
      } finally { setBusy(false); }
    }
  }
  // Plain drag committed as pan during mousemove; nothing to do on release.
  drawMinimap();
});

// Mouse wheel: zoom in/out. Each Mercator zoom step doubles/halves area, so a
// single wheel tick is a big jump. Accumulate deltaY and only fire a zoom step
// when accumulated motion exceeds a threshold — feels less twitchy on trackpads
// (which spam many small wheel events per gesture) and chunky mice alike.
let miniWheelAccum = 0;
let miniWheelTimer = null;
const MINI_ZOOM_THRESHOLD = 50;  // higher = less sensitive
minimapCanvas.addEventListener('wheel', (e) => {
  if (!debugMode) return;
  e.preventDefault();
  const delta = -e.deltaY;
  // Reset accumulator on direction reversal so user can quickly reverse zoom
  if (miniWheelAccum * delta < 0) miniWheelAccum = 0;
  miniWheelAccum += delta;
  // Clear accumulator after idle so stale partial-deltas don't combine with later gestures
  if (miniWheelTimer) clearTimeout(miniWheelTimer);
  miniWheelTimer = setTimeout(() => { miniWheelAccum = 0; }, 400);
  if (Math.abs(miniWheelAccum) < MINI_ZOOM_THRESHOLD) return;

  const direction = miniWheelAccum > 0 ? 1 : -1;
  miniWheelAccum = 0;
  const curZ = miniZoomOverride != null ? miniZoomOverride : MINI_DEFAULT_Z;
  const newZ = Math.max(MINI_MIN_Z, Math.min(MINI_MAX_Z, curZ + direction));
  if (newZ === curZ) return;
  // Adjust pan so the point under the cursor stays under the cursor across zoom
  const c = minimapCanvasCoords(e);
  const W = minimapCanvas.width, H = minimapCanvas.height;
  const factor = Math.pow(2, newZ - curZ);
  miniPanX = (miniPanX - (c.x - W/2)) * factor + (c.x - W/2);
  miniPanY = (miniPanY - (c.y - H/2)) * factor + (c.y - H/2);
  miniZoomOverride = newZ;
  drawMinimap();
}, {passive: false});

// Task picker
const taskPicker = document.getElementById('task-picker');
// Native <select> hijacks scroll-wheel and arrow keys when focused — which means
// trying to scroll-zoom the map or press M to open it accidentally cycles tasks.
// Suppress both, and blur the picker after every interaction so it never holds focus.
taskPicker.addEventListener('wheel', (e) => e.preventDefault(), {passive: false});
taskPicker.addEventListener('keydown', (e) => {
  if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','PageUp','PageDown','Home','End'].includes(e.key)) {
    e.preventDefault();
  }
});
taskPicker.addEventListener('change', async () => {
  const target = taskPicker.value;
  taskPicker.blur();  // release focus so subsequent scroll/keys don't cycle
  cachedGraph = null; cachedGraphTaskId = null;  // invalidate debug minimap cache
  miniPanX = 0; miniPanY = 0; miniZoomOverride = null;  // reset minimap viewport
  setBusy(true);
  try {
    const r = await fetch('/init?task_id=' + encodeURIComponent(target));
    applyResponse(await r.json());
    document.getElementById('submit-guess-btn').disabled = false;
    document.getElementById('guess-result').style.display = 'none';
  } finally { setBusy(false); }
});

async function populateTaskPicker(currentTaskId) {
  const r = await fetch('/tasks');
  const data = await r.json();
  taskPicker.innerHTML = '';
  // Group by city for cleaner display
  const byCity = {};
  for (const t of data.tasks) {
    (byCity[t.city] = byCity[t.city] || []).push(t);
  }
  for (const city of Object.keys(byCity).sort()) {
    const group = document.createElement('optgroup');
    group.label = city;
    for (const t of byCity[city]) {
      const opt = document.createElement('option');
      opt.value = t.task_id;
      const band = t.difficulty ? `[${t.difficulty}]` : '';
      opt.textContent = `${t.task_id}  ${band}  ${t.optimal_distance_m.toFixed(0)}m / ${t.optimal_steps} hops`;
      group.appendChild(opt);
    }
    taskPicker.appendChild(group);
  }
  taskPicker.value = currentTaskId;
}

// Bootstrap
fetch('/init').then(r => r.json()).then(d => {
  renderTask(d.task);
  populateTaskPicker(d.task.task_id);
  applyResponse(d);
});
</script>
</body></html>
"""


def _frame_b64(sim: WorldSim) -> str:
    img = sim.render().image
    buf = io.BytesIO()
    # JPEG q88: ~74× faster than PNG and ~6× smaller payload. Visible artifacts on
    # overlays at this resolution are negligible.
    img.save(buf, format="JPEG", quality=88, optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _state_dict(sim: WorldSim) -> dict:
    from core.tasks import _haversine_m
    err = (_haversine_m(sim.guess_lat, sim.guess_lng,
                        sim.task.goal_lat, sim.task.goal_lng)
           if sim.guess_submitted else 0.0)
    return {
        "turn_count": sim.turn_count,
        "steps_taken": sim.steps_taken,
        "view_mode": sim.view_mode,
        "mouse_is_down": sim.mouse_is_down,
        "yaw_deg": sim.yaw_deg,
        "pitch_deg": sim.pitch_deg,
        "fov_deg": sim.fov_deg,
        "current_pano_id": sim.current_pano_id,
        "distance_to_goal_m": sim.distance_to_goal_m(),
        "done": sim.done,
        "guess_submitted": sim.guess_submitted,
        "guess_error_m": err,
        "initial_distance_m": sim.task.initial_distance_m,
    }


def _response(sim: WorldSim) -> dict:
    return {
        "image_b64": _frame_b64(sim),
        "cursor_x": sim.cursor_x,
        "cursor_y": sim.cursor_y,
        "state": _state_dict(sim),
        "history_len": len(STATE.get("history", [])),
        "history_idx": STATE.get("history_idx", -1),
    }


# ---- replay history ------------------------------------------------------

_SNAPSHOT_FIELDS = (
    "current_pano_id", "yaw_deg", "pitch_deg", "fov_deg", "view_mode",
    "cursor_x", "cursor_y", "mouse_is_down", "mouse_down_x", "mouse_down_y",
    "drag_distance_px", "turn_count", "steps_taken", "last_action",
    "last_action_was_valid", "done", "guess_submitted", "guess_lat", "guess_lng",
    "map_zoom", "map_center_lat", "map_center_lng",
)


def _snapshot(sim: WorldSim) -> dict:
    snap = {k: getattr(sim, k) for k in _SNAPSHOT_FIELDS}
    snap["visited_panos"] = list(sim.visited_panos)
    return snap


def _restore(sim: WorldSim, snap: dict) -> None:
    for k, v in snap.items():
        setattr(sim, k, v)


def _record_action(sim: WorldSim, label: str) -> None:
    """Append a snapshot AFTER an action ran. If user was replaying (history_idx
    not at end), discards the future first (standard undo/redo branch behavior).
    """
    hist = STATE["history"]; acts = STATE["actions"]
    idx = STATE["history_idx"]
    if idx != -1 and idx < len(hist) - 1:
        # User branched — discard future
        del hist[idx + 1:]
        del acts[idx:]  # acts[i] produced hist[i+1], so trim from idx
    hist.append(_snapshot(sim))
    acts.append(label)
    STATE["history_idx"] = -1  # back to "live"


def _reset_history(sim: WorldSim) -> None:
    STATE["history"] = [_snapshot(sim)]
    STATE["actions"] = []  # actions[i] produced history[i+1]
    STATE["history_idx"] = -1


@app.route("/")
def index() -> str:
    return HTML


@app.route("/coverage")
def coverage_route():
    """Serve the interactive Leaflet coverage map for the current task."""
    task_id = STATE.get("task_id") or "sf_pac_heights_medium_01"
    html_path = REPO_ROOT / "data" / "maps" / f"{task_id}_interactive.html"
    if not html_path.exists():
        return (f"<p>No interactive map for {task_id}. Run "
                f"<code>python scripts/interactive_map.py {task_id}</code> first.</p>"), 404
    return html_path.read_text()


@app.route("/debug_set_pose", methods=["POST"])
def debug_set_pose_route():
    """Debug-only: teleport to a pano_id and/or override world_look. Bypasses
    the normal navigation rules — no cone check, no step count. Used by the
    minimap-click and minimap-drag debug features.
    Payload keys (all optional):
      - pano_id:       teleport to that pano (preserves world_look)
      - world_look_deg: rotate camera so it faces this world bearing
    """
    if STATE["sim"] is None:
        return jsonify({"error": "no sim"}), 400
    sim = STATE["sim"]
    payload = request.get_json(force=True) or {}
    pid = payload.get("pano_id")
    if pid and sim._graph is not None and pid in sim._graph:
        old_node = sim._graph.get(sim.current_pano_id)
        new_node = sim._graph.get(pid)
        world_look = (old_node.compass_angle + sim.yaw_deg) % 360.0
        sim.current_pano_id = pid
        sim.yaw_deg = (world_look - new_node.compass_angle) % 360.0
    if "world_look_deg" in payload:
        world_look = float(payload["world_look_deg"]) % 360.0
        if sim._graph is not None and sim.current_pano_id in sim._graph:
            n = sim._graph.get(sim.current_pano_id)
            sim.yaw_deg = (world_look - n.compass_angle) % 360.0
    # Label depending on which fields were set
    label_parts = []
    if pid: label_parts.append(f"teleport→{pid[:10]}")
    if "world_look_deg" in payload: label_parts.append(f"set_yaw={payload['world_look_deg']:.0f}°")
    _record_action(sim, "debug:" + ",".join(label_parts) if label_parts else "debug:noop")
    return jsonify(_response(sim))


_ROLLOUT_DIRS = [
    REPO_ROOT / "data" / "rollouts",
    REPO_ROOT / "data" / "rollouts_compass",       # compass + map-self runs
    REPO_ROOT / "data" / "rollouts_compass_v2",
]


@app.route("/rollouts", methods=["GET"])
def rollouts_route():
    """List saved benchmark rollouts across all known dirs (newest first).
    `filename` is "<dir-name>/<file>" so the loader knows where to find it."""
    out = []
    for rdir in _ROLLOUT_DIRS:
        if not rdir.exists():
            continue
        for f in rdir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                out.append({
                    "filename": f"{rdir.name}/{f.name}",
                    "task_id": d.get("task_id"),
                    "model": d.get("model"),
                    "n_turns": d.get("n_turns_taken"),
                    "reward": d.get("guess_reward"),
                    "guess_error_m": d.get("guess_error_m"),
                    "started_at": d.get("started_at"),
                    "_mtime": f.stat().st_mtime,
                })
            except Exception:
                continue
    out.sort(key=lambda r: r.pop("_mtime"), reverse=True)
    return jsonify({"rollouts": out})


@app.route("/load_rollout", methods=["POST"])
def load_rollout_route():
    """Load a saved rollout into the sim's history. `filename` is "<dir>/<file>"
    (e.g. "rollouts_compass_v2/foo.json")."""
    if STATE["sim"] is None:
        return jsonify({"error": "no sim"}), 400
    payload = request.get_json(force=True) or {}
    fname = payload.get("filename", "")
    # Resolve "<dir>/<file>" against the known rollout dirs.
    fpath = None
    if "/" in fname:
        dirname, base = fname.split("/", 1)
        for rdir in _ROLLOUT_DIRS:
            if rdir.name == dirname and (rdir / base).exists():
                fpath = rdir / base; break
    else:   # legacy: bare filename → look in each dir
        for rdir in _ROLLOUT_DIRS:
            if (rdir / fname).exists(): fpath = rdir / fname; break
    if fpath is None:
        return jsonify({"error": f"rollout not found: {fname}"}), 404
    rollout = json.loads(fpath.read_text())

    # If rollout is for a different task, swap the sim to that task first
    if rollout.get("task_id") and rollout["task_id"] != STATE.get("task_id"):
        tasks = {t.task_id: t for t in load_tasks(_tasks_file())}
        if rollout["task_id"] in tasks:
            sim = _new_sim(tasks[rollout["task_id"]])
            STATE["sim"] = sim
            STATE["task_id"] = rollout["task_id"]

    sim = STATE["sim"]
    STATE["history"] = list(rollout.get("snapshots") or [])
    STATE["actions"] = list(rollout.get("actions") or [])
    STATE["history_idx"] = 0  # start at initial state of the rollout
    if STATE["history"]:
        _restore(sim, STATE["history"][0])
    return jsonify({
        **_response(sim),
        "rollout_loaded": {
            "model": rollout.get("model"),
            "n_turns": len(STATE["history"]) - 1,
            "reward": rollout.get("guess_reward"),
            "guess_error_m": rollout.get("guess_error_m"),
        },
    })


@app.route("/history_seek", methods=["POST"])
def history_seek_route():
    """Jump to a snapshot in the action history. body: {"idx": int} where idx
    is the snapshot index (0 = initial state, len-1 = latest). Restores the
    sim to that state without executing actions in between — just sets state."""
    if STATE["sim"] is None:
        return jsonify({"error": "no sim"}), 400
    sim = STATE["sim"]
    hist = STATE["history"]
    if not hist:
        return jsonify({"error": "no history"}), 400
    payload = request.get_json(force=True) or {}
    idx = int(payload.get("idx", 0))
    idx = max(0, min(len(hist) - 1, idx))
    _restore(sim, hist[idx])
    STATE["history_idx"] = idx
    out = _response(sim)
    out["history_idx"] = idx  # explicit (recompute since _response reads STATE)
    return jsonify(out)


@app.route("/history", methods=["GET"])
def history_route():
    """Return the action labels for the current history (for slider tooltips)."""
    return jsonify({
        "actions": STATE.get("actions", []),
        "len": len(STATE.get("history", [])),
        "idx": STATE.get("history_idx", -1),
    })


@app.route("/graph", methods=["GET"])
def graph_route():
    """Lightweight graph dump for the debug minimap. Returns nodes with the
    fields the client needs to draw the road network + current-pose marker."""
    task_id = request.args.get("task_id", STATE.get("task_id"))
    tasks = {t.task_id: t for t in load_tasks(_tasks_file())}
    if task_id not in tasks:
        return jsonify({"error": f"unknown task_id {task_id}"}), 400
    task = tasks[task_id]
    graph_path = REPO_ROOT / task.world_graph_path
    nodes = []
    for line in graph_path.open():
        if not line.strip(): continue
        row = json.loads(line)
        nodes.append({
            "pano_id": row["pano_id"],
            "lat": row["lat"], "lng": row["lng"],
            "image_lat": row.get("image_lat", 0.0),
            "image_lng": row.get("image_lng", 0.0),
            "compass_angle": row.get("compass_angle", 0.0),
            "neighbors": row.get("neighbors", []),
        })
    return jsonify({
        "task_id": task_id,
        "bbox": task.info.get("bbox", []) if task.info else [],
        "start_lat": task.start_lat, "start_lng": task.start_lng,
        "goal_lat": task.goal_lat, "goal_lng": task.goal_lng,
        "nodes": nodes,
    })


@app.route("/tasks", methods=["GET"])
def tasks_route():
    tasks = load_tasks(_tasks_file())
    return jsonify({"tasks": [
        {
            "task_id": t.task_id,
            "city": t.city,
            "optimal_steps": t.optimal_steps,
            "optimal_distance_m": t.optimal_distance_m,
            "difficulty": t.info.get("difficulty", "") if t.info else "",
        }
        for t in tasks if not t.task_id.startswith("synthetic")
    ]})


@app.route("/init", methods=["GET"])
def init_route():
    task_id = request.args.get("task_id") or STATE.get("task_id") or _CONFIG.get("default_task")
    tasks = {t.task_id: t for t in load_tasks(_tasks_file())}
    if task_id not in tasks:
        # Fall back to the first non-synthetic task. Keeps the preview working
        # when the launcher cached an old --task-id arg pointing at a now-removed task.
        fallback = next((tid for tid in tasks if not tid.startswith("synthetic")), None)
        if fallback is None:
            return jsonify({"error": f"unknown task_id {task_id} and no fallback"}), 400
        print(f"[init] requested task_id={task_id!r} not found, falling back to {fallback}",
              file=sys.stderr, flush=True)
        task_id = fallback
    task = tasks[task_id]
    sim = _new_sim(task)
    STATE["sim"] = sim
    STATE["task_id"] = task_id
    STATE["_t_start"] = _time.time()
    STATE["_started_at"] = datetime.now(timezone.utc).isoformat()
    STATE["_saved_for"] = None
    _reset_history(sim)
    out = _response(sim)
    out["task"] = {
        "task_id": task.task_id,
        "city": task.city,
        "optimal_steps": task.optimal_steps,
        "initial_distance_m": task.initial_distance_m,
        "goal_radius_m": task.goal_radius_m,
    }
    return jsonify(out)


def _save_session(reason: str = "manual") -> Path | None:
    """Write the current human session (history + actions + metadata) to disk
    in the same JSON shape that play.py's /load_rollout consumes, so saved
    human runs are replayable in the same UI."""
    sim = STATE.get("sim")
    save_dir = STATE.get("save_dir")
    if sim is None or save_dir is None:
        return None
    task = sim.task
    from core.tasks import _haversine_m
    err_m = (
        _haversine_m(sim.guess_lat, sim.guess_lng,
                     task.goal_lat, task.goal_lng)
        if sim.guess_submitted else None
    )
    payload = {
        "task_id": task.task_id,
        "model": "human",
        "provider": "local",
        "started_at": STATE.get("_started_at"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "saved_reason": reason,
        "wall_time_s": _time.time() - STATE.get("_t_start", _time.time()),
        "n_turns_taken": len(STATE.get("actions", [])),
        "guess_submitted": bool(sim.guess_submitted),
        "guess_lat": float(sim.guess_lat),
        "guess_lng": float(sim.guess_lng),
        "guess_error_m": (float(err_m) if err_m is not None else None),
        "guess_reward": None,   # scored post-hoc by the verifier path-progress logic
        "optimal_steps": task.optimal_steps,
        "initial_distance_m": getattr(task, "initial_distance_m",
                                      getattr(task, "optimal_distance_m", None)),
        "snapshots": STATE.get("history", []),
        "actions": STATE.get("actions", []),
    }
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    fn = save_dir / f"{task.task_id}__human__{stamp}.json"
    fn.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[save] {fn.name} ({reason})", file=sys.stderr, flush=True)
    return fn


def _maybe_autosave(sim: WorldSim) -> None:
    """Auto-save once per task once submit_guess fires. Guarded so a user
    can keep poking after submit_guess without rewriting the saved file."""
    if not STATE.get("save_dir") or not sim.guess_submitted:
        return
    if STATE.get("_saved_for") == STATE.get("task_id"):
        return
    _save_session(reason="submit_guess")
    STATE["_saved_for"] = STATE.get("task_id")


@app.route("/save_session", methods=["POST"])
def save_session_route():
    """Manual save. Returns the saved file path. No-op if --save-dir wasn't set."""
    if STATE.get("save_dir") is None:
        return jsonify({"error": "no --save-dir configured at launch"}), 400
    fp = _save_session(reason="manual")
    if fp is None:
        return jsonify({"error": "save failed (no sim?)"}), 500
    return jsonify({"ok": True, "path": str(fp)})


@app.route("/completed_tasks", methods=["GET"])
def completed_tasks_route():
    """List task_ids that already have at least one saved session in --save-dir.
    Lets the UI dim tasks the user has already done."""
    save_dir = STATE.get("save_dir")
    if save_dir is None:
        return jsonify({"completed": []})
    done = set()
    for fp in Path(save_dir).glob("*__human__*.json"):
        done.add(fp.name.split("__human__")[0])
    return jsonify({"completed": sorted(done)})


def _dispatch_one(sim: WorldSim, tool: str, args: dict) -> None:
    if tool == "open_map":
        sim.open_map()
    elif tool == "close_map":
        sim.close_map()
    elif tool == "mouse_down":
        sim.mouse_down()
    elif tool == "mouse_up":
        sim.mouse_up()
    elif tool == "move_cursor":
        sim.move_cursor(
            direction_deg=float(args.get("direction_deg", 0.0)),
            distance_px=int(args.get("distance_px", 0)),
        )
    elif tool == "scroll_wheel":
        sim.scroll_wheel(delta_y=int(args.get("delta_y", 0)))
    elif tool == "submit_guess":
        sim.submit_guess()
    else:
        raise ValueError(f"unknown tool {tool}")


@app.route("/action", methods=["POST"])
def action_route():
    if STATE["sim"] is None:
        return jsonify({"error": "no sim — call /init first"}), 400
    payload = request.get_json(force=True)
    sim = STATE["sim"]
    try:
        _dispatch_one(sim, payload.get("tool"), payload.get("args", {}) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    _record_action(sim, payload.get("tool", "?"))
    _maybe_autosave(sim)
    return jsonify(_response(sim))


def _log_click_diagnostic(sim: WorldSim, actions: list[dict]) -> None:
    """If this batch is a click (has mouse_up), dump a one-line snapshot of what the
    algorithm saw so we can see WHY a click did / didn't move the user. Goes to stderr."""
    if not any(a.get("tool") == "mouse_up" for a in actions):
        return
    if sim._graph is None:
        return
    from core.world import bearing_deg, angular_diff_deg
    import math as _math
    node = sim._graph.get(sim.current_pano_id)
    world_look = (node.compass_angle + sim.yaw_deg) % 360.0
    nbrs = []
    for nid in node.neighbors:
        nb = sim._graph.get(nid)
        b = bearing_deg(node.lat, node.lng, nb.lat, nb.lng)
        rel = (b - world_look + 540) % 360 - 180
        d = _math.sqrt((nb.lat-node.lat)**2 + (nb.lng-node.lng)**2) * 111000
        nbrs.append(f"{nid[:10]}|rel{rel:+.0f}|d{d:.0f}m")
    print(
        f"[click] at={sim.current_pano_id[:10]} cursor=({sim.cursor_x},{sim.cursor_y}) "
        f"yaw={sim.yaw_deg:.0f} look={world_look:.0f} steps={sim.steps_taken} "
        f"nbrs=[{', '.join(nbrs)}]",
        file=sys.stderr, flush=True,
    )


@app.route("/action_batch", methods=["POST"])
def action_batch_route():
    """Run a sequence of tool calls atomically and return one response. Used by the
    play UI to bundle e.g. [move_cursor, mouse_down, mouse_up] for a click — avoids
    N×roundtrip latency."""
    if STATE["sim"] is None:
        return jsonify({"error": "no sim — call /init first"}), 400
    payload = request.get_json(force=True)
    actions = payload.get("actions", [])
    sim = STATE["sim"]
    before_pid = sim.current_pano_id
    before_steps = sim.steps_taken
    try:
        for a in actions:
            _dispatch_one(sim, a.get("tool"), a.get("args", {}) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    # If this batch was a click (mouse_up present), log what happened so we can
    # diagnose stuck/wrong-direction cases.
    if any(a.get("tool") == "mouse_up" for a in actions):
        hops = sim.steps_taken - before_steps
        moved = sim.current_pano_id != before_pid
        print(
            f"[click] cursor=({sim.cursor_x},{sim.cursor_y}) yaw={sim.yaw_deg:.0f} "
            f"before={before_pid[:10]} after={sim.current_pano_id[:10]} "
            f"hops={hops} moved={moved}",
            file=sys.stderr, flush=True,
        )
        if not moved:
            # Dump neighbor analysis so we can see why nothing was picked
            _log_click_diagnostic(sim, actions)
    # Record a single history entry per batch (a click batch = one logical
    # user action, even if internally [move_cursor, mouse_down, mouse_up])
    label = " → ".join(a.get("tool", "?") for a in actions)
    _record_action(sim, label)
    _maybe_autosave(sim)
    return jsonify(_response(sim))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", default="sf_pac_heights_medium_01")
    p.add_argument("--tasks-path", default=None,
                   help="tasks jsonl to load (default data/tasks.jsonl)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    # Difficulty toggles (default off = current/hardest setup)
    p.add_argument("--compass", action="store_true", help="heading compass in pano view")
    p.add_argument("--map-self", action="store_true", help="show current location + heading on the map")
    p.add_argument("--save-dir", default=None,
                   help="auto-save human sessions here on submit_guess (one JSON per task)")
    args = p.parse_args()
    # Config is global (not session): write straight to _CONFIG so this is safe
    # to call at startup (no request context).
    _CONFIG["default_task"] = args.task_id
    _CONFIG["tasks_path"] = Path(args.tasks_path) if args.tasks_path else None
    _CONFIG["opts"] = {"show_compass": args.compass, "map_show_self": args.map_self}
    _CONFIG["save_dir"] = Path(args.save_dir) if args.save_dir else None
    if _CONFIG["save_dir"]:
        _CONFIG["save_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[save] human sessions will auto-save to {_CONFIG['save_dir']}", file=sys.stderr)
    on = [n for n, v in (("compass", args.compass), ("map-self", args.map_self)) if v] or ["none (hardest)"]
    print(f"Open http://{args.host}:{args.port} in a browser. Task: {args.task_id}  "
          f"| toggles: {', '.join(on)}", file=sys.stderr)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
