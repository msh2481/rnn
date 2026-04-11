#!/usr/bin/env python3
"""
Interactive data explorer for RNN training sequences.
Usage: python explore.py [--frac-order ALPHA] [--port PORT]
"""

import argparse
import json
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor


def gl_weights(alpha, n):
    """Grünwald-Letnikov fractional derivative weights."""
    w = np.zeros(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1 - alpha) / k
    return w


def fractional_derivative(data_3d, alpha):
    """Apply GL fractional derivative along axis=1 (time)."""
    if alpha == 0:
        return data_3d.copy()
    n_steps = data_3d.shape[1]
    w = gl_weights(alpha, n_steps)
    return lfilter(w, [1.0], data_3d, axis=1)


def compute_adaptive_alpha(data_3d, alpha_grid, ema_span=5):
    """Estimate best local fractional-diff alpha at each timestep.

    For each alpha, compute one-step prediction error on the raw data,
    then EMA-smooth the squared errors and pick the best alpha per step.

    data_3d: (n_seqs, steps, 32) — raw (not differenced) data
    alpha_grid: 1-D array of alpha candidates
    ema_span: EMA span for smoothing squared errors

    Returns:
        best_alpha: (n_seqs, steps) — best alpha index at each step
        alpha_grid: the grid used (for labeling)
    """
    n_seqs, steps, n_feat = data_3d.shape
    n_alpha = len(alpha_grid)
    decay = 2.0 / (ema_span + 1)  # EMA decay factor

    # For each alpha, build the prediction filter.
    # If D^alpha x[t] ≈ 0 is a good model, then:
    #   sum_{k=0}^{t} w[k] * x[t-k] ≈ 0
    #   w[0]*x[t] + sum_{k=1} w[k]*x[t-k] ≈ 0
    #   x[t] ≈ -sum_{k=1} w[k]*x[t-k]   (since w[0]=1)
    # So the prediction of x[t] from past is: -sum_{k=1}^{K} w[k]*x[t-k]
    # And the prediction error is: x[t] - x̂[t] = D^alpha x[t]
    # i.e. the fractional derivative IS the prediction error.

    # Compute squared prediction error for each alpha: just |D^alpha x[t]|^2
    # averaged over features.
    # Shape per alpha: (n_seqs, steps)
    print(f"Computing adaptive alpha ({n_alpha} candidates)...")

    # pred_err_sq[a, seq, t] = mean over features of (D^alpha x[t])^2
    pred_err_sq = np.zeros((n_alpha, n_seqs, steps))
    for i, alpha in enumerate(alpha_grid):
        fd = fractional_derivative(data_3d, alpha)  # (n_seqs, steps, 32)
        pred_err_sq[i] = (fd ** 2).mean(axis=2)     # (n_seqs, steps)

    # EMA smooth along time axis
    ema = np.zeros_like(pred_err_sq)
    ema[:, :, 0] = pred_err_sq[:, :, 0]
    for t in range(1, steps):
        ema[:, :, t] = decay * pred_err_sq[:, :, t] + (1 - decay) * ema[:, :, t - 1]

    # Best alpha at each (seq, step): argmin over alpha grid
    best_alpha_idx = ema.argmin(axis=0)  # (n_seqs, steps)

    return best_alpha_idx, alpha_grid


def main():
    parser = argparse.ArgumentParser(description="Interactive RNN data explorer")
    parser.add_argument("--frac-order", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=str, default="datasets/train.parquet")
    parser.add_argument("--tsne-sub", type=int, default=5000)
    parser.add_argument("--alpha-grid", type=int, default=11,
                        help="Number of alpha candidates (0 to 1 inclusive)")
    parser.add_argument("--ema-span", type=int, default=5,
                        help="EMA span for smoothing prediction errors")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_parquet(args.data)
    feature_cols = [str(i) for i in range(32)]
    n_seqs = df["seq_ix"].nunique()
    steps = 1000
    seq_ids = sorted(df["seq_ix"].unique())

    data_3d = df[feature_cols].values.reshape(n_seqs, steps, 32)

    # Adaptive alpha estimation (on raw data, before any global frac-diff)
    alpha_grid = np.linspace(0, 1, args.alpha_grid)
    best_alpha_idx, _ = compute_adaptive_alpha(data_3d, alpha_grid, args.ema_span)

    if args.frac_order > 0:
        print(f"Fractional derivative (order={args.frac_order})...")
        data_3d = fractional_derivative(data_3d, args.frac_order)

    data_flat = data_3d.reshape(-1, 32)
    mu, sigma = data_flat.mean(0), data_flat.std(0) + 1e-10
    data_flat = (data_flat - mu) / sigma

    rng = np.random.RandomState(42)
    n_sub = min(args.tsne_sub, len(data_flat))
    sub_idx = rng.choice(len(data_flat), n_sub, replace=False)
    sub = data_flat[sub_idx]

    # Random projection
    print("Random projection...")
    R = rng.randn(32, 2)
    R /= np.linalg.norm(R, axis=0, keepdims=True)
    proj_rand = data_flat @ R

    # PCA
    print("PCA...")
    pca = PCA(n_components=2, random_state=42).fit(sub)
    proj_pca = pca.transform(data_flat)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # t-SNE on subsample, then MLP for out-of-sample
    print(f"t-SNE on {n_sub} points...")
    tsne_sub = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(sub)
    print("MLP for out-of-sample t-SNE projection...")
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=2000,
                       random_state=42, early_stopping=True, validation_fraction=0.1)
    mlp.fit(sub, tsne_sub)
    proj_tsne = mlp.predict(data_flat)

    # Normalize to [0.05, 0.95]
    def norm(p):
        mn, mx = p.min(0), p.max(0)
        return (p - mn) / (mx - mn + 1e-10) * 0.9 + 0.05

    proj_rand, proj_pca, proj_tsne = norm(proj_rand), norm(proj_pca), norm(proj_tsne)

    # Write output
    out_dir = "_explorer"
    os.makedirs(out_dir, exist_ok=True)

    print("Writing data...")
    viz = {
        "nSeqs": int(n_seqs),
        "steps": int(steps),
        "ctxEnd": 100,
        "fracOrder": args.frac_order,
        "seqIds": [int(s) for s in seq_ids],
        "alphaGrid": np.round(alpha_grid, 3).tolist(),
        "bestAlpha": best_alpha_idx.flatten().tolist(),  # (n_seqs * steps) ints
        "rand": np.round(proj_rand, 4).flatten().tolist(),
        "pca": np.round(proj_pca, 4).flatten().tolist(),
        "tsne": np.round(proj_tsne, 4).flatten().tolist(),
    }
    with open(os.path.join(out_dir, "data.json"), "w") as f:
        json.dump(viz, f)

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(HTML)

    sz = os.path.getsize(os.path.join(out_dir, "data.json")) / 1024 / 1024
    print(f"data.json: {sz:.1f} MB")

    # Serve
    os.chdir(out_dir)
    handler = SimpleHTTPRequestHandler
    handler.log_message = lambda *a, **k: None
    server = HTTPServer(("localhost", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"\n→ {url}   (Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Data Explorer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#111; color:#ccc; font-family: 'SF Mono',Menlo,Consolas,monospace;
       font-size:13px; overflow:hidden; height:100vh; display:flex; flex-direction:column; }

#controls {
  padding: 10px 16px; display:flex; align-items:center; gap:18px;
  background:#1a1a2e; border-bottom:1px solid #333; flex-shrink:0; flex-wrap:wrap;
}
#controls label { color:#888; }
#controls select, #controls input[type=range] { background:#222; color:#eee; border:1px solid #444;
  border-radius:4px; padding:2px 6px; }
#controls select { width: 80px; }
#controls input[type=range] { width:100px; vertical-align:middle; }
.btn { background:#333; color:#ddd; border:1px solid #555; border-radius:4px;
       padding:4px 12px; cursor:pointer; font-family:inherit; font-size:13px; }
.btn:hover { background:#444; }
.btn.active { background:#4a6; color:#fff; border-color:#4a6; }
#step-display { min-width:120px; text-align:center; font-variant-numeric:tabular-nums; }
#zone { padding:2px 8px; border-radius:3px; font-weight:bold; }
.zone-ctx { background:#2a4a8a; color:#8af; }
.zone-pred { background:#8a5a00; color:#fa0; }

#canvases { flex:1; display:flex; gap:0; min-height:0; }
.panel { flex:1; display:flex; flex-direction:column; border-right:1px solid #222; min-width:0; }
.panel:last-child { border-right:none; }
.panel-label { text-align:center; padding:6px 0 2px; color:#777; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.panel canvas { flex:1; width:100%; }

#alpha-row { height:140px; flex-shrink:0; display:flex; flex-direction:column; border-top:1px solid #333; }
#alpha-row .panel-label { padding:4px 0 0; }
#alpha-row canvas { flex:1; width:100%; }

#footer { padding:4px 16px; color:#555; font-size:11px; background:#1a1a2e; border-top:1px solid #222; }
</style>
</head>
<body>

<div id="controls">
  <label>Seq <select id="seq-sel"></select></label>
  <button class="btn" id="prev" title="← or A">◀</button>
  <span id="step-display">0 / 999</span>
  <button class="btn" id="next" title="→ or D">▶</button>
  <button class="btn" id="play">Play</button>
  <label>Trail <input type="range" id="trail" min="10" max="999" value="200"></label>
  <label>Speed <input type="range" id="speed" min="1" max="100" value="50"></label>
  <span id="zone" class="zone-ctx">CONTEXT</span>
  <span id="frac-info"></span>
</div>

<div id="canvases">
  <div class="panel"><div class="panel-label">Random Projection</div><canvas id="c-rand"></canvas></div>
  <div class="panel"><div class="panel-label">PCA</div><canvas id="c-pca"></canvas></div>
  <div class="panel"><div class="panel-label">t-SNE (MLP approx)</div><canvas id="c-tsne"></canvas></div>
</div>

<div id="alpha-row">
  <div class="panel-label">Adaptive α  <span id="alpha-val" style="color:#0f0">—</span></div>
  <canvas id="c-alpha"></canvas>
</div>

<div id="footer">Keys: ←/→ step ∙ Shift±10 ∙ Ctrl±100 ∙ Space play/pause ∙ Home/End jump</div>

<script>
let D = null;
let seq = 0, step = 0, playing = false, timer = null;

const $ = id => document.getElementById(id);
const canvases = [
  { id: 'rand', el: $('c-rand'), ctx: null },
  { id: 'pca',  el: $('c-pca'),  ctx: null },
  { id: 'tsne', el: $('c-tsne'), ctx: null },
];
const alphaCanvas = { el: $('c-alpha'), ctx: null, w: 0, h: 0 };

fetch('data.json').then(r => r.json()).then(d => { D = d; init(); });

function init() {
  // Populate seq selector
  const sel = $('seq-sel');
  D.seqIds.forEach((id, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = id;
    sel.appendChild(o);
  });
  sel.onchange = () => { seq = +sel.value; step = 0; render(); };

  // Frac info
  $('frac-info').textContent = D.fracOrder > 0 ? `frac_d=${D.fracOrder}` : '';

  // Buttons
  $('prev').onclick = () => moveStep(-1);
  $('next').onclick = () => moveStep(1);
  $('play').onclick = togglePlay;

  // Setup canvases
  canvases.forEach(c => { c.ctx = c.el.getContext('2d'); });
  alphaCanvas.ctx = alphaCanvas.el.getContext('2d');
  window.addEventListener('resize', () => { sizeCanvases(); render(); });
  sizeCanvases();

  // Keys
  document.addEventListener('keydown', e => {
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    else if (e.key === 'ArrowRight' || e.key === 'd') moveStep(e.ctrlKey||e.metaKey ? 100 : e.shiftKey ? 10 : 1);
    else if (e.key === 'ArrowLeft'  || e.key === 'a') moveStep(e.ctrlKey||e.metaKey ? -100 : e.shiftKey ? -10 : -1);
    else if (e.key === 'Home') { step = 0; render(); }
    else if (e.key === 'End')  { step = D.steps - 1; render(); }
  });

  render();
}

function sizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  [...canvases, alphaCanvas].forEach(c => {
    const rect = c.el.getBoundingClientRect();
    c.el.width = rect.width * dpr;
    c.el.height = rect.height * dpr;
    c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.w = rect.width;
    c.h = rect.height;
  });
}

function moveStep(delta) {
  step = Math.max(0, Math.min(D.steps - 1, step + delta));
  render();
}

function togglePlay() {
  playing = !playing;
  $('play').textContent = playing ? 'Pause' : 'Play';
  $('play').classList.toggle('active', playing);
  if (playing) startPlay(); else stopPlay();
}

function startPlay() {
  stopPlay();
  const ms = Math.max(5, 205 - $('speed').value * 2);
  timer = setInterval(() => {
    step++;
    if (step >= D.steps) { step = 0; }
    render();
  }, ms);
}

function stopPlay() { if (timer) { clearInterval(timer); timer = null; } }

// Re-read speed when slider changes during play
$('speed').oninput = () => { if (playing) startPlay(); };

function getXY(proj, seqIdx, s) {
  const arr = D[proj];
  const i = (seqIdx * D.steps + s) * 2;
  return [arr[i], arr[i + 1]];
}

function render() {
  const trail = +$('trail').value;

  // Update text
  $('step-display').textContent = `${step} / ${D.steps - 1}`;
  const zone = $('zone');
  if (step < D.ctxEnd) {
    zone.textContent = 'CONTEXT'; zone.className = 'zone-ctx';
  } else {
    zone.textContent = 'PREDICTION'; zone.className = 'zone-pred';
  }

  canvases.forEach(c => drawPanel(c, trail));
  drawAlpha();
}

function getAlpha(seqIdx, s) {
  return D.bestAlpha[seqIdx * D.steps + s];
}

function drawAlpha() {
  const c = alphaCanvas, ctx = c.ctx, w = c.w, h = c.h;
  const nAlpha = D.alphaGrid.length;
  const padL = 40, padR = 10, padT = 8, padB = 8;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  ctx.clearRect(0, 0, w, h);

  // Y-axis maps: alpha index 0 (α=0) at bottom, max at top
  function yForIdx(ai) { return padT + plotH - (ai / (nAlpha - 1)) * plotH; }
  function xForStep(s) { return padL + (s / (D.steps - 1)) * plotW; }

  // Y-axis labels
  ctx.fillStyle = '#555';
  ctx.font = '9px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < nAlpha; i += Math.max(1, Math.floor(nAlpha / 4))) {
    ctx.fillText(D.alphaGrid[i].toFixed(1), padL - 4, yForIdx(i));
  }

  // Context boundary
  const ctxX = xForStep(D.ctxEnd);
  ctx.strokeStyle = '#333';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(ctxX, padT); ctx.lineTo(ctxX, padT + plotH); ctx.stroke();
  ctx.setLineDash([]);

  // Draw alpha time series
  ctx.beginPath();
  for (let s = 0; s < D.steps; s++) {
    const ai = getAlpha(seq, s);
    const x = xForStep(s);
    const y = yForIdx(ai);
    if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#0c6';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Current step cursor
  const curX = xForStep(step);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(curX, padT); ctx.lineTo(curX, padT + plotH); ctx.stroke();

  // Dot at current value
  const curAlphaIdx = getAlpha(seq, step);
  const curAlphaVal = D.alphaGrid[curAlphaIdx];
  ctx.beginPath();
  ctx.arc(curX, yForIdx(curAlphaIdx), 4, 0, 6.2832);
  ctx.fillStyle = '#0f0';
  ctx.fill();

  $('alpha-val').textContent = `α=${curAlphaVal.toFixed(2)}`;
}

function drawPanel(c, trail) {
  const ctx = c.ctx, w = c.w, h = c.h;
  ctx.clearRect(0, 0, w, h);

  const start = Math.max(0, step - trail);
  const tau = trail / 3;

  // Draw connecting line segments with fading
  for (let s = start; s < step; s++) {
    const [x0, y0] = getXY(c.id, seq, s);
    const [x1, y1] = getXY(c.id, seq, s + 1);
    const age = step - s;
    const alpha = Math.max(0.03, Math.exp(-age / tau));
    const inCtx = s < D.ctxEnd;
    ctx.beginPath();
    ctx.moveTo(x0 * w, y0 * h);
    ctx.lineTo(x1 * w, y1 * h);
    ctx.strokeStyle = inCtx
      ? `rgba(100,149,237,${alpha * 0.5})`
      : `rgba(255,165,0,${alpha * 0.5})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Draw points
  for (let s = start; s <= step; s++) {
    const [x, y] = getXY(c.id, seq, s);
    const age = step - s;
    const alpha = Math.max(0.05, Math.exp(-age / tau));
    const inCtx = s < D.ctxEnd;
    const cur = s === step;

    ctx.beginPath();
    ctx.arc(x * w, y * h, cur ? 5 : 2, 0, 6.2832);
    if (cur) {
      ctx.fillStyle = '#fff';
      ctx.shadowColor = inCtx ? '#48f' : '#fa0';
      ctx.shadowBlur = 12;
    } else {
      ctx.fillStyle = inCtx
        ? `rgba(100,149,237,${alpha})`
        : `rgba(255,165,0,${alpha})`;
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
