"""!
@package results_viewer
@brief One page to explore every per-node result of a run, in a browser: outputs/<run_id>/analysis/results-viewer.html.

The nodes over the street centrelines and the buildings (footprints extruded to their roof height), on a map you
pan, zoom, tilt and turn. Colour the nodes by any result, raise them as columns by any other, switch the buildings
off or flat, hover a node for its main scores and click it for all of them. Every per-node column the run has:
the index and pillars, every rated term, the outcome, Space Syntax, land use, seating records, canyon geometry,
the segmentation's view indices, the street type and lingering (paper_figures.node_frame()).

A single self-contained file (the data inline; deck.gl from a CDN, pinned): no imagery and no source segments, so
it can be shared like the run's tables. Run by pipeline step make_results_viewer.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.paper_figures import Basemap, node_frame
from common.map_style import rings

DECK_GL = "https://unpkg.com/deck.gl@8.9.35/dist.min.js"
# per-node columns that are bookkeeping, not results
SKIP = {"seq_fwd", "seq_rev", "chain_pos_m", "heading_fwd_deg", "heading_rev_deg", "street_axis_deg", "snap_m",
        "facade_sides_hit", "facade_sides_open", "facade_empty_bearing", "image_raw", "form_raw", "stay_raw", "lat", "lon"}


def variables(nf, cfg):
    """!
    @brief The results a node can be coloured or raised by: {column: label}, the index first.
    """
    first = {"M": "M, the Street Interface Matrix score", **cfg["metric"]["names"],
             "volume": "pedestrian volume (people per hour)", "dwell_potential": "lingering (footfall x M / 7)",
             "street_type": "street type"}
    num = [c for c in nf.select_dtypes("number") if c not in SKIP and nf[c].notna().any() and nf[c].nunique() > 1]
    out = {c: label for c, label in first.items() if c in num}
    out |= {c: c.replace("_", " ") for c in num if c not in out}
    return out


def build(cfg, paths):
    """!
    @brief Write the page.
    @return Its path.
    """
    nf = node_frame(cfg, paths)
    base = Basemap(cfg, paths, nf)
    V = variables(nf, cfg)
    ll = nf.to_crs(4326)
    vals = ll[list(V)].astype(float).round(4)
    nodes = [{"id": r.node_id, "street": r.osm_name if isinstance(getattr(r, "osm_name", None), str) else r.street,
              "pos": [round(r.geometry.x, 6), round(r.geometry.y, 6)],
              "v": [None if np.isnan(x) else float(x) for x in vals.loc[i]]}
             for i, r in ll.iterrows()]
    streets = [[np.round(c, 6).tolist() for c in g.coords]
               for geom in base.streets.to_crs(4326).geometry for g in getattr(geom, "geoms", [geom])]
    buildings = []
    if base.buildings is not None:
        x0, x1, y0, y1 = base.extent
        b = base.buildings.cx[x0 - 200:x1 + 200, y0 - 200:y1 + 200].to_crs(4326)
        buildings = [{"p": np.round(r, 6).tolist(), "h": round(float(h), 1)}
                     for g, h in zip(b.geometry, b.height_m) for r in rings(g)]
    data = {"title": f"{cfg['name']}: Street Interface Matrix results ({cfg['run_id']})", "vars": list(V),
            "labels": list(V.values()), "nodes": nodes, "streets": streets, "buildings": buildings,
            "centre": [float(ll.geometry.x.mean()), float(ll.geometry.y.mean())]}
    out = paths.results_viewer_html
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DECK__", DECK_GL).replace("__TITLE__", data["title"])
                   .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"{len(nodes)} nodes x {len(V)} results, {len(buildings):,} buildings -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB)")
    return out


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__DECK__"></script>
<style>
  html, body { margin: 0; height: 100%; background: #f4f3f0; font: 13px/1.4 system-ui, sans-serif; color: #1f2a37; }
  #panel { position: absolute; top: 12px; left: 12px; z-index: 2; width: 300px; max-height: calc(100% - 24px);
           overflow: auto; background: rgba(255,255,255,.95); padding: 12px 14px; border-radius: 8px;
           box-shadow: 0 1px 6px rgba(0,0,0,.15); box-sizing: border-box; }
  h1 { font-size: 14px; margin: 0 0 8px; }
  label { display: block; margin: 8px 0 2px; font-weight: 600; font-size: 12px; }
  select { width: 100%; font: inherit; }
  .row { display: flex; gap: 12px; margin-top: 8px; }
  .row label { display: inline; font-weight: 400; margin: 0; }
  #legend { margin-top: 10px; }
  #ramp { height: 10px; border-radius: 2px; }
  #ends { display: flex; justify-content: space-between; font-size: 11px; color: #52514e; }
  #detail { margin-top: 12px; font-size: 12px; border-top: 1px solid #ddd; padding-top: 8px; }
  #detail table { border-collapse: collapse; width: 100%; }
  #detail td { padding: 1px 0; vertical-align: top; }
  #detail td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  .hint { color: #52514e; font-size: 11px; }
  @media (max-width: 600px) { #panel { width: calc(100% - 24px); max-height: 45%; } }
</style></head>
<body>
<div id="panel">
  <h1>__TITLE__</h1>
  <div class="hint">Drag to pan; right-drag or Ctrl-drag to tilt and turn; scroll to zoom. Hover a node; click it for
  every value.</div>
  <label for="colour">Colour nodes by</label><select id="colour"></select>
  <label for="height">Raise nodes as columns by</label><select id="height"></select>
  <label for="bld">Buildings</label>
  <select id="bld"><option value="3d">3D, roof height</option><option value="flat">flat footprints</option>
    <option value="off">off</option></select>
  <div id="legend"><div id="ramp"></div><div id="ends"><span id="lo"></span><span id="hi"></span></div></div>
  <div id="detail"><span class="hint">Click a node to list all of its results.</span></div>
</div>
<script>
const D = __DATA__;
const RAMP = [[240, 222, 214], [226, 217, 184], [169, 196, 160], [78, 138, 114], [28, 90, 73]];
const TYPES = [[42, 120, 214], [235, 104, 52], [27, 175, 122], [163, 162, 156], [120, 90, 160]];
const $ = id => document.getElementById(id);
D.vars.forEach((v, i) => { $('colour').add(new Option(D.labels[i], i)); });
$('height').add(new Option('none (flat dots)', -1));
D.vars.forEach((v, i) => { $('height').add(new Option(D.labels[i], i)); });
if (D.vars.includes('volume')) $('height').value = D.vars.indexOf('volume');   // start: columns of footfall
const pick = (n, i) => n.v[i];
function range(i) {
  const xs = D.nodes.map(n => pick(n, i)).filter(x => x !== null).sort((a, b) => a - b);
  return [xs[Math.floor(xs.length * 0.02)], xs[Math.ceil(xs.length * 0.98) - 1]];   // 2nd-98th percentile
}
function ramp(t) {
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const k = Math.min(Math.floor(t), RAMP.length - 2), f = t - k;
  return RAMP[k].map((c, j) => Math.round(c + (RAMP[k + 1][j] - c) * f));
}
const fmt = x => x === null ? '-' : Math.abs(x) >= 100 ? x.toLocaleString('en-US', {maximumFractionDigits: 0})
  : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(3);
function layers() {
  const ci = +$('colour').value, hi = +$('height').value, bld = $('bld').value;
  const typed = D.vars[ci] === 'street_type', [lo, up] = range(ci);
  const colour = n => { const x = pick(n, ci); if (x === null) return [200, 200, 200];
    return typed ? TYPES[(x - 1) % TYPES.length] : ramp((x - lo) / (up - lo || 1)); };
  $('ramp').style.background = typed ? 'none' : `linear-gradient(90deg, ${RAMP.map(c => `rgb(${c})`).join(',')})`;
  $('lo').textContent = typed ? 'types: blue 1, orange 2, green 3' : fmt(lo);
  $('hi').textContent = typed ? '' : fmt(up);
  const out = [new deck.PathLayer({id: 'streets', data: D.streets, getPath: d => d, getColor: [170, 167, 160],
    widthUnits: 'pixels', getWidth: 1})];
  if (bld !== 'off') out.push(new deck.PolygonLayer({id: 'buildings', data: D.buildings, getPolygon: d => d.p,
    extruded: bld === '3d', getElevation: d => d.h, getFillColor: [214, 211, 204], getLineColor: [190, 186, 178],
    wireframe: bld === '3d', opacity: bld === '3d' ? 0.4 : 0.9}));
  if (hi < 0) out.push(new deck.ScatterplotLayer({id: 'nodes', data: D.nodes, getPosition: d => d.pos,
    getFillColor: colour, radiusUnits: 'meters', getRadius: 5, pickable: true, updateTriggers: {getFillColor: [ci]}}));
  else { const [hlo, hup] = range(hi);
    out.push(new deck.ColumnLayer({id: 'columns', data: D.nodes, getPosition: d => d.pos, diskResolution: 6,
      radius: 4, extruded: true, pickable: true, getFillColor: colour,
      getElevation: d => { const x = pick(d, hi); return x === null ? 0 : 5 + 250 * Math.max(0, (x - hlo) / (hup - hlo || 1)); },
      updateTriggers: {getFillColor: [ci], getElevation: [hi]}})); }
  return out;
}
function detail(n) {
  $('detail').innerHTML = `<b>${n.id}</b> (${n.street})<table>` +
    D.labels.map((l, i) => `<tr><td>${l}</td><td>${fmt(pick(n, i))}</td></tr>`).join('') + '</table>';
}
const app = new deck.DeckGL({
  initialViewState: {longitude: D.centre[0], latitude: D.centre[1], zoom: 15.4, pitch: 50, bearing: -29},
  controller: true, layers: layers(),
  getTooltip: ({object}) => object && object.id && {html: `<b>${object.id}</b><br>` +
    [0, 1, 2, 3].filter(i => i < D.labels.length).map(i => `${D.labels[i]}: ${fmt(pick(object, i))}`).join('<br>')},
  onClick: ({object}) => object && object.id && detail(object)
});
for (const id of ['colour', 'height', 'bld']) $(id).onchange = () => app.setProps({layers: layers()});
</script></body></html>
"""
