"""!
@package street_interface_matrix
@brief Compose the Street Interface Matrix per view, in the form of the UN's Human Development Index.

    index_k = (x_k - worst_k) / (best_k - worst_k), clipped to [0, 1]
                  a rating: worst 1, best 7; a record: the config's goalposts, on ln(1 + x) where log: true
    D_d     = 1 + 6 mean[index_k, k in pillar d]             (the dimensions of metric.dimensions)
    M       = (D_1 D_2 D_3)^(1/3)                             the Street Interface Matrix score
The pillars and their indicators are the config's (configs/metric/street_interface_matrix/<design>.yaml).

Steps, in this order:
  0. derived     metric.derived_ratings, if any: a rating made from two others, 1 + 6 a / (a + b)
  1. terms       each rating's expected value over the seven rungs on the [0, 1] term scale by (EV - 1) / 6 (every
                 rated term, for the tables); a term no side of which could be judged takes its street's mean
  2. indicators  index_<name> per indicator of metric.dimensions: a rating, or a record from the node table (step
  measure_city_records's
                 physical attributes and the other per-node measurements); a missing value takes its street's mean
  3. index       each dimension on the strictly positive 1-7 scale (1 + 6 x, the rating scale's own ends), then M
  4. checks      dimension-pca-weights.csv, weight-robustness.csv (how far the dimensions and M depend on the weights),
                 exponent-robustness.csv, dimension-coherence.csv (Cronbach's alpha, with each indicator dropped) and
                 indicator-factor-loadings.csv (do the indicators group into their dimensions?)
  5. tables      all-rated-views-with-inputs.csv (every rated view, with its distribution, node measurements and the
                 segmenter's shares over the view's arc), scores-per-usable-view.csv (usable views),
                 mean-score-per-block-section.csv (mean M per block-length run of a street), column-dictionary.md
  6. run         the usable nodes' ratings with their records, composed, then the checks and the tables
All in outputs/<run_id>/metric/street_interface_matrix/. Run by pipeline step compose_metric.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.data_readers import (FIELDS, TERMS, TermSharesByBearing, arc_shares, dimension_columns, indicator_name,
                                 indicator_ratings, rated_fields)
from common.node_table import node_table

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # columns are added one per term
K = np.arange(1, 8)
ID = ["file", "node_id", "street", "walk", "seq", "cardinal"]
# per-node columns carried into the tables: the sampling nodes (sample_street_nodes), the view indices
# (calculate_segmentation_pixel_shares), the street canyons (measure_street_canyons)
NODE = ["exclude_reason", "typology", "osm_name", "face_id", "H_m", "W_facade", "HW_facade", "HW_effective",
        "HW_source", "SVF_canyon", "GVI", "VEI", "SVF_band"]


def fill_by_street(d, col):
    """!
    @brief A missing value takes its street's mean, else the city's.
    """
    if d[col].isna().any():
        d[col] = d[col].fillna(d.groupby("street")[col].transform("mean")).fillna(d[col].mean())


# ---------------------------------------------------------------- 0. derived ratings
def derived(d, C):
    """!
    @brief Ratings made from two others (metric.derived_ratings, {name: [a, b]}): <name>_ev = 1 + 6 a / (a + b), on
           the rating scale from the two expected values (e.g. the greenery ratio V_nat / (V_nat + V_built)).
    @return d with a <name>_ev column per derived rating.
    """
    for name, (a, b) in (C.get("derived_ratings") or {}).items():
        d[name + "_ev"] = 1 + 6 * d[a + "_ev"] / (d[a + "_ev"] + d[b + "_ev"])
    return d


# ---------------------------------------------------------------- 1. terms
def terms(d):
    """!
    @brief Every rated term's expected value on [0, 1].
    @param d  Ratings with <term>_ev columns (run_vlm, step rate_views_with_vlm).
    @return d with the terms added.
    """
    for field in rated_fields(d):
        col = TERMS.get(field, field)
        d[col] = (d[field + "_ev"].clip(1, 7) - 1) / 6.0
        fill_by_street(d, col)
    return d


# ---------------------------------------------------------------- 2. indicators
def records_needed(C):
    """!
    @brief The node-table columns the indicators read.
    """
    return sorted({i["record"] for items in C["dimensions"].values() for i in items if "record" in i})


def scaled(d, x, lo, hi):
    """!
    @brief x between goalposts lo (worst) and hi (best) on [0, 1]; a missing value takes its street's mean.
    """
    s = ((x.astype(float) - lo) / (hi - lo)).clip(0, 1)
    return s.fillna(s.groupby(d.street).transform("mean")).fillna(s.mean())


def indicators(d, C):
    """!
    @brief Each indicator on [0, 1]: its ratings scaled between rungs 1 and 7 (or rating_goalposts, e.g. [7, 1] to
           count downward) and averaged; its record between the config's goalposts [worst, best], on ln(1 + x) where
           log: true. An indicator with both is half the ratings' mean, half the record.
    @param d  Ratings (<term>_ev) with the records the dimensions name, one row per view.
    @return d with an index_<name> column per indicator.
    """
    for items in C["dimensions"].values():
        for it in items:
            parts = []
            if rs := indicator_ratings(it):
                lo, hi = map(float, it.get("rating_goalposts", [1, 7]))
                parts.append(pd.concat([scaled(d, d[r + "_ev"], lo, hi) for r in rs], axis=1).mean(axis=1))
            if "record" in it:
                x, (lo, hi) = d[it["record"]].astype(float), map(float, it["goalposts"])
                if it.get("log"):
                    x, lo, hi = np.log1p(x), np.log1p(lo), np.log1p(hi)
                parts.append(scaled(d, x, lo, hi))
            d[indicator_name(it)] = pd.concat(parts, axis=1).mean(axis=1)
    return d


# ---------------------------------------------------------------- 3. index
def equal_weights(P):
    """!
    @brief Each dimension's indicators at equal weights.
    @param P  {dimension: [indicator columns]}.
    @return {dimension: {column: weight}}.
    """
    return {k: {t: 1 / len(ts) for t in ts} for k, ts in P.items()}


def dimension(d, w):
    """!
    @brief One dimension: the weighted sum of its indicators.
    """
    return sum(wt * d[t] for t, wt in w.items())


def index(d, W):
    """!
    @brief The dimensions and M from the indicators of `d` with the weights W.
    @param W  {dimension: {column: weight}}.
    @return DataFrame <dimension>_raw (on [0, 1]), <dimension> (1-7) and M on d's index.
    """
    out = pd.DataFrame({f"{k}_raw": dimension(d, w) for k, w in W.items()}, index=d.index)
    for k in W:
        out[k] = 1.0 + 6.0 * out[f"{k}_raw"]
    out["M"] = np.exp(np.log(out[list(W)]).mean(axis=1))     # the geometric mean
    return out


def compose(d, C):
    """!
    @brief Terms, indicators, the dimensions and M, added to a ratings frame that carries the records it needs.
    @param C  The config's `metric`.
    @return d with the new columns.
    """
    d = derived(d, C)
    missing = [r + "_ev" for items in C["dimensions"].values() for i in items for r in indicator_ratings(i)
               if r + "_ev" not in d] + [c for c in records_needed(C) if c not in d]
    if missing:
        raise SystemExit(f"the metric needs {missing}: rate the views with the current prompts (rate_views_with_vlm) and measure the "
                         "physical attributes (measure_city_records) first")
    d = indicators(terms(d), C)
    out = index(d, equal_weights(dimension_columns(C)))
    for col in out:
        d[col] = out[col]
    return d


# ---------------------------------------------------------------- 4. checks
def pca_weights(d, P):
    """!
    @brief |PC1 loadings| of each dimension's standardised indicators, normalised: a diagnostic beside the equal
    weights.
    @return ({dimension: {column: weight}}, the table of weights, signed loadings and PC1's share of variance).
    """
    rows, W = [], {}
    for k, cols in P.items():
        Z = d[cols].to_numpy(float)
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
        val, vec = np.linalg.eigh(np.cov(Z, rowvar=False))
        load = np.abs(vec[:, -1])
        W[k] = dict(zip(cols, load / load.sum()))
        pc1 = val[-1] / val.sum()
        rows += [{"dimension": k, "indicator": t, "weight": W[k][t], "loading": vec[i, -1], "pc1_variance": pc1}
                 for i, t in enumerate(cols)]
    return W, pd.DataFrame(rows)


def cronbach_alpha(x):
    """!
    @brief Cronbach's alpha of the columns of x: how far they measure one thing (1 = the same, 0 = unrelated).
    """
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var().sum() / x.sum(axis=1).var())


def dimension_coherence(d, P):
    """!
    @brief Whether each dimension's indicators hang together: Cronbach's alpha, alpha without each indicator, and
           their mean correlation. A low alpha suits a formative index (separate ingredients, as the HDI's are), not a
           reflective one. vif_max is the largest variance inflation among the indicators: what a formative index needs
           instead of a high alpha is indicators that are not redundant (VIF well below 5).
    @return DataFrame dimension, indicators ("all", or "without <column>"), alpha, mean_r, vif_max.
    """
    rows = []
    for k, t in P.items():
        r = d[t].corr().to_numpy()
        rows.append({"dimension": k, "indicators": "all", "alpha": cronbach_alpha(d[t]),
                     "mean_r": r[np.triu_indices(len(t), 1)].mean(), "vif_max": np.diag(np.linalg.pinv(r)).max()})
        rows += [{"dimension": k, "indicators": f"without {x}", "alpha": cronbach_alpha(d[[c for c in t if c != x]]),
                  "mean_r": np.nan} for x in t if len(t) > 2]
    return pd.DataFrame(rows)


def indicator_factors(d, P, seed):
    """!
    @brief Exploratory factor analysis of the indicators (standardised; as many factors as dimensions, varimax): do they
           group into the dimensions they are averaged in?
    @return DataFrame, one row per indicator: its dimension and its loading on each factor.
    """
    from sklearn.decomposition import FactorAnalysis
    T = [t for ts in P.values() for t in ts]
    Z = (d[T] - d[T].mean()) / d[T].std().replace(0, 1)
    fa = FactorAnalysis(n_components=len(P), rotation="varimax", random_state=seed).fit(Z)
    out = pd.DataFrame(fa.components_.T, index=T, columns=[f"F{i + 1}" for i in range(len(P))])
    out.insert(0, "dimension", [k for k, ts in P.items() for _ in ts])
    return out.rename_axis("indicator").reset_index()


def weight_robustness(d, P, seed, draws=1000):
    """!
    @brief How far the dimensions and M depend on the indicator weights: their rank agreement (Spearman) with equal
           weights under PCA weights and `draws` random weight sets (uniform over each dimension's possible weights),
           and the share of the top 10% of views by M that stays there.
    @param d  The composed frame (indicators present).
    @param P  {dimension: [indicator columns]}.
    @return DataFrame, one row per scheme (random: median and 5th percentile).
    """
    ref = index(d, equal_weights(P))
    top = ref.M >= ref.M.quantile(0.9)
    keys = [*P, "M"]

    def compare(Wx):
        out = index(d, Wx)
        return ([spearmanr(ref[c], out[c]).statistic for c in keys]
                + [(top & (out.M >= out.M.quantile(0.9))).sum() / top.sum()])

    cols = [f"rho_{c}" for c in keys] + ["top10_M_kept"]
    rows = [{"scheme": "pca", **dict(zip(cols, compare(pca_weights(d, P)[0])))}]
    rng = np.random.default_rng(seed)
    r = np.array([compare({k: dict(zip(t, rng.dirichlet(np.ones(len(t))))) for k, t in P.items()})
                  for _ in range(draws)])
    rows.append({"scheme": f"random ({draws}), median", **dict(zip(cols, np.median(r, 0)))})
    rows.append({"scheme": f"random ({draws}), 5th percentile", **dict(zip(cols, np.percentile(r, 5, 0)))})
    return pd.DataFrame(rows)


def exponent_robustness(d, W, step=0.05):
    """!
    @brief How far M depends on its equal exponents: M's rank agreement (Spearman) with unequal ones, and the share of
           the top 10% of views by M kept, over every set of exponents >= step on a step grid summing to 1, plus equal
           exponents and each dimension alone.
    @return DataFrame: one row per named set, then the grid's median, 5th percentile and minimum.
    """
    out = index(d, W)
    dims = list(W)
    L = np.log(out[dims].to_numpy())
    ref = out.M
    top = ref >= ref.quantile(0.9)

    def compare(e):
        M = pd.Series(np.exp(L @ np.asarray(e)), index=out.index)
        return spearmanr(ref, M).statistic, (top & (M >= M.quantile(0.9))).sum() / top.sum()

    n = len(dims)
    named = {f"equal 1/{n} each": [1 / n] * n} | {f"{k} alone": list(np.eye(n)[i]) for i, k in enumerate(dims)}
    rows = [{"exponents": k, **dict(zip(dims, e)), **dict(zip(["rho_M", "top10_M_kept"], compare(e)))}
            for k, e in named.items()]
    g = np.round(np.arange(step, 1, step), 6)
    grid = np.array([compare([*e, 1 - sum(e)]) for e in itertools.product(g, repeat=n - 1) if 1 - sum(e) >= step - 1e-9])
    for name, f in (("grid median", np.median), ("grid 5th percentile", lambda x, axis: np.percentile(x, 5, axis)),
                    ("grid minimum", np.min)):
        rows.append({"exponents": f"{name} ({len(grid)} sets, step {step})", **dict(zip(["rho_M", "top10_M_kept"],
                                                                                         f(grid, axis=0)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. tables
def arcs(cfg, paths, sc):
    """!
    @brief The segmenter's vegetation / sky / building share over each view's own arc
    (vlm_rating_validation.arc_shares).
    @return DataFrame arc_vegetation, arc_sky, arc_building per row of sc; NaN without views-180.csv or
            the term shares by bearing.
    """
    cols = ["vegetation", "sky", "building"]
    zpath, rpath = paths.term_shares_by_bearing_npz, paths.views_csv
    if not (zpath.exists() and rpath.exists()):
        return pd.DataFrame(np.nan, index=sc.index, columns=["arc_" + c for c in cols])
    a = arc_shares(sc, dict(np.load(zpath)), pd.read_csv(rpath).set_index("file").bearing, TermSharesByBearing(cfg),
                   cfg["stitch_180_views"]["fov"])
    return a[cols].add_prefix("arc_")


def sections(m, crossing_m):
    """!
    @brief Cut each street into block-length runs between crossings.
    @param m           Nodes (osm_name, geometry).
    @param crossing_m  A node this close to another street is at a crossing.
    @return m with `section`: a run number, -1 at a crossing.
    """
    xy = np.c_[m.geometry.x, m.geometry.y]
    nm = m.osm_name.to_numpy()
    at = np.array([(nm != nm[i]).any() and np.hypot(*(xy[nm != nm[i]] - xy[i]).T).min() < crossing_m
                   for i in range(len(m))])
    out, sec = np.full(len(m), -1), 0
    for _, g in m.groupby("osm_name"):
        idx = m.index.get_indexer(g.index)
        c = xy[idx] - xy[idx].mean(0)
        v = np.linalg.svd(c, full_matrices=False)[2][0]
        prev_at = True
        for k in idx[np.argsort(c @ v)]:              # along the street's principal axis
            if at[k]:
                prev_at = True
                continue
            sec += prev_at
            out[k], prev_at = sec, False
    return m.assign(section=out)


def write_tables(cfg, paths, sc, dropped, nodes):
    """!
    @brief all-rated-views-with-inputs.csv, scores-per-usable-view.csv, mean-score-per-block-section.csv and
           column-dictionary.md.
    @param sc       Composed ratings on usable nodes.
    @param dropped  Ratings on unusable nodes (observations only).
    @param nodes    node_table().
    """
    C = cfg["metric"]
    dims = list(C["dimensions"])
    sc = pd.concat([sc.assign(usable=True), dropped.assign(usable=False)], ignore_index=True)
    sc = sc.drop(columns=[c for c in NODE if c in sc]).merge(
        nodes[["node_id"] + [c for c in NODE if c in nodes]], on="node_id", how="left")
    sc = sc.reindex(columns=list(dict.fromkeys([*sc.columns, *NODE])))   # columns of stages not yet run: NaN
    sc["exclude_reason"] = sc.exclude_reason.fillna("")
    sc = sc.join(arcs(cfg, paths, sc))

    fields = rated_fields(sc)
    obs = sc[ID + ["usable", "exclude_reason", "osm_name", "typology", "face_id"]
             + [f + "_ev" for f in fields]
             + ["arc_vegetation", "arc_sky", "arc_building", "H_m", "W_facade", "HW_facade",
                "HW_effective", "HW_source", "SVF_canyon", "GVI", "VEI", "SVF_band"]
             + [c for c in [f + "_argmax" for f in fields] + [f"{f}_p{k}" for f in fields for k in K] if c in sc]
             + [c for c in sc if c.endswith(("_left_rating", "_right_rating", "_left_p0", "_right_p0"))]
             ].rename(columns={"GVI": "node_GVI", "VEI": "node_VEI", "SVF_band": "node_SVF_band"})
    obs = obs.sort_values(["osm_name", "walk", "seq"])
    term_cols = [TERMS.get(f, f) for f in fields]
    index_cols = [c for cs in dimension_columns(C).values() for c in cs]
    calc = sc[sc.usable][ID + term_cols + records_needed(C) + index_cols + [f"{k}_raw" for k in dims] + dims
                         + ["HW_effective", "M"]]
    calc = calc.loc[:, ~calc.columns.duplicated()].sort_values("file")
    obs.insert(0, "city", cfg["name"])
    calc.insert(0, "city", cfg["name"])
    obs.to_csv(paths.all_views_csv, index=False)
    calc.to_csv(paths.scores_csv, index=False)
    print(f"all-rated-views-with-inputs.csv {obs.shape}, scores-per-usable-view.csv {calc.shape}")

    if len(calc):
        g = sections(nodes[nodes.node_id.isin(set(calc.node_id))].reset_index(drop=True), C["section_crossing_m"])
        g = g.merge(calc.groupby("node_id").M.mean().rename("M"), on="node_id", how="left")
        agg = (g[g.section > 0].groupby("section")
               .agg(street=("osm_name", "first"), n=("M", "count"), M=("M", "mean"), sd=("M", "std"))
               .reset_index())
        agg = agg[agg.n > 0]
        agg.to_csv(paths.block_sections_csv, index=False)
        print(f"mean-score-per-block-section.csv {len(agg)} sections")
    write_dictionary(cfg, paths, obs, calc)


DICT = {
    "file": "view path under data/<city>/data_prep/views-180; the join key",
    "node_id": "node id, positional to this city's sampling nodes",
    "street / walk / seq / cardinal": "street folder, direction of travel, position along the walk, compass letter",
    "usable / exclude_reason": "False rows are observations only; they are absent from scores-per-usable-view",
    "<term>_ev": "the rating: expected value over the 7-rung distribution (probability-weighted); empty where neither side of a facade term could be judged (the indicator then takes the street's mean)",
    "<field>_argmax / _p1.._p7": "most likely rung, and the full distribution",
    "arc_vegetation / arc_sky / arc_building": "segmenter shares over this view's own arc",
    "H_m / W_facade / HW_facade": "median building height, facade-to-facade width, their ratio (measured only)",
    "HW_effective / HW_source": "H/W including inherited values, and how it was obtained",
    "SVF_canyon": "measured sky view factor of the node's canyon, cos(arctan(2 HW_effective)) (footprints; validation only)",
    "node_GVI / node_VEI / node_SVF_band": "node-level 360-degree green view, enclosure, band sky share (not SVF)",
    "V_nat, GVI_eye, ..., and every other rated term": "each rating's expected value on [0, 1] by (EV - 1) / 6 (named as in the paper where it has a paper name)",
    "setback_m, storefronts, ... (the records the indicators name)": "step measure_city_records's physical attributes and other node measurements, as the indicators read them",
    "index_<name>": "an indicator scaled between its goalposts on [0, 1] (metric.dimensions)",
    "<dimension>_raw": "a dimension on [0, 1]: the equal-weight mean of its indicators",
    "<dimension> (e.g. identity / dependence / comfort)": "the pillars on the 1-7 scale (1 + 6 x)",
    "M": "Street Interface Matrix score: the geometric mean of the pillars, on 1-7",
}


def write_dictionary(cfg, paths, obs, calc):
    """!
    @brief column-dictionary.md: what each column means, and the ranges of the dimensions and M in this run.
    """
    dims = list(cfg["metric"]["dimensions"])
    lines = [f"# {cfg['name']} -- Street Interface Matrix tables ({cfg['run_id']})", "",
             f"Generated by pipeline step compose_metric. `all-rated-views-with-inputs.csv` {obs.shape[0]} rows x "
             f"{obs.shape[1]} cols; `scores-per-usable-view.csv` {calc.shape[0]} rows x {calc.shape[1]} cols "
             f"({calc.node_id.nunique()} nodes). Join on `file`.", "",
             "| column | meaning |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in DICT.items()]
    lines += ["", "Ranges in this run:", "", "| column | min | median | max |", "|---|---|---|---|"]
    for c in [*dims, "M"]:
        v = calc[c].dropna()
        lines.append(f"| `{c}` | {v.min():.3f} | {v.median():.3f} | {v.max():.3f} |")
    lines += ["", "Rated term -> the paper's term: " + ", ".join(f"`{f}` {t}" for f, t in FIELDS.items()), ""]
    paths.column_dictionary.write_text("\n".join(lines))


# ---------------------------------------------------------------- 6. run
def run(cfg, paths):
    """!
    @brief Compose the index on the usable nodes' ratings (view-ratings.csv) with their records, then write the checks
           and tables.
    """
    nodes = node_table(cfg, paths)
    C = cfg["metric"]
    need = [c for c in records_needed(C) if c not in nodes]
    if need:
        raise SystemExit(f"the node table lacks {need}: run measure_city_records first")
    d = pd.read_csv(paths.ratings_csv)
    clash = sorted(set(records_needed(C)) & set(rated_fields(d)))
    if clash:
        raise SystemExit(f"records named like rated terms (the ratings table has a column of that name): {clash}")
    bad = d.node_id.isin(set(nodes.node_id[~nodes.usable.astype(bool)]))
    d, dropped = d[~bad].reset_index(drop=True), d[bad]
    d = d.merge(pd.DataFrame(nodes[["node_id", *records_needed(C)]]), on="node_id", how="left")
    print(f"{len(d)} views rated, {len(dropped)} on unusable nodes set aside")
    P = dimension_columns(C)
    print("\n".join(f"{C['names'][k]} ({k}) = mean of {', '.join(c[6:] for c in v)}" for k, v in P.items()))
    d = compose(d, C)
    for c in [*P, "M"]:
        v = d[c].dropna()
        print(f"  {c:<8} min {v.min():.3f}  median {v.median():.3f}  max {v.max():.3f}")
    pca_weights(d, P)[1].to_csv(paths.pca_csv, index=False)
    rob = weight_robustness(d, P, cfg["seed"])
    rob.to_csv(paths.weight_robustness_csv, index=False)
    print("weights robustness (rank agreement with equal weights):\n" + rob.round(3).to_string(index=False))
    ex = exponent_robustness(d, equal_weights(P))
    ex.to_csv(paths.exponent_robustness_csv, index=False)
    print("exponent robustness (M's rank agreement with equal exponents):\n" + ex.round(3).to_string(index=False))
    coh = dimension_coherence(d, P)
    coh.to_csv(paths.dimension_coherence_csv, index=False)
    fac = indicator_factors(d, P, cfg["seed"])
    fac.to_csv(paths.indicator_factors_csv, index=False)
    print("dimension coherence (Cronbach's alpha):\n" + coh.round(2).to_string(index=False)
          + "\nindicator factor loadings (varimax):\n" + fac.round(2).to_string(index=False))
    write_tables(cfg, paths, d, dropped, nodes)
