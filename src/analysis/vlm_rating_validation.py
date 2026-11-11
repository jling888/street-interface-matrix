"""!
@package vlm_rating_validation
@brief How far the VLM's ratings follow the photos: each term's expected rating against its measured counterpart
       in the label maps (term shares by bearing), over the view's own arc (Spearman rho). A rating read from the photo
       rises
       with what the photo shows; one the model makes up does not. The enclosure rating is also checked against
       the canyon measured from building footprints (step measure_street_canyons), which the photos cannot show.

Writes outputs/<run_id>/analysis/vlm-rating-validation/: view-ratings-vs-segmentation.csv (with each rating's spread)
and
enclosure-vs-canyon-geometry.csv.

The ratings against what the city records and the footprints measure (step measure_city_records's city records), none of
which a photo's
segmentation can give (city_records_checks):

  ratings_vs_records         the earlier terms' checks: sidewalk sheds, building age, landmarks
  ratings_vs_physical_records  each agreement term against the record it reads (vlm_rating_validation.agreement): rank
  agreement,
                             its sign, and a suggested role in M (rating or record)
  rating_repeatability       every term: how far it repeats on the next view along the walk, and how bunched it is
  enclosure_by_height        the enclosure rating, the visible sky and the canyon sky view per band of building height

Writes outputs/<run_id>/analysis/vlm-rating-validation/: ratings-vs-city-records.csv, ratings-vs-physical-records.csv,
rating-repeatability.csv and enclosure-by-height.csv. Run by pipeline step validate_vlm_ratings.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.data_readers import FIELDS, TermSharesByBearing, arc_shares, rated_fields
from common.node_table import node_table, read_nodes

# term -> its measured counterpart: a term share over the whole view, <term>_eye within the
# paper's foveal band (0-15 degrees below the horizon), green_share(_eye) = vegetation / (vegetation +
# building), or enclosure = 1 - sky share. IAS has benches and stoops but not the ledges it also counts.
ANCHOR = {
    "v_nat": "V_nat",
    "gvi_eye": "vegetation_eye",
    "gmi": "green_share_eye",
    "v_built": "V_built",
    "enclosure": "enclosure",
    "v_sign": "signage",
    "v_pave": "V_pave",
    "gfapi": "openings_eye",
    "ias": "seating",
}



def anchor(d, node_shares, bearing, by_bearing, fov, frame_fov=90):
    """!
    @brief Rank agreement of each field's expected rating with its counterpart, on usable views, and how far the
           ratings spread: a rating bunched on one rung cannot follow the photos however well it is read. A term rated
           on the facade frames (the table has its per-side columns) is checked over the arcs of those two frames.
    @param frame_fov  The facade frames' field of view (imagery.fov).
    @return One row per field of ANCHOR the table has: term, counterpart, views, rho, sd_ev (the expected rating's
            standard deviation, in rungs), modal_share (the share of views on the most common rounded rung).
    """
    along = arc_shares(d, node_shares, bearing, by_bearing, fov)
    sides = None
    if any(f + "_left_rating" in d for f in ANCHOR):
        sides = (arc_shares(d, node_shares, bearing - 90, by_bearing, frame_fov)
                 + arc_shares(d, node_shares, bearing + 90, by_bearing, frame_fov)) / 2
    rows = []
    for f, c in ((f, c) for f, c in ANCHOR.items() if f + "_ev" in d):
        s = sides if f + "_left_rating" in d else along
        ok = s[c].notna() & d[f + "_ev"].notna()
        rho = spearmanr(d.loc[ok, f + "_ev"], s.loc[ok, c]).statistic if ok.sum() > 2 else np.nan
        rung = d[f + "_ev"].round()
        rows.append({"field": f, "term": FIELDS.get(f, f), "counterpart": c, "views": int(ok.sum()),
                     "rho": round(float(rho), 3), "sd_ev": round(float(d[f + "_ev"].std()), 3),
                     "modal_share": round(float((rung == rung.mode().iloc[0]).mean()), 3)})
    return pd.DataFrame(rows)


# the measured canyon (step measure_street_canyons, per node) the enclosure rating is checked against
GEOMETRY = {"1 - SVF_canyon": lambda n: 1 - n.SVF_canyon, "H/W measured": lambda n: n.HW_facade,
            "building height": lambda n: n.H_m, "street width": lambda n: n.W_facade}


def enclosure_vs_geometry(d, nodes):
    """!
    @brief Rank agreement of the enclosure rating with the canyon measured from building footprints: the
           counterpart the label maps cannot give (the photos see the sky only up to 45 degrees up).
    @param nodes  node_table() (H_m, W_facade, HW_facade, SVF_canyon from step measure_street_canyons).
    @return One row per GEOMETRY measure: measure, views, rho; empty without step measure_street_canyons's columns.
    """
    if "SVF_canyon" not in nodes:
        return pd.DataFrame(columns=["measure", "views", "rho"])
    v = d[["node_id", "enclosure_ev"]].merge(pd.DataFrame(nodes.drop(columns="geometry")), on="node_id")
    rows = []
    for name, f in GEOMETRY.items():
        x = f(v)
        ok = x.notna()
        rows.append({"measure": name, "views": int(ok.sum()),
                     "rho": round(float(spearmanr(v.enclosure_ev[ok], x[ok]).statistic), 3)})
    return pd.DataFrame(rows)


def enclosure_curve(d, nodes, V):
    """!
    @brief The shape of the enclosure rating against the measured H/W: its mean per H/W band, its rank agreement
           in open streets and in canyons (split at vlm_rating_validation.hw_split), and how well a straight line, a log
           and a
           saturating curve fit it. A level photo shows the sky only up to 45 degrees up, so from mid-street the
           roofline leaves the frame once H/W exceeds 0.5: a rating that flattens below the split, not a poor
           reading of the photo, is what that predicts.
    @param V  The config's `validate` (hw_bands, hw_split).
    @return DataFrame what (band / spearman / fit), label, views, mean_rating, sd, rho, r2; empty without step
    measure_street_canyons.
    """
    from scipy.optimize import curve_fit
    if "HW_effective" not in nodes:
        return pd.DataFrame(columns=["what", "label", "views", "mean_rating", "sd", "rho", "r2"])
    v = d[["node_id", "enclosure_ev"]].merge(pd.DataFrame(nodes[["node_id", "HW_effective"]]), on="node_id").dropna()
    x, y = v.HW_effective.to_numpy(), v.enclosure_ev.to_numpy()
    rows = [{"what": "band", "label": str(b), "views": len(g), "mean_rating": g.mean(), "sd": g.std()}
            for b, g in v.groupby(pd.cut(v.HW_effective, V["hw_bands"]), observed=True).enclosure_ev]
    for label, m in ((f"H/W <= {V['hw_split']}", x <= V["hw_split"]), (f"H/W > {V['hw_split']}", x > V["hw_split"]),
                     ("all", x > 0)):
        rows.append({"what": "spearman", "label": label, "views": int(m.sum()), "rho": spearmanr(x[m], y[m]).statistic})
    tss = (y - y.mean()) @ (y - y.mean())
    for label, f, p0 in (("linear: a + b H/W", lambda x, a, b: a + b * x, [4, 0.3]),
                         ("log: a + b ln H/W", lambda x, a, b: a + b * np.log(x), [4, 1]),
                         ("saturating: a + b (1 - exp(-H/W / c))", lambda x, a, b, c: a + b * (1 - np.exp(-x / c)),
                          [3, 2, 1])):
        p, _ = curve_fit(f, x, y, p0=p0, maxfev=20000)
        r = y - f(x, *p)
        rows.append({"what": "fit", "label": label, "views": len(v), "r2": 1 - r @ r / tss})
    return pd.DataFrame(rows)


# a VLM rating -> the city's record that should rise with it (step measure_city_records's seating), per node
FURNITURE = {"ias": ["seating_formal", "seating_public", "seating_cafes"], "gfapi": ["seating_cafes"],
             "v_sign": ["seating_cafes"],
             # the final terms: free seating against the city's benches (cafe seats are for customers: spill-out's)
             "free_seating": ["seating_public"], "business_spill_out": ["seating_cafes"],
             "ground_floor_openness": ["seating_cafes"], "signage_and_detail": ["seating_cafes"]}


def ratings_vs_street_seating(d, nodes):
    """!
    @brief The VLM's seating, shopfront and signage ratings against the formal seating the city records near each
           node (benches, approved sidewalk cafes): rank agreement, and the mean rating where there is any vs none.
           The seating rating counts informal seating (stoops, ledges), so a modest agreement is the expected one;
           cafes mark active ground floors, the shopfront term's subject.
    @return DataFrame field, record, views, rho, mean_with, mean_without; empty without the seating records.
    """
    if "seating_formal" not in nodes:
        return pd.DataFrame(columns=["field", "record", "views", "rho", "mean_with", "mean_without"])
    v = d.merge(pd.DataFrame(nodes[["node_id", *{c for cs in FURNITURE.values() for c in cs}]]), on="node_id")
    rows = []
    for f, records in ((f, r) for f, r in FURNITURE.items() if f + "_ev" in v):
        for c in records:
            has = v[c] > 0
            judged = v[f + "_ev"].notna()
            rows.append({"field": f, "record": c, "views": int(judged.sum()), "views_with": int((has & judged).sum()),
                         "rho": spearmanr(v.loc[judged, f + "_ev"], v.loc[judged, c]).statistic,
                         "mean_with": v.loc[has, f + "_ev"].mean(), "mean_without": v.loc[~has, f + "_ev"].mean()})
    return pd.DataFrame(rows)


def cannot_judge_vs_vehicles(d, shares):
    """!
    @brief Does the model say it cannot judge a frontage where vehicles fill the photos? Per term with a rung 0: the
           node mean of P(rung 0) over both sides against the node's segmented vehicle share (Spearman).
    @param shares  class-shares-per-node.csv (node_id, vehicle).
    @return DataFrame field, nodes, rho, share_of_sides_not_judged (P(rung 0) above 0.5); empty without a rung 0.
    """
    rows = []
    for f in (c[:-len("_left_p0")] for c in d if c.endswith("_left_p0")):
        p0 = d[[f + "_left_p0", f + "_right_p0"]]
        v = p0.mean(axis=1).groupby(d.node_id).mean().rename("p0").to_frame().join(shares.set_index("node_id").vehicle,
                                                                                    how="inner").dropna()
        rows.append({"field": f, "nodes": len(v), "rho": spearmanr(v.p0, v.vehicle).statistic,
                     "share_of_sides_not_judged": float((p0 > 0.5).to_numpy().mean())})
    return pd.DataFrame(rows, columns=["field", "nodes", "rho", "share_of_sides_not_judged"])



# ---------------------------------------------------------------- against the city's records
# V_sign -> the records that should follow it (v_pave against the sheds is an agreement pair:
# vlm_rating_validation.agreement)
RECORDS = {"v_sign": ["median_building_year", "landmark_building_share", "buildings_fronting"]}


def node_means(ratings, field):
    """!
    @brief A rating's mean over each node's views.
    """
    return ratings.groupby("node_id")[field + "_ev"].mean().rename("rating").reset_index()


def ratings_vs_records(ratings, records):
    """!
    @brief Rank agreement of each rating with its records, per node.
    @param ratings  DataFrame node_id, <term>_ev (one row per view).
    @return DataFrame field, record, nodes, rho.
    """
    rows = []
    for f, names in RECORDS.items():
        if f + "_ev" not in ratings:
            continue
        v = node_means(ratings, f).merge(records, on="node_id")
        for c in names:
            ok = v.rating.notna() & v[c].notna()
            rows.append({"field": f, "record": c, "nodes": int(ok.sum()), "rho": spearmanr(v.rating[ok], v.loc[ok, c]).statistic})
    return pd.DataFrame(rows, columns=["field", "record", "nodes", "rho"])


def rating_repeatability(ratings):
    """!
    @brief Per term: the correlation between a view and the next one along the same walk (20 m on, facing the same
           way; an upper bound on the rating's reliability, since the two images overlap), the same within streets
           (each street's mean removed), and the share of views on the most common rung.
    @return DataFrame field, next_view_r, next_view_r_within_street, modal_share, sd_rungs.
    """
    d = ratings.sort_values(["street", "walk", "seq"])
    g = d.groupby(["street", "walk"])
    step = (g.seq.shift(-1) - d.seq) == 1
    rows = []
    for f in rated_fields(d):
        x, nxt = d[f + "_ev"], g[f + "_ev"].shift(-1)
        ok = step & x.notna() & nxt.notna()
        m = d.groupby("street")[f + "_ev"].transform("mean")
        mode = d[f + "_ev"].round().value_counts(normalize=True)
        rows.append({"field": f, "next_view_r": np.corrcoef(x[ok], nxt[ok])[0, 1] if ok.sum() > 2 else np.nan,
                     "next_view_r_within_street": np.corrcoef((x - m)[ok], (nxt - m)[ok])[0, 1] if ok.sum() > 2 else np.nan,
                     "modal_share": mode.iloc[0] if len(mode) else np.nan, "sd_rungs": x.std()})
    return pd.DataFrame(rows, columns=["field", "next_view_r", "next_view_r_within_street", "modal_share", "sd_rungs"])


def ratings_vs_physical_records(ratings, nodes, repeat, V):
    """!
    @brief Each agreement term against the record it reads: rank agreement per node, whether its sign is the one
           expected, and a suggested role in M. A usable rating (vlm_rating_validation.max_modal_share,
           min_repeatability) whose
           repeatability exceeds rho squared by min_beyond_record carries perception the record misses: "rating";
           otherwise the record stands for it: "record". Only a suggestion: compose takes what its config names.
    @param repeat  rating_repeatability().
    @return DataFrame field, record, nodes, rho, sign_as_expected, next_view_r, beyond_record, suggested_role.
    """
    rep = repeat.set_index("field")
    rows = []
    for f, a in V["agreement"].items():
        if f + "_ev" not in ratings or a["record"] not in nodes:
            continue
        v = node_means(ratings, f).merge(pd.DataFrame(nodes[["node_id", a["record"]]]), on="node_id").dropna()
        rho = spearmanr(v.rating, v[a["record"]]).statistic if len(v) > 2 else np.nan
        r = rep.loc[f]
        beyond = r.next_view_r - rho ** 2
        usable = r.modal_share <= V["max_modal_share"] and r.next_view_r >= V["min_repeatability"]
        rows.append({"field": f, "record": a["record"], "nodes": len(v), "rho": rho,
                     "sign_as_expected": bool(np.sign(rho) == a["sign"]), "next_view_r": r.next_view_r,
                     "beyond_record": beyond,
                     "suggested_role": ("rating" if beyond >= V["min_beyond_record"] else "record") if usable
                     else "record (rating unusable)"})
    return pd.DataFrame(rows, columns=["field", "record", "nodes", "rho", "sign_as_expected", "next_view_r",
                                       "beyond_record", "suggested_role"])


def nat_built_vs_segmentation(ratings, shares, built):
    """!
    @brief The VLM's greenery-to-walls ratio against the same ratio in the label maps, per node.
    @param shares  class-shares-per-node.csv.
    @param built   The classes that are walls; the greenery classes are the term `vegetation`'s.
    """
    green = shares[["tree", "hedge", "planter", "vegetation"]].sum(axis=1)
    measured = green / (green + shares[built].sum(axis=1))
    v = ratings.groupby("node_id")[["v_nat_ev", "v_built_ev"]].mean()
    v = (v.v_nat_ev / (v.v_nat_ev + v.v_built_ev)).rename("rating").reset_index().merge(
        shares[["node_id"]].assign(measured=measured), on="node_id").dropna()
    return {"field": "nat_built", "record": "segmented_green_to_walls_ratio", "nodes": len(v),
            "rho": spearmanr(v.rating, v.measured).statistic}


def enclosure_by_height(ratings, nodes, shares, storey_m, bands):
    """!
    @brief The enclosure rating, the sky the level photos show and the canyon's sky view per band of building height:
           what a level view can tell apart as streets get taller.
    @param bands  Band edges, in storeys of storey_m metres.
    @return DataFrame storeys, nodes, enclosure_rating, visible_sky_share, canyon_sky_view.
    """
    v = ratings.groupby("node_id").enclosure_ev.mean().rename("rating").reset_index().merge(
        pd.DataFrame(nodes[["node_id", "H_m", "SVF_canyon"]]), on="node_id").merge(shares[["node_id", "sky"]], on="node_id")
    g = v.groupby(pd.cut(v.H_m / storey_m, bands), observed=True)
    return pd.DataFrame({"storeys": [str(k) for k in g.groups], "nodes": g.size().values, "enclosure_rating": g.rating.mean().values,
                         "visible_sky_share": g.sky.mean().values, "canyon_sky_view": g.SVF_canyon.mean().values})


def city_records_checks(cfg, paths, d, nodes):
    """!
    @brief Write ratings-vs-city-records.csv, rating-repeatability.csv, ratings-vs-physical-records.csv and
           enclosure-by-height.csv.
    @param d      The run's ratings on usable nodes.
    @param nodes  node_table(), with step measure_city_records's physical attributes.
    """
    V = cfg["vlm_rating_validation"]
    repeat = rating_repeatability(d)
    repeat.to_csv(paths.rating_repeatability_csv, index=False)
    print("\nrating repeatability (next view along the walk) and spread\n" + repeat.round(3).to_string(index=False))
    if "active_sidewalk_shed" not in nodes or "SVF_canyon" not in nodes or not paths.class_shares_csv.exists():
        print("skipped the record checks: they need the physical attributes (measure_city_records), street canyons (measure_street_canyons) and class shares (calculate_segmentation_pixel_shares)")
        return
    usable = pd.DataFrame(nodes[nodes.usable])
    out = ratings_vs_records(d, usable)
    shares = pd.read_csv(paths.class_shares_csv)
    if "v_nat_ev" in d:
        out = pd.concat([out, pd.DataFrame([nat_built_vs_segmentation(d, shares, V["wall_classes"])])], ignore_index=True)
    out.to_csv(paths.city_records_check_csv, index=False)
    print("\nratings vs the city's records (Spearman, per node)\n" + out.round(3).to_string(index=False))
    a = ratings_vs_physical_records(d, usable, repeat, V)
    a.to_csv(paths.physical_records_check_csv, index=False)
    print("\nagreement terms vs the physical records (Spearman, per node) and a suggested role in M\n"
          + a.round(3).to_string(index=False))
    if "enclosure_ev" in d:
        h = enclosure_by_height(d, nodes, shares, V["storey_m"], V["storey_bands"])
        h.to_csv(paths.enclosure_by_height_csv, index=False)
        print("\nenclosure rating by building height\n" + h.round(3).to_string(index=False))


# ---------------------------------------------------------------- run
def run(cfg, paths):
    """!
    @brief The run's ratings against the label maps' term shares (view-ratings-vs-segmentation.csv), the
           enclosure rating against the measured canyons (enclosure-vs-canyon-geometry.csv) and its curve
           (enclosure-vs-hw-curve.csv), and against the city's records (city_records).
    """
    if not (paths.ratings_csv.exists() and paths.term_shares_by_bearing_npz.exists()):
        print(f"skipped: needs {paths.ratings_csv.name} (rate_views_with_vlm) and {paths.term_shares_by_bearing_npz.name} (calculate_segmentation_pixel_shares)")
        return
    d = pd.read_csv(paths.ratings_csv)
    d = d[d.node_id.isin(read_nodes(paths).query("usable").node_id)].reset_index(drop=True)
    a = anchor(d, dict(np.load(paths.term_shares_by_bearing_npz)),
               pd.read_csv(paths.views_csv).set_index("file").bearing, TermSharesByBearing(cfg), cfg["stitch_180_views"]["fov"],
               cfg["download_gsv_panos"]["fov"])
    a.to_csv(paths.anchor_csv, index=False)
    print(f"\nrating vs segmentation over each view's arc (Spearman; near 0 = not read from the photo)\n"
          f"{a.to_string(index=False)}")
    if paths.class_shares_csv.exists():
        j = cannot_judge_vs_vehicles(d, pd.read_csv(paths.class_shares_csv))
        if len(j):
            j.to_csv(paths.cannot_judge_vs_vehicles_csv, index=False)
            print("\ncannot judge (rung 0) vs segmented vehicle share (Spearman)\n" + j.round(3).to_string(index=False))
    nodes = node_table(cfg, paths)                      # with the city's seating records (measure_city_records), when configured
    f = ratings_vs_street_seating(d, nodes)
    f.to_csv(paths.street_seating_check_csv, index=False)
    print("\nratings vs the city's seating records (Spearman; mean rating with / without)\n"
          + (f.round(3).to_string(index=False) if len(f) else "no seating records (city_records_per_node.seating)"))
    g = enclosure_vs_geometry(d, nodes)
    g.to_csv(paths.enclosure_vs_geometry_csv, index=False)
    print("\nenclosure rating vs the measured canyon (Spearman)\n"
          + (g.to_string(index=False) if len(g) else "no canyon geometry (measure_street_canyons)"))
    c = enclosure_curve(d, nodes, cfg["vlm_rating_validation"])
    c.to_csv(paths.enclosure_curve_csv, index=False)
    print("\nenclosure rating against H/W: bands, rank agreement, curve fits\n" + c.round(3).to_string(index=False))
    city_records_checks(cfg, paths, d, nodes)
