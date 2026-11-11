"""Pedestrian demand on the dimensions of M, with network and land-use controls: a test of M, never a fit of it.

    ln(V+1) = b0 + b_image ln image + b_form ln form + b_stay ln stay                    Model 1
                 [+ the network controls: ln(Choice+1), ln Integration, at each radius]  Model 2
                 [+ the land-use controls: retail, office shares, use mix, ln subway distance]  Model 3
    OLS network only, OLS network and land use (Models 2 and 3 without the dimensions): what the street adds
    Model 3, indicators: Model 3 with the indicators in place of the dimensions (which part of one carries it)

Each OLS model's residual Moran's I is computed under the Conley kernel. The outcome is step measure_pedestrian_volume's
pedestrian-volume-per-node.csv. Writes outputs/<run_id>/analysis/footfall-regression/: model-fit.csv and
regression-coefficients.csv.
Run by pipeline step fit_footfall_regression.

The models: one coefficient per predictor for the whole study area, with the inference the
spatial data needs. Nearby views share their errors (OLS residual Moran's I about 0.7), so plain standard
       errors are far too small; each model is reported with several corrections, and spatial models confirm the
       coefficients themselves.

  ols_coefficients  OLS per view: plain, block-face-clustered, Conley (spatial HAC) and wild-cluster-bootstrap
                    inference; standardised coefficients and VIFs
  spatial_models    per node (the mean of its views): spatial error (SEM), spatial lag (SAR), SLX (neighbours'
                    predictors added) and eigenvector spatial filtering (ESF), on k-nearest-neighbour weights

Predictors are named as in the config; predictor() gives each its scale: ln of the dimensions (on 1-7), ln(1 + 6 x)
of an indicator (index_<name>, on 0-1), ln(1 + choice), ln of integration, distance and floor area, the land-use
shares as they are.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import norm

from common.data_readers import dimension_columns
from common.node_table import node_table


def walking_distances(paths, d):
    """!
    @brief Walking distance on the street network between the rows' nodes.
    """
    z = np.load(paths.walking_distances, allow_pickle=True)
    pos = {n: i for i, n in enumerate(z["node_id"])}
    idx = d.node_id.map(pos).to_numpy()
    return z["D"][np.ix_(idx, idx)]


def run(cfg, paths):
    """!
    @brief The outcome, the OLS models, and the spatial models per node.
    """
    GC = cfg["footfall_regression"]
    dims = list(cfg["metric"]["dimensions"])
    network, landuse = GC["controls"], GC["land_use_controls"]
    nodes = pd.DataFrame(node_table(cfg, paths).drop(columns="geometry"))
    controls = ["face_id", *network, *landuse]
    d = (pd.read_csv(paths.scores_csv).drop(columns=controls, errors="ignore")    # a control may also be a record of M
         .merge(pd.read_csv(paths.modelled_volume_csv)[["node_id", "value"]], on="node_id")
         .merge(nodes[["node_id", *controls]], on="node_id")
         .dropna(subset=[*dims, "value", *network]).reset_index(drop=True))
    D = walking_distances(paths, d)
    print(f"{len(d)} views, {d.node_id.nunique()} nodes")

    richest = dims + network + landuse
    indicators = [c for cs in dimension_columns(cfg["metric"]).values() for c in cs]
    models = {"OLS network only": network, "OLS Model 1": dims, "OLS Model 2": dims + network,
              "OLS network and land use": network + landuse, "OLS Model 3": richest,
              "OLS Model 3, indicators": indicators + network + landuse}
    coef = []
    for name, names in models.items():
        m = d.dropna(subset=names)
        X, cols = design(m, names, dims)
        idx = m.index.to_numpy()
        coef.append(ols_coefficients(X, cols, np.log1p(m.value.to_numpy(float)), D[np.ix_(idx, idx)],
                                          m.face_id.fillna("none"), GC, name, cfg["seed"]))

    g = gpd.read_file(paths.nodes_gpkg).to_crs(int(cfg["projected_crs"])).set_index("node_id")
    nd = node_means(d.dropna(subset=richest).assign(y=lambda t: np.log1p(t.value)), ["y", *richest])
    xy = np.c_[g.loc[nd.node_id].geometry.x, g.loc[nd.node_id].geometry.y]
    spatial_coef, spatial_fit = spatial_models(nd, richest, nd.y.to_numpy(), xy, GC, cfg["seed"], dims)

    W = np.clip(1 - D / GC["conley_cutoff_m"], 0, None)
    rows = []
    for (name, names), c in zip(models.items(), coef):
        m = d.dropna(subset=names)
        X, _ = design(m, names, dims)
        y = np.log1p(m.value.to_numpy(float))
        e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        idx = m.index.to_numpy()
        I, z, p = morans_i(e, W[np.ix_(idx, idx)], GC["moran_permutations"], cfg["seed"])
        rows.append({"model": name, "n": len(m), "r2": c.r2.iloc[0], "r2_adjusted": c.r2_adjusted.iloc[0],
                     "aic": c.aic.iloc[0], "moran_I": I, "moran_z": z, "moran_p": p, "vif_max": np.nanmax(c.vif)})
    summary = pd.DataFrame(rows + spatial_fit)
    summary.to_csv(paths.model_fit_csv, index=False)
    coef = pd.concat(coef + spatial_coef, ignore_index=True)
    coef.to_csv(paths.coefficients_csv, index=False)
    show = ["model", "term", "b", "b_standardised", "t_clustered", "t_conley", "p_wild_bootstrap", "t"]
    print(summary.round(3).to_string(index=False))
    print(coef[coef.term != "const"][show].round(3).to_string(index=False))


# ---------------------------------------------------------------- the models
# ---------------------------------------------------------------- predictors
def predictor(d, name, dims=()):
    """!
    @brief One predictor on its regression scale, and its column name.
    @param dims  The dimensions' columns (on 1-7).
    @return (values, name): e.g. (ln stay, "ln_stay"), (ln(1 + choice), "ln_choice"), (retail_share, "retail_share").
    """
    x = d[name].astype(float)
    if name in dims:
        return np.log(x), f"ln_{name}"
    if name.startswith("index_"):
        return np.log(1 + 6 * x), f"ln_{name}"
    if name.startswith("choice"):
        return np.log1p(x), f"ln_{name}"
    if name.startswith("integration") or name in ("subway_distance_m", "floor_area"):
        return np.log(x.clip(lower=1e-9)), f"ln_{name}"
    return x, name


def design(d, names, dims=()):
    """!
    @brief The design matrix: a constant and each predictor on its scale.
    @return (X, column names).
    """
    cols = [predictor(d, n, dims) for n in names]
    return np.column_stack([np.ones(len(d))] + [c for c, _ in cols]), ["const"] + [n for _, n in cols]


# ---------------------------------------------------------------- OLS and its inference
def conley_se(X, e, D, cutoff):
    """!
    @brief Spatial HAC (Conley 1999) standard errors: residuals of views within `cutoff` of each other may
           correlate, weighted by the Bartlett kernel 1 - d / cutoff (two views of one node count fully).
    """
    K = np.clip(1 - D / cutoff, 0, None)
    Xe = X * e[:, None]
    bread = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(bread @ (Xe.T @ K @ Xe) @ bread))


def cluster_t(X, y, g, G):
    """!
    @brief OLS coefficients and their cluster-robust (CR1) t-values; g = cluster index per row, G clusters.
    """
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    b = bread @ X.T @ y
    s = np.zeros((G, k))
    np.add.at(s, g, X * (y - X @ b)[:, None])
    V = G / (G - 1) * (n - 1) / (n - k) * bread @ (s.T @ s) @ bread
    return b, b / np.sqrt(np.diag(V))


def wild_cluster_p(X, y, g, reps, seed):
    """!
    @brief Wild cluster bootstrap p-values (Cameron, Gelbach & Miller 2008; restricted, Rademacher weights): the
           clustered t-test with few clusters (29 block faces) rejects too often; this one does not.
    @return One p-value per column of X.
    """
    G = g.max() + 1
    t_obs = cluster_t(X, y, g, G)[1]
    rng = np.random.default_rng(seed)
    p = np.ones(X.shape[1])
    for j in range(1, X.shape[1]):
        Xr = np.delete(X, j, axis=1)
        fit = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        u = y - fit
        t = np.array([cluster_t(X, fit + u * rng.choice([-1.0, 1.0], G)[g], g, G)[1][j] for _ in range(reps)])
        p[j] = (np.abs(t) >= abs(t_obs[j])).mean()
    return p


def vifs(X):
    """!
    @brief Variance inflation factor of each predictor (NaN for the constant).
    """
    if X.shape[1] < 3:
        return np.r_[np.nan, np.ones(X.shape[1] - 1)]
    return np.r_[np.nan, np.diag(np.linalg.inv(np.corrcoef(X[:, 1:], rowvar=False)))]


def ols_coefficients(X, names, y, D, groups, GC, model, seed):
    """!
    @brief OLS coefficients with plain, block-face-clustered, Conley and wild-cluster-bootstrap inference.
    @param groups  Block face per row (calculate_segmentation_pixel_shares' face_id).
    @param GC      The config's `footfall_regression` (conley_cutoff_m, wild_bootstrap).
    @return DataFrame, one row per coefficient, with the model's R2, adjusted R2 and AIC on every row.
    """
    import statsmodels.api as sm
    plain = sm.OLS(y, X).fit()
    g = pd.factorize(groups)[0]
    clus = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    se_c = conley_se(X, plain.resid, D, GC["conley_cutoff_m"])
    t_c = plain.params / se_c
    sd = X.std(0)
    out = pd.DataFrame({"model": model, "term": names, "b": plain.params,
                        "b_standardised": np.where(sd > 0, plain.params * sd / y.std(), np.nan),
                        "vif": vifs(X), "se": plain.bse, "t": plain.tvalues,
                        "se_clustered": clus.bse, "t_clustered": clus.tvalues, "p_clustered": clus.pvalues,
                        "se_conley": se_c, "t_conley": t_c, "p_conley": 2 * (1 - norm.cdf(np.abs(t_c))),
                        "p_wild_bootstrap": wild_cluster_p(X, y, g, GC["wild_bootstrap"], seed)})
    return out.assign(r2=plain.rsquared, r2_adjusted=plain.rsquared_adj, aic=plain.aic, n=len(y),
                      clusters=g.max() + 1)


# ---------------------------------------------------------------- per node
def node_means(d, cols):
    """!
    @brief The views' columns averaged per node (two views per node), with the first face_id.
    """
    return d.groupby("node_id").agg({**{c: "mean" for c in cols}, "face_id": "first"}).reset_index()


def knn_weights(xy, k):
    """!
    @brief Row-standardised k-nearest-neighbour weights (libpysal) and their dense matrix.
    """
    from libpysal.weights import KNN
    w = KNN.from_array(xy, k=k)
    w.transform = "r"
    return w, w.full()[0]


def esf_vectors(Wd, threshold=0.25):
    """!
    @brief Moran eigenvectors (Griffith): eigenvectors of M C M (C the symmetrised weights, M the centring
           matrix) whose Moran's I is above threshold x the largest; each a map pattern of positive
           autocorrelation, added to the regression to absorb it.
    """
    n = len(Wd)
    C = (Wd + Wd.T) / 2
    M = np.eye(n) - 1 / n
    val, vec = np.linalg.eigh(M @ C @ M)
    keep = val > threshold * val.max()
    return vec[:, keep]


def morans_i(e, W, nperm, seed):
    """!
    @brief Moran's I of residuals under row-standardised W (diagonal zero); permutation z and p.
    """
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    W = W / np.where(W.sum(1, keepdims=True) > 0, W.sum(1, keepdims=True), 1)
    stat = lambda z: len(z) / W.sum() * (z @ W @ z) / (z @ z)
    z = e - e.mean()
    I = stat(z)
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(z)) for _ in range(nperm)])
    zs = (I - null.mean()) / null.std()
    return I, zs, 2 * (1 - norm.cdf(abs(zs)))


def spatial_models(nd, names, y, xy, GC, seed, dims=()):
    """!
    @brief SEM, spatial lag, SLX and ESF of y on `names`, per node.
    @param nd        Per-node table with the predictors.
    @return (coefficient rows, fit rows).
    """
    import spreg
    import statsmodels.api as sm
    X, cols = design(nd, names, dims)
    w, Wd = knn_weights(xy, GC["spatial_k"])
    Y = y[:, None]
    coef, fit = [], []
    for kind in ("sem", "lag", "slx", "esf"):
        if kind == "sem":
            m = spreg.ML_Error(Y, X[:, 1:], w)
            extra, z = ("lambda", m.lam), np.array([s for s, _ in m.z_stat])
            b, se, e, r2, aic = m.betas.ravel(), m.std_err, m.e_filtered, m.pr2, m.aic
            terms = cols + ["lambda"]
        elif kind == "lag":
            m = spreg.ML_Lag(Y, X[:, 1:], w)
            extra, z = ("rho", m.rho), np.array([s for s, _ in m.z_stat])
            b, se, e, r2, aic = m.betas.ravel(), m.std_err, m.u, m.pr2, m.aic
            terms = cols + ["rho"]
        elif kind == "slx":
            WX = Wd @ X[:, 1:]
            f = sm.OLS(y, np.column_stack([X, WX])).fit()
            extra, z = ("", np.nan), f.tvalues
            b, se, e, r2, aic = f.params, f.bse, f.resid, f.rsquared, f.aic
            terms = cols + [f"W_{c}" for c in cols[1:]]
        else:
            E = esf_vectors(Wd)
            f = sm.OLS(y, np.column_stack([X, E])).fit()
            k = X.shape[1]
            extra, z = ("eigenvectors", E.shape[1]), f.tvalues[:k]
            b, se, e, r2, aic = f.params[:k], f.bse[:k], f.resid, f.rsquared, f.aic
            terms = cols
        name = f"{kind.upper()} (nodes)"
        coef.append(pd.DataFrame({"model": name, "term": terms, "b": b[:len(terms)], "se": se[:len(terms)],
                                  "t": z[:len(terms)] if np.ndim(z) else np.nan,
                                  "r2": r2, "aic": aic, "n": len(y)}))
        fit.append({"model": name, "n": len(y), "r2": r2, "aic": aic, "moran_I": morans_i(np.asarray(e).ravel(), Wd, GC["moran_permutations"], seed)[0],
                    "spatial_parameter": f"{extra[0]} {extra[1]:.3f}" if extra[0] else ""})
    return coef, fit
