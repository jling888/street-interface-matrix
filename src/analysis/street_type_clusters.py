"""Street types: k-means on the standardised dimensions of M, numbered by descending mean M.

Reports the types' silhouette and bootstrap stability, a one-way MANOVA of the dimensions on type (significant by
construction: the types were cut on them), and Tukey HSD between the types on measures independent of the clustering
(`clusters.tukey_on`). Writes outputs/<run_id>/analysis/street-type-clusters/: cluster-quality.csv,
cluster-per-view.csv,
cluster-centroids.csv, manova-test.csv and tukey-pairwise-tests.csv. Run by pipeline step cluster_street_types.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from common.data_readers import read_outcome
from common.node_table import node_table, read_nodes


def cluster(Z, k, seed):
    """!
    @brief K-means labels for the standardised columns Z.
    @return An integer label per row.
    """
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(Z)


def quality(Z, k, bootstrap, seed):
    """!
    @brief Silhouette, and stability: the mean adjusted Rand index between the full clustering and refits on
           `bootstrap` random 80% samples (1 = the same types every time).
    @return DataFrame of one row: k, silhouette, stability.
    """
    rng = np.random.default_rng(seed)
    full = cluster(Z, k, seed)
    ari = []
    for b in range(bootstrap):
        s = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
        ari.append(adjusted_rand_score(full[s], cluster(Z[s], k, seed + b + 1)))
    return pd.DataFrame([{"k": k, "silhouette": silhouette_score(Z, full), "stability": np.mean(ari)}])


def run(cfg, paths):
    """!
    @brief K-means street types on the run's scores, with MANOVA and Tukey HSD between them.
    """
    K = cfg["street_type_clusters"]
    d = pd.read_csv(paths.scores_csv)
    vol = read_outcome(paths)
    d = d.merge(vol.assign(ln_volume=np.log1p(vol.volume))[["node_id", "ln_volume"]], on="node_id", how="left")
    d = d.merge(pd.DataFrame(node_table(cfg, paths)[["node_id", "integration"]]), on="node_id", how="left")
    dims = list(cfg["metric"]["dimensions"])
    d = d.dropna(subset=dims).reset_index(drop=True)
    Z = ((d[dims] - d[dims].mean()) / d[dims].std()).to_numpy()

    q = quality(Z, K["k"], K["bootstrap"], cfg["seed"])
    q.to_csv(paths.cluster_quality_csv, index=False)
    print(q.round(3).to_string(index=False))
    labels = cluster(Z, K["k"], cfg["seed"])
    rank = pd.Series(d.M.groupby(labels).mean()).rank(ascending=False).astype(int)
    d["cluster"] = rank.reindex(labels).to_numpy()

    d = d.merge(read_nodes(paths)[["node_id", "typology"]], on="node_id", how="left")
    cent = d.groupby("cluster").agg(views=("file", "size"), nodes=("node_id", "nunique"),
                                    **{c: (c, "mean") for c in [*dims, "M", "ln_volume", "HW_effective"]})
    cent = cent.join(pd.crosstab(d.cluster, d.typology).add_prefix("n_"))
    print(cent.round(3).to_string())

    mv = MANOVA.from_formula(" + ".join(dims) + " ~ C(cluster)", data=d).mv_test()
    res = mv.results["C(cluster)"]["stat"].reset_index().rename(columns={"index": "test"})
    print(res.to_string(index=False))

    rows = []
    for c in K["tukey_on"]:
        x = d[[c, "cluster"]].dropna()
        t = pairwise_tukeyhsd(x[c], x.cluster, alpha=K["tukey_alpha"])
        g1, g2 = np.triu_indices(len(t.groupsunique), 1)     # the pairs, in the order of t's results
        # full precision (the printed summary rounds to 4 decimals, which zeroes integration's differences)
        rows.append(pd.DataFrame({"group1": t.groupsunique[g1], "group2": t.groupsunique[g2], "meandiff": t.meandiffs,
                                  "p-adj": t.pvalues, "lower": t.confint[:, 0], "upper": t.confint[:, 1],
                                  "reject": t.reject, "variable": c}))
    tukey = pd.concat(rows, ignore_index=True)
    print(tukey.to_string(index=False))

    d[["file", "node_id", "cluster"]].to_csv(paths.clusters_csv, index=False)
    cent.to_csv(paths.centroids_csv)
    res.to_csv(paths.manova_csv, index=False)
    tukey.to_csv(paths.tukey_csv, index=False)
