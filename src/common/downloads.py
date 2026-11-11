"""!
@package downloads
@brief How steps download: every step fetches its own sources, through these, and records each file it fetched.

  record         one row per file in data/<city>/downloaded-sources.csv: source, file, url, when, size, sha256; a
                 file whose content changes gets a new row, so the log is the experiment's data provenance
  overpass       OpenStreetMap through Overpass, as of osm_date, cached by query (steps sample_street_nodes and
                 measure_space_syntax)
  socrata_*      an open-data portal's (Socrata) rows around the nodes: points by lat_field / lon_field, shapes by
                 geometry_field (step measure_city_records's city records, step measure_street_canyons's footprints)
  download_file  a whole file (step measure_pedestrian_volume's NYCWalks)
"""
import hashlib
import sys
import time
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape


# ---------------------------------------------------------------- the log
def record(paths, source, path, url):
    """!
    @brief Add `path` to the download log unless the log already has it with this content.
    """
    log = paths.downloads_log
    seen = pd.read_csv(log, dtype=str) if log.exists() else pd.DataFrame(columns=["file", "sha256"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rel = str(path.relative_to(paths.city_data))
    if ((seen.file == rel) & (seen.sha256 == digest)).any():
        return
    when = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = pd.DataFrame([{"source": source, "file": rel, "url": url, "downloaded_utc": when,
                         "bytes": path.stat().st_size, "sha256": digest}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)


# ---------------------------------------------------------------- OpenStreetMap
OVERPASS = "https://overpass-api.de/api/interpreter"           # OpenStreetMap contributors, ODbL 1.0


def overpass(query, cfg, paths, cache, fmt="json"):
    """!
    @brief Run an Overpass query as of `osm_date`.

    Always query by bounding box: Overpass times out on dated polygon queries (which is what osmnx sends). The
    response stays in `cache` (the asking stage's downloads/openstreetmap-cache/), named by the query's hash, as the
    record of the OSM data used.
    @return The cached response file.
    """
    osm_date, url = cfg.get("osm_date"), OVERPASS
    date = f'[date:"{osm_date}"]' if osm_date else ""
    q = f"[out:{fmt}][timeout:180]{date};{query}"
    path = cache / f"osm_{hashlib.sha1(q.encode()).hexdigest()[:10]}.{fmt}"
    for attempt in range(5):
        if path.exists():
            break
        try:
            r = requests.post(url, data={"data": q}, timeout=300,
                              headers={"User-Agent": "street-interface-matrix/1.0 (research)"})
            r.raise_for_status()
            path.write_bytes(r.content)
        except requests.RequestException as ex:   # Overpass rate-limits; wait and retry
            print(f"  Overpass attempt {attempt + 1} failed: {ex}")
            time.sleep(60)
    if not path.exists():
        sys.exit("Overpass unreachable")
    record(paths, "openstreetmap", path, url)
    return path


# ---------------------------------------------------------------- open-data portals (Socrata)
# Every dataset has a URL, https://<portal>/resource/<dataset id>.json, that answers SoQL queries: $where filters rows
# (within_box for a geometry column), $select picks columns, $limit / $offset page through the answer.
PAGE = 5000                                         # rows per request


def socrata_url(portal, dataset):
    """!
    @brief A dataset's URL on an open-data portal (data.cityofnewyork.us, data.ny.gov, ...).
    """
    return f"https://{portal}/resource/{dataset}.json"


def around_nodes(nodes, margin_deg):
    """!
    @brief The nodes' box in degrees, grown by margin_deg on every side.
    @return (south, west, north, east).
    """
    w, s, e, n = nodes.to_crs(4326).total_bounds
    return s - margin_deg, w - margin_deg, n + margin_deg, e + margin_deg


def socrata_rows(url, params):
    """!
    @brief Every row a Socrata query answers, a page at a time. Stops the run on any failed request, so a dataset is
           never saved with pages missing.
    """
    rows = []
    while True:
        r = requests.get(url, params={**params, "$limit": PAGE, "$offset": len(rows)}, timeout=180)
        r.raise_for_status()
        page = r.json()
        if not page:
            return rows
        rows += page


def first_answering(urls):
    """!
    @brief The first dataset URL that answers with data: NYC republishes some datasets (the footprints) under new ids.
    """
    for url in urls:
        try:
            r = requests.get(url, params={"$limit": 1}, timeout=30)
            if r.status_code == 200 and r.json():
                return url
        except requests.RequestException:
            pass
    sys.exit(f"no dataset answered: {urls}")


def socrata_points(url, S, nodes, path, margin_deg=0.01):
    """!
    @brief Save a dataset's points around the nodes as CSV: the rows whose lat_field / lon_field lie in the nodes' box
           and match the source's `where`, with those two columns and its `fields`.
    """
    s, w, n, e = around_nodes(nodes, S.get("margin_deg", margin_deg))
    lat, lon = S["lat_field"], S["lon_field"]
    in_box = f"{lat} between {s} and {n} and {lon} between {w} and {e}"
    rows = socrata_rows(url, {"$select": ",".join([lat, lon, *S.get("fields", {}).values()]),
                              "$where": " and ".join(filter(None, [in_box, S.get("where")]))})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"{len(rows)} rows from {url} -> {path}")


def socrata_shapes(urls, S, nodes, path, margin_deg=0.01):
    """!
    @brief Save a dataset's shapes around the nodes as GeoJSON: the rows whose geometry_field lies in the nodes' box
           and match the source's `where`, from the first of `urls` that answers.
    @return The URL used.
    """
    s, w, n, e = around_nodes(nodes, S.get("margin_deg", margin_deg))
    geom = S["geometry_field"]
    in_box = f"within_box({geom}, {n}, {w}, {s}, {e})"
    url = first_answering(urls)
    rows = socrata_rows(url, {"$where": " and ".join(filter(None, [in_box, S.get("where")]))})
    # every column but the geometry and Socrata's own (:@...)
    columns = [{k: v for k, v in row.items() if k != geom and not k.startswith(":@")} for row in rows]
    shapes = gpd.GeoDataFrame(columns, geometry=[shape(row[geom]) for row in rows], crs=4326)
    shapes.to_file(path, driver="GeoJSON")
    print(f"{len(rows)} shapes from {url} -> {path}")
    return url


def download_file(url, path):
    """!
    @brief Download `url` to `path`.
    """
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(path)
    print(f"downloaded {path} ({path.stat().st_size / 1e6:.0f} MB)")
