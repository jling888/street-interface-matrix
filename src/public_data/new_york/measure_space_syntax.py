"""!
@package measure_space_syntax
@brief Space Syntax controls and network distances (paper 2.8): angular segment analysis at a metric radius.

On the undirected graph of street centrelines people can walk (OSM; no motorways or road tunnels; the nodes' extent
plus the largest radius, so values near the edge are not truncated), each segment between junctions is a place in the
graph, and a route costs the angle it turns: 90 degrees is a depth of 1. Routes are the least-angle ones, as long as
they
stay within radius_m metres along the route, midpoint to midpoint, as in depthmapX's segment analysis:
  choice       how many least-angle routes between other segments pass through this one
  integration  n^2 / total angular depth, n the segments within the radius, this one included
Each node takes the values of its nearest segment; network.extra_radii_m adds choice_<r> and integration_<r> at more
radii. Also writes the node-to-node walking distance matrix the Conley errors use.

Writes data/<city>/public_data/: space-syntax-per-node.csv (choice, integration and the snap distance per usable node)
and walking-distances-between-nodes.npz. Run by pipeline step measure_space_syntax.
"""
import heapq
import shutil
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from common.downloads import overpass
from common.node_table import read_nodes

TURN_BINS = 1024         # a half turn in this many steps: routes this close in angle are equal (depthmapX's tulip bins)
BEARING_SAMPLE_M = 10    # a segment's direction at a junction: towards its point this far along
STEP = 1 << 20           # a route's cost: angle bins, then segments, so equal angles take the fewer segments


def walking_network(cfg, paths, crs):
    """!
    @brief The OSM street centrelines people can walk (space_syntax_network.highways; no road tunnels), around the
           nodes and at least the largest radius beyond them, else trips from the edge nodes are cut off where the map
           ends; saved as walking-network.osm. Every sampled node sets the extent: which are usable is only
           known after the imagery (download_street_view). In Murray Hill both give the same box.
    """
    N = cfg["space_syntax_network"]
    nodes = gpd.read_file(paths.nodes_gpkg).to_crs(crs)
    buffer_m = max(N["buffer_m"], N["radius_m"], *(N.get("extra_radii_m") or []))
    w, s, e, n = nodes.buffer(buffer_m).to_crs(4326).total_bounds
    streets = f'["highway"~"^({"|".join(N["highways"])})$"]["tunnel"!~"yes"]["area"!~"yes"]["access"!~"private"]'
    cached = overpass(f"(way{streets}({s:.6f},{w:.6f},{n:.6f},{e:.6f});>;);out;", cfg, paths, paths.public_data_osm_cache, "xml")
    if not paths.walking_network.exists() or paths.walking_network.read_bytes() != cached.read_bytes():
        shutil.copyfile(cached, paths.walking_network)


def build_graph(path, crs):
    """!
    @brief The street graph of the walking network (walking-network.osm).
    @return (graph, edges GeoDataFrame u, v, geometry, vertices GeoDataFrame).
    """
    import osmnx as ox
    G = ox.graph_from_xml(path, simplify=True)
    G = ox.project_graph(G, to_crs=crs).to_undirected()
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()[["u", "v", "geometry"]]
    return G, edges[edges.u != edges.v].reset_index(drop=True), ox.graph_to_gdfs(G, edges=False)


def segment_turns(edges, vertices):
    """!
    @brief The ways on from each segment. A segment has two states, one per direction of travel: 2 i is along it from
           its u to its v, 2 i + 1 back. Turning round on a segment is not a way on.
    @param edges     GeoDataFrame u, v, geometry, in metres.
    @param vertices  Frame of x, y by vertex.
    @return ({state: [(next state, angle bins turned)]}, segment lengths in metres).
    """
    leaving = defaultdict(list)                    # vertex: (state leaving it, its direction there)
    arriving = {}                                  # state: (vertex it reaches, its direction on arrival)
    for i, r in enumerate(edges.itertuples()):
        line, length = r.geometry, r.geometry.length
        if np.hypot(line.coords[0][0] - vertices.x[r.u], line.coords[0][1] - vertices.y[r.u]) > np.hypot(
                line.coords[0][0] - vertices.x[r.v], line.coords[0][1] - vertices.y[r.v]):
            line = line.reverse()                  # OSM's direction may run from v to u
        for state, start, end, frm, to in ((2 * i, 0.0, length, r.u, r.v), (2 * i + 1, length, 0.0, r.v, r.u)):
            step = min(BEARING_SAMPLE_M, length)
            sign = 1 if end > start else -1
            p, q = line.interpolate(start), line.interpolate(start + sign * step)
            leaving[frm].append((state, np.arctan2(q.y - p.y, q.x - p.x)))
            p, q = line.interpolate(end - sign * step), line.interpolate(end)
            arriving[state] = (to, np.arctan2(q.y - p.y, q.x - p.x))
    turns = {}
    for state, (vertex, heading) in arriving.items():
        turns[state] = []
        for other, direction in leaving[vertex]:
            if other // 2 != state // 2:
                angle = abs((direction - heading + np.pi) % (2 * np.pi) - np.pi)
                turns[state].append((other, round(angle / np.pi * TURN_BINS)))
    return turns, edges.geometry.length.to_numpy()


def least_angle_routes(turns, lengths, s, radius):
    """!
    @brief The least-angle routes from segment s, within `radius` metres along the route.
    @return (cost, sigma (routes), pred, order), by state.
    """
    cost, metres, sigma, pred = {}, {}, defaultdict(float), defaultdict(list)
    heap = [(0, 2 * s), (0, 2 * s + 1)]
    for _, state in heap:
        cost[state], metres[state], sigma[state] = 0, 0.0, 1.0
    seen, order = set(), []
    while heap:
        c, state = heapq.heappop(heap)
        if state in seen:
            continue
        seen.add(state)
        order.append(state)
        for other, bins in turns[state]:
            length = metres[state] + (lengths[state // 2] + lengths[other // 2]) / 2
            if length > radius or other // 2 == s:
                continue
            new = c + bins * STEP + 1
            if new < cost.get(other, np.inf):
                cost[other], metres[other], sigma[other], pred[other] = new, length, sigma[state], [state]
                heapq.heappush(heap, (new, other))
            elif new == cost[other]:
                metres[other] = min(metres[other], length)
                sigma[other] += sigma[state]
                pred[other].append(state)
    return cost, sigma, pred, order


def choice_integration(turns, lengths, radius):
    """!
    @brief Angular choice and integration of every segment, at a metric radius (Brandes' accumulation on the states).
    @return (choice, integration), arrays by segment.
    """
    choice, integration = np.zeros(len(lengths)), np.zeros(len(lengths))
    for s in range(len(lengths)):
        cost, sigma, pred, order = least_angle_routes(turns, lengths, s, radius)
        best = {}                                  # segment: its cheaper direction's cost
        for state, c in cost.items():
            best[state // 2] = min(c, best.get(state // 2, c))
        depth = sum(c // STEP for c in best.values()) / (TURN_BINS / 2)         # 90 degrees = 1
        integration[s] = len(best) ** 2 / depth if depth > 0 else 0.0
        ends = defaultdict(list)                   # a segment reached by both directions at equal cost shares its count
        for state in order:
            if state // 2 != s and cost[state] == best[state // 2]:
                ends[state // 2].append(state)
        target = {t: sigma[t] / sum(sigma[x] for x in states) for states in ends.values() for t in states}
        through = defaultdict(float)
        for state in reversed(order):
            for p in pred[state]:
                through[p] += sigma[p] / sigma[state] * (target.get(state, 0.0) + through[state])
            if state // 2 != s:
                choice[state // 2] += through[state]
    return choice / 2, integration                 # each route counted from both ends


def run(cfg, paths):
    """!
    @brief Choice and integration per usable node, and the walking distances between them.
    """
    N, crs = cfg["space_syntax_network"], int(cfg["projected_crs"])
    nodes = gpd.read_file(paths.nodes_gpkg).to_crs(crs)
    nodes = nodes[nodes.node_id.isin(read_nodes(paths).query("usable").node_id)].reset_index(drop=True)
    walking_network(cfg, paths, crs)
    G, edges, verts = build_graph(paths.walking_network, crs)
    print(f"graph: {len(G)} vertices, {len(edges)} segments")
    turns, lengths = segment_turns(edges, verts)

    near = gpd.sjoin_nearest(nodes[["node_id", "geometry"]], edges, distance_col="snap_m")
    near = near.drop_duplicates("node_id").set_index("node_id").reindex(nodes.node_id)
    out = pd.DataFrame({"node_id": nodes.node_id})
    # choice / integration at radius_m, and choice_<r> / integration_<r> at each of extra_radii_m
    for r, suffix in [(N["radius_m"], ""), *((r, f"_{r}") for r in N.get("extra_radii_m") or [])]:
        choice, integration = choice_integration(turns, lengths, r)
        out["choice" + suffix] = choice[near.index_right.to_numpy()]
        out["integration" + suffix] = integration[near.index_right.to_numpy()]
        print(f"  radius {r} m done")
    out["snap_m"] = near.snap_m.values
    out.to_csv(paths.space_syntax_csv, index=False)
    print(out.drop(columns=["node_id", "snap_m"]).describe().round(4).to_string())

    # node-to-node walking distance: snap to the nearer end of the nearest segment
    vxy = verts[["x", "y"]]
    ends = []
    for r, g in zip(near.itertuples(), nodes.geometry):
        du, dv = (np.hypot(vxy.loc[k, "x"] - g.x, vxy.loc[k, "y"] - g.y) for k in (r.u, r.v))
        ends.append((r.u, du) if du <= dv else (r.v, dv))
    D = np.zeros((len(nodes), len(nodes)))
    for i, (src, off) in enumerate(ends):
        dist = nx.single_source_dijkstra_path_length(G, src, weight="length")
        D[i] = [off + dist.get(t, np.inf) + o2 for t, o2 in ends]
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0.0)
    np.savez_compressed(paths.walking_distances, node_id=nodes.node_id.to_numpy(), D=D)
    print(f"wrote space-syntax-per-node.csv and walking-distances-between-nodes.npz ({len(nodes)} x {len(nodes)}); "
          f"unreachable pairs: {int(np.isinf(D).sum())}")
