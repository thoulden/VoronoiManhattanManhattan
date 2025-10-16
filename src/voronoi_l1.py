#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Keep Overpass polite but avoid hanging forever
ox.settings.overpass_rate_limit = True
ox.settings.overpass_timeout = 180
ox.settings.requests_timeout = 180


# ---------- helpers ----------

def parse_figsize(s: str):
    m = re.match(r"^\s*(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError("figsize must be like '24x36'")
    return float(m.group(1)), float(m.group(2))

def parse_kv(query: str):
    if "=" not in query:
        raise argparse.ArgumentTypeError("query must be key=value")
    k, v = query.split("=", 1)
    return {k.strip(): v.strip()}

def polygons_list(geom):
    if isinstance(geom, Polygon): return [geom]
    if isinstance(geom, MultiPolygon): return list(geom.geoms)
    u = unary_union(geom)
    return list(u.geoms) if isinstance(u, MultiPolygon) else [u]

def rotate_clockwise(xy: np.ndarray, deg: float) -> np.ndarray:
    """Rotate points xy by +deg degrees clockwise."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, s], [-s, c]], dtype=np.float32)
    return xy @ R.T

def nearest_labels(points_xy: np.ndarray, sites_xy: np.ndarray, metric: str) -> np.ndarray:
    """
    Assign each point in points_xy the index of its nearest site in sites_xy.
    Uses cKDTree (fast, low-memory). We already applied the L1 axis rotation
    upstream; here we just do nearest neighbors with Minkowski p.
    """
    P = np.asarray(points_xy, dtype=np.float32, order="C")
    S = np.asarray(sites_xy, dtype=np.float32, order="C")
    p = 1 if metric == "l1" else 2
    tree = cKDTree(S)
    try:
        _, idx = tree.query(P, k=1, p=p, workers=-1)  # SciPy ≥1.6
    except TypeError:
        _, idx = tree.query(P, k=1, p=p)
    return idx.astype(np.int32, copy=False)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Generate Voronoi poster (L1 or L2) as SVG")
    ap.add_argument("--city", default="New York County, New York, USA")
    ap.add_argument("--query", default="railway=station")
    ap.add_argument("--metric", choices=["l1", "l2"], default="l1")
    ap.add_argument("--grid-res", type=float, default=10.0, help="meters between grid samples")
    ap.add_argument("--figsize", type=parse_figsize, default=(24.0, 36.0))
    ap.add_argument("--max_sites", type=int, default=1500)
    ap.add_argument("--out", default="out/poster.svg")
    ap.add_argument("--draw-sites", action="store_true")
    ap.add_argument("--bg", default="#f8f7f2")
    ap.add_argument("--stroke", default="#111111")
    ap.add_argument("--linewidth", type=float, default=0.6)
    ap.add_argument("--l1-rotate-deg", type=float, default=29.0,
                    help="Rotate axes clockwise before L1 (≈29 for Manhattan)")
    ap.add_argument("--row-block", type=int, default=2000,
                    help="rows per block when rasterizing (controls memory)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Boundary (tight working bbox + crisp admin shape if available)
    print(f"[1/5] Getting boundary for: {args.city}")

    place_gdf = ox.geocode_to_gdf(args.city)[["geometry"]].to_crs(3857)
    place_poly = place_gdf.iloc[0].geometry

    try:
        admin = ox.features_from_place(args.city, {"boundary": "administrative"}).to_crs(3857)
        admin = admin[admin.geometry.intersects(place_poly)]
        if len(admin):
            overlap_area = admin.geometry.intersection(place_poly).area
            city_poly = admin.loc[overlap_area.idxmax(), "geometry"]
        else:
            city_poly = place_poly
    except Exception:
        city_poly = place_poly

    # 2) Sites from OSM (query by place → no giant Overpass polygons)
    kv = parse_kv(args.query)
    print(f"[2/5] Fetching OSM features for {kv} in {args.city} …")
    try:
        feats = ox.features_from_place(args.city, kv)
    except Exception as e:
        print(f"[skip] Overpass returned no data for {kv}: {e}")
        return

    feats_3857 = feats.to_crs(3857)
    geom = feats_3857.geometry
    if geom.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"]).any():
        geom = geom.centroid
    pts = gpd.GeoDataFrame(geometry=geom, crs=3857).dropna()
    pts = pts[pts.within(city_poly)].drop_duplicates(subset=["geometry"])

    if args.max_sites and len(pts) > args.max_sites:
        print(f"Thinning sites: {len(pts)} → {args.max_sites}")
        pts = pts.sample(args.max_sites, random_state=42)

    if len(pts) < 3:
        print(f"[skip] Only {len(pts)} site(s) for {kv}; not generating {args.out}")
        return

    print(f"Using {len(pts)} sites.")

    # Sites array (float32)
    S = np.c_[pts.geometry.x.values, pts.geometry.y.values].astype(np.float32)
    # Optional axis rotation for L1
    if args.metric == "l1" and abs(args.l1_rotate_deg) > 1e-6:
        S = rotate_clockwise(S, args.l1_rotate_deg)

    # 3) Grid (row-blocked to avoid huge allocations)
    print(f"[3/5] Building grid at ~{args.grid_res} m …")
    minx, miny, maxx, maxy = place_poly.bounds
    xs = np.arange(minx, maxx, args.grid_res, dtype=np.float32)
    ys = np.arange(miny, maxy, args.grid_res, dtype=np.float32)
    H, W = len(ys), len(xs)
    label_grid = np.empty((H, W), dtype=np.int32)

    # 4) Nearest site (KD-tree) in blocks
    print(f"[4/5] Assigning nearest site ({args.metric.upper()})…")
    row_block = max(1, int(args.row_block))

    for r0 in range(0, H, row_block):
        r1 = min(r0 + row_block, H)
        rows = r1 - r0

        # Build block points as [x,y] without a giant mesh:
        # repeat ys[r0:r1] across columns, tile xs across rows
        block_y = np.repeat(ys[r0:r1], W)
        block_x = np.tile(xs, rows)
        P = np.column_stack((block_x, block_y)).astype(np.float32)

        if args.metric == "l1" and abs(args.l1_rotate_deg) > 1e-6:
            P = rotate_clockwise(P, args.l1_rotate_deg)

        labels = nearest_labels(P, S, args.metric)
        label_grid[r0:r1, :] = labels.reshape(rows, W)

    # Mask outside boundary using the *detailed* city polygon
    centers_inside = np.zeros((H, W), dtype=bool)
    for r0 in range(0, H, row_block):
        r1 = min(r0 + row_block, H)
        rows = r1 - r0
        block_y = np.repeat(ys[r0:r1], W)
        block_x = np.tile(xs, rows)
        block_pts = gpd.GeoSeries(gpd.points_from_xy(block_x, block_y), crs=3857)
        centers_inside[r0:r1, :] = block_pts.within(city_poly).values.reshape(rows, W)

    label_grid = np.where(centers_inside, label_grid, -1)

    # 5) Extract boundaries and plot
    print("[5/5] Extracting contours and rendering …")
    edges = np.zeros_like(label_grid, dtype=np.uint8)
    edges[:-1, :] |= (label_grid[:-1, :] != label_grid[1:, :]).astype(np.uint8)
    edges[:, :-1] |= (label_grid[:, :-1] != label_grid[:, 1:]).astype(np.uint8)
    contours = measure.find_contours(edges.astype(float), 0.5)

    fig = plt.figure(figsize=args.figsize, dpi=300)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_facecolor(args.bg)

    for poly in polygons_list(city_poly):
        bx, by = poly.exterior.xy
        ax.plot(bx, by, linewidth=args.linewidth * 1.8, color=args.stroke)

    for c in contours:
        yy, xx = c[:, 0], c[:, 1]
        xw = xs[0] + (xx / (W - 1)) * (xs[-1] - xs[0]) if W > 1 else xs[0]
        yw = ys[0] + (yy / (H - 1)) * (ys[-1] - ys[0]) if H > 1 else ys[0]
        ax.plot(xw, yw, linewidth=args.linewidth, color=args.stroke, solid_joinstyle="miter")

    if args.draw_sites:
        ax.scatter(S[:, 0], S[:, 1], s=2, color=args.stroke, alpha=0.5)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print("Done.")

if __name__ == "__main__":
    main()

