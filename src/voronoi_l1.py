#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure
import matplotlib.pyplot as plt

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

def nearest_labels(points_xy, sites_xy, metric: str, chunk=50000):
    labels = np.empty(points_xy.shape[0], dtype=np.int32)
    for i in range(0, points_xy.shape[0], chunk):
        P = points_xy[i:i+chunk][:, None, :]
        S = sites_xy[None, :, :]
        D = P - S
        d = (np.abs(D).sum(axis=2) if metric == "l1"
             else np.sqrt((D**2).sum(axis=2)))
        labels[i:i+chunk] = d.argmin(axis=1)
    return labels

def rotate_clockwise(xy: np.ndarray, deg: float) -> np.ndarray:
    """Rotate (x,y) by +deg degrees clockwise."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, s], [-s, c]])
    return xy @ R.T

def main():
    ap = argparse.ArgumentParser(description="Generate Voronoi poster (L1 or L2) as SVG")
    ap.add_argument("--city", default="Manhattan, New York, USA")
    ap.add_argument("--query", default="railway=station")
    ap.add_argument("--metric", choices=["l1", "l2"], default="l1")
    ap.add_argument("--grid-res", type=float, default=20.0)
    ap.add_argument("--figsize", type=parse_figsize, default=(24.0, 36.0))
    ap.add_argument("--max_sites", type=int, default=1500)
    ap.add_argument("--out", default="out/poster.svg")
    ap.add_argument("--draw-sites", action="store_true")
    ap.add_argument("--bg", default="#f8f7f2")
    ap.add_argument("--stroke", default="#111111")
    ap.add_argument("--linewidth", type=float, default=0.6)
    ap.add_argument("--l1-rotate-deg", type=float, default=0.0,
                    help="Rotate axes clockwise (degrees) before L1 distance; ~29 for Manhattan")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Boundary (prefer detailed admin relation)
    print(f"[1/5] Getting boundary for: {args.city}")
    try:
        admin = ox.features_from_place(args.city, {"boundary": "administrative"}).to_crs(3857)
        geom = admin.geometry.unary_union
        city_poly = geom if geom.geom_type != "MultiPolygon" else max(geom.geoms, key=lambda g: g.area)
    except Exception:
        boundary = ox.geocode_to_gdf(args.city)[["geometry"]].to_crs(3857)
        city_poly = boundary.iloc[0].geometry

    # 2) Sites from OSM
    kv = parse_kv(args.query)
    print(f"[2/5] Fetching OSM features with {kv} …")
    city_poly_wgs84 = gpd.GeoSeries([city_poly], crs=3857).to_crs(4326).iloc[0]
    try:
        feats = ox.features_from_polygon(city_poly_wgs84, kv)
    except Exception as e:
        print(f"[skip] Overpass returned no data for {kv}: {e}")
        return

    # Reproject first, THEN centroid for non-point geometries
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

    # 3) Grid
    print(f"[3/5] Building grid at ~{args.grid_res} m …")
    minx, miny, maxx, maxy = city_poly.bounds
    xs = np.arange(minx, maxx, args.grid_res)
    ys = np.arange(miny, maxy, args.grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.c_[X.ravel(), Y.ravel()]

    # 4) Distance labels (with optional L1 rotation)
    print(f"[4/5] Assigning nearest site ({args.metric.upper()})…")
    S = np.c_[pts.geometry.x.values, pts.geometry.y.values]
    P = grid_pts
    if args.metric == "l1" and abs(args.l1_rotate_deg) > 1e-6:
        P = rotate_clockwise(P, args.l1_rotate_deg)
        S = rotate_clockwise(S, args.l1_rotate_deg)
    labels = nearest_labels(P, S, args.metric)
    label_grid = labels.reshape(Y.shape)

    # Mask outside boundary
    centers = gpd.GeoSeries(gpd.points_from_xy(grid_pts[:, 0], grid_pts[:, 1]), crs=3857)
    inside = centers.within(city_poly).values.reshape(Y.shape)
    label_grid = np.where(inside, label_grid, -1)

    # 5) Extract boundaries and plot
    print("[5/5] Extracting vector contours and rendering …")
    edges = np.zeros_like(label_grid, dtype=np.uint8)
    edges[:-1, :] |= (label_grid[:-1, :] != label_grid[1:, :]).astype(np.uint8)
    edges[:, :-1] |= (label_grid[:, :-1] != label_grid[:, 1:]).astype(np.uint8)
    contours = measure.find_contours(edges.astype(float), 0.5)

    fig = plt.figure(figsize=args.figsize, dpi=300); ax = plt.gca()
    ax.set_aspect("equal"); ax.set_facecolor(args.bg)

    for poly in polygons_list(city_poly):
        bx, by = poly.exterior.xy
        ax.plot(bx, by, linewidth=args.linewidth * 1.8, color=args.stroke)

    for c in contours:
        yy, xx = c[:, 0], c[:, 1]
        xw = xs[0] + (xx / (len(xs) - 1)) * (xs[-1] - xs[0])
        yw = ys[0] + (yy / (len(ys) - 1)) * (ys[-1] - ys[0])
        ax.plot(xw, yw, linewidth=args.linewidth, color=args.stroke, solid_joinstyle="miter")

    if args.draw_sites:
        ax.scatter(S[:, 0], S[:, 1], s=2, color=args.stroke, alpha=0.5)

    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(args.out, format="svg", bbox_inches="tight")
    print("Done.")

if __name__ == "__main__":
    main()

