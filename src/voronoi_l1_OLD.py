#!/usr/bin/env python3
# Generate an L1 (Manhattan) Voronoi diagram for a city from OSM points.
# Exports a poster-ready SVG.

import argparse
import os
import re
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
        raise argparse.ArgumentTypeError("query must be key=value, e.g. railway=station")
    k, v = query.split("=", 1)
    return {k.strip(): v.strip()}

def nearest_labels_L1(points_xy: np.ndarray, sites_xy: np.ndarray, chunk: int = 50000) -> np.ndarray:
    labels = np.empty(points_xy.shape[0], dtype=np.int32)
    for i in range(0, points_xy.shape[0], chunk):
        P = points_xy[i:i+chunk]
        diff = P[:, None, :] - sites_xy[None, :, :]
        d = np.abs(diff).sum(axis=2)  # L1 distance
        labels[i:i+chunk] = d.argmin(axis=1)
    return labels

def polygons_list(geom):
    """Return a list of Polygons from a Polygon or MultiPolygon."""
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    # Fallback: unify and try again
    u = unary_union(geom)
    return list(u.geoms) if isinstance(u, MultiPolygon) else [u]

def main():
    ap = argparse.ArgumentParser(description="Generate Manhattan (L1) Voronoi as SVG")
    ap.add_argument("--city", default="Manhattan, New York, USA")
    ap.add_argument("--query", default="railway=station")
    ap.add_argument("--grid-res", type=float, default=20.0, help="meters between grid samples")
    ap.add_argument("--figsize", type=parse_figsize, default=(24.0, 36.0), help='WxH inches, e.g. "24x36"')
    ap.add_argument("--max_sites", type=int, default=1500, help="cap sites for speed (0 = no cap)")
    ap.add_argument("--out", default="out/manhattan_voronoi_L1.svg")
    ap.add_argument("--draw-sites", action="store_true")
    ap.add_argument("--bg", default="#f8f7f2")
    ap.add_argument("--stroke", default="#111111")
    ap.add_argument("--linewidth", type=float, default=0.6)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Boundary -> EPSG:3857 (meters)
    print(f"[1/5] Geocoding boundary for: {args.city}")
    boundary = ox.geocode_to_gdf(args.city)[["geometry"]].to_crs(3857)
    # Some places return MultiPolygon; keep as-is
    city_poly = boundary.iloc[0].geometry

    # 2) Sites from OSM (query must use WGS84 polygon)
    kv = parse_kv(args.query)
    print(f"[2/5] Fetching OSM features with {kv} …")
    city_poly_wgs84 = gpd.GeoSeries([city_poly], crs=3857).to_crs(4326).iloc[0]
    feats = ox.features_from_polygon(city_poly_wgs84, kv)

    # Use centroids for non-point features, then project to 3857
    geom = feats.geometry
    if geom.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"]).any():
        # (Warning about centroid in geographic CRS is fine; we reproject right after.)
        geom = geom.centroid
    pts = gpd.GeoDataFrame(geometry=geom, crs=feats.crs).to_crs(3857).dropna()

    # Keep points within the city polygon(s)
    pts = pts[pts.within(city_poly)]
    pts = pts.drop_duplicates(subset=["geometry"])

    if args.max_sites and len(pts) > args.max_sites:
        print(f"Thinning sites: {len(pts)} → {args.max_sites}")
        pts = pts.sample(args.max_sites, random_state=42)

    if len(pts) < 3:
        raise SystemExit("Need at least 3 sites. Try a different query or city.")

    print(f"Using {len(pts)} sites.")

    # 3) Grid over city bounds (meters)
    print(f"[3/5] Building grid at ~{args.grid_res} m …")
    minx, miny, maxx, maxy = city_poly.bounds
    xs = np.arange(minx, maxx, args.grid_res)
    ys = np.arange(miny, maxy, args.grid_res)
    X, Y = np.meshgrid(xs, ys)
    grid_pts = np.c_[X.ravel(), Y.ravel()]

    # 4) L1 assignment
    print("[4/5] Assigning nearest site (L1)…")
    S = np.c_[pts.geometry.x.values, pts.geometry.y.values]
    labels = nearest_labels_L1(grid_pts, S, chunk=50000)
    label_grid = labels.reshape(Y.shape)

    # Mask outside city (sample grid cell centers)
    centers = gpd.GeoSeries(gpd.points_from_xy(grid_pts[:, 0], grid_pts[:, 1]), crs=3857)
    inside = centers.within(city_poly).values.reshape(Y.shape)
    label_grid = np.where(inside, label_grid, -1)

    # 5) Extract boundaries as contours where neighboring labels differ
    print("[5/5] Extracting vector contours…")
    edges = np.zeros_like(label_grid, dtype=np.uint8)
    edges[:-1, :] |= (label_grid[:-1, :] != label_grid[1:, :]).astype(np.uint8)
    edges[:, :-1] |= (label_grid[:, :-1] != label_grid[:, 1:]).astype(np.uint8)
    contours = measure.find_contours(edges.astype(float), 0.5)

    # Plot to SVG
    print(f"Rendering SVG → {args.out}")
    fig = plt.figure(figsize=args.figsize, dpi=300)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_facecolor(args.bg)

    # City outline (handle Polygon or MultiPolygon)
    for poly in polygons_list(city_poly):
        bx, by = poly.exterior.xy
        ax.plot(bx, by, linewidth=args.linewidth * 1.4, color=args.stroke)

    # Voronoi boundaries
    for c in contours:
        yy, xx = c[:, 0], c[:, 1]  # contour in (row, col)
        xw = xs[0] + (xx / (len(xs) - 1)) * (xs[-1] - xs[0])
        yw = ys[0] + (yy / (len(ys) - 1)) * (ys[-1] - ys[0])
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
