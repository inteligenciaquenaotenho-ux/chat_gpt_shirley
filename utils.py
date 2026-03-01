from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union


def log(message: str) -> None:
    print(f"[LOG] {message}")


def qc(name: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"[QC:{status}] {name}{suffix}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_bairro(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    txt = str(value).upper().strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def to_utm_valid(gdf: gpd.GeoDataFrame, crs_utm: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if out.crs is None:
        raise ValueError("Layer sem CRS definido.")
    if str(out.crs) != crs_utm:
        out = out.to_crs(crs_utm)
    out["geometry"] = out.geometry.make_valid()
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    return out


def clip_to_muni(gdf: gpd.GeoDataFrame, muni: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    out = gpd.clip(gdf, muni)
    out["geometry"] = out.geometry.make_valid()
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    return out


def extract_lines(geom) -> list[LineString]:
    lines: list[LineString] = []
    if geom is None or geom.is_empty:
        return lines
    if isinstance(geom, LineString):
        if len(geom.coords) >= 2 and geom.length > 0 and geom.coords[0] != geom.coords[-1]:
            lines.append(geom)
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            lines.extend(extract_lines(part))
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            lines.extend(extract_lines(part))
    elif hasattr(geom, "boundary"):
        lines.extend(extract_lines(geom.boundary))
    return lines


def clean_line_layer(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for geom in gdf.geometry:
        rows.extend(extract_lines(geom))
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    return gpd.GeoDataFrame(geometry=rows, crs=gdf.crs)


def polygonize_cells(lines_gdf: gpd.GeoDataFrame, muni: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    merged = unary_union(lines_gdf.geometry.tolist())
    polys = list(polygonize(merged))
    cells = gpd.GeoDataFrame({"geometry": polys}, crs=lines_gdf.crs)
    cells = gpd.clip(cells, muni)
    cells = cells[~cells.geometry.is_empty & cells.geometry.notna()].copy()
    cells["cell_id"] = np.arange(1, len(cells) + 1)
    return cells


def choose_region_by_overlap(cells: gpd.GeoDataFrame, regions: gpd.GeoDataFrame, region_col: str) -> gpd.GeoDataFrame:
    overlay = gpd.overlay(
        cells[["cell_id", "geometry"]], regions[[region_col, "geometry"]], how="intersection"
    )
    overlay["ov_area"] = overlay.geometry.area
    best = (
        overlay.sort_values(["cell_id", "ov_area"], ascending=[True, False])
        .drop_duplicates("cell_id")[["cell_id", region_col]]
    )
    out = cells.merge(best, on="cell_id", how="left")
    return out


def mode_or_empty(values: Iterable[str]) -> str:
    ser = pd.Series(list(values), dtype="string").dropna()
    ser = ser[ser != ""]
    if ser.empty:
        return ""
    return str(ser.mode().iloc[0])


def fill_missing_by_neighbors(
    cells: gpd.GeoDataFrame,
    value_col: str,
    max_iter: int = 15,
) -> gpd.GeoDataFrame:
    out = cells.copy()
    for _ in range(max_iter):
        missing_mask = out[value_col].fillna("") == ""
        if not missing_mask.any():
            break
        progress = 0
        missing_ids = out.loc[missing_mask, "cell_id"].tolist()
        for cid in missing_ids:
            geom = out.loc[out["cell_id"] == cid, "geometry"].iloc[0]
            neighbors = out[out.geometry.touches(geom)]
            values = neighbors[value_col].dropna().tolist()
            values = [v for v in values if str(v).strip() != ""]
            if values:
                out.loc[out["cell_id"] == cid, value_col] = mode_or_empty(values)
                progress += 1
        if progress == 0:
            break
    return out


def build_adjacency(cells: gpd.GeoDataFrame) -> dict[int, list[tuple[int, float]]]:
    adjacency: dict[int, list[tuple[int, float]]] = {int(cid): [] for cid in cells["cell_id"].tolist()}
    sindex = cells.sindex
    recs = cells[["cell_id", "geometry"]].reset_index(drop=True)
    for i, row in recs.iterrows():
        cid = int(row["cell_id"])
        geom = row.geometry
        cand_idx = list(sindex.intersection(geom.bounds))
        for j in cand_idx:
            if j <= i:
                continue
            other = recs.iloc[j]
            oid = int(other["cell_id"])
            og = other.geometry
            if not geom.touches(og):
                continue
            shared = geom.boundary.intersection(og.boundary)
            weight = shared.length
            if weight <= 0:
                continue
            adjacency[cid].append((oid, weight))
            adjacency[oid].append((cid, weight))
    return adjacency


def connected_components_for_key(
    cell_ids: set[int], adjacency: dict[int, list[tuple[int, float]]]
) -> list[set[int]]:
    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in cell_ids:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.add(node)
            for nxt, _ in adjacency.get(node, []):
                if nxt in cell_ids and nxt not in seen:
                    stack.append(nxt)
        comps.append(comp)
    return comps


def explode_singleparts(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.explode(index_parts=False, ignore_index=True)
    out["geometry"] = out.geometry.make_valid()
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    return out


def sum_holes_area_km2(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    hole_area = 0.0
    if geom.geom_type == "Polygon":
        hole_area += sum(abs(Polygon(ring).area) for ring in geom.interiors)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            hole_area += sum(abs(Polygon(ring).area) for ring in poly.interiors)
    return hole_area / 1_000_000
