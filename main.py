from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from utils import (
    build_adjacency,
    choose_region_by_overlap,
    clean_line_layer,
    clip_to_muni,
    connected_components_for_key,
    ensure_dir,
    explode_singleparts,
    fill_missing_by_neighbors,
    log,
    mode_or_empty,
    normalize_bairro,
    polygonize_cells,
    qc,
    to_utm_valid,
)


CRS_UTM = "EPSG:31983"
BASE_DIR = Path("/content/drive/MyDrive")
OUT_DIR = BASE_DIR / "R_GERADOR_DE_BAIRROS" / "_OUT_BAIRROS"
OUT_GPKG = OUT_DIR / "result_bairros_maua.gpkg"
OUT_CSV = OUT_DIR / "metricas_bairros.csv"

FILES = {
    "cnefe_gpkg": BASE_DIR / "R_GERADOR_DE_BAIRROS" / "CNEFE_MAUA.gpkg",
    "cnefe_layer": "pontos_cnefe_2022",
    "principais_bairros": BASE_DIR
    / "ANÁLISE DE DADOS ADMINISTRATIVOS"
    / "BANCO SIG Fonte Prefeitura 2019"
    / "Export"
    / "Parcelamento"
    / "MAUA_Principais_Bairros.shp",
    "regioes_planejamento": BASE_DIR
    / "ANÁLISE DE DADOS ADMINISTRATIVOS"
    / "BANCO SIG Fonte Prefeitura 2019"
    / "Export"
    / "Planejamento"
    / "MAUA_Regioes_Planejamento.shp",
    "logradouros": BASE_DIR / "R_GERADOR_DE_BAIRROS" / "Logradouros_eixos.shp",
    "ferrovia_gpkg": BASE_DIR / "R_GERADOR_DE_BAIRROS" / "trabalho_final.gpkg",
    "ferrovia_layer": "ferrovia",
    "parques": BASE_DIR / "R_GERADOR_DE_BAIRROS" / "MAUA_Parques.shp",
    "hidrografia": BASE_DIR
    / "ANÁLISE DE DADOS ADMINISTRATIVOS"
    / "BANCO SIG Fonte Prefeitura 2019"
    / "Export"
    / "BASE_2010"
    / "MAUA_Hidrografia_2010.shp",
}


def read_inputs() -> dict[str, gpd.GeoDataFrame]:
    log("Carregando camadas...")
    layers = {
        "regioes": gpd.read_file(FILES["regioes_planejamento"]),
        "cnefe": gpd.read_file(FILES["cnefe_gpkg"], layer=FILES["cnefe_layer"]),
        "logradouros": gpd.read_file(FILES["logradouros"]),
        "ferrovia": gpd.read_file(FILES["ferrovia_gpkg"], layer=FILES["ferrovia_layer"]),
        "hidrografia": gpd.read_file(FILES["hidrografia"]),
        "parques": gpd.read_file(FILES["parques"]),
        "validacao": gpd.read_file(FILES["principais_bairros"]),
    }
    for key, gdf in layers.items():
        layers[key] = to_utm_valid(gdf, CRS_UTM)
        empty_rate = 1 - (len(layers[key]) / max(1, len(gdf)))
        qc(f"{key} CRS {CRS_UTM}", str(layers[key].crs) == CRS_UTM)
        qc(f"{key} geometria vazia <=5%", empty_rate <= 0.05, f"taxa={empty_rate:.2%}")
    return layers


def dissolve_muni(regioes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    muni = regioes.dissolve().reset_index(drop=True)
    muni = to_utm_valid(muni, CRS_UTM)
    qc("len(muni)==1", len(muni) == 1, f"len={len(muni)}")
    return muni


def build_area_util(muni: gpd.GeoDataFrame, parques: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    parques_diss = parques.dissolve().reset_index(drop=True)
    parques_diss = clip_to_muni(parques_diss, muni)
    area_util_geom = muni.geometry.iloc[0].difference(unary_union(parques_diss.geometry.tolist()))
    area_util = gpd.GeoDataFrame({"geometry": [area_util_geom]}, crs=CRS_UTM)
    qc("area_util > 0", area_util.area.iloc[0] > 0)
    return area_util


def pick_region_col(regioes: gpd.GeoDataFrame) -> str:
    for candidate in ["region_id", "REGION_ID", "ID", "id"]:
        if candidate in regioes.columns:
            return candidate
    regioes["region_id"] = np.arange(1, len(regioes) + 1)
    return "region_id"


def assign_cnefe_to_cells(cells: gpd.GeoDataFrame, cnefe: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float]:
    cnefe = cnefe.copy()
    cnefe["bairro_norm"] = cnefe["DSC_LOCALIDADE"].map(normalize_bairro)
    cnefe["dom_w"] = 1

    pts_cells = gpd.sjoin(
        cnefe[["bairro_norm", "dom_w", "geometry"]],
        cells[["cell_id", "geometry"]],
        how="left",
        predicate="within",
    )
    coverage = pts_cells["cell_id"].notna().mean() if len(pts_cells) else 0.0

    grouped = (
        pts_cells.dropna(subset=["cell_id"])
        .groupby("cell_id")
        .agg(
            bairro_mode=("bairro_norm", mode_or_empty),
            dom_sum=("dom_w", "sum"),
            pts_n=("dom_w", "count"),
        )
        .reset_index()
    )

    out = cells.merge(grouped, on="cell_id", how="left")
    out["bairro_mode"] = out["bairro_mode"].fillna("")
    out["dom_sum"] = out["dom_sum"].fillna(0)
    out["pts_n"] = out["pts_n"].fillna(0)

    out = fill_missing_by_neighbors(out, value_col="bairro_mode")
    return out, float(coverage)


def fix_connectivity(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = cells.copy()
    adjacency = build_adjacency(out)
    max_iter = 10
    for it in range(max_iter):
        reassignments = 0
        for key in sorted(out["bairro_key"].dropna().unique().tolist()):
            key_ids = set(out.loc[out["bairro_key"] == key, "cell_id"].astype(int).tolist())
            if not key_ids:
                continue
            comps = connected_components_for_key(key_ids, adjacency)
            if len(comps) <= 1:
                continue
            largest = max(comps, key=len)
            for comp in comps:
                if comp == largest:
                    continue
                for cid in comp:
                    neigh = adjacency.get(int(cid), [])
                    best_key = None
                    best_weight = -1.0
                    for nid, w in neigh:
                        nkey = out.loc[out["cell_id"] == nid, "bairro_key"].iloc[0]
                        if nkey == key:
                            continue
                        if w > best_weight:
                            best_weight = w
                            best_key = nkey
                    if best_key is not None:
                        out.loc[out["cell_id"] == cid, "bairro_key"] = best_key
                        reassignments += 1
        log(f"Iteracao conectividade {it + 1}: reassignment={reassignments}")
        if reassignments == 0:
            break
    return out


def fill_gaps(bairros: gpd.GeoDataFrame, muni: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    out = bairros.copy()
    union_geom = unary_union(out.geometry.tolist())
    diff = muni.geometry.iloc[0].difference(union_geom)
    gaps = gpd.GeoDataFrame(geometry=[diff], crs=CRS_UTM)
    gaps = explode_singleparts(gaps)
    gaps = gaps[gaps.area > 1].copy()

    if gaps.empty:
        return out, gaps

    for _, gap in gaps.iterrows():
        ggeom = gap.geometry
        best_idx = None
        best_len = -1.0
        for idx, row in out.iterrows():
            shared = ggeom.boundary.intersection(row.geometry.boundary).length
            if shared > best_len:
                best_len = shared
                best_idx = idx
        if best_idx is not None:
            out.at[best_idx, "geometry"] = out.at[best_idx, "geometry"].union(ggeom)

    out["geometry"] = out.geometry.make_valid()
    out = out.dissolve(by="bairro_key", as_index=False, aggfunc="sum")
    return out, gaps


def metrics_table(bairros: gpd.GeoDataFrame, area_util: gpd.GeoDataFrame) -> pd.DataFrame:
    util_parts = gpd.overlay(
        bairros[["bairro_key", "dom_sum", "geometry"]],
        area_util[["geometry"]],
        how="intersection",
    )
    util_area = util_parts.groupby("bairro_key").geometry.area.sum().rename("area_util_m2")
    m = bairros[["bairro_key", "dom_sum", "geometry"]].copy()
    m["area_total_km2"] = m.geometry.area / 1_000_000
    m = m.merge(util_area, on="bairro_key", how="left")
    m["area_util_m2"] = m["area_util_m2"].fillna(0)
    m["area_util_km2"] = m["area_util_m2"] / 1_000_000
    m["flag_dom"] = np.where(m["dom_sum"] < 800, "LOW", np.where(m["dom_sum"] > 2400, "HIGH", "OK"))
    m["flag_area_util"] = np.where(
        m["area_util_km2"] < 0.3,
        "LOW",
        np.where(m["area_util_km2"] > 1.2, "HIGH", "OK"),
    )
    return pd.DataFrame(m.drop(columns="geometry"))


def validate_points(validacao: gpd.GeoDataFrame, bairros: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(validacao, bairros[["bairro_key", "geometry"]], how="left", predicate="within")
    joined["ok_in_polygon"] = joined["bairro_key"].notna()
    coverage = joined["ok_in_polygon"].mean() if len(joined) else 0.0
    qc("validacao pontos em bairro >=90%", coverage >= 0.90, f"taxa={coverage:.2%}")

    centroids = bairros[["bairro_key", "geometry"]].copy()
    centroids["centroid"] = centroids.geometry.centroid
    joined = joined.merge(centroids[["bairro_key", "centroid"]], on="bairro_key", how="left")
    joined["dist_centro_m"] = joined.geometry.distance(joined["centroid"])
    return joined.drop(columns=["centroid"])


def run_pipeline() -> None:
    ensure_dir(OUT_DIR)
    layers = read_inputs()

    regioes = layers["regioes"].copy()
    region_col = pick_region_col(regioes)
    if region_col != "region_id":
        regioes = regioes.rename(columns={region_col: "region_id"})

    muni = dissolve_muni(regioes)

    for key in ["cnefe", "logradouros", "ferrovia", "hidrografia", "parques", "validacao", "regioes"]:
        layers[key] = clip_to_muni(layers[key], muni)

    area_util = build_area_util(muni, layers["parques"])

    barreiras = gpd.GeoDataFrame(
        pd.concat(
            [
                clean_line_layer(layers["logradouros"]),
                clean_line_layer(layers["ferrovia"]),
                clean_line_layer(layers["hidrografia"]),
                clean_line_layer(muni),
                clean_line_layer(layers["regioes"]),
            ],
            ignore_index=True,
        ),
        crs=CRS_UTM,
    )
    qc("n_linhas_limpas > 0", len(barreiras) > 0, f"n={len(barreiras)}")
    _ = unary_union(barreiras.geometry.tolist())
    qc("unary_union barreiras", True)

    cells = polygonize_cells(barreiras, muni)
    qc("len(cells)>0", len(cells) > 0, f"n={len(cells)}")

    muni_minus_cells = muni.geometry.iloc[0].difference(unary_union(cells.geometry.tolist()))
    miss_area_km2 = muni_minus_cells.area / 1_000_000
    qc("muni-union(cells)<0.05km2", miss_area_km2 < 0.05, f"area={miss_area_km2:.4f}")

    cells = choose_region_by_overlap(cells, layers["regioes"].rename(columns={"region_id": "region_id"}), "region_id")
    fill_region = cells["region_id"].notna().mean() if len(cells) else 0
    qc("cells region_id >=99%", fill_region >= 0.99, f"taxa={fill_region:.2%}")

    cells, pts_cov = assign_cnefe_to_cells(cells, layers["cnefe"])
    filled_bairro = (cells["bairro_mode"].fillna("") != "").mean() if len(cells) else 0
    qc("cells bairro_mode >=99%", filled_bairro >= 0.99, f"taxa={filled_bairro:.2%}")
    qc("pontos em cells >=95%", pts_cov >= 0.95, f"taxa={pts_cov:.2%}")

    cells["bairro_key"] = "R" + cells["region_id"].fillna(-1).astype(int).astype(str) + "__" + cells["bairro_mode"]
    fill_key = (cells["bairro_key"].fillna("") != "").mean() if len(cells) else 0
    qc("bairro_key preenchido >=99%", fill_key >= 0.99, f"taxa={fill_key:.2%}")

    cells = fix_connectivity(cells)

    bairros_raw = cells.dissolve(by="bairro_key", as_index=False, aggfunc={"dom_sum": "sum", "pts_n": "sum"})
    bairros_raw["geometry"] = bairros_raw.geometry.make_valid()
    qc("len(bairros_raw)>0", len(bairros_raw) > 0, f"n={len(bairros_raw)}")

    bairros_fixed, qc_gaps = fill_gaps(bairros_raw, muni)
    leftover = muni.geometry.iloc[0].difference(unary_union(bairros_fixed.geometry.tolist())).area / 1_000_000
    qc("muni-union(bairros_fixed)<0.001km2", leftover < 0.001, f"area={leftover:.6f}")

    bairros_fixed = explode_singleparts(bairros_fixed)
    non_multi = (bairros_fixed.geom_type != "MultiPolygon").all()
    qc("sem multiparte", bool(non_multi))

    metrics = metrics_table(bairros_fixed, area_util)
    metrics.to_csv(OUT_CSV, index=False)

    validacao = validate_points(layers["validacao"], bairros_fixed)

    log("Salvando camadas no GPKG...")
    muni.to_file(OUT_GPKG, layer="muni_limite", driver="GPKG")
    area_util.to_file(OUT_GPKG, layer="area_util_mask", driver="GPKG")
    barreiras.to_file(OUT_GPKG, layer="barreiras_unificadas", driver="GPKG")
    cells.to_file(OUT_GPKG, layer="cells_atribuicao", driver="GPKG")
    cells[["cell_id", "geometry"]].to_file(OUT_GPKG, layer="cells", driver="GPKG")
    bairros_raw.to_file(OUT_GPKG, layer="bairros_raw", driver="GPKG")
    bairros_fixed.to_file(OUT_GPKG, layer="bairros_fixed", driver="GPKG")
    qc_gaps.to_file(OUT_GPKG, layer="qc_gaps", driver="GPKG")
    validacao.to_file(OUT_GPKG, layer="validacao_prefeitura", driver="GPKG")
    gpd.GeoDataFrame(metrics).to_file(OUT_GPKG, layer="metricas_bairros", driver="GPKG")

    log(f"Pipeline finalizado. GPKG: {OUT_GPKG}")
    log(f"CSV de metricas: {OUT_CSV}")


if __name__ == "__main__":
    run_pipeline()
