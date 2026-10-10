# -*- coding: utf-8 -*-
"""Phase F-1 制图准备：栅格档案检查 + 样地点提取核验（栅格值 vs 样地表值一致性）。
输出: 06_raster_profiles.csv / 06_extract_check.csv / 06_map_prep_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.RASTER
report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase F-1 制图准备报告\n")

# ---------- 1. 栅格档案 ----------
layers = {
    # var: (path, rasterio resampling 用途说明)
    "soil_SOC": ("土壤\\广东林地土壤SOC.tif",),
    "soil_TN": ("土壤\\广东林地土壤TN.tif",),
    "soil_TP": ("土壤\\广东林地土壤TP.tif",),
    "soil_Tk": ("土壤\\广东林地土壤Tk.tif",),
    "soil_pH": ("土壤\\广东林地土壤pH.tif",),
    "soil_BD": ("土壤\\广东林地土壤容重.tif",),
    "soil_depth": ("土壤\\土层厚度.tif",),
    "dem": ("地形水文参数\\GD_DEM.tif",),
    "slope": ("地形水文参数\\GD_SLP.tif",),
    "aspect_deg": ("地形水文参数\\GD_ASP.tif",),
    "curvature_total": ("地形水文参数\\GD_CU.tif",),
    "dist_stream": ("地形水文参数\\GD_DHN.tif",),
    "relief": ("地形水文参数\\GD_DXQFD.tif",),
    "flow_length": ("地形水文参数\\GD_FL.tif",),
    "heat_load": ("地形水文参数\\GD_HL.tif",),
    "SPI": ("地形水文参数\\GD_SPI.tif",),
    "TPI": ("地形水文参数\\GD_TPI.tif",),
    "TSI": ("地形水文参数\\GD_TSI.tif",),
    "TWI": ("地形水文参数\\GD_TWI.tif",),
    "stand_density_r": ("林分\\林分密度.tif",),
    "stand_age_r": ("林分\\林龄.tif",),
    "forest_mask": ("林分\\全省林地分布栅格.tif",),
}
for i in (1, 2, 4, 12, 15, 18, 19):
    layers[f"bio{i}"] = (f"气候\\peng_bio_{i}.tif",)

rows = []
for var, (rel,) in layers.items():
    path = os.path.join(R, rel)
    with rasterio.open(path) as src:
        rows.append({"var": var, "file": rel,
                     "width": src.width, "height": src.height,
                     "crs": str(src.crs), "dtype": str(src.dtypes[0]),
                     "res_x": round(src.res[0], 6), "res_y": round(src.res[1], 6),
                     "nodata": src.nodata})
prof = pd.DataFrame(rows)
prof.to_csv(os.path.join(C.OUT, "06_raster_profiles.csv"), index=False, encoding="utf-8-sig")
say(prof[["var", "width", "height", "crs", "res_x", "nodata"]].to_string(index=False))

# ---------- 2. 样地点提取核验 ----------
say("\n## 样地点提取核验（栅格值 vs 样地表已提取值）")
tab, vd, folds = C.load_model_data()
plot_xy = tab.groupby("plot_id")[["lon", "lat"]].first()
plot_tab = tab.drop_duplicates("plot_id").set_index("plot_id")

check_pairs = [
    ("soil_SOC", "soil_SOC"), ("soil_TN", "soil_TN"), ("soil_TP", "soil_TP"),
    ("soil_Tk", "soil_Tk"), ("soil_pH", "soil_pH"), ("soil_BD", "soil_BD"),
    ("soil_depth", "soil_depth"),
    ("dem", "dem"), ("slope", "slope"), ("TWI", "TWI"), ("TPI", "TPI"),
    ("bio1", "bio1"), ("bio12", "bio12"),
    ("stand_density_r", "stand_density"), ("stand_age_r", "stand_age"),
]
ck_rows = []
for rv, pv in check_pairs:
    path = os.path.join(R, layers[rv][0])
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(
            list(zip(plot_xy["lon"], plot_xy["lat"])))], dtype=float)
    vals[vals == src.nodata if src.nodata is not None else False] = np.nan
    tabv = plot_tab[pv].reindex(plot_xy.index).values.astype(float)
    ok = ~(np.isnan(vals) | np.isnan(tabv))
    if ok.sum() > 10:
        rho = spearmanr(vals[ok], tabv[ok]).statistic
        agree = np.mean(np.isclose(vals[ok], tabv[ok], rtol=1e-3, atol=1e-3))
    else:
        rho, agree = np.nan, np.nan
    ck_rows.append({"raster": rv, "plot_col": pv, "n_ok": int(ok.sum()),
                    "spearman": round(float(rho), 4) if rho == rho else None,
                    "pct_near_equal": round(float(agree) * 100, 1) if agree == agree else None})
ck = pd.DataFrame(ck_rows)
ck.to_csv(os.path.join(C.OUT, "06_extract_check.csv"), index=False, encoding="utf-8-sig")
say(ck.to_string(index=False))

# ---------- 2b. 投影坐标系层的 CRS 感知核验 ----------
say("\n## 投影层核验（坐标转换后采样）")
from rasterio.warp import transform as warp_transform

proj_pairs = [
    ("soil_depth", "soil_depth"), ("dem", "dem"), ("slope", "slope"),
    ("TWI", "TWI"), ("TPI", "TPI"), ("SPI", "SPI"), ("heat_load", "heat_load"),
    ("relief", "relief"), ("dist_stream", "dist_stream"),
    ("flow_length", "flow_length"), ("curvature_total", "curvature_total"),
    ("stand_density_r", "stand_density"),
]
ck2_rows = []
for rv, pv in proj_pairs:
    path = os.path.join(R, layers[rv][0])
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.is_geographic:
            xs, ys = plot_xy["lon"].values, plot_xy["lat"].values
        else:
            xs, ys = warp_transform("EPSG:4326", src.crs,
                                    plot_xy["lon"].tolist(), plot_xy["lat"].tolist())
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        vals[vals == (src.nodata if src.nodata is not None else np.nan)] = np.nan
    tabv = plot_tab[pv].reindex(plot_xy.index).values.astype(float)
    ok = ~(np.isnan(vals) | np.isnan(tabv))
    if ok.sum() > 10:
        rho = spearmanr(vals[ok], tabv[ok]).statistic
        agree = np.mean(np.isclose(vals[ok], tabv[ok], rtol=1e-3, atol=1e-3))
    else:
        rho, agree = np.nan, np.nan
    ck2_rows.append({"raster": rv, "plot_col": pv, "n_ok": int(ok.sum()),
                     "spearman": round(float(rho), 4) if rho == rho else None,
                     "pct_near_equal": round(float(agree) * 100, 1) if agree == agree else None})
ck2 = pd.DataFrame(ck2_rows)
ck2.to_csv(os.path.join(C.OUT, "06_extract_check_proj.csv"), index=False, encoding="utf-8-sig")
say(ck2.to_string(index=False))

say("\nDONE Phase F-1")
with open(os.path.join(C.OUT, "06_map_prep_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
