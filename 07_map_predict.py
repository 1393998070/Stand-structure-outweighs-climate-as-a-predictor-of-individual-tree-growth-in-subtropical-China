# -*- coding: utf-8 -*-
"""Phase F-2 省域制图：样地级栅格一致模型 → 250m 预测 + 5 折 ensemble 不确定性 + 格局方差分解。
输出: map/ 目录 GeoTIFF + 07_map_summary.json + 07_map_report.md + 07_map_pattern_varpart.csv
"""
import os
import sys
import time
import itertools
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform
from rasterio.transform import from_bounds
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.RASTER
MAPDIR = os.path.join(C.OUT, "map")
os.makedirs(MAPDIR, exist_ok=True)

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase F-2 省域制图报告\n")

# ---------- 栅格清单（制图可用 = 有对应栅格的变量） ----------
LAYER_FILE = {
    "stand_age_r": "林分\\林龄.tif",
    "stand_density": "林分\\林分密度.tif",
    "soil_SOC": "土壤\\广东林地土壤SOC.tif", "soil_TN": "土壤\\广东林地土壤TN.tif",
    "soil_TP": "土壤\\广东林地土壤TP.tif", "soil_Tk": "土壤\\广东林地土壤Tk.tif",
    "soil_pH": "土壤\\广东林地土壤pH.tif", "soil_BD": "土壤\\广东林地土壤容重.tif",
    "soil_depth": "土壤\\土层厚度.tif",
    "bio1": "气候\\peng_bio_1.tif", "bio2": "气候\\peng_bio_2.tif",
    "bio4": "气候\\peng_bio_4.tif", "bio12": "气候\\peng_bio_12.tif",
    "bio15": "气候\\peng_bio_15.tif", "bio18": "气候\\peng_bio_18.tif",
    "bio19": "气候\\peng_bio_19.tif",
    "dem": "地形水文参数\\GD_DEM.tif", "slope": "地形水文参数\\GD_SLP.tif",
    "aspect_deg": "地形水文参数\\GD_ASP.tif", "curvature_total": "地形水文参数\\GD_CU.tif",
    "dist_stream": "地形水文参数\\GD_DHN.tif", "relief": "地形水文参数\\GD_DXQFD.tif",
    "flow_length": "地形水文参数\\GD_FL.tif", "heat_load": "地形水文参数\\GD_HL.tif",
    "SPI": "地形水文参数\\GD_SPI.tif", "TPI": "地形水文参数\\GD_TPI.tif",
    "TSI": "地形水文参数\\GD_TSI.tif", "TWI": "地形水文参数\\GD_TWI.tif",
    "forest_prop": "林分\\全省林地分布栅格.tif",
}
MAP_FEATS = ["stand_age_r", "stand_density",
             "bio1", "bio2", "bio4", "bio12", "bio15", "bio18", "bio19",
             "soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth",
             "dem", "slope", "aspect_cos", "aspect_sin", "curvature_total", "dist_stream",
             "relief", "flow_length", "heat_load", "SPI", "TPI", "TSI", "TWI"]
GROUP_OF = {**{v: "stand" for v in ("stand_age_r", "stand_density")},
           **{v: "climate" for v in ("bio1", "bio2", "bio4", "bio12", "bio15", "bio18", "bio19")},
           **{v: "soil" for v in ("soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth")},
           **{v: "terrain" for v in ("dem", "slope", "aspect_cos", "aspect_sin", "curvature_total",
                                     "dist_stream", "relief", "flow_length", "heat_load", "SPI",
                                     "TPI", "TSI", "TWI")}}

# ---------- 1. 输出网格 ----------
with rasterio.open(os.path.join(R, LAYER_FILE["forest_prop"])) as msrc:
    W, S, E, N = msrc.bounds.left, msrc.bounds.bottom, msrc.bounds.right, msrc.bounds.top
RES = 0.0025  # ~250m
ow = int((E - W) / RES); oh = int((N - S) / RES)
otrans = from_bounds(W, S, W + ow * RES, S + oh * RES, ow, oh)
say(f"- 输出网格: {ow}×{oh} @ {RES}°（~250m），范围 [{W:.3f},{S:.3f},{E:.3f},{N:.3f}]")

def to_grid(path_or_arr, src_crs=None, src_transform=None, resampling=Resampling.average,
            dtype=np.float32, nodata=None):
    """把栅格文件或数组重投影到输出网格。"""
    dst = np.full((oh, ow), np.nan, dtype=np.float32)
    if isinstance(path_or_arr, str):
        with rasterio.open(path_or_arr) as src:
            reproject(rasterio.band(src, 1), dst,
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=otrans, dst_crs="EPSG:4326",
                      dst_nodata=np.nan, resampling=resampling)
    else:
        reproject(path_or_arr, dst, src_transform=src_transform, src_crs=src_crs,
                  src_nodata=nodata, dst_transform=otrans, dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=resampling)
    return dst

t0 = time.time()
# ---------- 2. 逐层重投影 ----------
say("\n## 1. 栅格重投影到统一网格")
grids = {}
for v in MAP_FEATS:
    if v == "aspect_cos" or v == "aspect_sin":
        continue
    grids[v] = to_grid(os.path.join(R, LAYER_FILE[v]))
    say(f"  {v}: 有效像元 {np.isfinite(grids[v]).sum()/grids[v].size*100:.1f}%")

# 坡向 → cos/sin（先读原栅格转 cos/sin，再均值重投影，避免 0/360 环绕）
with rasterio.open(os.path.join(R, LAYER_FILE["aspect_deg"])) as src:
    asp_full = src.read(1).astype(np.float32)
    nd = src.nodata
    asp_full[asp_full == (nd if nd is not None else -9999)] = np.nan
    src_transform, src_crs = src.transform, src.crs
grids["aspect_cos"] = to_grid(np.cos(np.deg2rad(asp_full)), src_crs=src_crs,
                              src_transform=src_transform, nodata=np.nan)
grids["aspect_sin"] = to_grid(np.sin(np.deg2rad(asp_full)), src_crs=src_crs,
                              src_transform=src_transform, nodata=np.nan)
del asp_full
say(f"  aspect_cos/sin 完成")

# 林地覆盖比例（average 重投影 = 250m 内林地占比）
fprop = to_grid(os.path.join(R, LAYER_FILE["forest_prop"]))
forest_mask = fprop >= 0.5
say(f"- 林地占比≥0.5 的像元: {forest_mask.sum()}（{forest_mask.mean()*100:.1f}%）")
say(f"（重投影用时 {time.time()-t0:.0f}s）")

# ---------- 3. 样地训练数据（栅格一致提取） ----------
say("\n## 2. 样地级训练数据")
tab, vd, folds = C.load_model_data()
plot_xy = tab.groupby("plot_id")[["lon", "lat"]].first()
plot_resp = tab.groupby("plot_id")["RGR"].mean().rename("plot_mean_RGR")
pf = folds.set_index("plot_id").reindex(plot_xy.index)
pdata = pd.DataFrame({"lon": plot_xy["lon"], "lat": plot_xy["lat"],
                      "fold_spatial": pf["fold_spatial"]}).join(plot_resp)

def sample_raster(var, path):
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.is_geographic:
            xs, ys = plot_xy["lon"].tolist(), plot_xy["lat"].tolist()
        else:
            xs, ys = warp_transform("EPSG:4326", src.crs,
                                    plot_xy["lon"].tolist(), plot_xy["lat"].tolist())
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        vals[vals == (src.nodata if src.nodata is not None else -9999)] = np.nan
    return pd.Series(vals, index=plot_xy.index)

for v in MAP_FEATS:
    if v in ("aspect_cos", "aspect_sin"):
        with rasterio.open(os.path.join(R, LAYER_FILE["aspect_deg"])) as src:
            if src.crs is None or src.crs.is_geographic:
                xs, ys = plot_xy["lon"].tolist(), plot_xy["lat"].tolist()
            else:
                xs, ys = warp_transform("EPSG:4326", src.crs,
                                        plot_xy["lon"].tolist(), plot_xy["lat"].tolist())
            a = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
            a[a == (src.nodata if src.nodata is not None else -9999)] = np.nan
        pdata["aspect_cos" if v == "aspect_cos" else "aspect_sin"] = (
            np.cos(np.deg2rad(a)) if v == "aspect_cos" else np.sin(np.deg2rad(a)))
    else:
        pdata[v] = sample_raster(v, os.path.join(R, LAYER_FILE[v]))
cc = pdata.dropna(subset=MAP_FEATS + ["plot_mean_RGR", "fold_spatial"])
say(f"- 完整案例样地: {len(cc)}/{len(pdata)}")
say(f"- 栅格年龄 vs CFI 年龄 Spearman: "
    f"{pd.Series(pdata['stand_age_r']).corr(pdata.groupby('plot_id').apply(lambda g: 0), method='spearman') if False else '见06报告'}")

# ---------- 4. 样地级 XGB（小随机搜索，空间块 CV） ----------
say("\n## 3. 样地级制图模型（250m 空间块 CV 随机搜索）")
rng = np.random.default_rng(C.SEED)
space = {"max_depth": [2, 3, 4, 5, 6], "learning_rate": [0.02, 0.05, 0.1],
         "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
         "min_child_weight": [1, 3, 8], "reg_lambda": [1, 3, 8, 15], "reg_alpha": [0, 0.5, 2]}
best_cfg, best_rmse = None, np.inf
for ci in range(12):
    params = {k: rng.choice(v) for k, v in space.items()}
    params = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
              for k, v in params.items()}
    oof = pd.Series(np.nan, index=cc.index)
    for k in range(5):
        tr = cc[cc["fold_spatial"] != k]; te = cc[cc["fold_spatial"] == k]
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60, eval_metric="rmse",
                             tree_method="hist", random_state=C.SEED + k, n_jobs=-1, **params)
        m.fit(tr[MAP_FEATS], tr["plot_mean_RGR"], eval_set=[(te[MAP_FEATS], te["plot_mean_RGR"])],
              verbose=False)
        oof.loc[te.index] = m.predict(te[MAP_FEATS])
    r2 = C.metrics(cc["plot_mean_RGR"], oof)["R2"]
    rmse = float(np.sqrt(np.mean((cc["plot_mean_RGR"] - oof) ** 2)))
    if rmse < best_rmse:
        best_rmse, best_cfg, best_oof = rmse, params, oof
    say(f"  cand {ci+1:02d}: R2={r2:.4f} rmse={rmse:.5f}")
map_r2 = C.metrics(cc["plot_mean_RGR"], best_oof)
say(f"- 最优: {best_cfg}")
say(f"- 制图模型空间块 CV: R2={map_r2['R2']:.4f}, rho={map_r2['Spearman']:.4f}, RMSE={map_r2['RMSE']:.5f}")

# 无年龄变体（敏感性：年龄代理质量）
feats_noage = [f for f in MAP_FEATS if f != "stand_age_r"]
oof_na = pd.Series(np.nan, index=cc.index)
for k in range(5):
    tr = cc[cc["fold_spatial"] != k]; te = cc[cc["fold_spatial"] == k]
    m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60, eval_metric="rmse",
                         tree_method="hist", random_state=C.SEED + k, n_jobs=-1, **best_cfg)
    m.fit(tr[feats_noage], tr["plot_mean_RGR"], eval_set=[(te[feats_noage], te["plot_mean_RGR"])],
          verbose=False)
    oof_na.loc[te.index] = m.predict(te[feats_noage])
r2_na = C.metrics(cc["plot_mean_RGR"], oof_na)["R2"]
say(f"- 无年龄制图模型空间块 CV: R2={r2_na:.4f}")

# ---------- 5. 全图预测（5 折 ensemble） ----------
say("\n## 4. 全图预测（5 折模型 ensemble：均值 + 离散度）")
valid = forest_mask.copy()
for v in MAP_FEATS:
    valid &= np.isfinite(grids[v])
say(f"- 可预测林地像元: {valid.sum()}（占林地 {valid.sum()/forest_mask.sum()*100:.1f}%）")
rows_idx, cols_idx = np.where(valid)
Xmap = np.column_stack([grids[v][rows_idx, cols_idx] for v in MAP_FEATS]).astype(np.float32)
preds = np.empty((5, len(Xmap)), dtype=np.float32)
ens_models = []
for k in range(5):
    tr = cc[cc["fold_spatial"] != k]; te = cc[cc["fold_spatial"] == k]
    m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60,
                         eval_metric="rmse", tree_method="hist",
                         random_state=C.SEED + k, n_jobs=-1, **best_cfg)
    m.fit(tr[MAP_FEATS], tr["plot_mean_RGR"],
          eval_set=[(te[MAP_FEATS], te["plot_mean_RGR"])], verbose=False)
    preds[k] = m.predict(Xmap)
    ens_models.append(m)
    say(f"  fold {k} 模型完成 (best_iter={m.best_iteration})")
map_mean = preds.mean(axis=0)
map_sd = preds.std(axis=0)
say(f"- 预测 RGR: mean={map_mean.mean():.5f} (per yr), sd={map_mean.std():.5f}, "
    f"P5={np.percentile(map_mean,5):.5f}, P95={np.percentile(map_mean,95):.5f}")
say(f"- 不确定性 (ensemble sd): mean={map_sd.mean():.5f}, P95={np.percentile(map_sd,95):.5f}")

# 写 GeoTIFF
def write_tif(arr2d, name, dtype=np.float32, nodata=np.nan):
    prof = dict(driver="GTiff", height=oh, width=ow, count=1, dtype=dtype,
                crs="EPSG:4326", transform=otrans, nodata=nodata, compress="lzw")
    with rasterio.open(os.path.join(MAPDIR, name), "w", **prof) as dst:
        dst.write(arr2d.astype(dtype), 1)

mean_grid = np.full((oh, ow), np.nan, dtype=np.float32); mean_grid[rows_idx, cols_idx] = map_mean
sd_grid = np.full((oh, ow), np.nan, dtype=np.float32); sd_grid[rows_idx, cols_idx] = map_sd
fprop_out = np.where(np.isfinite(fprop), fprop, np.nan).astype(np.float32)
write_tif(mean_grid, "RGR_mean_250m.tif")
write_tif(sd_grid, "RGR_sd_250m.tif")
write_tif(fprop_out, "forest_prop_250m.tif")
say(f"- 已写出: RGR_mean_250m.tif / RGR_sd_250m.tif / forest_prop_250m.tif")
np.savez_compressed(os.path.join(MAPDIR, "map_arrays.npz"),
                    mean=mean_grid, sd=sd_grid, forest_prop=fprop_out,
                    transform=np.array(otrans[:6]), rows=rows_idx, cols=cols_idx,
                    map_mean=map_mean, map_sd=map_sd, Xmap_sample=Xmap[rng.choice(
                        len(Xmap), size=min(100000, len(Xmap)), replace=False)] if len(Xmap) > 100000 else Xmap,
                    map_feats=np.array(MAP_FEATS), groups=np.array([GROUP_OF[f] for f in MAP_FEATS]))

# ---------- 6. 格局归因（预测值的 SHAP 类组占比，10 万像元样本） ----------
say("\n## 5. 预测格局归因（mean|SHAP| 类组占比）")
samp = rng.choice(len(Xmap), size=min(100000, len(Xmap)), replace=False)
dm = xgb.DMatrix(Xmap[samp], feature_names=MAP_FEATS)
sv = ens_models[0].get_booster().predict(dm, pred_contribs=True)[:, :-1]
gs_arr = np.abs(sv).mean(axis=0)
vp_rows = []
for g in ("stand", "climate", "soil", "terrain"):
    idx = [i for i, f in enumerate(MAP_FEATS) if GROUP_OF[f] == g]
    vp_rows.append({"group": g, "shap_share": round(float(gs_arr[idx].sum() / gs_arr.sum()), 4)})
vp = pd.DataFrame(vp_rows)
vp.to_csv(os.path.join(C.OUT, "07_map_pattern_varpart.csv"), index=False, encoding="utf-8-sig")
say(vp.to_string(index=False))
# 类组内 top 变量（制图模型格局）
top_map = pd.DataFrame({"var": MAP_FEATS, "group": [GROUP_OF[f] for f in MAP_FEATS],
                        "mean_abs_shap": gs_arr}).sort_values(
    "mean_abs_shap", ascending=False).head(10)
say("- 制图格局 top-10 变量:")
say(top_map.to_string(index=False))

C.save_json({"grid_res_deg": RES, "grid_w": ow, "grid_h": oh,
             "n_forest_px": int(forest_mask.sum()), "n_pred_px": int(valid.sum()),
             "map_model_R2_spatialCV": map_r2["R2"], "map_model_Spearman": map_r2["Spearman"],
             "map_model_RMSE": map_r2["RMSE"],
             "map_model_R2_noage": r2_na,
             "map_mean": float(map_mean.mean()), "map_sd_between_px": float(map_mean.std()),
             "ens_sd_mean": float(map_sd.mean()), "best_params": best_cfg},
            "07_map_summary.json")
say("\nDONE Phase F-2")
with open(os.path.join(C.OUT, "07_map_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
