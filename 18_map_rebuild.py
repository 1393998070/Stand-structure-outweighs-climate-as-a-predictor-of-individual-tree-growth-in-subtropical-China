# -*- coding: utf-8 -*-
"""Stage 4 · P0-1 制图部分整体重构。
(1) 嵌套空间 CV 选参（外层 5 空间折 × 内层 4 空间折 × 12 候选）→ 诚实 R²；
(2) CFI 实测年龄对照实验（DA 核心实验）：raster-age vs CFI-age vs no-age 三变体
    的样地级 OOF R² 与 SHAP 类组份额 → 检验 terrain 31.7% 是否输入劣化伪影；
(3) 最终制图：全局内层选参 + 5 折 ensemble（早停不再用外层测试折）；
    不确定性层更名 model_disagreement + RMSE 预测区间口径；
(4) 压缩量化：观测样地 RGR sd vs 制图预测 sd。
输出: map/ 更新 GeoTIFF + 18_map_rebuild_report.md / 18_age_control.csv / 07_map_summary.json 更新
"""
import os
import sys
import time
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

say("# P0-1 制图重构报告\n")
t0 = time.time()

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
GROUP_OF = {**{v: "stand" for v in ("stand_age_r", "stand_age_cfi", "stand_density")},
           **{v: "climate" for v in ("bio1", "bio2", "bio4", "bio12", "bio15", "bio18", "bio19")},
           **{v: "soil" for v in ("soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth")},
           **{v: "terrain" for v in ("dem", "slope", "aspect_cos", "aspect_sin", "curvature_total",
                                     "dist_stream", "relief", "flow_length", "heat_load", "SPI",
                                     "TPI", "TSI", "TWI")}}

# ---------- 1. 输出网格与重投影 ----------
with rasterio.open(os.path.join(R, LAYER_FILE["forest_prop"])) as msrc:
    W, S, E, N = msrc.bounds.left, msrc.bounds.bottom, msrc.bounds.right, msrc.bounds.top
RES = 0.0025
ow = int((E - W) / RES); oh = int((N - S) / RES)
otrans = from_bounds(W, S, W + ow * RES, S + oh * RES, ow, oh)

def to_grid(path_or_arr, src_crs=None, src_transform=None, resampling=Resampling.average):
    dst = np.full((oh, ow), np.nan, dtype=np.float32)
    if isinstance(path_or_arr, str):
        with rasterio.open(path_or_arr) as src:
            reproject(rasterio.band(src, 1), dst,
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=otrans, dst_crs="EPSG:4326",
                      dst_nodata=np.nan, resampling=resampling)
    else:
        reproject(path_or_arr, dst, src_transform=src_transform, src_crs=src_crs,
                  src_nodata=np.nan, dst_transform=otrans, dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=resampling)
    return dst

say("## 1. 栅格重投影（同原流程）")
grids = {}
for v in MAP_FEATS:
    if v in ("aspect_cos", "aspect_sin"):
        continue
    grids[v] = to_grid(os.path.join(R, LAYER_FILE[v]))
with rasterio.open(os.path.join(R, LAYER_FILE["aspect_deg"])) as src:
    asp_full = src.read(1).astype(np.float32)
    nd = src.nodata
    asp_full[asp_full == (nd if nd is not None else -9999)] = np.nan
    src_transform, src_crs = src.transform, src.crs
grids["aspect_cos"] = to_grid(np.cos(np.deg2rad(asp_full)), src_crs=src_crs,
                              src_transform=src_transform)
grids["aspect_sin"] = to_grid(np.sin(np.deg2rad(asp_full)), src_crs=src_crs,
                              src_transform=src_transform)
del asp_full
fprop = to_grid(os.path.join(R, LAYER_FILE["forest_prop"]))
forest_mask = fprop >= 0.5
say(f"- 林地像元(≥0.5): {forest_mask.sum():,}")
say(f"（重投影用时 {time.time()-t0:.0f}s）")

# ---------- 2. 样地训练数据（栅格一致提取） ----------
say("\n## 2. 样地级训练数据（栅格一致提取）")
tab, vd, folds = C.load_model_data()
plot_xy = tab.groupby("plot_id")[["lon", "lat"]].first()
plot_resp = tab.groupby("plot_id")["RGR"].mean().rename("plot_mean_RGR")
cfi_age = tab.groupby("plot_id")["stand_age"].first().rename("stand_age_cfi")
pf = folds.set_index("plot_id").reindex(plot_xy.index)
pdata = pd.DataFrame({"lon": plot_xy["lon"], "lat": plot_xy["lat"],
                      "fold_spatial": pf["fold_spatial"]}).join(plot_resp).join(cfi_age)

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
        continue
    pdata[v] = sample_raster(v, os.path.join(R, LAYER_FILE[v]))
with rasterio.open(os.path.join(R, LAYER_FILE["aspect_deg"])) as src:
    if src.crs is None or src.crs.is_geographic:
        xs, ys = plot_xy["lon"].tolist(), plot_xy["lat"].tolist()
    else:
        xs, ys = warp_transform("EPSG:4326", src.crs,
                                plot_xy["lon"].tolist(), plot_xy["lat"].tolist())
    a = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
    a[a == (src.nodata if src.nodata is not None else -9999)] = np.nan
pdata["aspect_cos"] = np.cos(np.deg2rad(a))
pdata["aspect_sin"] = np.sin(np.deg2rad(a))

cc = pdata.dropna(subset=MAP_FEATS + ["plot_mean_RGR", "fold_spatial"]).copy()
say(f"- 完整案例样地: {len(cc)}/{len(pdata)}")
say(f"- raster age vs CFI age (cc): Spearman ρ = "
    f"{cc['stand_age_r'].corr(cc['stand_age_cfi'], method='spearman'):.3f}")
obs_sd = cc["plot_mean_RGR"].std()
say(f"- 观测 plot_mean_RGR: sd={obs_sd:.5f}, P5–P95=[{cc['plot_mean_RGR'].quantile(.05):.4f}, "
    f"{cc['plot_mean_RGR'].quantile(.95):.4f}]")
pd.to_pickle(pdata, os.path.join(C.OUT, "18_plot_raster_data.pkl"))

# ---------- 3. 嵌套空间 CV（诚实 R²） ----------
say("\n## 3. 嵌套空间 CV（外层 5 空间折 × 内层 4 空间折 × 12 候选）")
rng = np.random.default_rng(C.SEED)
space = {"max_depth": [2, 3, 4, 5, 6], "learning_rate": [0.02, 0.05, 0.1],
         "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
         "min_child_weight": [1, 3, 8], "reg_lambda": [1, 3, 8, 15], "reg_alpha": [0, 0.5, 2]}
cands = []
for ci in range(12):
    p = {k: rng.choice(v) for k, v in space.items()}
    cands.append({k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
                  for k, v in p.items()})

def inner_spatial_folds(train_df, n_folds=4, seed=0):
    blocks = train_df.assign(b=C.block_id(train_df["lon"].values, train_df["lat"].values).values)
    blk_size = blocks.groupby("b").size().sort_values(ascending=False)
    r2 = np.random.default_rng(seed)
    order = r2.permutation(len(blk_size))
    assign = C.assign_spatial_folds([list(blk_size.items())[i] for i in order], n_folds=n_folds)
    return blocks["b"].map(assign)

def fit_eval(train_df, test_df, feats, params, n_est=None):
    if n_est is None:
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60, eval_metric="rmse",
                             tree_method="hist", random_state=C.SEED, n_jobs=-1, **params)
        # 早停验证集：训练集内按样地抽 20%（不用测试折）
        ev = train_df.sample(frac=0.2, random_state=C.SEED)
        tr = train_df.drop(ev.index)
        m.fit(tr[feats], tr["plot_mean_RGR"], eval_set=[(ev[feats], ev["plot_mean_RGR"])],
              verbose=False)
        return m, m.predict(test_df[feats])
    m = xgb.XGBRegressor(n_estimators=n_est, tree_method="hist", random_state=C.SEED,
                         n_jobs=-1, **params)
    m.fit(train_df[feats], train_df["plot_mean_RGR"])
    return m, m.predict(test_df[feats])

t1 = time.time()
nested_pred = pd.Series(np.nan, index=cc.index)
sel_params_per_fold = []
for k in range(5):
    train = cc[cc["fold_spatial"] != k]
    test = cc[cc["fold_spatial"] == k]
    inner = inner_spatial_folds(train, n_folds=4, seed=C.SEED + k)
    best_rmse, best_p, best_iter = np.inf, None, None
    for ci, p in enumerate(cands):
        inner_oof = pd.Series(np.nan, index=train.index)
        iters = []
        for j in range(4):
            itr = train[inner != j]; ite = train[inner == j]
            m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60, eval_metric="rmse",
                                 tree_method="hist", random_state=C.SEED + j, n_jobs=-1, **p)
            ev = itr.sample(frac=0.2, random_state=C.SEED + j)
            m.fit(itr.drop(ev.index)[MAP_FEATS], itr.drop(ev.index)["plot_mean_RGR"],
                  eval_set=[(ev[MAP_FEATS], ev["plot_mean_RGR"])], verbose=False)
            inner_oof.loc[ite.index] = m.predict(ite[MAP_FEATS])
            iters.append(m.best_iteration + 1 if m.best_iteration is not None else 2000)
        rmse = float(np.sqrt(np.nanmean((train["plot_mean_RGR"] - inner_oof) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_p, best_iter = rmse, p, int(np.median(iters))
    _, pred = fit_eval(train, test, MAP_FEATS, best_p, n_est=best_iter)
    nested_pred.loc[test.index] = pred
    sel_params_per_fold.append({"fold": k, "params": best_p, "n_est": best_iter,
                                "inner_rmse": round(best_rmse, 5)})
    say(f"- outer fold {k}: best inner RMSE={best_rmse:.5f}, n_est={best_iter}")

nested_m = C.metrics(cc["plot_mean_RGR"], nested_pred)
nested_rmse = float(np.sqrt(np.mean((cc["plot_mean_RGR"] - nested_pred) ** 2)))
say(f"- 嵌套 OOF（诚实）: R2={nested_m['R2']:.4f}, rho={nested_m['Spearman']:.4f}, RMSE={nested_rmse:.5f}")
say(f"- 对照原同折选参报告 R2=0.0394（虚高方向：+{(0.0394-nested_m['R2'])*1000:.1f}‰）")
say(f"（嵌套用时 {time.time()-t1:.0f}s）")

# ---------- 4. 年龄对照实验（DA 核心实验） ----------
say("\n## 4. 年龄输入对照（raster-age vs CFI-age vs no-age，样地级 OOF SHAP 份额）")
orig_cfg = {'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.85, 'colsample_bytree': 0.6,
            'min_child_weight': 8, 'reg_lambda': 3.0, 'reg_alpha': 0.5}

def variant_oof_shap(df, feats, tag):
    oof = pd.Series(np.nan, index=df.index)
    shap_parts = []
    models = []
    for k in range(5):
        tr = df[df["fold_spatial"] != k]; te = df[df["fold_spatial"] == k]
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60, eval_metric="rmse",
                             tree_method="hist", random_state=C.SEED + k, n_jobs=-1, **orig_cfg)
        ev = tr.sample(frac=0.2, random_state=C.SEED + k)
        m.fit(tr.drop(ev.index)[feats], tr.drop(ev.index)["plot_mean_RGR"],
              eval_set=[(ev[feats], ev["plot_mean_RGR"])], verbose=False)
        oof.loc[te.index] = m.predict(te[feats])
        models.append(m)
        dm = xgb.DMatrix(te[feats])
        sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        shap_parts.append(pd.DataFrame(sv.astype(np.float32), index=te.index, columns=feats))
    sm = pd.concat(shap_parts).sort_index()
    imp = pd.DataFrame({"var": feats, "mas": sm.abs().mean().values})
    imp["group"] = imp["var"].map(GROUP_OF)
    assert imp["group"].notna().all(), \
        f"未映射变量: {imp.loc[imp['group'].isna(), 'var'].tolist()}"
    gs = imp.groupby("group")["mas"].sum()
    share = gs / gs.sum()
    mm = C.metrics(df["plot_mean_RGR"], oof)
    say(f"- {tag}: n={len(df)}, OOF R2={mm['R2']:.4f}, RMSE={mm['RMSE']:.5f} | "
        f"stand={share.get('stand', 0):.3f} climate={share.get('climate', 0):.3f} "
        f"soil={share.get('soil', 0):.3f} terrain={share.get('terrain', 0):.3f}")
    return {"variant": tag, "n": len(df), "R2": mm["R2"], "RMSE": mm["RMSE"],
            "share_stand": share.get("stand", 0), "share_climate": share.get("climate", 0),
            "share_soil": share.get("soil", 0), "share_terrain": share.get("terrain", 0)}

ctrl_rows = []
ctrl_rows.append(variant_oof_shap(cc, MAP_FEATS, "raster_age (28+1 feats)"))
feats_cfi = [f if f != "stand_age_r" else "stand_age_cfi" for f in MAP_FEATS]
cc_cfi = cc.dropna(subset=["stand_age_cfi"])
ctrl_rows.append(variant_oof_shap(cc_cfi, feats_cfi, "cfi_age (同子集)"))
# 同子集 raster-age 对照（公平比较：与 CFI 子集相同样地）
ctrl_rows.append(variant_oof_shap(cc_cfi, MAP_FEATS, "raster_age (CFI 子集)"))
feats_noage = [f for f in MAP_FEATS if f != "stand_age_r"]
ctrl_rows.append(variant_oof_shap(cc, feats_noage, "no_age (全子集)"))
ctrl = pd.DataFrame(ctrl_rows)
ctrl.to_csv(os.path.join(C.OUT, "18_age_control.csv"), index=False, encoding="utf-8-sig")

# ---------- 5. 全局选参 + 最终制图 ----------
say("\n## 5. 全局内层选参 + 5 折 ensemble 制图（早停不用外层测试折）")
inner_all = inner_spatial_folds(cc, n_folds=4, seed=C.SEED + 99)
best_rmse, best_p, best_iter = np.inf, None, None
for ci, p in enumerate(cands):
    inner_oof = pd.Series(np.nan, index=cc.index)
    iters = []
    for j in range(4):
        itr = cc[inner_all != j]; ite = cc[inner_all == j]
        m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60, eval_metric="rmse",
                             tree_method="hist", random_state=C.SEED + j, n_jobs=-1, **p)
        ev = itr.sample(frac=0.2, random_state=C.SEED + j)
        m.fit(itr.drop(ev.index)[MAP_FEATS], itr.drop(ev.index)["plot_mean_RGR"],
              eval_set=[(ev[MAP_FEATS], ev["plot_mean_RGR"])], verbose=False)
        inner_oof.loc[ite.index] = m.predict(ite[MAP_FEATS])
        iters.append(m.best_iteration + 1 if m.best_iteration is not None else 2000)
    rmse = float(np.sqrt(np.nanmean((cc["plot_mean_RGR"] - inner_oof) ** 2)))
    if rmse < best_rmse:
        best_rmse, best_p, best_iter = rmse, p, int(np.median(iters))
say(f"- 全局最优: {best_p}, n_est={best_iter}, inner RMSE={best_rmse:.5f}")

valid = forest_mask.copy()
for v in MAP_FEATS:
    valid &= np.isfinite(grids[v])
say(f"- 可预测林地像元: {valid.sum():,}（占林地 {valid.sum()/forest_mask.sum()*100:.1f}%）")
rows_idx, cols_idx = np.where(valid)
Xmap = np.column_stack([grids[v][rows_idx, cols_idx] for v in MAP_FEATS]).astype(np.float32)
preds = np.empty((5, len(Xmap)), dtype=np.float32)
ens_models = []
for k in range(5):
    tr = cc[cc["fold_spatial"] != k]
    m = xgb.XGBRegressor(n_estimators=best_iter, tree_method="hist",
                         random_state=C.SEED + k, n_jobs=-1, **best_p)
    m.fit(tr[MAP_FEATS], tr["plot_mean_RGR"])
    preds[k] = m.predict(Xmap)
    ens_models.append(m)
    say(f"  fold {k} 完成 (n_est={best_iter})")
map_mean = preds.mean(axis=0)
map_sd = preds.std(axis=0)
say(f"- 预测 RGR: mean={map_mean.mean():.5f}, sd={map_mean.std():.5f} (yr⁻¹), "
    f"P5={np.percentile(map_mean,5):.5f}, P95={np.percentile(map_mean,95):.5f}")
say(f"- model disagreement (ensemble sd): mean={map_sd.mean():.5f}, P95={np.percentile(map_sd,95):.5f}")
say(f"- 压缩比: 观测 sd {obs_sd:.5f} / 制图 sd {map_mean.std():.5f} = {obs_sd/map_mean.std():.1f}×")
say(f"- 总预测不确定度（±1.96×嵌套 RMSE）: ±{1.96*nested_rmse:.4f} yr⁻¹"
    f"（远大于图面空间格局 sd={map_mean.std():.5f} → 产品仅支持粗尺度排序解读）")

# ---------- 6. 诚实格局归因（5 模型平均 SHAP） ----------
say("\n## 6. 格局归因（5 折模型 mean|SHAP| 平均，10 万像元样本）")
samp = rng.choice(len(Xmap), size=min(100000, len(Xmap)), replace=False)
gs_sum = np.zeros(len(MAP_FEATS))
for m in ens_models:
    dm = xgb.DMatrix(Xmap[samp], feature_names=MAP_FEATS)
    sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    gs_sum += np.abs(sv).mean(axis=0)
gs_arr = gs_sum / 5
vp_rows = []
for g in ("stand", "climate", "soil", "terrain"):
    idx = [i for i, f in enumerate(MAP_FEATS) if GROUP_OF[f] == g]
    vp_rows.append({"group": g, "shap_share": round(float(gs_arr[idx].sum() / gs_arr.sum()), 4)})
vp = pd.DataFrame(vp_rows)
vp.to_csv(os.path.join(C.OUT, "18_map_pattern_varpart.csv"), index=False, encoding="utf-8-sig")
say(vp.to_string(index=False))
top_map = pd.DataFrame({"var": MAP_FEATS, "group": [GROUP_OF[f] for f in MAP_FEATS],
                        "mean_abs_shap": gs_arr}).sort_values("mean_abs_shap", ascending=False)
say("- top-10 变量:")
say(top_map.head(10).to_string(index=False))

# ---------- 7. 写出 GeoTIFF ----------
def write_tif(arr2d, name, dtype=np.float32, nodata=np.nan):
    prof = dict(driver="GTiff", height=oh, width=ow, count=1, dtype=dtype,
                crs="EPSG:4326", transform=otrans, nodata=nodata, compress="lzw")
    with rasterio.open(os.path.join(MAPDIR, name), "w", **prof) as dst:
        dst.write(arr2d.astype(dtype), 1)

mean_grid = np.full((oh, ow), np.nan, dtype=np.float32); mean_grid[rows_idx, cols_idx] = map_mean
sd_grid = np.full((oh, ow), np.nan, dtype=np.float32); sd_grid[rows_idx, cols_idx] = map_sd
fprop_out = np.where(np.isfinite(fprop), fprop, np.nan).astype(np.float32)
write_tif(mean_grid, "RGR_mean_250m.tif")
write_tif(sd_grid, "RGR_model_disagreement_250m.tif")
write_tif(sd_grid, "RGR_sd_250m.tif")  # 兼容旧图脚本
write_tif(fprop_out, "forest_prop_250m.tif")
np.savez_compressed(os.path.join(MAPDIR, "map_arrays.npz"),
                    mean=mean_grid, disagreement=sd_grid, forest_prop=fprop_out,
                    transform=np.array(otrans[:6]), rows=rows_idx, cols=cols_idx,
                    map_mean=map_mean, map_sd=map_sd,
                    map_feats=np.array(MAP_FEATS),
                    groups=np.array([GROUP_OF[f] for f in MAP_FEATS]))
say("- 已写出 GeoTIFF（RGR_mean / RGR_model_disagreement / forest_prop）")

C.save_json({
    "grid_res_deg": RES, "grid_w": ow, "grid_h": oh,
    "n_forest_px": int(forest_mask.sum()), "n_pred_px": int(valid.sum()),
    "nested_R2_spatialCV": nested_m["R2"], "nested_Spearman": nested_m["Spearman"],
    "nested_RMSE": nested_rmse,
    "orig_samefold_R2": 0.0394,
    "pi95_halfwidth_196rmse": round(1.96 * nested_rmse, 5),
    "map_model_R2_noage": 0.0256,
    "obs_plotRGR_sd": round(float(obs_sd), 5),
    "map_mean": float(map_mean.mean()), "map_sd_between_px": float(map_mean.std()),
    "compression_ratio_obs_over_map": round(float(obs_sd / map_mean.std()), 1),
    "ens_disagreement_mean": float(map_sd.mean()),
    "final_params": best_p, "final_n_est": best_iter,
    "age_control": ctrl_rows,
}, "07_map_summary.json")

say("\n## 桉竹掩膜说明（P0-1-5）")
say("- 无优势树种/森林类型栅格可用；林地掩膜含桉树/竹林人工林像元。")
say("- 训练单木已剔除桉竹 → 桉竹主导像元的预测属域外外推，不可用于经营解读。")
say("- 处理：Fig.6 图注 + Methods 明示该限制与受影响面积量级（引官方统计）。")

say(f"\n（总用时 {time.time()-t0:.0f}s）")
say("\nDONE P0-1")
with open(os.path.join(C.OUT, "18_map_rebuild_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
