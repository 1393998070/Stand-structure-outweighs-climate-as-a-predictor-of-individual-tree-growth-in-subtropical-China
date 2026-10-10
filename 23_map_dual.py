# -*- coding: utf-8 -*-
"""P0-G + P0-H 功能群双图制图 + 森林掩膜构造修正。

P0-H 修正：原实现把 4 类【分类】栅格用 Resampling.average 聚合到 250 m 后以 >=0.5
阈值化，导致有效森林覆盖率阈值随类别漂移（class4 仅需 12.5% 覆盖即计入林地）。
本脚本改为：先二值化（任一类 ∈ {1,2,3,4} → 1，nodata → 0），再按覆盖率聚合，
最后施加统一的 >=0.5 覆盖率阈值。

P0-G 新增：按功能群分层建模出两张省域图——针叶 RGR 与阔叶 RGR。
另保留合并（pooled）图作为对照。

输出:
  23_mask_fix.csv                  掩膜修正前后像元数对比
  23_dual_map_results.csv          三种响应的嵌套 CV 性能
  23_dual_map_attribution.csv      三张图的格局归因
  23_dual_map_report.md
  map/RGR_conifer_250m.tif / RGR_broadleaf_250m.tif / RGR_pooled_250m.tif
  map/*_disagreement_250m.tif
  map/forest_frac_250m.tif
  map/dual_map_arrays.npz          （供图件脚本）
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform
from rasterio.transform import from_bounds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.RASTER
MAPDIR = os.path.join(C.OUT, "map")
os.makedirs(MAPDIR, exist_ok=True)

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P0-G/P0-H 功能群双图制图报告\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say(f"- Python {sys.version.split()[0]}；xgboost {xgb.__version__}")
say("")

LAYER_FILE = {
    "stand_age_r": "林分\\林龄.tif", "stand_density": "林分\\林分密度.tif",
    "forest_cls": "林分\\全省林地分布栅格.tif",
    "dem": "地形水文参数\\GD_DEM.tif", "slope": "地形水文参数\\GD_SLP.tif",
    "aspect_deg": "地形水文参数\\GD_ASP.tif", "curvature_total": "地形水文参数\\GD_CU.tif",
    "dist_stream": "地形水文参数\\GD_DHN.tif", "relief": "地形水文参数\\GD_DXQFD.tif",
    "flow_length": "地形水文参数\\GD_FL.tif", "heat_load": "地形水文参数\\GD_HL.tif",
    "SPI": "地形水文参数\\GD_SPI.tif", "TPI": "地形水文参数\\GD_TPI.tif",
    "TSI": "地形水文参数\\GD_TSI.tif", "TWI": "地形水文参数\\GD_TWI.tif",
    "soil_SOC": "土壤\\广东林地土壤SOC.tif", "soil_TN": "土壤\\广东林地土壤TN.tif",
    "soil_TP": "土壤\\广东林地土壤TP.tif", "soil_Tk": "土壤\\广东林地土壤Tk.tif",
    "soil_pH": "土壤\\广东林地土壤pH.tif", "soil_BD": "土壤\\广东林地土壤容重.tif",
    "soil_depth": "土壤\\土层厚度.tif",
}
for i in (1, 2, 4, 12, 15, 18, 19):
    LAYER_FILE[f"bio{i}"] = f"气候\\peng_bio_{i}.tif"

MAP_FEATS = ["stand_age_r", "stand_density",
             "bio1", "bio2", "bio4", "bio12", "bio15", "bio18", "bio19",
             "soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth",
             "dem", "slope", "aspect_cos", "aspect_sin", "curvature_total", "dist_stream",
             "relief", "flow_length", "heat_load", "SPI", "TPI", "TSI", "TWI"]
GROUP_OF = {"stand_age_r": "stand", "stand_density": "stand",
            **{f"bio{i}": "climate" for i in (1, 2, 4, 12, 15, 18, 19)},
            **{v: "soil" for v in ("soil_SOC", "soil_TN", "soil_TP", "soil_Tk",
                                   "soil_pH", "soil_BD", "soil_depth")},
            **{v: "terrain" for v in ("dem", "slope", "aspect_cos", "aspect_sin",
                                      "curvature_total", "dist_stream", "relief",
                                      "flow_length", "heat_load", "SPI", "TPI", "TSI", "TWI")}}
MIN_TREES = 3   # 每功能群每样地最少株数（保证样地均值可靠）

# ---------- 1. 输出网格 ----------
t0 = time.time()
with rasterio.open(os.path.join(R, LAYER_FILE["forest_cls"])) as msrc:
    W, S, E, N = msrc.bounds.left, msrc.bounds.bottom, msrc.bounds.right, msrc.bounds.top
RES = 0.0025
ow = int((E - W) / RES); oh = int((N - S) / RES)
otrans = from_bounds(W, S, W + ow * RES, S + oh * RES, ow, oh)
say("## 1. 输出网格与掩膜修正（P0-H）")
say(f"- 网格 {ow} × {oh} @ {RES}°（~250 m）")

def to_grid(path_or_arr, src_crs=None, src_transform=None,
            resampling=Resampling.average):
    """重投影到输出网格。

    稳健做法：先把 nodata 显式置 NaN，再以 src_nodata=NaN 重投影——
    避免 rasterio 1.4.3 在 int8 + average 下把 -128 当作 +128 参与平均。
    """
    dst = np.full((oh, ow), np.nan, dtype=np.float32)
    if isinstance(path_or_arr, str):
        with rasterio.open(path_or_arr) as src:
            a = src.read(1).astype(np.float32)
            ndv = src.nodata
            if ndv is not None:
                a[a == np.float32(ndv)] = np.nan
            src_crs, src_transform = src.crs, src.transform
    else:
        a = np.asarray(path_or_arr, dtype=np.float32)
    reproject(a, dst, src_transform=src_transform, src_crs=src_crs,
              src_nodata=np.nan, dst_transform=otrans, dst_crs="EPSG:4326",
              dst_nodata=np.nan, resampling=resampling)
    return dst

# --- 旧口径（复现，供对比）：分类码直接 average 后 >=0.5 ---
# 注意：必须显式把 nodata 置 NaN 后再重投影。rasterio 1.4.3 在 int8 + average
# 下不会正确排除 src_nodata=-128（会被当作 +128 参与平均），故不依赖该路径。
with rasterio.open(os.path.join(R, LAYER_FILE["forest_cls"])) as src:
    cls = src.read(1)
    nd = src.nodata if src.nodata is not None else -128
    cls_transform, cls_crs = src.transform, src.crs
    cls_counts = pd.Series(cls[cls != nd]).value_counts().sort_index()
    cls_f = cls.astype(np.float32)
    cls_f[cls == nd] = np.nan
    binarr = ((cls != nd) & (cls >= 1) & (cls <= 4)).astype(np.float32)
fprop_old = to_grid(cls_f, src_crs=cls_crs, src_transform=cls_transform)
mask_old = fprop_old >= 0.5

# --- 新口径：先二值化，再 average 得真实森林覆盖率，再 >=0.5 ---
forest_frac = to_grid(binarr, src_crs=cls_crs, src_transform=cls_transform)
mask_new = forest_frac >= 0.5

say(f"- 栅格类别计数（原始分辨率）: "
    + ", ".join(f"class{int(k)}={v:,}" for k, v in cls_counts.items()))
say(f"- 旧口径（分类码 average >=0.5）林地像元: {int(mask_old.sum()):,}")
say(f"- 新口径（二值化→覆盖率 >=0.5）林地像元: {int(mask_new.sum()):,}")
say(f"- 差异: {int(mask_old.sum()) - int(mask_new.sum()):,} 像元 "
    f"（−{(mask_old.sum() - mask_new.sum()) / mask_old.sum() * 100:.1f}%；旧口径因类别漂移多计入）")
pd.DataFrame([
    {"scheme": "old: class-code average >=0.5", "forest_px": int(mask_old.sum())},
    {"scheme": "new: binarize then cover-fraction >=0.5", "forest_px": int(mask_new.sum())},
]).to_csv(os.path.join(C.OUT, "23_mask_fix.csv"), index=False, encoding="utf-8-sig")

# 其余预测层
grids = {}
for v in MAP_FEATS:
    if v in ("aspect_cos", "aspect_sin"):
        continue
    grids[v] = to_grid(os.path.join(R, LAYER_FILE[v]))
with rasterio.open(os.path.join(R, LAYER_FILE["aspect_deg"])) as src:
    asp = src.read(1).astype(np.float32)
    asp[asp == (src.nodata if src.nodata is not None else -9999)] = np.nan
    a_tr, a_crs = src.transform, src.crs
grids["aspect_cos"] = to_grid(np.cos(np.deg2rad(asp)), src_crs=a_crs, src_transform=a_tr)
grids["aspect_sin"] = to_grid(np.sin(np.deg2rad(asp)), src_crs=a_crs, src_transform=a_tr)
del asp
say(f"（重投影用时 {time.time()-t0:.0f}s）")

# ---------- 2. 样地训练数据（含功能群响应） ----------
say("\n## 2. 样地级训练数据（栅格提取 + 功能群响应）")
tab, vd, folds = C.load_model_data()
plot_xy = tab.groupby("plot_id")[["lon", "lat"]].first()
pf = folds.set_index("plot_id").reindex(plot_xy.index)

grp = tab.groupby(["plot_id", "fg_conifer"])["RGR"].agg(["mean", "size"]).reset_index()
def resp(cflag, suffix):
    s = grp[grp["fg_conifer"] == cflag].set_index("plot_id")
    s = s[s["size"] >= MIN_TREES]
    return (s["mean"].rename(f"RGR_grp_{suffix}"),
            s["size"].rename(f"n_{suffix}"))

con_r, con_n = resp(1, "con")
brd_r, brd_n = resp(0, "brd")
pool_r = tab.groupby("plot_id")["RGR"].mean().rename("RGR_pool")
pool_n = tab.groupby("plot_id")["RGR"].size().rename("n_pool")

pdata = pd.DataFrame({"lon": plot_xy["lon"], "lat": plot_xy["lat"],
                      "fold_spatial": pf["fold_spatial"]}).join(
    [pool_r, pool_n, con_r, con_n, brd_r, brd_n])

def sample_raster(path):
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.is_geographic:
            xs, ys = plot_xy["lon"].tolist(), plot_xy["lat"].tolist()
        else:
            xs, ys = warp_transform("EPSG:4326", src.crs, plot_xy["lon"].tolist(),
                                    plot_xy["lat"].tolist())
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        vals[vals == (src.nodata if src.nodata is not None else -9999)] = np.nan
    return pd.Series(vals, index=plot_xy.index)

for v in MAP_FEATS:
    if v in ("aspect_cos", "aspect_sin"):
        continue
    pdata[v] = sample_raster(os.path.join(R, LAYER_FILE[v]))
with rasterio.open(os.path.join(R, LAYER_FILE["aspect_deg"])) as src:
    if src.crs is None or src.crs.is_geographic:
        xs, ys = plot_xy["lon"].tolist(), plot_xy["lat"].tolist()
    else:
        xs, ys = warp_transform("EPSG:4326", src.crs, plot_xy["lon"].tolist(),
                                plot_xy["lat"].tolist())
    a = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
    a[a == (src.nodata if src.nodata is not None else -9999)] = np.nan
pdata["aspect_cos"] = np.cos(np.deg2rad(a))
pdata["aspect_sin"] = np.sin(np.deg2rad(a))
pd.to_pickle(pdata, os.path.join(C.OUT, "23_plot_raster_data.pkl"))

say(f"- 样地总数: {len(pdata)}")
for tag, rcol, ncol in [("pooled", "RGR_pool", "n_pool"),
                        ("conifer", "RGR_grp_con", "n_con"),
                        ("broadleaf", "RGR_grp_brd", "n_brd")]:
    sub = pdata.dropna(subset=MAP_FEATS + [rcol, "fold_spatial"])
    say(f"- {tag}: 可用样地 {len(sub)}（功能群最少 {MIN_TREES} 株）")

# ---------- 3. 嵌套空间 CV（三种响应） ----------
say("\n## 3. 嵌套空间 CV（外层 5 空间折 × 内层 4 空间折 × 12 候选）")
rng = np.random.default_rng(C.SEED)
space = {"max_depth": [2, 3, 4, 5, 6], "learning_rate": [0.02, 0.05, 0.1],
         "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
         "min_child_weight": [1, 3, 8], "reg_lambda": [1, 3, 8, 15], "reg_alpha": [0, 0.5, 2]}
cands = []
for _ in range(12):
    p = {k: rng.choice(v) for k, v in space.items()}
    cands.append({k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
                  for k, v in p.items()})

def inner_spatial_folds(train_df, n_folds=4, seed=0):
    b = C.block_id(train_df["lon"].values, train_df["lat"].values).values
    blk = pd.Series(b, index=train_df.index).value_counts().sort_values(ascending=False)
    r2 = np.random.default_rng(seed)
    order = r2.permutation(len(blk))
    assign = C.assign_spatial_folds([list(blk.items())[i] for i in order], n_folds=n_folds)
    return pd.Series(b, index=train_df.index).map(assign)

def nested_cv(df, feats, ycol):
    oof = pd.Series(np.nan, index=df.index)
    sel = []
    for k in range(5):
        train = df[df["fold_spatial"] != k]
        test = df[df["fold_spatial"] == k]
        inner = inner_spatial_folds(train, 4, C.SEED + k)
        best_rmse, best_p, best_iter = np.inf, None, 200
        for p in cands:
            ioof = pd.Series(np.nan, index=train.index)
            iters = []
            for j in range(4):
                itr = train[inner != j]; ite = train[inner == j]
                ev = itr.sample(frac=0.2, random_state=C.SEED + j)
                m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60,
                                     eval_metric="rmse", tree_method="hist",
                                     random_state=C.SEED + j, n_jobs=-1, **p)
                m.fit(itr.drop(ev.index)[feats], itr.drop(ev.index)[ycol],
                      eval_set=[(ev[feats], ev[ycol])], verbose=False)
                ioof.loc[ite.index] = m.predict(ite[feats])
                iters.append(m.best_iteration + 1 if m.best_iteration is not None else 2000)
            rmse = float(np.sqrt(np.nanmean((train[ycol] - ioof) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_p, best_iter = rmse, p, int(np.median(iters))
        m = xgb.XGBRegressor(n_estimators=best_iter, tree_method="hist",
                             random_state=C.SEED + k, n_jobs=-1, **best_p)
        m.fit(train[feats], train[ycol])
        oof.loc[test.index] = m.predict(test[feats])
        sel.append({"fold": k, "inner_rmse": round(best_rmse, 5), "n_est": best_iter, **best_p})
    return oof, pd.DataFrame(sel)

def global_params(df, feats, ycol):
    inner = inner_spatial_folds(df, 4, C.SEED + 99)
    best_rmse, best_p, best_iter = np.inf, None, 200
    for p in cands:
        ioof = pd.Series(np.nan, index=df.index)
        iters = []
        for j in range(4):
            itr = df[inner != j]; ite = df[inner == j]
            ev = itr.sample(frac=0.2, random_state=C.SEED + j)
            m = xgb.XGBRegressor(n_estimators=2000, early_stopping_rounds=60,
                                 eval_metric="rmse", tree_method="hist",
                                 random_state=C.SEED + j, n_jobs=-1, **p)
            m.fit(itr.drop(ev.index)[feats], itr.drop(ev.index)[ycol],
                  eval_set=[(ev[feats], ev[ycol])], verbose=False)
            ioof.loc[ite.index] = m.predict(ite[feats])
            iters.append(m.best_iteration + 1 if m.best_iteration is not None else 2000)
        rmse = float(np.sqrt(np.nanmean((df[ycol] - ioof) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_p, best_iter = rmse, p, int(np.median(iters))
    return best_p, best_iter, best_rmse

RESPONSES = [("pooled", "RGR_pool"), ("conifer", "RGR_grp_con"), ("broadleaf", "RGR_grp_brd")]
YCOL = dict(RESPONSES)
res_rows, models_all, params_all, data_all = [], {}, {}, {}
for tag, ycol in RESPONSES:
    sub = pdata.dropna(subset=MAP_FEATS + [ycol, "fold_spatial"]).copy()
    t1 = time.time()
    oof, sel = nested_cv(sub, MAP_FEATS, ycol)
    mm = C.metrics(sub[ycol].values, oof.values)
    rmse = float(np.sqrt(np.mean((sub[ycol].values - oof.values) ** 2)))
    bp, bi, brmse = global_params(sub, MAP_FEATS, ycol)
    res_rows.append({"response": tag, "n_plots": len(sub), "R2": mm["R2"],
                     "Spearman": mm["Spearman"], "RMSE": round(rmse, 5),
                     "n_est": bi, "seconds": round(time.time() - t1, 1)})
    say(f"- {tag}: n={len(sub)}, nested OOF R²={mm['R2']}, rho={mm['Spearman']}, "
        f"RMSE={rmse:.5f}, n_est={bi}（{time.time()-t1:.0f}s）")
    params_all[tag] = {"params": bp, "n_est": bi}
    data_all[tag] = sub
    # 5 折 ensemble 模型（用于制图）
    ens = []
    for k in range(5):
        tr = sub[sub["fold_spatial"] != k]
        m = xgb.XGBRegressor(n_estimators=bi, tree_method="hist",
                             random_state=C.SEED + k, n_jobs=-1, **bp)
        m.fit(tr[MAP_FEATS], tr[ycol])
        ens.append(m)
    models_all[tag] = ens
    sel.insert(0, "response", tag)
    sel.to_csv(os.path.join(C.OUT, f"23_nested_params_{tag}.csv"), index=False,
               encoding="utf-8-sig")

res = pd.DataFrame(res_rows)
res.to_csv(os.path.join(C.OUT, "23_dual_map_results.csv"), index=False, encoding="utf-8-sig")

# ---------- 4. 制图（三张图） ----------
say("\n## 4. 制图（针叶 / 阔叶 / 合并）")
valid = mask_new.copy()
for v in MAP_FEATS:
    valid &= np.isfinite(grids[v])
say(f"- 可预测林地像元: {int(valid.sum()):,}（占新口径林地 "
    f"{valid.sum()/mask_new.sum()*100:.1f}%）")
ri, ci = np.where(valid)
Xmap = np.column_stack([grids[v][ri, ci] for v in MAP_FEATS]).astype(np.float32)

arrays = {}
attr_rows = []
samp = rng.choice(len(Xmap), size=min(100000, len(Xmap)), replace=False)
for tag, _ in RESPONSES:
    ens = models_all[tag]
    preds = np.empty((5, len(Xmap)), dtype=np.float32)
    gs_sum = np.zeros(len(MAP_FEATS))
    for k, m in enumerate(ens):
        preds[k] = m.predict(Xmap)
        dm = xgb.DMatrix(Xmap[samp], feature_names=MAP_FEATS)
        sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        gs_sum += np.abs(sv).mean(axis=0)
    mean = preds.mean(axis=0); sd = preds.std(axis=0)
    gs_arr = gs_sum / 5
    shares = {g: float(gs_arr[[i for i, f in enumerate(MAP_FEATS)
                              if GROUP_OF[f] == g]].sum() / gs_arr.sum())
              for g in ("stand", "climate", "soil", "terrain")}
    obs_sd = data_all[tag][YCOL[tag]].std()
    nrmse = float(res[res["response"] == tag]["RMSE"].iloc[0])
    arrays[tag] = {"mean": mean, "sd": sd, "rows": ri, "cols": ci,
                   "obs_sd": float(obs_sd), "nested_rmse": nrmse}
    attr_rows.append({"response": tag, **{f"share_{k}": round(v, 4)
                                          for k, v in shares.items()}})
    say(f"- {tag}: map mean={mean.mean():.5f}, spatial sd={mean.std():.5f}, "
        f"ens disagreement={sd.mean():.5f}, 压缩比={obs_sd/mean.std():.1f}×")
    say(f"    attribution: " + ", ".join(f"{k}={v:.3f}" for k, v in shares.items()))

attr = pd.DataFrame(attr_rows)
attr.to_csv(os.path.join(C.OUT, "23_dual_map_attribution.csv"), index=False,
            encoding="utf-8-sig")

# ---------- 5. 写出 GeoTIFF ----------
say("\n## 5. 写出 GeoTIFF")
def write_tif(arr2d, name):
    prof = dict(driver="GTiff", height=oh, width=ow, count=1, dtype="float32",
                crs="EPSG:4326", transform=otrans, nodata=np.nan, compress="lzw")
    with rasterio.open(os.path.join(MAPDIR, name), "w", **prof) as dst:
        dst.write(arr2d.astype(np.float32), 1)

out_npz = {"transform": np.array(otrans[:6]), "rows": ri, "cols": ci,
           "forest_frac": np.where(np.isfinite(forest_frac), forest_frac, np.nan),
           "map_feats": np.array(MAP_FEATS),
           "groups": np.array([GROUP_OF[f] for f in MAP_FEATS])}
for tag, _ in RESPONSES:
    a = arrays[tag]
    mg = np.full((oh, ow), np.nan, dtype=np.float32); mg[ri, ci] = a["mean"]
    sg = np.full((oh, ow), np.nan, dtype=np.float32); sg[ri, ci] = a["sd"]
    write_tif(mg, f"RGR_{tag}_250m.tif")
    write_tif(sg, f"RGR_{tag}_disagreement_250m.tif")
    out_npz[f"mean_{tag}"] = mg
    out_npz[f"sd_{tag}"] = sg
    say(f"- RGR_{tag}_250m.tif + RGR_{tag}_disagreement_250m.tif")
write_tif(np.where(np.isfinite(forest_frac), forest_frac, np.nan),
          "forest_frac_250m.tif")
np.savez_compressed(os.path.join(MAPDIR, "dual_map_arrays.npz"), **out_npz)
say("- forest_frac_250m.tif + dual_map_arrays.npz")

C.save_json({"mask_old_px": int(mask_old.sum()), "mask_new_px": int(mask_new.sum()),
             "n_pred_px": int(valid.sum()), "min_trees_per_group": MIN_TREES,
             "responses": res_rows, "attribution": attr_rows,
             "global_params": params_all}, "23_dual_map_summary.json")

say(f"\n（总用时 {time.time()-t0:.0f}s）")
say("\nDONE P0-G/P0-H")
with open(os.path.join(C.OUT, "23_dual_map_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
