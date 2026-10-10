# -*- coding: utf-8 -*-
"""Stage 4 · P0-5 soil_depth 保真度核查 + 剔除敏感性。
背景：正文称 "ρ=1.00 for all directly matching layers"，但 06_extract_check_proj.csv 实测
soil_depth ρ=0.13（R1-M5 证伪项）。本脚本 (1) 深查栅格定义与数值分布，(2) 排查替代解释，
(3) 重训无 soil_depth 模型检验 soil 类组份额稳定性。
输出: 11_soil_depth_report.md / 11_soil_depth_extract.csv / 11_soil_depth_sensitivity.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from rasterio.warp import transform as warp_transform
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P0-5 soil_depth 保真度核查与剔除敏感性\n")

path = os.path.join(C.RASTER, "土壤", "土层厚度.tif")
with rasterio.open(path) as src:
    say("## 1. 栅格档案")
    say(f"- size: {src.width}x{src.height}, res: {src.res}, dtype: {src.dtypes[0]}")
    say(f"- CRS: {src.crs}")
    say(f"- nodata: {src.nodata}, bounds: {src.bounds}")
    arr = src.read(1, masked=True)
    v = arr.compressed()
    say(f"- 有效像元: {len(v):,}；值域 [{v.min():.1f}, {v.max():.1f}]，"
        f"均值 {v.mean():.1f}，中位 {np.median(v):.1f}")
    qs = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    say(f"- 分位数 p1/p5/p25/p50/p75/p95/p99: {[round(float(x),1) for x in qs]}")
    say(f"- 零值像元占比: {(v == 0).mean()*100:.1f}%（0 值可能表示无土层/非林地掩膜外的退化值）")
    say(f"- CRS 为投影坐标（Albers, 单位米），与样地经纬度需坐标转换后采样")

# ---------- 2. 样地点提取与分布对比 ----------
tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
plot_xy = tab.groupby("plot_id")[["lon", "lat"]].first()
plot_tab = tab.drop_duplicates("plot_id").set_index("plot_id")

say("\n## 2. 样地点提取对比（栅格值 vs 样地表值）")
with rasterio.open(path) as src:
    xs, ys = warp_transform("EPSG:4326", src.crs,
                            plot_xy["lon"].tolist(), plot_xy["lat"].tolist())
    vals = np.array([r[0] for r in src.sample(list(zip(xs, ys)))], dtype=float)
    vals[vals == (src.nodata if src.nodata is not None else np.nan)] = np.nan
tabv = plot_tab["soil_depth"].reindex(plot_xy.index).values.astype(float)
ok = ~(np.isnan(vals) | np.isnan(tabv))
rv, tv = vals[ok], tabv[ok]
say(f"- n配对: {ok.sum()}")
say(f"- 栅格值: 均值 {rv.mean():.1f}, 中位 {np.median(rv):.1f}, 范围 [{rv.min():.1f}, {rv.max():.1f}]")
say(f"- 表值:   均值 {tv.mean():.1f}, 中位 {np.median(tv):.1f}, 范围 [{tv.min():.1f}, {tv.max():.1f}]")
say(f"- Spearman ρ = {spearmanr(rv, tv).statistic:.4f}, Pearson r = {pearsonr(rv, tv).statistic:.4f}")
ext = pd.DataFrame({"plot_id": plot_xy.index[ok], "raster": rv, "table": tv})
ext.to_csv(os.path.join(C.OUT, "11_soil_depth_extract.csv"), index=False, encoding="utf-8-sig")

# 单位/倍数关系探测
ratios = tv[rv > 0] / rv[rv > 0]
say(f"- 表值/栅格值 比值分布: 中位 {np.median(ratios):.3f}, IQR [{np.percentile(ratios,25):.3f}, {np.percentile(ratios,75):.3f}]")

# 3x3 窗口均值提取（检验"表值来自邻域均值"假设）
say("\n## 3. 替代解释排查")
with rasterio.open(path) as src:
    win_vals = []
    for x, y in zip(xs, ys):
        try:
            r, c = src.index(x, y)
            w = src.read(1, window=rasterio.windows.Window(c-1, r-1, 3, 3), masked=True)
            win_vals.append(float(np.ma.compressed(w).mean()) if w.count() else np.nan)
        except Exception:
            win_vals.append(np.nan)
wv = np.array(win_vals)
okw = ~(np.isnan(wv) | np.isnan(tabv))
say(f"- 3x3 窗口均值 vs 表值: ρ = {spearmanr(wv[okw], tabv[okw]).statistic:.4f} (n={okw.sum()})")
# 与其他土壤层栅格对比：土层厚度栅格是否与 SOC 栅格同源同投影
with rasterio.open(os.path.join(C.RASTER, "土壤", "广东林地土壤SOC.tif")) as s2:
    say(f"- SOC 栅格 CRS: {s2.crs}, res: {s2.res}（对照：土层厚度 CRS 不同则属不同投影族）")
with rasterio.open(os.path.join(C.RASTER, "林分", "林龄.tif")) as s3:
    xs3, ys3 = warp_transform("EPSG:4326", s3.crs,
                              plot_xy["lon"].tolist(), plot_xy["lat"].tolist())
    v3 = np.array([r[0] for r in s3.sample(list(zip(xs3, ys3)))], dtype=float)
    v3[v3 == (s3.nodata if s3.nodata is not None else np.nan)] = np.nan
    ta = plot_tab["stand_age"].reindex(plot_xy.index).values.astype(float)
    ok3 = ~(np.isnan(v3) | np.isnan(ta))
    say(f"- 同法核验 stand_age: ρ = {spearmanr(v3[ok3], ta[ok3]).statistic:.4f} (n={ok3.sum()})"
        f" —— age raster 同样不匹配，映射到 P0-1-6")

# ---------- 4. 剔除 soil_depth 敏感性（重训 + OOF-SHAP 类组份额） ----------
say("\n## 4. 剔除 soil_depth 敏感性（空间块折重训 XGB + OOF-SHAP）")
feats_all = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
group_of = dict(zip(vd["var"], vd["group"]))
t0 = time.time()

def family_shares(feats, tag):
    oof, models = C.fit_xgb_folds(tab, feats, params, "fold_spatial", return_models=True)
    parts = []
    for k, m in enumerate(models):
        te = tab[tab["fold_spatial"] == k]
        dm = xgb.DMatrix(te[feats])
        sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        parts.append(pd.DataFrame(sv.astype(np.float32), index=te.index, columns=feats))
    sm = pd.concat(parts).sort_index()
    imp = pd.DataFrame({"var": feats, "mas": sm.abs().mean().values})
    imp["group"] = imp["var"].map(group_of)
    gs = imp.groupby("group")["mas"].sum()
    share = gs / gs.sum()
    m_perf = C.metrics(tab["RGR"], oof)
    say(f"- {tag}: tree R2={m_perf['R2']:.4f} (Δ vs 全模型见下表), 类组份额: "
        + share.round(4).to_dict().__str__())
    return share, m_perf

share_full, perf_full = family_shares(feats_all, "全模型（含 soil_depth, 38 特征）")
feats_nosd = [f for f in feats_all if f != "soil_depth"]
share_nosd, perf_nosd = family_shares(feats_nosd, "剔除 soil_depth（37 特征）")

sens = pd.DataFrame({"group": share_full.index,
                     "share_full": share_full.values,
                     "share_no_soil_depth": share_nosd.reindex(share_full.index).values})
sens["abs_diff"] = (sens["share_full"] - sens["share_no_soil_depth"]).round(4)
sens.loc[len(sens)] = {"group": "tree_R2_full", "share_full": perf_full["R2"],
                       "share_no_soil_depth": perf_nosd["R2"], "abs_diff": round(perf_full["R2"]-perf_nosd["R2"],4)}
sens.to_csv(os.path.join(C.OUT, "11_soil_depth_sensitivity.csv"), index=False, encoding="utf-8-sig")
say(f"\n{sens.to_string(index=False)}")
say(f"（用时 {time.time()-t0:.0f}s）")

say("\nDONE P0-5")
with open(os.path.join(C.OUT, "11_soil_depth_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
