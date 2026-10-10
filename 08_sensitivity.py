# -*- coding: utf-8 -*-
"""Phase G 敏感性套件：极端值阈值 / 低密度剔除阈值 / 共线性筛选 / 空间块大小 / 分层vs合并 / ΔD 备用响应。
输出: 08_sensitivity_results.csv / 08_sensitivity_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase G 敏感性套件报告\n")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "block05", "fold_spatial", "fold_group"]],
                on="plot_id", how="left")
params = C.load_json("02_best_params.json")["xgb_best"]
feats = C.selected_features(vd)
group_of = dict(zip(vd["var"], vd["group"]))
groups = ["tree", "stand", "climate", "soil", "terrain"]

def run_variant(sub, f_list, fold_col, tag, y_col="RGR", collect_shap=True):
    """跑一个变体：空间块 CV + OOF SHAP 类组占比。返回指标行。"""
    oof, models = C.fit_xgb_folds(sub, f_list, params, fold_col, y_col=y_col,
                                  return_models=True)
    m = C.metrics(sub[y_col], oof)
    pm = sub.assign(pred=oof).groupby("plot_id")[[y_col, "pred"]].mean()
    m["plot_R2"] = C.metrics(pm[y_col], pm["pred"])["R2"]
    row = {"variant": tag, "n": len(sub), "tree_R2": m["R2"], "plot_R2": m["plot_R2"],
           "RMSE": m["RMSE"], "Spearman": m["Spearman"]}
    if collect_shap:
        parts = []
        for k, mm in enumerate(models):
            te = sub[sub[fold_col] == k]
            dm = xgb.DMatrix(te[f_list])
            sv = mm.get_booster().predict(dm, pred_contribs=True)[:, :-1]
            parts.append(pd.DataFrame(sv, index=te.index, columns=f_list))
        sm = pd.concat(parts)
        gs = sm.abs().mean()
        for g in groups:
            share = gs[[f for f in f_list if group_of.get(f) == g]].sum() / gs.sum()
            row[f"share_{g}"] = round(float(share), 4)
    return row

rows = []

# ---------- 1. 极端值阈值 ----------
say("## 1. 极端增量阈值（原始清洗: growth ≤ 200 mm/5yr）")
t0 = tab.dropna(subset=["fold_spatial"])
for thr in (150, 200, 250, 300):
    sub = t0[t0["D1"] - t0["D0"] <= thr]
    rows.append(run_variant(sub, feats, "fold_spatial", f"threshold_{thr}mm"))
    say(f"- ≤{thr}mm: n={len(sub)}, R2={rows[-1]['tree_R2']:.4f}, shares: "
        + ", ".join(f"{g}={rows[-1][f'share_{g}']:.3f}" for g in groups))

# ---------- 2. 低密度样地剔除阈值（新数据已剔除密度<105 株/ha） ----------
say("\n## 2. 低密度样地剔除阈值（基准数据已剔除密度<105 株/ha）")
for thr in (105, 200, 400):
    keep_ids = set(tab.loc[tab["stand_density"] >= thr, "plot_id"].unique())
    sub = tab[tab["plot_id"].isin(keep_ids)].dropna(subset=["fold_spatial"])
    rows.append(run_variant(sub, feats, "fold_spatial", f"density_ge_{thr}"))
    say(f"- 密度≥{thr} 株/ha: n={len(sub)} 株 / {len(keep_ids)} 样地, "
        f"R2={rows[-1]['tree_R2']:.4f}, shares: "
        + ", ".join(f"{g}={rows[-1][f'share_{g}']:.3f}" for g in groups))

# ---------- 3. 共线性筛选（预筛选 vs 全候选变量） ----------
say("\n## 3. 共线性筛选（预筛选35 vs 全候选51）")
all_cands = (C.TREE_VARS + C.STAND_VARS
             + C.SOIL_VARS + C.TERRAIN_VARS + C.CLIMATE_CANDIDATES)
all_cands = [v for v in dict.fromkeys(all_cands) if v in tab.columns]
sub = tab.dropna(subset=["fold_spatial"])
rows.append(run_variant(sub, all_cands, "fold_spatial", "all_candidates_51"))
say(f"- 全候选: n={len(sub)}, R2={rows[-1]['tree_R2']:.4f}, shares: "
    + ", ".join(f"{g}={rows[-1][f'share_{g}']:.3f}" for g in groups))

# ---------- 4. 空间块大小 ----------
say("\n## 4. 空间块大小（0.25/0.5/1.0°）")
for size in (0.25, 0.5, 1.0):
    blk = C.block_id(tab["lon"].values, tab["lat"].values, size)
    tree_per_block = blk.value_counts()
    assign = C.assign_spatial_folds(list(tree_per_block.items()), n_folds=5)
    tab[f"fold_spa_{size}"] = blk.map(assign).values
    sub = tab.dropna(subset=[f"fold_spa_{size}"])
    rows.append(run_variant(sub, feats, f"fold_spa_{size}", f"block_{size}deg"))
    say(f"- {size}° 块: n_blocks={len(tree_per_block)}, R2={rows[-1]['tree_R2']:.4f}")

# ---------- 5. 分层 vs 合并（汇总 Phase C 结果对比） ----------
say("\n## 5. 分层 vs 合并建模")
strat = pd.read_csv(os.path.join(C.OUT, "03_stratified.csv"))
for _, r in strat.iterrows():
    rows.append({"variant": f"stratified_{r['subset']}", "n": int(r["n_trees"]),
                 "tree_R2": r["R2"], "plot_R2": r["plot_R2"], "RMSE": r["RMSE"],
                 "Spearman": r["Spearman"]})
base = [r for r in rows if r["variant"] == "threshold_200mm"][0]
say(f"- 合并模型: R2={base['tree_R2']:.4f}；分层: 针叶 {strat.iloc[0]['R2']:.4f} / "
    f"阔叶 {strat.iloc[1]['R2']:.4f}")

# ---------- 6. ΔD 备用响应（绝对增量, mm/yr） ----------
say("\n## 6. 备用响应：绝对增量 ΔD/5 (mm/yr)")
sub = tab.copy()
sub["dinc"] = (sub["D1"] - sub["D0"]) / 5.0
rows.append(run_variant(sub, feats, "fold_spatial", "response_abs_increment", y_col="dinc"))
say(f"- ΔD 模型: R2={rows[-1]['tree_R2']:.4f}, shares: "
    + ", ".join(f"{g}={rows[-1][f'share_{g}']:.3f}" for g in groups))
rgr_row = base
say(f"- RGR 模型 shares: " + ", ".join(f"{g}={rgr_row[f'share_{g}']:.3f}" for g in groups))
from scipy.stats import spearmanr
share_rgr = np.array([rgr_row[f"share_{g}"] for g in groups])
share_dinc = np.array([rows[-1][f"share_{g}"] for g in groups])
say(f"- 两模型类组占比 Spearman ρ = {spearmanr(share_rgr, share_dinc).statistic:.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(C.OUT, "08_sensitivity_results.csv"), index=False, encoding="utf-8-sig")
say("\nDONE Phase G")
with open(os.path.join(C.OUT, "08_sensitivity_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
