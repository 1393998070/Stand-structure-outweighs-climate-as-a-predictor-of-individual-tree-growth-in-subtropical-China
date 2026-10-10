# -*- coding: utf-8 -*-
"""Phase C 三重验证（RQ5）：随机/样地分组/空间块 5 折 CV 对比 + 分层建模 + bootstrap CI。
输出: 03_triple_cv_results.csv / 03_oof_predictions.pkl / 03_stratified.csv / 03_triple_cv_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial", "fold_group"]], on="plot_id", how="left")
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
rf_params = C.load_json("02_rf_params.json")

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase C 三重交叉验证报告\n")
say(f"- XGB 最优参数: {params}")
say(f"- RF 参数: {rf_params}")

def plot_level_metrics(tab, oof):
    pm = tab.assign(pred=oof).groupby("plot_id")[["RGR", "pred"]].mean()
    return C.metrics(pm["RGR"], pm["pred"])

rows = []
t0 = time.time()
# ---------- 1. XGB 三重 CV ----------
for scheme, col in [("random", "fold_random"), ("group", "fold_group"), ("spatial", "fold_spatial")]:
    oof = C.fit_xgb_folds(tab, feats, params, col)
    m_tree = C.metrics(tab["RGR"], oof)
    m_plot = plot_level_metrics(tab, oof)
    ci = C.cluster_bootstrap_ci(tab["RGR"], oof, tab["plot_id"])
    rows.append({"model": "XGBoost", "cv": scheme, **m_tree,
                 "plot_R2": m_plot["R2"], "plot_Spearman": m_plot["Spearman"],
                 "ci_tree_R2": str(ci["tree_R2"]), "ci_tree_Spearman": str(ci["tree_Spearman"]),
                 "ci_plot_R2": str(ci["plot_R2"])})
    say(f"- XGB {scheme:8s}: tree R2={m_tree['R2']:.4f} [{ci['tree_R2'][0]:.4f},{ci['tree_R2'][1]:.4f}] "
        f"rho={m_tree['Spearman']:.4f} | plot R2={m_plot['R2']:.4f} [{ci['plot_R2'][0]:.4f},{ci['plot_R2'][1]:.4f}]")
    if scheme == "spatial":
        oof_spatial = oof.copy()

# ---------- 2. RF 三重 CV ----------
for scheme, col in [("random", "fold_random"), ("group", "fold_group"), ("spatial", "fold_spatial")]:
    oof = C.fit_rf_folds(tab, feats, rf_params, col)
    m_tree = C.metrics(tab["RGR"], oof)
    m_plot = plot_level_metrics(tab, oof)
    rows.append({"model": "RandomForest", "cv": scheme, **m_tree,
                 "plot_R2": m_plot["R2"], "plot_Spearman": m_plot["Spearman"],
                 "ci_tree_R2": "", "ci_tree_Spearman": "", "ci_plot_R2": ""})
    say(f"- RF  {scheme:8s}: tree R2={m_tree['R2']:.4f} rho={m_tree['Spearman']:.4f} | plot R2={m_plot['R2']:.4f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(C.OUT, "03_triple_cv_results.csv"), index=False, encoding="utf-8-sig")

# ---------- 3. 分层建模（RQ3：针叶/阔叶子集，空间块 CV） ----------
say("\n## 分层建模（空间块 CV）")
strat_rows = []
for fg_name, sub in [("conifer", tab[tab["fg_conifer"] == 1]), ("broadleaf", tab[tab["fg_conifer"] == 0])]:
    oof = C.fit_xgb_folds(sub, feats, params, "fold_spatial")
    m_tree = C.metrics(sub["RGR"], oof)
    m_plot = plot_level_metrics(sub, oof)
    strat_rows.append({"subset": fg_name, "n_trees": len(sub), "n_plots": sub["plot_id"].nunique(),
                       **m_tree, "plot_R2": m_plot["R2"]})
    say(f"- {fg_name:10s} (n={len(sub)}): tree R2={m_tree['R2']:.4f} rho={m_tree['Spearman']:.4f} plot R2={m_plot['R2']:.4f}")
pd.DataFrame(strat_rows).to_csv(os.path.join(C.OUT, "03_stratified.csv"), index=False, encoding="utf-8-sig")

# ---------- 4. 泄漏量化 ----------
xr = res[(res["model"] == "XGBoost")]
r_ran = float(xr[xr["cv"] == "random"]["R2"].iloc[0])
r_grp = float(xr[xr["cv"] == "group"]["R2"].iloc[0])
r_spa = float(xr[xr["cv"] == "spatial"]["R2"].iloc[0])
say(f"\n## 泄漏量化")
say(f"- 随机 vs 空间块: R2 虚高 {r_ran - r_spa:.4f}（相对 {(r_ran/r_spa-1)*100:.0f}%）")
say(f"- 分组 vs 空间块: R2 虚高 {r_grp - r_spa:.4f}（相对 {(r_grp/r_spa-1)*100:.0f}%）")

# ---------- 5. 保存 OOF ----------
oof_df = tab[["plot_id", "fg", "RGR", "lnD0", "fold_random", "fold_group", "fold_spatial"]].copy()
oof_df["oof_spatial_xgb"] = oof_spatial
oof_df.to_pickle(os.path.join(C.OUT, "03_oof_predictions.pkl"))
say(f"\n用时 {time.time()-t0:.0f}s")
say("DONE Phase C")
with open(os.path.join(C.OUT, "03_triple_cv_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
