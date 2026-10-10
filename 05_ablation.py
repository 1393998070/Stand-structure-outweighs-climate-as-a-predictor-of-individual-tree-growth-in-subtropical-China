# -*- coding: utf-8 -*-
"""Phase E 驱动层级推断强化（RQ1 稳健性）：类组消融 + 年龄切片 + 样地级方差分解 + 逐样地主导类组。
依赖 04_shap_matrix.pkl。输出: 05_ablation.csv / 05_age_origin_slicing.csv / 05_varpart_plot.csv
      05_plot_dominance.csv / 05_ablation_report.md
"""
import os
import sys
import time
import itertools
import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
groups = ["tree", "stand", "climate", "soil", "terrain"]
feats_by_group = {g: C.selected_features(vd, [g]) for g in groups}

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase E 消融与驱动层级推断报告\n")

def plot_metrics(sub, oof):
    pm = sub.assign(pred=oof).groupby("plot_id")[["RGR", "pred"]].mean()
    return C.metrics(pm["RGR"], pm["pred"])

# ---------- 1. 类组消融 ----------
say("## 1. 类组消融（空间块 CV，ΔR² = 全模型 − 去除该类组）")
t0 = time.time()
oof_full = C.fit_xgb_folds(tab, feats, params, "fold_spatial")
m_full = C.metrics(tab["RGR"], oof_full)
p_full = plot_metrics(tab, oof_full)
say(f"- 全模型: tree R2={m_full['R2']:.4f}, plot R2={p_full['R2']:.4f}")

abl_rows = [{"model": "full", "removed": "none", "n_feats": len(feats),
             "tree_R2": m_full["R2"], "plot_R2": p_full["R2"],
             "d_tree_R2": 0.0, "d_plot_R2": 0.0}]
for g in groups:
    f_sub = [f for f in feats if f not in feats_by_group[g]]
    oof = C.fit_xgb_folds(tab, f_sub, params, "fold_spatial")
    m = C.metrics(tab["RGR"], oof)
    pm = plot_metrics(tab, oof)
    abl_rows.append({"model": f"without_{g}", "removed": g, "n_feats": len(f_sub),
                     "tree_R2": m["R2"], "plot_R2": pm["R2"],
                     "d_tree_R2": round(m_full["R2"] - m["R2"], 4),
                     "d_plot_R2": round(p_full["R2"] - pm["R2"], 4)})
    say(f"- 去 {g:8s} ({len(f_sub):2d} feats): tree R2={m['R2']:.4f} (Δ={m_full['R2']-m['R2']:+.4f}) "
        f"plot R2={pm['R2']:.4f} (Δ={p_full['R2']-pm['R2']:+.4f})")
for g in groups:
    oof = C.fit_xgb_folds(tab, feats_by_group[g], params, "fold_spatial")
    m = C.metrics(tab["RGR"], oof)
    pm = plot_metrics(tab, oof)
    abl_rows.append({"model": f"only_{g}", "removed": f"all_but_{g}",
                     "n_feats": len(feats_by_group[g]),
                     "tree_R2": m["R2"], "plot_R2": pm["R2"],
                     "d_tree_R2": "", "d_plot_R2": ""})
    say(f"- 仅 {g:8s} ({len(feats_by_group[g]):2d} feats): tree R2={m['R2']:.4f}, plot R2={pm['R2']:.4f}")
pd.DataFrame(abl_rows).to_csv(os.path.join(C.OUT, "05_ablation.csv"),
                              index=False, encoding="utf-8-sig")
say(f"（用时 {time.time()-t0:.0f}s）")

# ---------- 2. 年龄 × 起源切片 ----------
say("\n## 2. 年龄 × 起源切片（剥离起源-年龄混杂）")
pt = tab.groupby("plot_id").agg(
    plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"),
    origin_natural=("origin_natural", "first"), conifer_prop=("conifer_prop", "first"),
    fg_conifer=("fg_conifer", "mean")).reset_index()
# 第 3 节方差分解所需的样地级环境变量（样地内恒定，取 first）
_env_cols = [c for c in dict.fromkeys(["stand_density"]
                                      + feats_by_group["climate"]
                                      + feats_by_group["soil"]
                                      + feats_by_group["terrain"])]
pt = pt.merge(tab.groupby("plot_id")[_env_cols].first().reset_index(),
              on="plot_id", how="left")
pt["fg_conifer"] = (pt["fg_conifer"] > 0.5).astype(int)
aged = pt[pt["stand_age"].notna() & pt["origin_natural"].isin([0.0, 1.0])].copy()
aged["age_band"] = pd.qcut(aged["stand_age"], 5,
                           labels=["Q1 youngest", "Q2", "Q3", "Q4", "Q5 oldest"])
sl_rows = []
for band, grp in aged.groupby("age_band", observed=True):
    nat = grp[grp["origin_natural"] == 1]["plot_RGR"]
    pla = grp[grp["origin_natural"] == 0]["plot_RGR"]
    t, p = ttest_ind(nat, pla, equal_var=False)
    sl_rows.append({"age_band": str(band), "age_range": f"{grp['stand_age'].min():.0f}-{grp['stand_age'].max():.0f}",
                    "n_natural": len(nat), "n_planted": len(pla),
                    "RGR_natural": round(nat.mean(), 5), "RGR_planted": round(pla.mean(), 5),
                    "diff_permille": round((nat.mean() - pla.mean()) * 1000, 2),
                    "t": round(t, 2), "p_welch": round(p, 4)})
sl = pd.DataFrame(sl_rows)
sl.to_csv(os.path.join(C.OUT, "05_age_origin_slicing.csv"), index=False, encoding="utf-8-sig")
say(sl.to_string(index=False))

# ---------- 3. 样地级方差分解（stand/climate/soil/terrain 四类组） ----------
say("\n## 3. 样地级方差分解（plot_mean_RGR；调整 R²，分层分割 + 唯一贡献）")
pt["mean_lnD0"] = pt["plot_id"].map(tab.groupby("plot_id")["lnD0"].mean())
sets = {
    "stand": ["stand_age", "stand_density", "origin_natural", "conifer_prop",
              "mean_lnD0"],
    "climate": feats_by_group["climate"],
    "soil": feats_by_group["soil"],
    "terrain": feats_by_group["terrain"],
}
d = pt.dropna(subset=["plot_RGR"] + sum(sets.values(), [])).copy()
say(f"- 完整案例样地: {len(d)}")

def adj_r2(y, X):
    Xc = np.column_stack([np.ones(len(X))] + [X[c].values.astype(float) for c in X.columns])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    ssr = np.sum(resid ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    n, k = len(y), Xc.shape[1] - 1
    r2 = 1 - ssr / sst
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

y = d["plot_RGR"].values
names = list(sets)
sub_r2 = {}
for r in range(1, len(names) + 1):
    for combo in itertools.combinations(names, r):
        cols = sum([sets[c] for c in combo], [])
        sub_r2[combo] = adj_r2(y, d[cols])
full_r2 = sub_r2[tuple(names)]
rows = []
for g in names:
    others = tuple(n for n in names if n != g)
    unique = full_r2 - sub_r2[others]
    rows.append({"group": g, "unique_adjR2": round(unique, 4),
                 "alone_adjR2": round(sub_r2[(g,)], 4)})
# 分层分割（所有排序下增量 R² 平均）
hp = {g: 0.0 for g in names}
for perm in itertools.permutations(names):
    used = []
    for g in perm:
        cols = sum([sets[c] for c in used], [])
        r_before = adj_r2(y, d[cols]) if used else 0.0
        cols2 = cols + sets[g]
        r_after = adj_r2(y, d[cols2])
        hp[g] += (r_after - r_before) / math.factorial(len(names))
for row, g in zip(rows, names):
    row["hier_share"] = round(hp[g] / full_r2, 4)
vp = pd.DataFrame(rows)
vp.loc[len(vp)] = {"group": "TOTAL", "unique_adjR2": "", "alone_adjR2": "",
                   "hier_share": round(full_r2, 4)}
vp.to_csv(os.path.join(C.OUT, "05_varpart_plot.csv"), index=False, encoding="utf-8-sig")
say(vp.to_string(index=False))
say(f"- 全模型调整 R² = {full_r2:.4f}")

# ---------- 4. 逐样地主导类组（OOF-SHAP） ----------
say("\n## 4. 逐样地主导类组（按样地内各类组 mean|SHAP| 占比）")
shap_mat = pd.read_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))
grp_cols = {g: feats_by_group[g] for g in groups}
per_plot = {}
for g, cols in grp_cols.items():
    per_plot[g] = shap_mat[cols].abs().mean(axis=1)
pp = pd.DataFrame(per_plot)
pp["plot_id"] = tab["plot_id"].values
plot_dom = pp.groupby("plot_id").mean()
dominant = plot_dom.idxmax(axis=1)
share_rows = []
for g in groups:
    share_rows.append({"group": g, "mean_share": round(plot_dom[g].div(plot_dom.sum(axis=1)).mean(), 4),
                       "n_plots_dominant": int((dominant == g).sum()),
                       "pct_plots_dominant": round((dominant == g).mean() * 100, 1)})
dom_df = pd.DataFrame(share_rows)
dom_df.to_csv(os.path.join(C.OUT, "05_plot_dominance.csv"), index=False, encoding="utf-8-sig")
say(dom_df.to_string(index=False))

say("\nDONE Phase E")
with open(os.path.join(C.OUT, "05_ablation_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
