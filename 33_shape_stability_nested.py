# -*- coding: utf-8 -*-
"""嵌套口径 SHAP 依赖形状的跨折稳定性（供 Supplementary Fig. 3 与正文使用）。

输入: 26_shap_matrix.pkl（嵌套模型 × 各自留出折的 OOF SHAP 矩阵）
输出: 33_shape_curves.pkl / 33_shape_stability.csv / 33_shape_stability_report.md

方法与旧口径（16_gam_stability.py）一致：每折内 15 分箱均值 → 公共 x 网格插值 →
10 个折对的 Pearson r。另附 stand_density 分箱均值跨度（嵌套口径）。
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []


def say(x):
    print(x, flush=True)
    report.append(str(x))


say("# 嵌套口径 SHAP 依赖形状跨折稳定性\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")

with open(os.path.join(C.OUT, "26_shap_matrix.pkl"), "rb") as f:
    blob = pickle.load(f)
S = blob["shap"]
fold_of = tab.loc[S.index, "fold_spatial"].values

order = ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]
curves, rows = {}, []
for v in order:
    x = tab.loc[S.index, v].values.astype(float)
    y = S[v].values.astype(float)
    fc = {}
    for k in range(5):
        m = fold_of == k
        xk, yk = x[m], y[m]
        ok = np.isfinite(xk) & np.isfinite(yk)
        xk, yk = xk[ok], yk[ok]
        q = pd.qcut(pd.Series(xk), 15, duplicates="drop")
        gm = pd.DataFrame({"x": xk, "y": yk}).groupby(q, observed=True).mean()
        fc[k] = {"x": gm["x"].values, "y": gm["y"].values}
    curves[v] = fc
    lo = max(c["x"].min() for c in fc.values())
    hi = min(c["x"].max() for c in fc.values())
    grid = np.linspace(lo, hi, 50)
    interp = np.array([np.interp(grid, c["x"], c["y"]) for c in fc.values()])
    rs = [float(np.corrcoef(interp[i], interp[j])[0, 1])
          for i in range(5) for j in range(i + 1, 5)]
    rows.append({"var": v,
                 "mean_pairwise_r": round(float(np.mean(rs)), 3),
                 "min_pairwise_r": round(float(np.min(rs)), 3),
                 "n_pairs": len(rs)})
    say(f"- {v}: mean r = {np.mean(rs):.3f}, min r = {np.min(rs):.3f}")

pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "33_shape_stability.csv"),
                          index=False, encoding="utf-8-sig")
with open(os.path.join(C.OUT, "33_shape_curves.pkl"), "wb") as f:
    pickle.dump(curves, f)

den = pd.DataFrame({"x": tab.loc[S.index, "stand_density"].values,
                    "y": S["stand_density"].values}).dropna()
b = den.groupby(pd.qcut(den["x"], 30, duplicates="drop"), observed=True)["y"].mean()
say("")
say(f"- stand_density 分箱均值跨度 = {b.max() - b.min():.6f} yr^-1 "
    f"（观测密度范围 {den['x'].min():.0f}–{den['x'].max():.0f} trees ha^-1）")

with open(os.path.join(C.OUT, "33_shape_stability_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE 33")
