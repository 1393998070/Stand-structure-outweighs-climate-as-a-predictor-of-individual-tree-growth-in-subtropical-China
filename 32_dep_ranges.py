# -*- coding: utf-8 -*-
"""效应形态与样本支撑的数值快照（供正文与图注引用）。

从嵌套口径 OOF SHAP 矩阵计算六个主变量的 30 分箱均值曲线端点、密度主效应跨度，
以及林龄分箱的样地计数（用于饱和段证据强度的如实描述）。

输出: 32_dep_ranges.csv / 32_dep_ranges_report.md
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# 效应形态数值快照（嵌套口径）\n")

tab, vd, folds = C.load_model_data()
S = pd.read_pickle(os.path.join(C.OUT, "26_shap_matrix.pkl"))["shap"].reindex(tab.index)

rows = []
say("## 1. 30 分箱均值曲线端点（SHAP 单位: yr⁻¹）")
for v in ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]:
    d = pd.DataFrame({"v": tab[v].values, "s": S[v].values}).dropna()
    g = d.groupby(pd.qcut(d["v"], 30, duplicates="drop"), observed=True).mean()
    rows.append({"var": v, "curve_first": round(float(g["s"].iloc[0]), 5),
                 "curve_last": round(float(g["s"].iloc[-1]), 5),
                 "curve_min": round(float(g["s"].min()), 5),
                 "curve_max": round(float(g["s"].max()), 5),
                 "span": round(float(g["s"].max() - g["s"].min()), 5)})
    say(f"- {v:16s}: {g['s'].iloc[0]:+.4f} → {g['s'].iloc[-1]:+.4f}  "
        f"(min {g['s'].min():+.4f}, max {g['s'].max():+.4f}, span {g['s'].max()-g['s'].min():.4f})")

say("")
d = pd.DataFrame({"v": tab["stand_density"].values, "s": S["stand_density"].values}).dropna()
g = d.groupby(pd.qcut(d["v"], 30, duplicates="drop"), observed=True).mean()
say(f"## 2. 林分密度主效应跨度（分箱均值）: {g['s'].max()-g['s'].min():.4f} yr⁻¹")
say(f"- 观测密度范围: {d['v'].min():.0f} – {d['v'].max():.0f} trees ha⁻¹")

say("")
pt = tab.groupby("plot_id").agg(a=("stand_age", "first")).dropna()
# 与 20_si_figures.py 的 SF1 口径一致：左闭右开 [0,10), [10,20), ...
b = pd.cut(pt["a"], bins=list(range(0, 90, 10)) + [200], right=False,
           labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70",
                   "70-80", ">80"])
say("## 3. 林龄分箱的样地计数（饱和段证据强度）")
for k, v in pt.groupby(b, observed=True).size().items():
    say(f"- {k} yr: {v} plots")
say(f"- ≥40 yr: {int((pt['a'] >= 40).sum())} plots; ≥50 yr: {int((pt['a'] >= 50).sum())} plots; "
    f">60 yr: {int((pt['a'] > 60).sum())} plots")

pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "32_dep_ranges.csv"), index=False,
                          encoding="utf-8-sig")
with open(os.path.join(C.OUT, "32_dep_ranges_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE dep ranges")
