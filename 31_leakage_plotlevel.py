# -*- coding: utf-8 -*-
"""嵌套口径的【样地级】泄漏分解（补 22 未覆盖的层级）。

22_nested_cv.py 的泄漏分解只做了 tree 级 R²；本脚本用同一批嵌套 OOF 预测，
在样地级（plot mean）重算三设计的 R² 及其差异，并给出 plot-cluster bootstrap CI。

输出: 31_leakage_plotlevel.csv / 31_leakage_plotlevel_report.md
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

say("# 嵌套口径样地级泄漏分解报告\n")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id"]], on="plot_id", how="left")
with open(os.path.join(C.OUT, "22_nested_oof.pkl"), "rb") as f:
    blob = pickle.load(f)
oof = blob["oof"]

# 样地级观测/预测均值
pdat = tab[["plot_id", "RGR"]].copy()
for d in ("random", "group", "spatial"):
    pdat[d] = oof[d].values
pm = pdat.groupby("plot_id")[["RGR", "random", "group", "spatial"]].mean()
say(f"- 样地数: {len(pm)}")
for d in ("random", "group", "spatial"):
    r2 = 1 - np.sum((pm["RGR"] - pm[d]) ** 2) / np.sum((pm["RGR"] - pm["RGR"].mean()) ** 2)
    say(f"- {d}: plot-level R² = {r2:.4f}")

# bootstrap（以样地为重采样单元）
rng = np.random.default_rng(C.SEED)
y = pm["RGR"].values
P = {d: pm[d].values for d in ("random", "group", "spatial")}
n = len(pm)
boot = {d: [] for d in P}
for _ in range(1000):
    idx = rng.integers(0, n, n)
    yy = y[idx]
    ss = np.sum((yy - yy.mean()) ** 2)
    for d in P:
        hh = P[d][idx]
        boot[d].append(1 - np.sum((yy - hh) ** 2) / ss)
q = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
abs_inf = [boot["random"][i] - boot["spatial"][i] for i in range(1000)]
rel = [100 * a / s for a, s in zip(abs_inf, boot["spatial"])]
within = [boot["random"][i] - boot["group"][i] for i in range(1000)]
blk = [boot["group"][i] - boot["spatial"][i] for i in range(1000)]
wshare = [100 * w / a for w, a in zip(within, abs_inf)]

rows = []
for d in ("random", "group", "spatial"):
    r2 = 1 - np.sum((y - P[d]) ** 2) / np.sum((y - y.mean()) ** 2)
    rows.append({"quantity": f"plot R2 {d}", "value": round(float(r2), 4),
                 "ci": str(q(boot[d]))})
rows += [
    {"quantity": "plot absolute inflation (random - spatial)",
     "value": round(float(rows[0]["value"] - rows[2]["value"]), 4), "ci": str(q(abs_inf))},
    {"quantity": "plot relative inflation (%)",
     "value": round(100 * (rows[0]["value"] - rows[2]["value"]) / rows[2]["value"], 1),
     "ci": str(q(rel))},
    {"quantity": "plot within-plot share of inflation (%)",
     "value": round(100 * (rows[0]["value"] - rows[1]["value"])
                    / (rows[0]["value"] - rows[2]["value"]), 1), "ci": str(q(wshare))},
    {"quantity": "plot block-scale component (group - spatial)",
     "value": round(float(rows[1]["value"] - rows[2]["value"]), 4), "ci": str(q(blk))},
]
res = pd.DataFrame(rows)
say("")
say(res.to_string(index=False))
res.to_csv(os.path.join(C.OUT, "31_leakage_plotlevel.csv"), index=False,
           encoding="utf-8-sig")

with open(os.path.join(C.OUT, "31_leakage_plotlevel_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE plot-level leakage")
