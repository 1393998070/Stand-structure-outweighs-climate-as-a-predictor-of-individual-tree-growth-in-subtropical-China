# -*- coding: utf-8 -*-
"""P0-D-a 嵌套交叉验证重跑：消除选参与早停的乐观偏倚。

对三种 CV 设计（random / plot-grouped / spatial block）分别执行嵌套 CV：
外层 5 折 × 内层 4 折 × 24 个候选超参数，内层完成选参与早停迭代数选择，
再以选中配置在外层训练集上重训（不再早停）并预测外层留出折。

输出:
  22_nested_cv_results.csv      三设计 OOF 性能 + 聚类 bootstrap CI
  22_nested_leakage.csv         泄漏分解（random - spatial 等）
  22_nested_params_log.csv      每折选中的超参数与迭代数
  22_nested_oof.pkl             OOF 预测 + 每折模型（供 SHAP 复用）
  22_nested_cv_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P0-D-a 嵌套 CV 重跑报告（诚实选参）\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say(f"- Python {sys.version.split()[0]}；xgboost "
    f"{__import__('xgboost').__version__}；numpy {np.__version__}；pandas {pd.__version__}")
say("- 协议: 外层 5 折 × 内层 4 折 × 24 候选；内层选参+早停，外层重训不早停")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial", "fold_group"]], on="plot_id", how="left")
feats = C.selected_features(vd)
say(f"- 建模表: {tab.shape}，特征 {len(feats)} 个，样地 {tab['plot_id'].nunique()} 个")
say("")

DESIGNS = [("random", "fold_random"), ("group", "fold_group"), ("spatial", "fold_spatial")]
oof_store, model_store, params_all = {}, {}, []
res_rows = []

for name, col in DESIGNS:
    say(f"\n## {name} CV（{col}）")
    t0 = time.time()
    oof, models, chosen, log = C.nested_xgb_oof(tab, feats, col, n_cand=24, verbose=True)
    dt = time.time() - t0
    m = C.metrics(tab["RGR"].values, oof.values)
    pm = tab.assign(_p=oof.values).groupby("plot_id")[["RGR", "_p"]].mean()
    plot_r2 = float(1 - np.sum((pm["RGR"] - pm["_p"]) ** 2)
                    / np.sum((pm["RGR"] - pm["RGR"].mean()) ** 2))
    ci = C.cluster_bootstrap_ci(tab["RGR"].values, oof.values, tab["plot_id"].values)
    res_rows.append({"model": "XGBoost-nested", "cv": name, **m,
                     "plot_R2": round(plot_r2, 4),
                     "ci_tree_R2": str(ci["tree_R2"]), "ci_plot_R2": str(ci["plot_R2"]),
                     "n_est_median": int(np.median([c["n_estimators"] for c in chosen])),
                     "seconds": round(dt, 1)})
    say(f"- tree R² = {m['R2']} (95% CI {ci['tree_R2']})；plot R² = {plot_r2:.4f} "
        f"(95% CI {ci['plot_R2']})；RMSE = {m['RMSE']}")
    say(f"- 每折选中 n_estimators: {[c['n_estimators'] for c in chosen]}（用时 {dt:.0f}s）")
    log.insert(0, "design", name)
    params_all.append(log)
    oof_store[name] = oof
    model_store[name] = models

res = pd.DataFrame(res_rows)
res.to_csv(os.path.join(C.OUT, "22_nested_cv_results.csv"), index=False, encoding="utf-8-sig")
plog = pd.concat(params_all, ignore_index=True)
plog.to_csv(os.path.join(C.OUT, "22_nested_params_log.csv"), index=False, encoding="utf-8-sig")

# ---------- 泄漏分解（嵌套口径） ----------
say("\n## 泄漏分解（嵌套口径，plot-cluster bootstrap 1000 次）")
r2 = res.set_index("cv")["R2"]
def infl(a, b):
    return float(r2[a] - r2[b])

rng = np.random.default_rng(C.SEED)
ug = tab["plot_id"].values
groups_idx = {}
for g, idx in tab.groupby("plot_id").indices.items():
    groups_idx[g] = idx
keys = list(groups_idx)
r2boot = {d: [] for d in r2.index}
y_all = tab["RGR"].values
for _ in range(1000):
    gs = rng.choice(keys, len(keys), replace=True)
    idx = np.concatenate([groups_idx[g] for g in gs])
    yy = y_all[idx]
    ss = np.sum((yy - yy.mean()) ** 2)
    for d in r2.index:
        hh = oof_store[d].values[idx]
        r2boot[d].append(1 - np.sum((yy - hh) ** 2) / ss)
q = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
abs_infl = [r2boot["random"][i] - r2boot["spatial"][i] for i in range(1000)]
within = [r2boot["random"][i] - r2boot["group"][i] for i in range(1000)]
blockc = [r2boot["group"][i] - r2boot["spatial"][i] for i in range(1000)]
rel = [100 * a / s for a, s in zip(abs_infl, r2boot["spatial"])]
within_share = [100 * w / a for w, a in zip(within, abs_infl)]

leak = pd.DataFrame([
    {"quantity": "R2 random", "value": round(r2["random"], 4), "ci": str(q(r2boot["random"]))},
    {"quantity": "R2 group", "value": round(r2["group"], 4), "ci": str(q(r2boot["group"]))},
    {"quantity": "R2 spatial", "value": round(r2["spatial"], 4), "ci": str(q(r2boot["spatial"]))},
    {"quantity": "absolute inflation (random - spatial)",
     "value": round(infl("random", "spatial"), 4), "ci": str(q(abs_infl))},
    {"quantity": "relative inflation (%)",
     "value": round(100 * infl("random", "spatial") / r2["spatial"], 1), "ci": str(q(rel))},
    {"quantity": "within-plot share of inflation (%)",
     "value": round(100 * infl("random", "group") / infl("random", "spatial"), 1),
     "ci": str(q(within_share))},
    {"quantity": "block-scale component (group - spatial)",
     "value": round(infl("group", "spatial"), 4), "ci": str(q(blockc))},
])
leak.to_csv(os.path.join(C.OUT, "22_nested_leakage.csv"), index=False, encoding="utf-8-sig")
say(leak.to_string(index=False))

# ---------- 与旧（同折选参）口径对比 ----------
old = pd.read_csv(os.path.join(C.OUT, "03_triple_cv_results.csv"))
old = old[old["model"] == "XGBoost"].set_index("cv")
say("\n## 与旧口径（同折选参+测试折早停）对比")
cmp_rows = []
for d in ["random", "group", "spatial"]:
    cmp_rows.append({"cv": d, "old_R2": round(float(old.loc[d, "R2"]), 4),
                     "nested_R2": round(float(r2[d]), 4),
                     "delta": round(float(old.loc[d, "R2"]) - float(r2[d]), 4)})
cmpdf = pd.DataFrame(cmp_rows)
say(cmpdf.to_string(index=False))
cmpdf.to_csv(os.path.join(C.OUT, "22_nested_vs_old.csv"), index=False, encoding="utf-8-sig")

import pickle
with open(os.path.join(C.OUT, "22_nested_oof.pkl"), "wb") as f:
    pickle.dump({"oof": oof_store, "models": model_store, "params_log": plog}, f)

with open(os.path.join(C.OUT, "22_nested_cv_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE P0-D-a")
