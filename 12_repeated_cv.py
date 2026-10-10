# -*- coding: utf-8 -*-
"""Stage 4 · P1-2 泄漏论断精度与外推收敛。
(1) 12 次重复空间 CV（块→折重分配，XGB）→ 空间基线 mean±SD；
(2) 完整泄漏分解（tree/plot × XGB/RF，固定折）+ 按样地聚类 bootstrap CI（含 120%/147% 比值 CI）；
(3) 种子稳定性（P2-2）：XGB 3 个额外随机种子。
输出: 12_repeated_cv_results.csv / 12_leakage_decomposition.csv / 12_repeated_cv_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "block05", "fold_spatial", "fold_group"]], on="plot_id", how="left")
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
rf_params = C.load_json("02_rf_params.json")

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-2 重复空间 CV 与泄漏分解报告\n")

def plot_metrics(sub, oof):
    pm = sub.assign(pred=oof).groupby("plot_id")[["RGR", "pred"]].mean()
    return C.metrics(pm["RGR"], pm["pred"])

def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

# ---------- 1. 12 次重复空间 CV ----------
say("## 1. 重复空间 CV（12 次块→折重分配，XGB）")
blk_trees = tab.groupby("block05").size().sort_values(ascending=False)
blocks = list(blk_trees.items())
rep_rows = []
t0 = time.time()
for rep in range(12):
    rng = np.random.default_rng(C.SEED + 1000 + rep)
    order = rng.permutation(len(blocks))
    assign = C.assign_spatial_folds([blocks[i] for i in order], n_folds=5)
    tab["fold_sp_rep"] = tab["block05"].map(assign)
    oof = C.fit_xgb_folds(tab, feats, params, "fold_sp_rep")
    m = C.metrics(tab["RGR"], oof)
    pm = plot_metrics(tab, oof)
    rep_rows.append({"rep": rep, "tree_R2": m["R2"], "tree_Spearman": m["Spearman"],
                     "plot_R2": pm["R2"]})
    say(f"- rep {rep:02d}: tree R2={m['R2']:.4f}, plot R2={pm['R2']:.4f}")
rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(os.path.join(C.OUT, "12_repeated_cv_results.csv"), index=False, encoding="utf-8-sig")
say(f"- 空间块 CV: tree R2 = {rep_df['tree_R2'].mean():.4f} ± {rep_df['tree_R2'].std():.4f} "
    f"(range {rep_df['tree_R2'].min():.4f}–{rep_df['tree_R2'].max():.4f})")
say(f"-            plot R2 = {rep_df['plot_R2'].mean():.4f} ± {rep_df['plot_R2'].std():.4f}")
say(f"（用时 {time.time()-t0:.0f}s）")

# ---------- 2. 固定折三重 CV OOF（XGB + RF） ----------
say("\n## 2. 固定折三重 CV（XGB + RF，供分解与 bootstrap）")
oofs = {}
for name, fc in [("random", "fold_random"), ("group", "fold_group"), ("spatial", "fold_spatial")]:
    t1 = time.time()
    oofs[("xgb", name)] = C.fit_xgb_folds(tab, feats, params, fc)
    say(f"- XGB {name:8s}: tree R2={r2(tab['RGR'], oofs[('xgb', name)]):.4f} ({time.time()-t1:.0f}s)")
for name, fc in [("random", "fold_random"), ("group", "fold_group"), ("spatial", "fold_spatial")]:
    t1 = time.time()
    oofs[("rf", name)] = C.fit_rf_folds(tab, feats, rf_params, fc)
    say(f"- RF  {name:8s}: tree R2={r2(tab['RGR'], oofs[('rf', name)]):.4f} ({time.time()-t1:.0f}s)")

# ---------- 3. 聚类 bootstrap 分解表 ----------
say("\n## 3. 泄漏分解（按样地聚类 bootstrap，1000 次）")
y = tab["RGR"].values
groups = tab["plot_id"].values
ug, inv = np.unique(groups, return_inverse=True)
idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
cnt = np.bincount(inv, minlength=len(ug)).astype(float)
pm_y = np.bincount(inv, weights=y, minlength=len(ug)) / cnt

def boot_stats(preds, n_boot=1000, seed=C.SEED):
    """preds: dict cv_name -> array; 返回各统计量的点估计与 95% CI。"""
    pm_p = {k: np.bincount(inv, weights=np.asarray(v, float), minlength=len(ug)) / cnt
            for k, v in preds.items()}
    rng = np.random.default_rng(seed)
    tr = {k: [] for k in preds}; pl = {k: [] for k in preds}
    for _ in range(n_boot):
        gs = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[g] for g in gs])
        yy = y[idx]
        for k, v in preds.items():
            vv = np.asarray(v, float)[idx]
            tr[k].append(r2(yy, vv))
            o, pr = pm_y[gs], pm_p[k][gs]
            pl[k].append(r2(o, pr))
    return tr, pl

tr_x, pl_x = boot_stats({k: oofs[("xgb", k)] for k in ("random", "group", "spatial")})
tr_r, pl_r = boot_stats({k: oofs[("rf", k)] for k in ("random", "group", "spatial")})

def ci(a):
    return (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))

rows = []
for model, tr, pl in [("XGB", tr_x, pl_x), ("RF", tr_r, pl_r)]:
    for lvl, d in [("tree", tr), ("plot", pl)]:
        for k in ("random", "group", "spatial"):
            rows.append({"model": model, "level": lvl, "cv": k,
                         "R2": round(float(np.mean(d[k])), 4), "ci_lo": ci(d[k])[0], "ci_hi": ci(d[k])[1]})
dec = pd.DataFrame(rows)
dec.to_csv(os.path.join(C.OUT, "12_leakage_decomposition.csv"), index=False, encoding="utf-8-sig")
say(dec.to_string(index=False))

# 派生量 CI
def derived(model, tr, pl):
    out = []
    for lvl, d, in [("tree", tr), ("plot", pl)]:
        infl = (np.array(d["random"]) - np.array(d["spatial"])) / np.array(d["spatial"])
        within = (np.array(d["random"]) - np.array(d["group"])) / (np.array(d["random"]) - np.array(d["spatial"]))
        beyond = (np.array(d["group"]) - np.array(d["spatial"]))
        out.append({"model": model, "level": lvl,
                    "inflation_abs": (round(float(np.mean(np.array(d['random'])-np.array(d['spatial']))),4),
                                      ci(np.array(d['random'])-np.array(d['spatial']))[0], ci(np.array(d['random'])-np.array(d['spatial']))[1]),
                    "inflation_rel_pct": (round(float(np.mean(infl*100)),1), ci(infl*100)[0], ci(infl*100)[1]),
                    "within_plot_share": (round(float(np.mean(within)),4), ci(within)[0], ci(within)[1]),
                    "spatial_beyond_plot_abs": (round(float(np.mean(beyond)),4), ci(beyond)[0], ci(beyond)[1])})
    return out

say("\n## 派生量（点估计 [95% CI]）")
for model, tr, pl in [("XGB", tr_x, pl_x), ("RF", tr_r, pl_r)]:
    for row in derived(model, tr, pl):
        say(f"- {model} {row['level']}: 随机CV虚高 {row['inflation_abs'][0]*1000:+.1f}‰ [{row['inflation_abs'][1]*1000:.1f},{row['inflation_abs'][2]*1000:.1f}] | "
            f"相对 {row['inflation_rel_pct'][0]:+.0f}% [{row['inflation_rel_pct'][1]:.0f},{row['inflation_rel_pct'][2]:.0f}] | "
            f"样地内共享占虚高比例 {row['within_plot_share'][0]*100:.1f}% [{row['within_plot_share'][1]*100:.1f},{row['within_plot_share'][2]*100:.1f}] | "
            f"块尺度分量 {row['spatial_beyond_plot_abs'][0]:+.4f} [{row['spatial_beyond_plot_abs'][1]:+.4f},{row['spatial_beyond_plot_abs'][2]:+.4f}]")

# ---------- 4. 种子稳定性（P2-2） ----------
say("\n## 4. 种子稳定性（XGB，空间块折，3 个附加种子）")
seed_rows = []
for s in (7, 123, 2024):
    oof = C.fit_xgb_folds(tab, feats, params, "fold_spatial", seed=s)
    m = C.metrics(tab["RGR"], oof)
    seed_rows.append({"seed": s, "tree_R2": m["R2"], "Spearman": m["Spearman"]})
    say(f"- seed {s}: tree R2={m['R2']:.4f}")
pd.DataFrame(seed_rows).to_csv(os.path.join(C.OUT, "12_seed_stability.csv"), index=False, encoding="utf-8-sig")

say("\nDONE P1-2")
with open(os.path.join(C.OUT, "12_repeated_cv_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
