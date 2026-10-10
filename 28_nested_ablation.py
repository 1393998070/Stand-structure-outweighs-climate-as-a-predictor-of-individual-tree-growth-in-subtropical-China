# -*- coding: utf-8 -*-
"""嵌套口径的类组消融与逐样地主导（P0-D-a 下游）。

消融沿用主模型【每折嵌套选出的超参数与迭代数】（不再重新选参，避免二次选参偏倚），
仅在去掉/仅保留某一类组后重训同一折模型，故 ΔR² 与主模型口径一致。

输出: 28_ablation_nested.csv / 28_plot_dominance_nested.csv / 28_ablation_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# 嵌套口径类组消融报告\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
groups = ["tree", "stand", "climate", "soil", "terrain"]
feats_by_group = {g: C.selected_features(vd, [g]) for g in groups}

plog = pd.read_csv(os.path.join(C.OUT, "22_nested_params_log.csv"))
plog = plog[plog["design"] == "spatial"].set_index("fold")
say(f"- 沿用主模型每折嵌套选参: {plog[['n_est', 'max_depth', 'learning_rate']].to_dict('index')}")
say("")

def fit_oof(feat_list, seed_off=0):
    oof = pd.Series(np.nan, index=tab.index)
    for k in range(5):
        row = plog.loc[k]
        p = {kk: (int(row[kk]) if kk in ("max_depth", "min_child_weight") else float(row[kk]))
             for kk in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                        "min_child_weight", "reg_alpha", "reg_lambda")}
        tr = tab[tab["fold_spatial"] != k]
        te = tab[tab["fold_spatial"] == k]
        m = xgb.XGBRegressor(n_estimators=int(row["n_est"]), tree_method="hist",
                             random_state=C.SEED + k + seed_off, n_jobs=-1, **p)
        m.fit(tr[feat_list], tr["RGR"], verbose=False)
        oof.loc[te.index] = m.predict(te[feat_list])
    return oof

def plot_metrics(sub, oof):
    pm = sub.assign(pred=oof).groupby("plot_id")[["RGR", "pred"]].mean()
    return C.metrics(pm["RGR"], pm["pred"])

say("## 1. 类组消融（ΔR² = 全模型 − 去该类组）")
oof_full = fit_oof(feats)
m_full = C.metrics(tab["RGR"], oof_full)
p_full = plot_metrics(tab, oof_full)
say(f"- 全模型: tree R²={m_full['R2']:.4f}, plot R²={p_full['R2']:.4f}")
rows = [{"model": "full", "removed": "none", "n_feats": len(feats),
         "tree_R2": m_full["R2"], "plot_R2": p_full["R2"], "d_tree_R2": 0.0, "d_plot_R2": 0.0}]
for g in groups:
    fs = [f for f in feats if f not in feats_by_group[g]]
    oof = fit_oof(fs)
    m = C.metrics(tab["RGR"], oof); pm = plot_metrics(tab, oof)
    rows.append({"model": f"without_{g}", "removed": g, "n_feats": len(fs),
                 "tree_R2": m["R2"], "plot_R2": pm["R2"],
                 "d_tree_R2": round(m_full["R2"] - m["R2"], 4),
                 "d_plot_R2": round(p_full["R2"] - pm["R2"], 4)})
    say(f"- 去 {g:8s} ({len(fs):2d} feats): tree R²={m['R2']:.4f} "
        f"(Δ={m_full['R2']-m['R2']:+.4f}), plot R²={pm['R2']:.4f} "
        f"(Δ={p_full['R2']-pm['R2']:+.4f})")
for g in groups:
    fs = feats_by_group[g]
    oof = fit_oof(fs)
    m = C.metrics(tab["RGR"], oof); pm = plot_metrics(tab, oof)
    rows.append({"model": f"only_{g}", "removed": f"all_but_{g}", "n_feats": len(fs),
                 "tree_R2": m["R2"], "plot_R2": pm["R2"], "d_tree_R2": "", "d_plot_R2": ""})
    say(f"- 仅 {g:8s} ({len(fs):2d} feats): tree R²={m['R2']:.4f}, plot R²={pm['R2']:.4f}")
pd.DataFrame(rows).to_csv(os.path.join(C.OUT, "28_ablation_nested.csv"), index=False,
                          encoding="utf-8-sig")

say("\n## 2. 逐样地主导类组（嵌套 OOF SHAP）")
S = pd.read_pickle(os.path.join(C.OUT, "26_shap_matrix.pkl"))["shap"]
per_plot = {g: S[feats_by_group[g]].abs().mean(axis=1) for g in groups}
pp = pd.DataFrame(per_plot)
pp["plot_id"] = tab.loc[S.index, "plot_id"].values
plot_dom = pp.groupby("plot_id").mean()
dominant = plot_dom.idxmax(axis=1)
dom_rows = []
for g in groups:
    dom_rows.append({"group": g,
                     "mean_share": round(plot_dom[g].div(plot_dom.sum(axis=1)).mean(), 4),
                     "n_plots_dominant": int((dominant == g).sum()),
                     "pct_plots_dominant": round((dominant == g).mean() * 100, 1)})
dom = pd.DataFrame(dom_rows)
dom.to_csv(os.path.join(C.OUT, "28_plot_dominance_nested.csv"), index=False,
           encoding="utf-8-sig")
say(dom.to_string(index=False))
say("")
say("- 注（R1-W5）：样地内只有 2 个树级变量（lnD0、fg_conifer），故'树大小在 X% 样地内主导'")
say("  是【两变量之间】的对比，不构成与样地级变量（组内恒定）的重要性比较；正文须弱化措辞。")

with open(os.path.join(C.OUT, "28_ablation_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE nested ablation")
