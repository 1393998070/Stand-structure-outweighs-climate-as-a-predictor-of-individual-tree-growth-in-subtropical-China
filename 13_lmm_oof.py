# -*- coding: utf-8 -*-
"""Stage 4 · P1-4 LMM 同口径 OOF 对比。
在相同空间块折上输出 LMM 固定效应 OOF R²（tree + plot 级），样本与 XGB 在完整案例上对齐；
同时报告 null ICC / 全模型隐含 ICC（P1-8）与密度系数（P0-2）。
输出: 13_lmm_oof.csv / 13_lmm_oof_report.md
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import statsmodels.formula.api as smf

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-4 LMM 同口径 OOF 对比报告\n")

BINARY = {"fg_conifer", "origin_nat01", "origin_missing", "age_missing"}
cc = tab.dropna(subset=feats + ["RGR"]).copy()
cc["origin_nat01"] = (cc["origin_natural"] == 1).astype(float)
feats_lmm = [f if f != "origin_natural" else "origin_nat01" for f in feats]
feats_lmm = [f for f in feats_lmm if f != "age_missing"]
say(f"- 完整案例: {len(cc)} 株 / {cc['plot_id'].nunique()} 样地（与 02b_lmm 一致）")
cc["RGR_permille"] = cc["RGR"] * 1000.0

# XGB OOF 对齐（完整案例子集；03 与本脚本对同一 tab 做同一 merge，行序一致）
oof_x = pd.read_pickle(os.path.join(C.OUT, "03_oof_predictions.pkl"))
assert (oof_x["plot_id"].values == tab["plot_id"].values).all(), "03 OOF 行序与 tab 不一致"
cc = cc.assign(oof_spatial_xgb=oof_x["oof_spatial_xgb"].reindex(cc.index).values)
say(f"- XGB OOF 覆盖完整案例: {cc['oof_spatial_xgb'].notna().mean()*100:.1f}%")

def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

# ---------- 逐折 LMM ----------
t0 = time.time()
pred_all = pd.Series(np.nan, index=cc.index)
fold_stats = []
for k in range(5):
    tr_mask = cc["fold_spatial"] != k
    tr, te = cc[tr_mask], cc[~tr_mask]
    # 训练折内标准化（避免泄漏）
    means, sds = {}, {}
    Xte = te[feats_lmm].copy()
    for f in feats_lmm:
        if f not in BINARY:
            means[f], sds[f] = tr[f].mean(), tr[f].std(ddof=0)
            tr = tr.assign(**{f"z_{f}": (tr[f] - means[f]) / sds[f]})
            Xte[f"z_{f}"] = (te[f] - means[f]) / sds[f]
    zvars = [f"z_{f}" if f not in BINARY else f for f in feats_lmm]
    # 秩检查（训练折）
    A = np.column_stack([np.ones(len(tr)), tr[zvars].values])
    from scipy.linalg import qr
    _, R, piv = qr(A, mode="full", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int((diag > diag[0] * 1e-10).sum())
    kept = [zvars[c - 1] for c in sorted(piv[:rank]) if c != 0]
    formula = "RGR_permille ~ " + " + ".join(kept)
    # 同 02b：statsmodels 0.15 的 lbfgs 会在退化点误报收敛，按「RE方差为正 + 似然最大」择优
    cands = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for meth in ("bfgs", "powell", "lbfgs"):
            try:
                cand = smf.mixedlm(formula, tr, groups=tr["plot_id"]).fit(
                    reml=False, method=meth, maxiter=1000)
                cands.append((float(cand.cov_re.iloc[0, 0]), float(cand.llf), cand))
            except Exception:
                pass
    ok_c = [c for c in cands if np.isfinite(c[1]) and c[0] > 1e-6] or \
           [c for c in cands if np.isfinite(c[1])]
    assert ok_c, f"fold {k} 所有优化器失败"
    res = max(ok_c, key=lambda c: c[1])[2]
    # 测试折预测：仅固定效应（新样地无随机效应信息）
    Xte_k = Xte.assign(const=1.0)[["const"] + kept].astype(float)
    pred = (Xte_k.values @ res.params[["Intercept"] + kept].values) / 1000.0
    pred_all.loc[te.index] = pred
    fold_stats.append({"fold": k, "n_test": len(te), "converged": bool(res.converged),
                       "n_kept": len(kept)})
    say(f"- fold {k}: n_test={len(te)}, kept={len(kept)}, converged={res.converged}")
cc["lmm_oof"] = pred_all

ok = cc["lmm_oof"].notna()
sub = cc[ok]
m_lmm_tree = r2(sub["RGR"], sub["lmm_oof"])
pm = sub.groupby("plot_id")[["RGR", "lmm_oof", "oof_spatial_xgb"]].mean()
m_lmm_plot = r2(pm["RGR"], pm["lmm_oof"])
m_xgb_tree = r2(sub["RGR"], sub["oof_spatial_xgb"])
m_xgb_plot = r2(pm["RGR"], pm["oof_spatial_xgb"])
# 全样本 XGB OOF（同一次 03 训练，含非完整案例行——新数据下两者等价）
full = tab.assign(oof=oof_x["oof_spatial_xgb"].reindex(tab.index).values).dropna(subset=["oof"])
m_xgb_full_tree = r2(full["RGR"], full["oof"])
_pmf = full.groupby("plot_id")[["RGR", "oof"]].mean()
m_xgb_full_plot = r2(_pmf["RGR"], _pmf["oof"])
summ = C.load_json("02_lmm_summary.json")
say(f"\n## 完整案例对齐比较（n={len(sub)} 株 / {len(pm)} 样地）")
say(f"- LMM 固定效应 OOF: tree R2={m_lmm_tree:.4f}, plot R2={m_lmm_plot:.4f}")
say(f"- XGB OOF（同一子集）: tree R2={m_xgb_tree:.4f}, plot R2={m_xgb_plot:.4f}")
say(f"- XGB 全样本（{len(full):,}）: tree R2={m_xgb_full_tree:.4f}, plot R2={m_xgb_full_plot:.4f}（03 报告）")
say(f"- LMM in-sample 边际 R2={summ['marginal_R2']:.4f}（02b 报告）→ OOF {m_lmm_tree:.4f}（过拟合幅度小）")

pd.DataFrame([{"model": "LMM_fixed_OOF", "tree_R2": round(m_lmm_tree, 4), "plot_R2": round(m_lmm_plot, 4)},
              {"model": "XGB_OOF_aligned", "tree_R2": round(m_xgb_tree, 4), "plot_R2": round(m_xgb_plot, 4)},
              {"model": "XGB_OOF_full", "tree_R2": round(m_xgb_full_tree, 4),
               "plot_R2": round(m_xgb_full_plot, 4)}]).to_csv(
    os.path.join(C.OUT, "13_lmm_oof.csv"), index=False, encoding="utf-8-sig")

# ---------- ICC 与密度系数引用（P1-8 / P0-2） ----------
say(f"\n## ICC 与关键系数（02b 全样本拟合）")
say(f"- null ICC (样地): {summ['ICC_plot']:.3f}（{summ['ICC_method']}）；全模型隐含 ICC: {summ['ICC_model_implied']:.3f}")
coefs = pd.read_csv(os.path.join(C.OUT, "02_lmm_coefs.csv"), index_col=0)
for v in ("z_stand_density", "Intercept"):
    if v in coefs.index:
        say(f"- {v}: coef={coefs.loc[v,'coef_RGR_per_SD']:+.5f}/SD, t={coefs.loc[v,'t']:.2f}, p={coefs.loc[v,'p']:.3f}")
say(f"（LMM 用时 {time.time()-t0:.0f}s）")

say("\nDONE P1-4")
with open(os.path.join(C.OUT, "13_lmm_oof_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
