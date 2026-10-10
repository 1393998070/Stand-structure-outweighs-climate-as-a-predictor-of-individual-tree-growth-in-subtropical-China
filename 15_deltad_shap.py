# -*- coding: utf-8 -*-
"""Stage 4 · P1-3 lnD₀ 循环性：ΔD（绝对增量）模型变量级 SHAP 分解。
(1) ΔD 模型（同折同参数）OOF-SHAP 变量级份额与类组份额；
(2) lnD₀ 依赖形状：ΔD vs RGR 模型并列对比（定义性警示的证据基础）；
(3) age–lnD₀ 精确相关；RGR vs ΔD 变量级份额差。
输出: 15_deltad_var_shares.csv / 15_deltad_report.md / 15_lnd0_curves.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
group_of = dict(zip(vd["var"], vd["group"]))
shap_rgr = pd.read_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-3 ΔD 模型变量级 SHAP 分解报告\n")
t0 = time.time()

# ---------- 1. ΔD 模型 OOF-SHAP ----------
oof_d, models = C.fit_xgb_folds(tab, feats, params, "fold_spatial", y_col="abs_inc",
                                return_models=True)
parts = []
for k, m in enumerate(models):
    te = tab[tab["fold_spatial"] == k]
    dm = xgb.DMatrix(te[feats])
    sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    parts.append(pd.DataFrame(sv.astype(np.float32), index=te.index, columns=feats))
shap_d = pd.concat(parts).sort_index()
m_d = C.metrics(tab["abs_inc"], oof_d)
say(f"- ΔD 模型 OOF: R2={m_d['R2']:.4f}, Spearman={m_d['Spearman']:.4f}, RMSE={m_d['RMSE']:.3f} cm/5yr")

# ---------- 2. 变量级与类组级份额 ----------
imp_d = pd.DataFrame({"var": feats, "mas_deltaD": shap_d.abs().mean().values})
imp_r = pd.read_csv(os.path.join(C.OUT, "04_shap_importance.csv"))[["var", "mean_abs_shap"]]
cmp = imp_r.merge(imp_d, on="var")
cmp["share_RGR"] = cmp["mean_abs_shap"] / cmp["mean_abs_shap"].sum()
cmp["share_deltaD"] = cmp["mas_deltaD"] / cmp["mas_deltaD"].sum()
cmp["abs_share_diff"] = (cmp["share_RGR"] - cmp["share_deltaD"]).abs()
cmp["group"] = cmp["var"].map(group_of)
cmp = cmp.sort_values("share_RGR", ascending=False)
cmp.to_csv(os.path.join(C.OUT, "15_deltad_var_shares.csv"), index=False, encoding="utf-8-sig")
rho_rank = spearmanr(cmp["share_RGR"], cmp["share_deltaD"]).statistic
say(f"- RGR vs ΔD 变量级份额排序 Spearman ρ = {rho_rank:.3f}")
say("- top-10 对比（share_RGR | share_deltaD | |diff|）:")
say(cmp.head(10)[["var", "group", "share_RGR", "share_deltaD", "abs_share_diff"]].round(4).to_string(index=False))
say("- |份额差| 最大 5 个变量:")
say(cmp.nlargest(5, "abs_share_diff")[["var", "group", "share_RGR", "share_deltaD", "abs_share_diff"]].round(4).to_string(index=False))

fam = cmp.groupby("group")[["share_RGR", "share_deltaD"]].sum()
say("- 类组份额对比:")
say(fam.round(4).to_string())

# ---------- 3. lnD0 依赖形状对比 ----------
say("\n## lnD₀ 依赖形状（RGR vs ΔD，30 分箱均值，形状相关）")
bins = pd.qcut(tab["lnD0"], 30, duplicates="drop")
cur_r = shap_rgr.assign(v=tab["lnD0"].values).groupby(bins, observed=True)["lnD0"].mean()
cur_r_s = shap_rgr.assign(v=tab["lnD0"].values).groupby(bins, observed=True)["lnD0"].mean()
# 正确分箱：值与 SHAP
df_r = pd.DataFrame({"v": tab["lnD0"].values, "s": shap_rgr["lnD0"].values})
df_d = pd.DataFrame({"v": tab["lnD0"].values, "s": shap_d["lnD0"].values})
binned = df_r.groupby(pd.qcut(df_r["v"], 30, duplicates="drop"), observed=True).mean()
binned_d = df_d.groupby(pd.qcut(df_d["v"], 30, duplicates="drop"), observed=True).mean()
def norm_shape(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() or 1.0)
r_shape_corr = np.corrcoef(norm_shape(binned["s"]), norm_shape(binned_d["s"]))[0, 1]
say(f"- 形状相关 r = {r_shape_corr:.3f}（ΔD 的 lnD₀ 效应应近单调正——生物量积累尺寸效应；"
    f"RGR 的 lnD₀ 效应为定义性负混杂）")
pd.to_pickle({"rgr": binned, "deltaD": binned_d}, os.path.join(C.OUT, "15_lnd0_curves.pkl"))
say(f"- RGR 曲线首尾: {binned['s'].iloc[0]:+.4f} → {binned['s'].iloc[-1]:+.4f}")
say(f"- ΔD  曲线首尾: {binned_d['s'].iloc[0]:+.4f} → {binned_d['s'].iloc[-1]:+.4f}")

# ---------- 4. age–lnD0 精确相关 ----------
cc_al = tab.dropna(subset=["stand_age", "lnD0"])
rho_al = spearmanr(cc_al["stand_age"], cc_al["lnD0"]).statistic
say(f"\n- age–lnD₀ Spearman ρ = {rho_al:.3f}（树级，n={len(cc_al)}）")
# 样地级
pt = tab.groupby("plot_id").agg(age=("stand_age", "first"), mlnD0=("lnD0", "mean")).dropna()
rho_pt = spearmanr(pt["age"], pt["mlnD0"]).statistic
say(f"- 样地级 mean(lnD₀)–stand_age Spearman ρ = {rho_pt:.3f}（n={len(pt)}）")

say(f"\n（用时 {time.time()-t0:.0f}s）")
say("\nDONE P1-3")
with open(os.path.join(C.OUT, "15_deltad_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
