# -*- coding: utf-8 -*-
"""ΔD 伴随模型的嵌套口径重算（P0-D-a 下游 + P1-3 单位修正）。

为与主模型口径一致，ΔD 模型沿用主模型【每折嵌套选出的超参数与迭代数】
（同参数、仅换响应），故两响应之间的差异只来自响应本身。

同时修正单位：模型响应 abs_inc = 直径增量(mm)/10 → **cm per 5 yr**。
v2 正文写作 "mm per 5 yr" 与 Methods 的 "ΔD/5, mm yr⁻¹" 均与实现不符。

输出: 29_deltad_nested.csv / 29_lnd0_curves.pkl / 29_deltad_report.md
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# ΔD 伴随模型（嵌套口径 + 单位修正）报告\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say("- 响应: abs_inc = 直径增量(mm) / 10 → 单位 **cm per 5 yr**")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
grp_of = dict(zip(vd["var"], vd["group"]))
plog = pd.read_csv(os.path.join(C.OUT, "22_nested_params_log.csv"))
plog = plog[plog["design"] == "spatial"].set_index("fold")

# ---------- 1. ΔD 模型（沿用主模型每折嵌套参数） ----------
oof_d = pd.Series(np.nan, index=tab.index)
parts = []
for k in range(5):
    row = plog.loc[k]
    p = {kk: (int(row[kk]) if kk in ("max_depth", "min_child_weight") else float(row[kk]))
         for kk in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                    "min_child_weight", "reg_alpha", "reg_lambda")}
    tr = tab[tab["fold_spatial"] != k]
    te = tab[tab["fold_spatial"] == k]
    m = xgb.XGBRegressor(n_estimators=int(row["n_est"]), tree_method="hist",
                         random_state=C.SEED + k, n_jobs=-1, **p)
    m.fit(tr[feats], tr["abs_inc"], verbose=False)
    oof_d.loc[te.index] = m.predict(te[feats])
    dm = xgb.DMatrix(te[feats], feature_names=feats)
    sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    parts.append(pd.DataFrame(sv.astype(np.float32), index=te.index, columns=feats))
shap_d = pd.concat(parts).sort_index()
m_d = C.metrics(tab["abs_inc"], oof_d)
say(f"- ΔD 模型 OOF（嵌套口径）: R²={m_d['R2']}, Spearman={m_d['Spearman']}, "
    f"RMSE={m_d['RMSE']} cm per 5 yr")

# ---------- 2. 份额对比（与嵌套 RGR 口径） ----------
imp_r = pd.read_csv(os.path.join(C.OUT, "26_shap_importance.csv"))[["var", "mean_abs_shap"]]
imp_d = pd.DataFrame({"var": feats, "mas_deltaD": shap_d.abs().mean().values})
cmp = imp_r.merge(imp_d, on="var")
cmp["share_RGR"] = cmp["mean_abs_shap"] / cmp["mean_abs_shap"].sum()
cmp["share_deltaD"] = cmp["mas_deltaD"] / cmp["mas_deltaD"].sum()
cmp["group"] = cmp["var"].map(grp_of)
rho_v = spearmanr(cmp["share_RGR"].rank(), cmp["share_deltaD"].rank()).statistic
say(f"\n- 变量级份额排序 Spearman ρ = {rho_v:.3f}")
fam = cmp.groupby("group")[["share_RGR", "share_deltaD"]].sum()
rho_f = spearmanr(fam["share_RGR"].rank(), fam["share_deltaD"].rank()).statistic
say(f"- 类组份额排序 Spearman ρ = {rho_f:.3f}")
say(fam.round(4).to_string())
cmp.to_csv(os.path.join(C.OUT, "29_deltad_nested.csv"), index=False, encoding="utf-8-sig")

# ---------- 3. lnD0 依赖形状（RGR vs ΔD） ----------
S_r = pd.read_pickle(os.path.join(C.OUT, "26_shap_matrix.pkl"))["shap"]
df_r = pd.DataFrame({"v": tab["lnD0"].values, "s": S_r["lnD0"].reindex(tab.index).values})
df_d = pd.DataFrame({"v": tab["lnD0"].values, "s": shap_d["lnD0"].values})
b_r = df_r.groupby(pd.qcut(df_r["v"], 30, duplicates="drop"), observed=True).mean()
b_d = df_d.groupby(pd.qcut(df_d["v"], 30, duplicates="drop"), observed=True).mean()
nrm = lambda x: (np.asarray(x, float) - np.mean(x)) / (np.std(x) or 1.0)
r_shape = np.corrcoef(nrm(b_r["s"]), nrm(b_d["s"]))[0, 1]
say(f"\n- lnD₀ 依赖形状相关 r = {r_shape:.3f}")
say(f"- RGR 模型 lnD₀ 曲线首尾: {b_r['s'].iloc[0]:+.4f} → {b_r['s'].iloc[-1]:+.4f} yr⁻¹")
say(f"- ΔD  模型 lnD₀ 曲线首尾: {b_d['s'].iloc[0]:+.4f} → {b_d['s'].iloc[-1]:+.4f} cm per 5 yr")
say(f"- 单位换算: ΔD 曲线首尾 = {b_d['s'].iloc[0]*10:+.2f} → {b_d['s'].iloc[-1]*10:+.2f} mm per 5 yr")
pd.to_pickle({"rgr": b_r, "deltaD": b_d}, os.path.join(C.OUT, "29_lnd0_curves.pkl"))

cc_al = tab.dropna(subset=["stand_age", "lnD0"])
say(f"\n- age–lnD₀ Spearman ρ = "
    f"{spearmanr(cc_al['stand_age'], cc_al['lnD0']).statistic:.3f}（树级, n={len(cc_al)}）")
pt = tab.groupby("plot_id").agg(a=("stand_age", "first"), l=("lnD0", "mean")).dropna()
say(f"- 样地级 mean(lnD₀)–stand_age ρ = {spearmanr(pt['a'], pt['l']).statistic:.3f}（n={len(pt)}）")

with open(os.path.join(C.OUT, "29_deltad_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE ΔD nested")
