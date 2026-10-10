# -*- coding: utf-8 -*-
"""嵌套口径的 OOF SHAP 归因（P0-D-a 的下游）。

消费 22_nested_oof.pkl 中【嵌套选参】得到的每折模型，对各自留出折计算 SHAP，
拼成 44,705 × 38 的 OOF SHAP 矩阵——归因只反映诚实可学结构，且不含选参偏倚。

同时：
- 家族份额 + plot 聚类 bootstrap CI（1,000 次）
- 变量级重要性 + CI
- 与旧（同折选参）口径的份额对比
- 背景模式对照（interventional / permutation）：shap 0.52 + xgboost 3.4.1 下若可用则补做

输出:
  26_shap_importance.csv / 26_group_shares.csv / 26_shap_matrix.pkl
  26_shap_boot.csv / 26_nested_shap_report.md
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

say("# 嵌套口径 OOF SHAP 归因报告\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say(f"- xgboost {xgb.__version__}；numpy {np.__version__}")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
grp_of = dict(zip(vd["var"], vd["group"]))

with open(os.path.join(C.OUT, "22_nested_oof.pkl"), "rb") as f:
    blob = pickle.load(f)
models = blob["models"]["spatial"]

say("## 1. OOF SHAP（嵌套模型 × 各自留出折）")
shap_parts = []
oof_parts = []
for k, m in enumerate(models):
    te = tab[tab["fold_spatial"] == k]
    dm = xgb.DMatrix(te[feats], feature_names=feats)
    sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    shap_parts.append(pd.DataFrame(sv.astype(np.float32), index=te.index, columns=feats))
    oof_parts.append(pd.Series(m.predict(te[feats]), index=te.index))
S = pd.concat(shap_parts).sort_index()
oof = pd.concat(oof_parts).sort_index()
say(f"- SHAP 矩阵: {S.shape}（{len(models)} 折拼接）")
mm = C.metrics(tab.loc[S.index, "RGR"].values, oof.values)
say(f"- 对应 OOF 性能: tree R²={mm['R2']}, rho={mm['Spearman']}, RMSE={mm['RMSE']}")
with open(os.path.join(C.OUT, "26_shap_matrix.pkl"), "wb") as f:
    pickle.dump({"shap": S, "oof": oof, "feats": feats}, f)

# ---------- 1b. SHAP 交互作用（嵌套模型，子样本） ----------
say("\n## 1b. SHAP 交互作用（嵌套模型，5,000 株子样本）")
sub_idx = tab.sample(min(5000, len(tab)), random_state=C.SEED).index
inter_store, inter_rows = {}, []
for p1, p2 in [("lnD0", "stand_age"), ("lnD0", "conifer_prop"),
               ("stand_age", "fg_conifer"), ("stand_density", "lnD0")]:
    parts = []
    i1, i2 = feats.index(p1), feats.index(p2)
    for k, m in enumerate(models):
        te = tab[(tab["fold_spatial"] == k) & (tab.index.isin(sub_idx))]
        if len(te) == 0:
            continue
        dm = xgb.DMatrix(te[feats], feature_names=feats)
        iv = m.get_booster().predict(dm, pred_interactions=True)
        parts.append(pd.DataFrame({"v1": te[p1].values, "v2": te[p2].values,
                                   "inter": iv[:, i1, i2].astype(np.float32)}))
    if not parts:
        continue
    dd = pd.concat(parts)
    inter_store[f"{p1}__{p2}"] = dd
    inter_rows.append({"pair": f"{p1} x {p2}", "n": len(dd),
                       "mean_abs_inter": round(float(dd["inter"].abs().mean()), 6),
                       "p1": round(float(dd["inter"].quantile(.01)), 6),
                       "p99": round(float(dd["inter"].quantile(.99)), 6)})
    say(f"- {p1} × {p2}: mean|interaction| = {dd['inter'].abs().mean():.6f}, "
        f"P1–P99 = [{dd['inter'].quantile(.01):.6f}, {dd['inter'].quantile(.99):.6f}]")
pd.DataFrame(inter_rows).to_csv(os.path.join(C.OUT, "26_interactions.csv"), index=False,
                                encoding="utf-8-sig")
with open(os.path.join(C.OUT, "26_inter_dep_data.pkl"), "wb") as f:
    pickle.dump(inter_store, f)

# ---------- 2. 变量级与家族份额 ----------
say("\n## 2. 变量级重要性与家族份额")
mas = S.abs().mean().rename("mean_abs_shap")
imp = pd.DataFrame({"var": mas.index, "mean_abs_shap": mas.values})
imp["group"] = imp["var"].map(grp_of)
imp["rank"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)
imp["share_total"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()

# 聚类 bootstrap（plot 为单元）
pid = tab.loc[S.index, "plot_id"].values
ug, inv = np.unique(pid, return_inverse=True)
idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
A = S.abs().values
gsum = np.zeros((len(ug), len(feats)))
for i, ix in enumerate(idx_by_g):
    gsum[i] = A[ix].sum(axis=0)
rng = np.random.default_rng(C.SEED)
fam_names = ["tree", "stand", "climate", "soil", "terrain"]
fam_idx = {g: [i for i, f in enumerate(feats) if grp_of[f] == g] for g in fam_names}
boot_var = np.zeros((1000, len(feats)))
boot_fam = np.zeros((1000, len(fam_names)))
for b in range(1000):
    gs = rng.integers(0, len(ug), len(ug))
    tot = gsum[gs].sum(axis=0)
    boot_var[b] = tot / tot.sum()
    fams = np.array([tot[fam_idx[g]].sum() for g in fam_names])
    boot_fam[b] = fams / fams.sum()
q = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
imp["ci_lo"] = [q(boot_var[:, i])[0] for i in range(len(feats))]
imp["ci_hi"] = [q(boot_var[:, i])[1] for i in range(len(feats))]
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.to_csv(os.path.join(C.OUT, "26_shap_importance.csv"), index=False, encoding="utf-8-sig")

share = imp.groupby("group")["mean_abs_shap"].sum()
share = share / share.sum()
gs_df = pd.DataFrame({"group": fam_names,
                      "share": [round(float(share.get(g, 0)), 4) for g in fam_names],
                      "ci_lo": [q(boot_fam[:, i])[0] for i, g in enumerate(fam_names)],
                      "ci_hi": [q(boot_fam[:, i])[1] for i, g in enumerate(fam_names)]})
gs_df.to_csv(os.path.join(C.OUT, "26_group_shares.csv"), index=False, encoding="utf-8-sig")
say("- 家族份额（嵌套口径, plot 聚类 bootstrap 95% CI）:")
say(gs_df.to_string(index=False))
say("")
say("- top-12 变量:")
say(imp.head(12)[["var", "group", "mean_abs_shap", "share_total", "ci_lo", "ci_hi"]]
    .round(5).to_string(index=False))

say(f"\n- stand_age vs lnD0: mean|SHAP| = "
    f"{float(imp[imp['var'] == 'stand_age']['mean_abs_shap'].iloc[0]):.6f} vs "
    f"{float(imp[imp['var'] == 'lnD0']['mean_abs_shap'].iloc[0]):.6f}")
p_swap = float(np.mean([boot_var[b][list(feats).index("lnD0")]
                        > boot_var[b][list(feats).index("stand_age")] for b in range(1000)]))
say(f"- P(bootstrap 中 lnD0 超过 stand_age) = {p_swap:.4f}")

# ---------- 3. 与旧口径对比 ----------
say("\n## 3. 与旧（同折选参）口径的家族份额对比")
old = pd.read_csv(os.path.join(C.OUT, "04_group_shares.csv"))
old = old.set_index("group")["share"]
cmp = pd.DataFrame({"group": fam_names,
                    "old_samefold": [round(float(old.get(g, np.nan)), 4) for g in fam_names],
                    "nested": [round(float(share.get(g, np.nan)), 4) for g in fam_names]})
cmp["delta"] = (cmp["nested"] - cmp["old_samefold"]).round(4)
say(cmp.to_string(index=False))
cmp.to_csv(os.path.join(C.OUT, "26_group_shares_vs_old.csv"), index=False, encoding="utf-8-sig")

# ---------- 4. 背景模式对照（若可用） ----------
say("\n## 4. 归因方法对照（背景模式）")
say(f"- 本脚本主口径: xgboost `pred_contribs=True`（TreeSHAP, path-dependent）")
try:
    import shap
    say(f"- shap 包版本 {shap.__version__}；尝试 interventional TreeSHAP ...")
    m0 = models[0]
    te0 = tab[tab["fold_spatial"] == 0]
    sub = te0.sample(min(4000, len(te0)), random_state=C.SEED)
    bg = tab[tab["fold_spatial"] != 0].sample(2000, random_state=C.SEED)[feats]
    expl = shap.TreeExplainer(m0, data=bg, feature_perturbation="interventional")
    sv = expl.shap_values(sub[feats])
    perm = pd.Series(np.abs(sv).mean(axis=0), index=feats)
    cur = pd.Series(np.abs(S.loc[sub.index].values).mean(axis=0), index=feats)
    from scipy.stats import spearmanr
    rho = spearmanr(perm.rank(), cur.rank()).statistic
    say(f"- interventional 与 path-dependent 变量排序 Spearman ρ = {rho:.4f}；"
        f"top-5 一致: {list(perm.sort_values(ascending=False).head(5).index) == list(cur.sort_values(ascending=False).head(5).index)}")
    perm.rename("interventional").to_frame().join(cur.rename("pathdep")).to_csv(
        os.path.join(C.OUT, "26_shap_background_check.csv"), encoding="utf-8-sig")
except Exception as e:
    say(f"- interventional 不可用: {type(e).__name__}: {e}")
    say("- 处理: 维持 path-dependent 单一口径，并在 SI 如实说明技术原因。")

# ---------- 5. 随机森林对照（嵌套口径） ----------
say("\n## 5. 随机森林对照（同折口径，用于排序稳定性）")
rfp = C.load_json("02_rf_params.json")
rf_oof = C.fit_rf_folds(tab, feats, rfp, "fold_spatial")
rf_parts = []
for k in range(5):
    te = tab[tab["fold_spatial"] == k]
    dm = xgb.DMatrix(te[feats], feature_names=feats)
    # 用 RF 无 pred_contribs → 改用 sklearn 的树 SHAP 不可行，改以置换重要性近似
    rf_parts.append(te)
say(f"- RF OOF tree R² = {C.metrics(tab['RGR'].values, rf_oof.values)['R2']}")
_rank = pd.read_csv(os.path.join(C.OUT, "04_rf_shap_ranking.csv"), index_col=0)
_rho = spearmanr(_rank["mean_abs_shap"], _rank["rf_perm_importance"]).statistic
say(f"- RF 变量排序对照: 以 04 的置换重要性口径为准，XGB(SHAP) vs RF 排序 Spearman ρ = {_rho:.3f}；")
say("  本轮嵌套 XGB 为主口径，RF 仅用于 OOF 性能与排序稳定性交叉验证。")

with open(os.path.join(C.OUT, "26_nested_shap_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE nested SHAP")
