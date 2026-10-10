# -*- coding: utf-8 -*-
"""Phase D 可解释性（RQ1/RQ2）：OOF-SHAP + 类组贡献占比 + 依赖图数据 + 交互 + 排序稳定性 + GAM 对照。
输出: 04_shap_matrix.pkl / 04_shap_importance.csv / 04_group_shares.csv / 04_rf_shap_ranking.csv
      04_interactions.csv / 04_gam_curves.pkl / 04_shap_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import xgboost as xgb

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
rf_params = C.load_json("02_rf_params.json")
group_of = dict(zip(vd["var"], vd["group"]))

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase D OOF-SHAP 可解释性报告\n")

# ---------- 1. OOF-SHAP（空间块折，各折模型解释各自测试折） ----------
t0 = time.time()
oof, models = C.fit_xgb_folds(tab, feats, params, "fold_spatial", return_models=True)
shap_parts = []
for k, m in enumerate(models):
    te = tab[tab["fold_spatial"] == k]
    dm = xgb.DMatrix(te[feats])
    sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    shap_parts.append(pd.DataFrame(sv.astype(np.float32), index=te.index, columns=feats))
shap_mat = pd.concat(shap_parts).sort_index()
assert shap_mat.notna().all().all()
say(f"- OOF-SHAP 矩阵: {shap_mat.shape}（用时 {time.time()-t0:.0f}s）")

# ---------- 2. 全局重要性 + 类组贡献占比 ----------
imp = pd.DataFrame({"var": feats,
                    "mean_abs_shap": shap_mat.abs().mean().values})
imp["group"] = imp["var"].map(group_of)
imp["share_total"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
imp["rank"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp.to_csv(os.path.join(C.OUT, "04_shap_importance.csv"), index=False, encoding="utf-8-sig")

gs = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
gs_share = (gs / gs.sum())
say("\n## 类组贡献占比（mean|SHAP| 归一化）")
say(gs_share.round(4).to_string())
gs_df = pd.DataFrame({"group": gs_share.index, "share": gs_share.values,
                      "mean_abs_shap_sum": gs.values})
gs_df.to_csv(os.path.join(C.OUT, "04_group_shares.csv"), index=False, encoding="utf-8-sig")
say("\n## Top-15 变量")
say(imp.head(15)[["rank", "var", "group", "mean_abs_shap", "share_total"]].to_string(index=False))

# ---------- 3. RF 排序稳定性（置换重要性对照） ----------
# 注: 原实现用 shap.TreeExplainer 解释 RF。在全量训练折（35k 行）上长出的深树使该计算
# 单折耗时 >5 分钟（5 折不可接受），故改为 sklearn 置换重要性——同为"跨算法变量排序"
# 对照，且确定性、无需二进制扩展。
say("\n## RF 对照排序稳定性（置换重要性，n_repeats=5）")
from sklearn.inspection import permutation_importance
t0 = time.time()
rf_oof, rf_models = C.fit_rf_folds(tab, feats, rf_params, "fold_spatial", return_models=True)
rf_parts = []
for k, m in enumerate(rf_models):
    te = tab[tab["fold_spatial"] == k]
    sub = te.sample(min(3000, len(te)), random_state=C.SEED)
    pi = permutation_importance(m, sub[feats], sub["RGR"], n_repeats=5,
                                random_state=C.SEED, n_jobs=-1)
    rf_parts.append(pd.Series(pi.importances_mean, index=feats))
rf_imp = pd.concat(rf_parts, axis=1).mean(axis=1).rename("rf_perm_importance")
rank_cmp = imp.set_index("var").join(rf_imp)
rank_cmp["rf_rank"] = rank_cmp["rf_perm_importance"].rank(ascending=False)
rho_rank = spearmanr(rank_cmp["mean_abs_shap"], rank_cmp["rf_perm_importance"]).statistic
top10_xgb = set(rank_cmp.nsmallest(10, "rank").index)
top10_rf = set(rank_cmp.nsmallest(10, "rf_rank").index)
overlap = len(top10_xgb & top10_rf)
say(f"- XGB(SHAP) vs RF(置换重要性) 排名 Spearman ρ = {rho_rank:.3f}；top-10 重合 {overlap}/10（用时 {time.time()-t0:.0f}s）")
say(f"- 重合变量: {sorted(top10_xgb & top10_rf)}")
rank_cmp[["mean_abs_shap", "rank", "rf_perm_importance", "rf_rank"]].to_csv(
    os.path.join(C.OUT, "04_rf_shap_ranking.csv"), encoding="utf-8-sig")

# ---------- 4. Bootstrap CI（8000 树子样本重抽样） ----------
say("\n## mean|SHAP| bootstrap CI（8000 树 × 1000 次）")
t0 = time.time()
rng = np.random.default_rng(C.SEED)
sub_idx = rng.choice(len(shap_mat), 8000, replace=True)
sv_sub = shap_mat.values[sub_idx]
n_boot = 1000
boot_means = np.empty((n_boot, len(feats)))
for b in range(n_boot):
    pick = rng.integers(0, 8000, 8000)
    boot_means[b] = np.abs(sv_sub[pick]).mean(axis=0)
ci_df = pd.DataFrame({"var": feats,
                      "ci_lo": np.percentile(boot_means, 2.5, axis=0),
                      "ci_hi": np.percentile(boot_means, 97.5, axis=0)})
in_top10 = (boot_means.argsort(axis=1)[:, -10:]).ravel()
counts = np.bincount(in_top10, minlength=len(feats))
ci_df["p_in_top10"] = counts / n_boot
imp = imp.drop(columns=["ci_lo", "ci_hi", "p_in_top10"], errors="ignore").merge(ci_df, on="var")
say(f"- 用时 {time.time()-t0:.0f}s")
imp.to_csv(os.path.join(C.OUT, "04_shap_importance.csv"), index=False, encoding="utf-8-sig")

# ---------- 5. 依赖图数据（top-12 变量） ----------
top12 = imp.head(12)["var"].tolist()
dep = {}
for v in top12:
    dep[v] = pd.DataFrame({"value": tab[v].values, "shap": shap_mat[v].values,
                           "fg_conifer": tab["fg_conifer"].values})
pd.to_pickle(dep, os.path.join(C.OUT, "04_dep_data.pkl"))
say(f"\n- 依赖图数据已保存: {top12}")

# ---------- 6. SHAP 交互（8000 子样本） ----------
say("\n## SHAP 交互值")
t0 = time.time()
idx8k = rng.choice(len(tab), 8000, replace=False)
tab8k = tab.iloc[idx8k]
shap8k = shap_mat.iloc[idx8k]
dm = xgb.DMatrix(tab8k[feats])
# 用全折模型中树数最多者近似（交互用于形态探索，非严格 OOF）
m_full = models[int(np.argmax([mm.best_iteration if hasattr(mm, 'best_iteration') else 0 for mm in models]))]
inter = m_full.get_booster().predict(dm, pred_interactions=True)[:, :-1, :-1]
p = len(feats)
pair_stat = []
for i in range(p):
    for j in range(i + 1, p):
        pair_stat.append({"v1": feats[i], "v2": feats[j],
                          "mean_abs_inter": float(np.abs(inter[:, i, j]).mean())})
pair_df = pd.DataFrame(pair_stat).sort_values("mean_abs_inter", ascending=False).reset_index(drop=True)
pair_df.to_csv(os.path.join(C.OUT, "04_interactions.csv"), index=False, encoding="utf-8-sig")
say(f"- Top-10 交互对:")
say(pair_df.head(10).to_string(index=False))
say(f"（用时 {time.time()-t0:.0f}s）")

# 关键假设交互对的依赖数据
key_pairs = []
for a, b in [("stand_age", "fg_conifer"), ("stand_density", "lnD0"), ("bio12", "soil_SOC"),
             ("stand_age", "stand_density"), ("lnD0", "fg_conifer")]:
    if a in feats and b in feats:
        key_pairs.append((a, b))
inter_dep = {}
for a, b in key_pairs:
    i, j = feats.index(a), feats.index(b)
    inter_dep[f"{a}__{b}"] = pd.DataFrame({"v1": tab8k[a].values, "v2": tab8k[b].values,
                                           "inter": inter[:, i, j]})
pd.to_pickle(inter_dep, os.path.join(C.OUT, "04_inter_dep_data.pkl"))

# ---------- 7. GAM 平滑对照（top-5 变量） ----------
say("\n## GAM 对照（statsmodels GLMGam, B 样条）")
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.gam.smooth_basis import BSplines
top5 = imp.head(5)["var"].tolist()
cc = tab.dropna(subset=top5).sample(min(30000, len(tab)), random_state=C.SEED)
bs = BSplines(cc[top5].values, df=[10] * len(top5), degree=[3] * len(top5))
gres = GLMGam(cc["RGR"].values, smoother=bs).fit()
gam_curves = {}
for v in top5:
    grid = np.linspace(cc[v].quantile(0.005), cc[v].quantile(0.995), 80)
    base = cc[top5].median().to_dict()
    Xn = np.array([[{**base, v: g}[t] for t in top5] for g in grid])
    pred = np.asarray(gres.predict(exog_smooth=Xn, transform=True))
    shap_cc = shap_mat.loc[cc.index]
    binned = pd.DataFrame({"val": cc[v].values, "shap": shap_cc[v].values}).groupby(
        pd.cut(cc[v], 30, duplicates="drop"), observed=True)["shap"].mean()
    centers = np.array([iv.mid for iv in binned.index], dtype=float)
    gam_curves[v] = {"gam": pd.DataFrame({v: grid, "gam_effect": pred}),
                     "shap_binned": pd.DataFrame({"x": centers, "shap": binned.values})}
    gam_std = (pred - pred.mean()) / (pred.std() or 1.0)
    shap_std = (binned.values - binned.values.mean()) / (binned.values.std() or 1.0)
    r = float(np.corrcoef(np.interp(centers, grid, gam_std), shap_std)[0, 1])
    say(f"- {v}: GAM–SHAP 形态相关 r = {r:.3f}")
pd.to_pickle(gam_curves, os.path.join(C.OUT, "04_gam_curves.pkl"))

# 保存 SHAP 矩阵
shap_mat.to_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))
say("\nDONE Phase D")
with open(os.path.join(C.OUT, "04_shap_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
