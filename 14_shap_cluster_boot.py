# -*- coding: utf-8 -*-
"""Stage 4 · P1-5 SHAP 置信区间（按样地聚类 bootstrap）+ P2-2 SHAP 背景模式对照。
(1) 按样地 cluster bootstrap（1000 次）重算全部 mean|SHAP| 与类组份额 → 95% CI；
    检验 stand_age vs lnD0 是否区间重叠（"single strongest predictor" 收敛判定）；
(2) 空间块级（更保守）重采样对照；
(3) SHAP 背景模式对照：XGB pred_contribs（tree_path_dependent）vs shap.TreeExplainer
    interventional（背景=训练集子样本）在 4000 树子样本上的排序一致性。
输出: 14_shap_cluster_boot.csv / 14_shap_background_check.csv / 14_shap_boot_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import xgboost as xgb

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "block05", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]
group_of = dict(zip(vd["var"], vd["group"]))
shap_mat = pd.read_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-5 SHAP 聚类 bootstrap CI 报告\n")
t0 = time.time()

groups = tab["plot_id"].values
blocks = tab["block05"].values
S = np.abs(shap_mat.values)  # 44705 x 38
fam_idx = {}
for g in ("tree", "stand", "climate", "soil", "terrain"):
    fam_idx[g] = np.array([i for i, f in enumerate(feats) if group_of[f] == g])

def cluster_boot(unit, n_boot=1000, seed=C.SEED):
    ug, inv = np.unique(unit, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    var_means = np.empty((n_boot, len(feats)))
    fam_shares = np.empty((n_boot, 5))
    for b in range(n_boot):
        gs = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[g] for g in gs])
        vm = S[idx].mean(axis=0)
        var_means[b] = vm
        tot = sum(vm[fam_idx[g]].sum() for g in fam_idx)
        for j, g in enumerate(("tree", "stand", "climate", "soil", "terrain")):
            fam_shares[b, j] = vm[fam_idx[g]].sum() / tot
    return var_means, fam_shares

for unit_name, unit in [("plot", groups), ("block", blocks)]:
    vm, fs = cluster_boot(unit)
    rows = []
    for j, g in enumerate(("tree", "stand", "climate", "soil", "terrain")):
        rows.append({"unit": unit_name, "item": f"family_{g}",
                     "mean": round(float(fs[:, j].mean()), 4),
                     "ci_lo": round(float(np.percentile(fs[:, j], 2.5)), 4),
                     "ci_hi": round(float(np.percentile(fs[:, j], 97.5)), 4)})
    for i, f in enumerate(feats):
        rows.append({"unit": unit_name, "item": f"var_{f}",
                     "mean": round(float(vm[:, i].mean()), 6),
                     "ci_lo": round(float(np.percentile(vm[:, i], 2.5)), 6),
                     "ci_hi": round(float(np.percentile(vm[:, i], 97.5)), 6)})
    boot_df = pd.DataFrame(rows)
    if unit_name == "plot":
        boot_all = boot_df
    else:
        boot_all = pd.concat([boot_all, boot_df])
    say(f"## 重采样单元: {unit_name}")
    for r in rows[:5]:
        say(f"- {r['item']}: {r['mean']} [{r['ci_lo']}, {r['ci_hi']}]")
boot_all.to_csv(os.path.join(C.OUT, "14_shap_cluster_boot.csv"), index=False, encoding="utf-8-sig")

# stand_age vs lnD0 重叠判定（plot 级单元）
vm_p, _ = cluster_boot(groups, n_boot=1000, seed=C.SEED + 1)
i_age, i_d0 = feats.index("stand_age"), feats.index("lnD0")
p_d0_higher = float((vm_p[:, i_d0] > vm_p[:, i_age]).mean())
ci_age = (np.percentile(vm_p[:, i_age], 2.5), np.percentile(vm_p[:, i_age], 97.5))
ci_d0 = (np.percentile(vm_p[:, i_d0], 2.5), np.percentile(vm_p[:, i_d0], 97.5))
overlap = max(0.0, min(ci_age[1], ci_d0[1]) - max(ci_age[0], ci_d0[0]))
say(f"\n## stand_age vs lnD0（plot 级 bootstrap）")
say(f"- stand_age mean|SHAP| CI: [{ci_age[0]:.6f}, {ci_age[1]:.6f}]")
say(f"- lnD0      mean|SHAP| CI: [{ci_d0[0]:.6f}, {ci_d0[1]:.6f}]")
say(f"- 区间重叠长度: {overlap:.6f}；P(bootstrap 中 lnD0 > stand_age) = {p_d0_higher:.3f}")
verdict = "区间重叠 → 'single strongest' 需收敛为 '并列最强两个之一'" if overlap > 0 else \
          "区间不重叠 → 'single strongest predictor' 成立"
say(f"- 判定: {verdict}")

# ---------- SHAP 背景模式对照（P2-2） ----------
say("\n## SHAP 背景模式对照（4000 树子样本）")
import shap
from scipy.stats import spearmanr
t1 = time.time()
sub = tab.sample(4000, random_state=C.SEED)
try:
    # xgboost 2.x 将自动估计的 base_score 以数组序列化，当前 shap 版本解析失败；
    # 改用原生 API + 显式标量 base_score 重训等价模型，同一模型内对照两种背景模式
    npar = {k: v for k, v in params.items()
            if k not in ("n_estimators", "random_state", "n_jobs",
                         "early_stopping_rounds", "eval_metric", "objective")}
    npar.update({"objective": "reg:squarederror",
                 "base_score": float(tab["RGR"].mean()),
                 "tree_method": "hist", "seed": C.SEED})
    tr_bg = tab.sample(30000, random_state=C.SEED + 7)
    bg = tab.sample(2000, random_state=C.SEED + 9)
    bst = xgb.train(npar, xgb.DMatrix(tr_bg[feats], label=tr_bg["RGR"]),
                    num_boost_round=300)
    ex = shap.TreeExplainer(bst, data=bg[feats], feature_perturbation="interventional")
    sv_int = np.asarray(ex.shap_values(sub[feats], check_additivity=False))
    sv_pd_bst = bst.predict(xgb.DMatrix(sub[feats]), pred_contribs=True)[:, :-1]
    mas_pd = np.abs(sv_pd_bst).mean(axis=0)
    mas_int = np.abs(sv_int).mean(axis=0)
    rho_bg = spearmanr(mas_pd, mas_int).statistic
    cmp_df = pd.DataFrame({"var": feats, "path_dependent": mas_pd, "interventional": mas_int})
    cmp_df.to_csv(os.path.join(C.OUT, "14_shap_background_check.csv"),
                  index=False, encoding="utf-8-sig")
    say(f"- 同一模型内排序 Spearman ρ (path-dependent vs interventional) = {rho_bg:.4f}")
    say("- top-5 (path-dependent): " + str([feats[i] for i in np.argsort(-mas_pd)[:5]]))
    say("- top-5 (interventional): " + str([feats[i] for i in np.argsort(-mas_int)[:5]]))
except Exception as e:
    say(f"- interventional 模式在当前 shap/xgboost 版本组合不可用（{type(e).__name__}: {e}）")
    say("- 处理：正文维持 path-dependent（TreeSHAP 默认）单一口径，SI 注明未做背景模式对照。")
say(f"-（本节用时 {time.time()-t1:.0f}s）")

say(f"\n（bootstrap 总用时 {time.time()-t0:.0f}s）")
say("\nDONE P1-5")
with open(os.path.join(C.OUT, "14_shap_boot_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
