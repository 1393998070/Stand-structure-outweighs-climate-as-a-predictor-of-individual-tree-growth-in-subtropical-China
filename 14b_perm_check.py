# -*- coding: utf-8 -*-
"""P1-5 附：SHAP 归因稳健性替代检验（Permutation SHAP）。
interventional TreeSHAP 因 xgboost 2.x base_score 数组序列化与当前 shap 版本不兼容
（见 14_shap_boot_report.md），改用模型无关的 Permutation SHAP
（背景 = 2000 树 Independent 掩码，等价于 interventional 式独立背景）
在同一模型、同一样本上与 TreeSHAP path-dependent 排序对照。
输出: 14b_perm_check.csv / 14b_perm_check_report.md
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
feats = C.selected_features(vd)
params = C.load_json("02_best_params.json")["xgb_best"]

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-5 附：Permutation SHAP 替代检验\n")
t0 = time.time()

X_all = tab[feats].values
m = xgb.XGBRegressor(n_estimators=600, tree_method="hist", random_state=C.SEED,
                     n_jobs=-1, **params)
m.fit(X_all, tab["RGR"].values)

sub = tab.sample(400, random_state=C.SEED)
bg = tab.sample(2000, random_state=C.SEED + 9)

sv_pd = m.get_booster().predict(xgb.DMatrix(sub[feats].values),
                                pred_contribs=True)[:, :-1]

import shap
from scipy.stats import spearmanr
try:
    masker = shap.maskers.Independent(bg[feats].values, max_samples=2000)
    pe = shap.PermutationExplainer(m.predict, masker)
    sv_perm = np.asarray(pe.shap_values(sub[feats].values))
    mas_pd = np.abs(sv_pd).mean(axis=0)
    mas_pm = np.abs(sv_perm).mean(axis=0)
    rho = spearmanr(mas_pd, mas_pm).statistic
    out = pd.DataFrame({"var": feats, "treeshap_path_dep": mas_pd, "permutation": mas_pm})
    out.to_csv(os.path.join(C.OUT, "14b_perm_check.csv"), index=False, encoding="utf-8-sig")
    say(f"- 排序 Spearman ρ (TreeSHAP path-dependent vs Permutation SHAP) = {rho:.4f}")
    say("- top-5 (path-dependent): " + str([feats[i] for i in np.argsort(-mas_pd)[:5]]))
    say("- top-5 (permutation):    " + str([feats[i] for i in np.argsort(-mas_pm)[:5]]))
    say("- 结论: " + ("变量排序对归因方法/背景掩码稳健" if rho >= 0.9 else
                    "排序存在差异，需在 SI 说明"))
except Exception as e:
    say(f"- Permutation SHAP 亦不可用（{type(e).__name__}: {e}）"
        "→ SI 仅报告 path-dependent 单一口径")

say(f"\n（用时 {time.time()-t0:.0f}s）")
say("DONE 14b")
with open(os.path.join(C.OUT, "14b_perm_check_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
