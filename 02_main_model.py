# -*- coding: utf-8 -*-
"""Phase B 主建模：XGBoost 空间块CV随机搜索调参 + Random Forest 对照 + LMM 传统基准。
输出: 02_best_params.json / 02_rf_params.json / 02_tuning_results.csv / 02_main_model_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial", "fold_group"]], on="plot_id", how="left")
feats = C.selected_features(vd)
y = tab["RGR"].values

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# Phase B 主建模报告\n")
say(f"- 建模表: {tab.shape}，特征 {len(feats)} 个")

# ---------- 1. XGBoost 随机搜索（空间块 5 折，早停） ----------
say("\n## 1. XGBoost 调参（空间块 5 折内嵌随机搜索，最严格方案）")
space = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "min_child_weight": [1, 5, 10, 20],
    "reg_alpha": [0, 0.5, 2, 5],
    "reg_lambda": [1, 2, 5, 10],
}
rng = np.random.default_rng(C.SEED)
n_cand = 24
results = []
t0 = time.time()
for ci in range(n_cand):
    params = {k: rng.choice(v) for k, v in space.items()}
    params = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
              for k, v in params.items()}
    rmses = []
    for k in range(5):
        tr = tab[tab["fold_spatial"] != k]
        te = tab[tab["fold_spatial"] == k]
        m = xgb.XGBRegressor(n_estimators=4000, early_stopping_rounds=80,
                             eval_metric="rmse", tree_method="hist",
                             random_state=C.SEED + k, n_jobs=-1, **params)
        m.fit(tr[feats], tr["RGR"], eval_set=[(te[feats], te["RGR"])], verbose=False)
        pred = m.predict(te[feats])
        rmses.append(float(np.sqrt(np.mean((te["RGR"].values - pred) ** 2))))
    results.append({**params, "mean_rmse": float(np.mean(rmses)),
                    "std_rmse": float(np.std(rmses))})
    say(f"  cand {ci+1:02d}/{n_cand}: rmse={np.mean(rmses):.5f}  {params}")
res_df = pd.DataFrame(results).sort_values("mean_rmse").reset_index(drop=True)
res_df.to_csv(os.path.join(C.OUT, "02_tuning_results.csv"), index=False, encoding="utf-8-sig")
best = res_df.iloc[0]
best_params = {k: (int(best[k]) if k in ("max_depth", "min_child_weight") else float(best[k]))
               for k in space}
C.save_json({"xgb_best": best_params, "xgb_best_rmse": float(best["mean_rmse"]),
             "n_candidates": n_cand, "search_space": space}, "02_best_params.json")
say(f"- 最优参数: {best_params}")
say(f"- 最优空间块 CV RMSE: {best['mean_rmse']:.5f}（用时 {time.time()-t0:.0f}s）")

# ---------- 2. Random Forest 对照（同一折，小网格） ----------
say("\n## 2. Random Forest 调参（同一空间块折）")
rf_grid = [{"n_estimators": 600, "max_features": mf, "min_samples_leaf": msl,
            "max_depth": md}
           for mf in (0.3, 0.5, "sqrt") for msl in (1, 5) for md in (None, 20)]
rf_results = []
t0 = time.time()
for gi, g in enumerate(rf_grid):
    rmses = []
    for k in range(5):
        tr = tab[tab["fold_spatial"] != k]
        te = tab[tab["fold_spatial"] == k]
        rf = RandomForestRegressor(random_state=C.SEED, n_jobs=-1, **g)
        rf.fit(tr[feats], tr["RGR"])
        pred = rf.predict(te[feats])
        rmses.append(float(np.sqrt(np.mean((te["RGR"].values - pred) ** 2))))
    rf_results.append({**{k: str(v) for k, v in g.items()},
                       "mean_rmse": float(np.mean(rmses))})
    say(f"  rf {gi+1:02d}/{len(rf_grid)}: rmse={np.mean(rmses):.5f}  {g}")
rf_df = pd.DataFrame(rf_results).sort_values("mean_rmse").reset_index(drop=True)
rf_best = rf_df.iloc[0]
try:
    mf = float(rf_best["max_features"])
except (ValueError, TypeError):
    mf = rf_best["max_features"]
rf_params = {"n_estimators": 600, "max_features": mf,
             "min_samples_leaf": int(rf_best["min_samples_leaf"]),
             "max_depth": None if rf_best["max_depth"] == "None" else int(rf_best["max_depth"])}
C.save_json(rf_params, "02_rf_params.json")
say(f"- RF 最优: {rf_params}，RMSE={rf_best['mean_rmse']:.5f}（用时 {time.time()-t0:.0f}s）")

# ---------- 3. LMM 传统基准（样地随机截距 + 全固定效应；完整案例） ----------
say("\n## 3. LMM 传统基准（statsmodels MixedLM）")
import statsmodels.formula.api as smf
cc = tab.dropna(subset=feats + ["RGR"]).copy()
say(f"- 完整案例: {len(cc)} 株 / {cc['plot_id'].nunique()} 样地")
try:
    t0 = time.time()
    formula = "RGR ~ " + " + ".join(feats)
    md = smf.mixedlm(formula, cc, groups=cc["plot_id"])
    res = md.fit(reml=False, method="lbfgs", maxiter=200)
    xb = np.dot(res.model.exog, res.fe_params.values)
    var_fix = float(np.var(xb))
    var_re = float(res.cov_re.iloc[0, 0])
    var_res = float(res.scale)
    r2_m = var_fix / (var_fix + var_re + var_res)
    r2_c = (var_fix + var_re) / (var_fix + var_re + var_res)
    null = smf.mixedlm("RGR ~ 1", cc, groups=cc["plot_id"]).fit(reml=False, method="lbfgs")
    icc = float(null.cov_re.iloc[0, 0] / (null.cov_re.iloc[0, 0] + null.scale))
    say(f"- 边际 R² (固定效应): {r2_m:.4f}")
    say(f"- 条件 R² (固定+随机): {r2_c:.4f}")
    say(f"- 样地内 ICC (零模型): {icc:.4f}")
    say(f"- 收敛: {res.converged}（用时 {time.time()-t0:.0f}s）")
except Exception as e:
    say(f"- 原始量纲 LMM 拟合失败（{type(e).__name__}: {e}），"
        f"权威 LMM 基准由 02b_lmm.py 提供（标准化 + 千分率 + 优化器回退链）")

say("\nDONE Phase B")
with open(os.path.join(C.OUT, "02_main_model_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
