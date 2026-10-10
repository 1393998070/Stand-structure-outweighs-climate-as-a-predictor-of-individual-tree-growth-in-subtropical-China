# -*- coding: utf-8 -*-
"""Stage 4 · P1-6 GAM 对照与 SHAP 形状稳定性。
(1) GAM held-out R²（同空间块折，top-5 变量 B 样条，与 Phase D 同规格）；
(2) SHAP 依赖形状跨折稳定性（top-6 变量 × 5 折曲线 + 两两 Spearman + 重叠带数据）；
(3) 五个 GAM–SHAP 形态相关 r 值如实全披露（stand_age=-0.037 等）+ 技术原因。
输出: 16_gam_stability_report.md / 16_shape_stability.csv / 16_shape_curves.pkl
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
shap_mat = pd.read_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-6 GAM 对照与 SHAP 形状稳定性报告\n")
t0 = time.time()

TOP5 = ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15"]

def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

# ---------- 1. GAM held-out R² ----------
say("## 1. GAM held-out R²（空间块折，top-5 变量 B 样条 df=10）")
from statsmodels.gam.generalized_additive_model import GLMGam
from statsmodels.gam.smooth_basis import BSplines
import warnings
cc = tab.dropna(subset=TOP5 + ["RGR"]).copy()
gam_pred = pd.Series(np.nan, index=cc.index)
for k in range(5):
    tr = cc[cc["fold_spatial"] != k]
    te = cc[cc["fold_spatial"] == k]
    trs = tr.sample(min(30000, len(tr)), random_state=C.SEED)
    bs = BSplines(trs[TOP5].values, df=[10] * 5, degree=[3] * 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gres = GLMGam(trs["RGR"].values, smoother=bs).fit()
        # B 样条不支持外推：测试点截断到训练子样本范围（边缘饱和效应）
        lo = trs[TOP5].min(); hi = trs[TOP5].max()
        te_clipped = te[TOP5].clip(lo, hi, axis=1)
        gam_pred.loc[te.index] = np.asarray(gres.predict(
            exog_smooth=te_clipped.values, transform=True))
ok = gam_pred.notna()
gam_oof_r2 = r2(cc.loc[ok, "RGR"], gam_pred[ok])
say(f"- GAM OOF tree R2 = {gam_oof_r2:.4f}（n={ok.sum()}）")
_xgb_r2 = pd.read_csv(os.path.join(C.OUT, "03_triple_cv_results.csv"))
_row = _xgb_r2[(_xgb_r2["model"] == "XGBoost") & (_xgb_r2["cv"] == "spatial")]
_xgb_tree_r2 = float(_row["R2"].iloc[0]) if len(_row) else float("nan")
_n_feat = len(C.selected_features(C.load_model_data()[1]))
say(f"- 对照: XGB OOF tree R2 = {_xgb_tree_r2:.4f}（全 {len(tab):,}，{_n_feat} 特征）；"
    f"GAM 仅用 top-5 变量 → 差距同时来自变量集与交互结构，不能单独归因于交互")

# ---------- 2. SHAP 形状跨折稳定性 ----------
say("\n## 2. SHAP 依赖形状跨折稳定性（top-6 变量）")
TOP6 = TOP5 + ["origin_natural"]
stability_rows = []
curves = {}
for v in TOP6:
    df = pd.DataFrame({"v": tab[v].values, "s": shap_mat[v].values, "f": tab["fold_spatial"].values})
    fold_curves = {}
    for k in range(5):
        sub = df[df["f"] == k]
        binned = sub.groupby(pd.qcut(sub["v"], 15, duplicates="drop"), observed=True).agg(
            x=("v", "mean"), y=("s", "mean"))
        fold_curves[k] = binned
    # 两两形状相关（在共同 x 范围插值）
    pair_rs = []
    for a in range(5):
        for b in range(a + 1, 5):
            ca, cb = fold_curves[a], fold_curves[b]
            lo = max(ca["x"].min(), cb["x"].min()); hi = min(ca["x"].max(), cb["x"].max())
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, 30)
            ya = np.interp(grid, ca["x"], ca["y"])
            yb = np.interp(grid, cb["x"], cb["y"])
            pair_rs.append(spearmanr(ya, yb).statistic)
    pair_rs = np.array(pair_rs)
    if pair_rs.size:
        mean_r, min_r = round(float(pair_rs.mean()), 3), round(float(pair_rs.min()), 3)
        say(f"- {v:14s}: 跨折形状相关 mean r={pair_rs.mean():.3f}, min r={pair_rs.min():.3f}"
            f"（{len(pair_rs)} 对折）")
    else:
        # 二值/退化变量（如 origin_natural）分箱后各折无共同 x 区间，形状相关不可计算
        mean_r = min_r = np.nan
        say(f"- {v:14s}: 分箱后各折无共同 x 区间（取值仅 {tab[v].nunique()} 个），形状相关记为 NA")
    stability_rows.append({"var": v, "mean_pairwise_r": mean_r,
                           "min_pairwise_r": min_r, "n_pairs": len(pair_rs)})
    curves[v] = fold_curves
pd.DataFrame(stability_rows).to_csv(os.path.join(C.OUT, "16_shape_stability.csv"),
                                    index=False, encoding="utf-8-sig")
pd.to_pickle(curves, os.path.join(C.OUT, "16_shape_curves.pkl"))

# ---------- 3. GAM–SHAP 形态相关全披露 ----------
say("\n## 3. GAM–SHAP 形态相关（Phase D 实测值，如实披露）")
for v, r in [("stand_age", -0.037), ("lnD0", 0.286), ("conifer_prop", 0.185),
             ("bio12", 0.449), ("bio15", 0.080)]:
    say(f"- {v:14s}: r = {r:+.3f}")
say("- 技术原因: GAM 平滑项为边际加性效应（以协变量中位数为基线、预测尺度中心化），"
    "SHAP 依赖曲线含交互再分配与垂直偏移（行中心化）；两者不可直接互换解读。"
    "stand_age 的 r=-0.037 主要源于 GAM 平滑自由度不足以复现陡降段位置 + 中心化差异，"
    "而非方向不一致（两者均单调下降）。")
say("- 删除正文 \"captured interaction structure invisible to additive specifications\" 类断言，"
    "改为报告 GAM OOF R² 并注明变量集差异。")

say(f"\n（用时 {time.time()-t0:.0f}s）")
say("\nDONE P1-6")
with open(os.path.join(C.OUT, "16_gam_stability_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
