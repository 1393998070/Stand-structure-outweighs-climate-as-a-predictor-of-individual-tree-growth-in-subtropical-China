# -*- coding: utf-8 -*-
"""Phase B 补遗：LMM 传统基准（v2：标准化+千分率量纲修复病态；优化器回退链）。
输出: 02_lmm_summary.json / 02_lmm_coefs.csv / 02_lmm_report.md
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
from scipy.linalg import qr

tab, vd, folds = C.load_model_data()
feats = C.selected_features(vd)

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# LMM 传统基准报告（v2 标准化拟合）\n")

BINARY = {"fg_conifer", "origin_nat01", "origin_missing", "age_missing"}
cc = tab.dropna(subset=feats + ["RGR"]).copy()
cc["origin_nat01"] = (cc["origin_natural"] == 1).astype(float)
feats_lmm = [f if f != "origin_natural" else "origin_nat01" for f in feats]
# 完整案例下 age_missing 恒为 0，剔除
feats_lmm = [f for f in feats_lmm if f != "age_missing"]
say(f"- 完整案例: {len(cc)} 株 / {cc['plot_id'].nunique()} 样地")

# 连续变量 z 标准化（系数即标准化效应）；RGR 用千分率修复数值条件
sd = {}
for f in feats_lmm:
    if f not in BINARY:
        sd[f] = cc[f].std(ddof=0)
        cc[f"z_{f}"] = (cc[f] - cc[f].mean()) / sd[f]
zvars = [f"z_{f}" if f not in BINARY else f for f in feats_lmm]
cc["RGR_permille"] = cc["RGR"] * 1000.0

# QR 秩检查
A = np.column_stack([np.ones(len(cc)), cc[zvars].values])
_, R, piv = qr(A, mode="full", pivoting=True)
diag = np.abs(np.diag(R))
rank = int((diag > diag[0] * 1e-10).sum())
kept = [zvars[c - 1] for c in sorted(piv[:rank]) if c != 0]
dropped = [v for v in zvars if v not in kept]
say(f"- 设计矩阵秩 {rank}/{len(zvars)+1}，剔除 {len(dropped)} 列: {dropped}")

formula = "RGR_permille ~ " + " + ".join(kept)
t0 = time.time()
# 注: statsmodels 0.15 的 lbfgs 会在退化点（随机效应方差=0、llf=inf、多系数为0）误报收敛，
# 故遍历全部优化器，按「随机效应方差为正 + 对数似然最大」择优，而非取首个收敛解
cands = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for meth in ("bfgs", "powell", "cg", "lbfgs"):
        try:
            cand = smf.mixedlm(formula, cc, groups=cc["plot_id"]).fit(
                reml=False, method=meth, maxiter=1000)
            re_v = float(cand.cov_re.iloc[0, 0])
            llf = float(cand.llf)
            cands.append((meth, cand, re_v, llf))
            say(f"- 优化器 {meth}: 收敛={cand.converged} RE方差={re_v:.3f} llf={llf:.1f}")
        except Exception as e:
            say(f"- 优化器 {meth} 失败: {type(e).__name__}")
ok = [c for c in cands if np.isfinite(c[3]) and c[2] > 1e-6]
pool = ok if ok else [c for c in cands if np.isfinite(c[3])]
assert pool, "所有优化器失败或均退化"
opt_used, res, _, _ = max(pool, key=lambda c: c[3])
say(f"- 采用优化器 {opt_used}（{len(ok)}/{len(cands)} 个非退化解）")
if not ok:
    say("- 警告: 无非退化解，采用似然最大的退化解")

var_fix = float(np.var(np.dot(res.model.exog, res.fe_params.values)))
var_re = float(res.cov_re.iloc[0, 0])
var_res = float(res.scale)
r2_m = var_fix / (var_fix + var_re + var_res)
r2_c = (var_fix + var_re) / (var_fix + var_re + var_res)
# 零模型 ICC（优化器回退 + 经验矩估计兜底）
null_re = 0.0
for meth in ("bfgs", "powell", "lbfgs"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nfit = smf.mixedlm("RGR_permille ~ 1", cc, groups=cc["plot_id"]).fit(
            reml=False, method=meth, maxiter=500)
    null_re = float(nfit.cov_re.iloc[0, 0])
    if null_re > 1e-6:
        icc = null_re / (null_re + float(nfit.scale))
        icc_method = f"null-model MLE ({meth})"
        break
if null_re <= 1e-6:  # 边界失败 → 单因素随机效应矩估计
    gm = cc.groupby("plot_id")["RGR_permille"]
    means, counts = gm.mean(), gm.size()
    grand = cc["RGR_permille"].mean()
    msb = float(np.sum(counts * (means - grand) ** 2) / (len(means) - 1))
    w = cc["RGR_permille"] - cc["plot_id"].map(means)
    msw = float(np.sum(w ** 2) / (len(cc) - len(means)))
    var_b = (msb - msw) / counts.mean()
    icc = var_b / (var_b + msw)
    icc_method = "empirical ANOVA (MLE boundary)"
icc_model = var_re / (var_re + var_res)
say(f"- 边际 R² (固定效应): {r2_m:.4f}")
say(f"- 条件 R² (固定+随机): {r2_c:.4f}")
say(f"- 样地 ICC: {icc:.4f}（{icc_method}）；全模型隐含 ICC: {icc_model:.4f}")
say(f"- 用时 {time.time()-t0:.0f}s；RE 方差成分: {var_re:.3f}（千分率²），残差 {var_res:.3f}")

coefs = pd.DataFrame({"coef_permille": res.params, "t": res.tvalues,
                      "p": res.pvalues}).drop("Group Var", errors="ignore")
coefs["coef_RGR_per_SD"] = coefs["coef_permille"] / 1000.0  # 连续变量为每 SD 效应
say("\n- 固定效应 |t| top-12:")
say(coefs.reindex(coefs["t"].abs().sort_values(ascending=False).index).head(12).round(3).to_string())
coefs.to_csv(os.path.join(C.OUT, "02_lmm_coefs.csv"), encoding="utf-8-sig")

C.save_json({"marginal_R2": round(r2_m, 4), "conditional_R2": round(r2_c, 4),
             "ICC_plot": round(float(icc), 4), "ICC_method": icc_method,
             "ICC_model_implied": round(float(icc_model), 4),
             "n_complete": int(len(cc)),
             "n_plots": int(cc["plot_id"].nunique()), "dropped_collinear": dropped,
             "converged": bool(res.converged)}, "02_lmm_summary.json")
with open(os.path.join(C.OUT, "02_lmm_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("DONE LMM")
