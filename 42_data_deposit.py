# -*- coding: utf-8 -*-
"""42_data_deposit.py — 导出公开归档的派生数据（国内可信仓储，如 ScienceDB）

对应正文 Data availability 承诺的公开产品：
  1. 不含坐标的样地水平汇总数据        -> plot_level_summary.csv（协变量舍入 2 位有效数字）
  2. 聚合折外性能与 SHAP 归因验证表    -> oof_performance_summary.csv /
                                          oof_observed_predicted_grid.csv /
                                          shap_variable_summary.csv /
                                          shap_decile_means.csv
  3. 250-m 预测与模型分歧 GeoTIFF       -> Source_Data_maps.zip（复制自 08_定稿，4 个 GeoTIFF）
另附 variable_dictionary.csv（35 个模型变量的分组字典）与 README.md（存档说明）。

安全约束：
- 输出一律不含 lon/lat 坐标列、不含原始清查测量列（D0/D1/生长量等）；
- 不输出任何单木（单个调查对象）级记录——树级折外预测与 SHAP 矩阵仅聚合后发布，
  聚合验证表足以复核文中每一个报告统计量。
"""
import os
import shutil

import numpy as np
import pandas as pd

import common as C

DEST = os.path.join(C.PIPE, "10_投稿文件包_20261010", "04_数据归档")
os.makedirs(DEST, exist_ok=True)
for _f in os.listdir(DEST):                      # 每次全量重建，清掉上轮残留
    _p = os.path.join(DEST, _f)
    if os.path.isfile(_p):
        os.remove(_p)

FORBIDDEN = {"lon", "lat", "经度", "纬度", "D0", "D1", "生长量", "方向", "距离"}


def guard(df, name):
    bad = FORBIDDEN & set(df.columns)
    assert not bad, f"{name} 泄露敏感列: {bad}"
    return df


# ==========================================================================
# 1. 样地水平汇总（01_model_table 按样地聚合；坐标列显式剔除）
# ==========================================================================
tab = pd.read_pickle(os.path.join(C.OUT, "01_model_table.pkl"))
tree_lvl = ["plot_id", "fg", "fg_conifer", "D0", "D1", "RGR", "lnD0", "abs_inc",
            "lon", "lat", "fold_random"]
preds = [c for c in tab.columns if c not in tree_lvl]      # 33 个样地水平预测因子
assert {"lon", "lat"}.issubset(tab.columns)                # 源表确有坐标——必须剔除
assert len(preds) == 33, len(preds)

g = tab.groupby("plot_id")
plot_sum = pd.DataFrame({
    "plot_id": g.size().index,
    "n_trees": g.size().values,
    "plot_mean_RGR_per_yr": g["RGR"].mean().values,
    "plot_median_RGR_per_yr": g["RGR"].median().values,
    "plot_sd_RGR_per_yr": g["RGR"].std(ddof=1).values,
    "mean_ln_D0": g["lnD0"].mean().values,
    "dominant_functional_group": np.where(
        g["fg_conifer"].mean().values > 0.5, "conifer", "broadleaf"),
})
for c in preds:
    plot_sum[c] = g[c].first().values
guard(plot_sum, "plot_level_summary.csv")


def _round_sig(v, sig=2):
    """舍入至 sig 位有效数字（防环境指纹再识别：公开表中的协变量不做全精度）。"""
    v = np.asarray(v, dtype=float)
    out = v.copy()
    nz = (v != 0) & np.isfinite(v)
    e = np.floor(np.log10(np.abs(v[nz])))
    out[nz] = np.round(v[nz] / 10.0 ** e, sig - 1) * 10.0 ** e
    return out


# 防指纹再识别：全部协变量舍入至 2 位有效数字；响应量汇总保持全精度
for c in preds:
    plot_sum[c] = _round_sig(plot_sum[c].values)
plot_sum.round(6).to_csv(os.path.join(DEST, "plot_level_summary.csv"), index=False)

# ==========================================================================
# 2. 聚合验证表（不提供单木级折外预测/SHAP 记录）
# ==========================================================================
oof = pd.read_pickle(os.path.join(C.OUT, "22_nested_oof.pkl"))["oof"]
shap_pack = pd.read_pickle(os.path.join(C.OUT, "26_shap_matrix.pkl"))
shap, feats = shap_pack["shap"], shap_pack["feats"]
n = len(tab)
assert len(shap) == n == 44_373 and list(shap.columns) == list(feats)
obs = tab["RGR"].values


def _r2(o, p):
    return 1.0 - float(np.sum((o - p) ** 2)) / float(np.sum((o - o.mean()) ** 2))


DESIGNS = [("random", "random"), ("plot_grouped", "group"), ("spatial_block", "spatial")]
_perf, _grid = [], []
for dname, key in DESIGNS:
    p = oof[key].values
    _pm = pd.DataFrame({"plot_id": tab["plot_id"].values, "o": obs, "p": p}).groupby("plot_id").mean()
    _perf.append({
        "cv_design": dname, "n_trees": n, "n_plots": len(_pm),
        "tree_R2": _r2(obs, p),
        "tree_RMSE_per_yr": float(np.sqrt(np.mean((obs - p) ** 2))),
        "tree_Spearman": float(pd.Series(obs).corr(pd.Series(p), method="spearman")),
        "plot_R2": _r2(_pm["o"].values, _pm["p"].values),
        "plot_RMSE_per_yr": float(np.sqrt(np.mean((_pm["o"] - _pm["p"]) ** 2))),
        "mean_bias_obs_minus_pred_per_yr": float(np.mean(obs - p)),
        "pred_q05_per_yr": float(np.quantile(p, 0.05)),
        "pred_q50_per_yr": float(np.quantile(p, 0.50)),
        "pred_q95_per_yr": float(np.quantile(p, 0.95)),
    })
    # 观测 × 预测二维计数（图面显示范围 0–0.22 yr⁻¹，22 × 22 网格）
    ke = (obs <= 0.22) & (p <= 0.22)
    ed = np.linspace(0, 0.22, 23)
    H, _, _ = np.histogram2d(obs[ke], p[ke], bins=[ed, ed])
    for i in range(22):
        for j in range(22):
            if H[i, j] > 0:
                _grid.append({"cv_design": dname,
                              "obs_bin_left_per_yr": ed[i], "obs_bin_right_per_yr": ed[i + 1],
                              "pred_bin_left_per_yr": ed[j], "pred_bin_right_per_yr": ed[j + 1],
                              "n_trees": int(H[i, j])})
perf = pd.DataFrame(_perf)
guard(perf, "oof_performance_summary.csv")
perf.round(6).to_csv(os.path.join(DEST, "oof_performance_summary.csv"), index=False)
grid = pd.DataFrame(_grid)
guard(grid, "oof_observed_predicted_grid.csv")
grid.round(6).to_csv(os.path.join(DEST, "oof_observed_predicted_grid.csv"), index=False)

vd = pd.read_csv(os.path.join(C.OUT, "01_var_dict.csv"))[["var", "group"]]
_abs = shap.abs().mean()
_rows = []
for v in feats:
    s = shap[v].values
    _rows.append({
        "variable": v, "family": vd.set_index("var")["group"].get(v, ""),
        "share_of_total_absSHAP": float(_abs[v] / _abs.sum()),
        "mean_absSHAP_per_yr": float(_abs[v]),
        "sd_SHAP_per_yr": float(np.std(s, ddof=1)),
        "SHAP_q05_per_yr": float(np.quantile(s, 0.05)),
        "SHAP_q50_per_yr": float(np.quantile(s, 0.50)),
        "SHAP_q95_per_yr": float(np.quantile(s, 0.95)),
    })
shap_sum = pd.DataFrame(_rows)
guard(shap_sum, "shap_variable_summary.csv")
shap_sum.round(8).to_csv(os.path.join(DEST, "shap_variable_summary.csv"), index=False)

_drows = []
for v in feats:
    d = pd.DataFrame({"value": tab[v].values, "shap": shap[v].values})
    q = pd.qcut(d["value"], 10, duplicates="drop")
    gg = d.groupby(q, observed=True).agg(x=("value", "mean"), y=("shap", "mean"),
                                         n=("shap", "size"))
    for k, (_, r) in enumerate(gg.iterrows(), 1):
        _drows.append({"variable": v, "decile": k, "bin_mean_value": r["x"],
                       "bin_mean_SHAP_per_yr": r["y"], "n_trees_in_bin": int(r["n"])})
dec = pd.DataFrame(_drows)
guard(dec, "shap_decile_means.csv")
dec.round(8).to_csv(os.path.join(DEST, "shap_decile_means.csv"), index=False)

# ==========================================================================
# 3. 250-m 预测与模型分歧 GeoTIFF（Source_Data_maps.zip 原样复制）+ 变量字典
# ==========================================================================
shutil.copyfile(os.path.join(C.PIPE, "08_定稿", "Source_Data_maps.zip"),
                os.path.join(DEST, "Source_Data_maps.zip"))
shutil.copyfile(os.path.join(C.OUT, "01_var_dict.csv"),
                os.path.join(DEST, "variable_dictionary.csv"))

sizes = {f: os.path.getsize(os.path.join(DEST, f))
         for f in sorted(os.listdir(DEST))}
readme = f"""# Derived-data deposit — coordinate-free, aggregate-only products

Manuscript: *Stand structure outweighs climate as a predictor of individual-tree
growth in subtropical China* (v4.0). This folder holds the derived products named
in the paper's Data availability statement, deposited in a domestic research data
repository (e.g. Science Data Bank). **None of these files contains confidential
plot coordinates, raw inventory measurements, or individual-tree records.**

| File | Content |
|---|---|
| `plot_level_summary.csv` | One row per plot ({len(plot_sum):,} plots): tree counts and RGR summaries, dominant functional group (binary majority rule), and the 33 plot-level model predictors. Covariates are rounded to two significant digits to prevent location re-identification by raster fingerprint matching. |
| `oof_performance_summary.csv` | Tree- and plot-level out-of-fold performance of the nested XGBoost models under the three cross-validation designs (random, plot-grouped, 0.5° spatial block): R², RMSE, Spearman correlation, bias and prediction quantiles. |
| `oof_observed_predicted_grid.csv` | Observed × predicted tree counts on a 22 × 22 grid spanning the drawn 0–0.22 yr⁻¹ axes, per design (aggregated replacement for the individual-tree prediction records, which are withheld as single-survey-object data). |
| `shap_variable_summary.csv` | Per-variable out-of-fold TreeSHAP attribution summary ({len(feats)} variables): family, share of total |SHAP|, mean |SHAP|, s.d. and quantiles. |
| `shap_decile_means.csv` | Per-variable effect-shape verification table: mean predictor value and mean SHAP per decile ({len(feats)} variables × ≤10 deciles). |
| `Source_Data_maps.zip` | The four 250-m GeoTIFFs plotted in Fig. 1a and Fig. 6a-c (forest-cover fraction, conifer and broadleaf predicted RGR, model disagreement; EPSG:4326, nodata −9999). |
| `variable_dictionary.csv` | The 35 model variables with predictor family and collinearity-screening status. |

These aggregates suffice to verify every performance, attribution and effect-shape
statistics reported in the paper and supplement. Provenance: produced by
`42_data_deposit.py` from the pipeline outputs (`01_model_table.pkl`,
`22_nested_oof.pkl`, `26_shap_matrix.pkl`); the maps zip is built by
`37_source_data.py`. All analyses use fixed random seeds (see code deposit).

Access to the raw Continuous Forest Inventory data is restricted by national
regulations governing confidential forest inventory plot coordinates and by the
data-licence agreement under which they were used; in accordance with those
regulations the raw records cannot be shared with third parties, including
reviewers and editors.

**DOI 占位：【投稿/发表时回填归档 DOI】**
"""
with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme)

for f, sz in sizes.items():
    print(f"  {f:36s} {sz/1024/1024:8.2f} MB")
print(f"DONE 42 data deposit export -> {DEST}")
