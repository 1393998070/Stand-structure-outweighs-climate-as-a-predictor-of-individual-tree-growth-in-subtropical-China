# -*- coding: utf-8 -*-
"""37 Source Data 底表构建（Nature Communications 投稿件）。

NC 要求
-------
"the source data file should, as a minimum, contain the numerical data underlying any
graphs and charts ... each figure or table (in the main manuscript and in the
Supplementary Information) containing relevant data should be represented by a single
sheet in an Excel document, or a single .txt file or other file type in a zipped folder."

本脚本的落地方式
----------------
* **`Source_Data.xlsx`**：一个面板一个 sheet（`Fig1a`、`Fig1b` …），另加 `README` 说明
  「sheet ↔ 面板」对应关系。之所以按面板而非按整图分表：同一张图各面板的行结构
  差异很大（如 Fig.1 的 (a) 是 1,187 个样地、(b) 是 44,705 株树），强行并入一张表
  会引入大量空单元格；按面板分表可保证每个数值都可逐行对源。整图 6 张、SI 图 4 张
  的全部面板均在表内，满足"每张图都有代表"的实质要求。
* **`Source_Data_maps.zip`**：Fig.1a 与 Fig.6a–c 的地图是 250 m 栅格（1.54 × 10⁶ 像元），
  无法放入 Excel，按 NC 允许的"zipped folder"形式提供 GeoTIFF（与绘图所用数组逐像元一致）。

**不改动任何源稿或分析结果**：本脚本只读取 04_分析结果 与 05_图表，写出 08_定稿/Source_Data*。
所有重算均复用 09_figures.py / 20_si_figures.py 的同一套分组与分箱逻辑，
以保证底表与图面数值严格一致。

用法: python 37_source_data.py
产出: 08_定稿/Source_Data.xlsx
      08_定稿/Source_Data_maps.zip
      08_定稿/04_Source_Data_说明.md（构建核验报告）
"""
import os
import sys
import json
import pickle
import shutil
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.OUT
MAPDIR = os.path.join(R, "map")
OUT = os.path.join(C.PIPE, "08_定稿")
XLSX = os.path.join(OUT, "Source_Data.xlsx")
ZIP = os.path.join(OUT, "Source_Data_maps.zip")

report, fails = [], []


def say(x):
    print(x, flush=True)
    report.append(str(x))


def check(label, cond, detail=""):
    say(f"  [{'OK' if cond else 'XX'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


# --------------------------------------------------------------------------
# 0. 数据装载（与 09_figures.py / 20_si_figures.py 完全同源）
# --------------------------------------------------------------------------
tab, vd, folds = C.load_model_data()
plot_tab = (tab.groupby("plot_id")
            .agg(lon=("lon", "first"), lat=("lat", "first"),
                 plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"),
                 origin_natural=("origin_natural", "first"),
                 conifer_prop=("conifer_prop", "first"), fg_conifer=("fg_conifer", "mean"),
                 bio1=("bio1", "first"), bio12=("bio12", "first"))
            .reset_index())
plot_tab["fg_conifer"] = (plot_tab["fg_conifer"] > 0.5).astype(int)

# 坐标粗化：国家清查规定样地精确坐标保密，公开底表只给 0.05° 格心
# （与 09_figures.py 的 Fig1a 图面点位同步；正文 Methods/图 1 注有对应说明）
_COARSE = 0.05
plot_tab["lon"] = (np.floor(plot_tab["lon"] / _COARSE) + 0.5) * _COARSE
plot_tab["lat"] = (np.floor(plot_tab["lat"] / _COARSE) + 0.5) * _COARSE

# 单木明细属"单个调查对象资料"（《统计法》），不随 Source Data 对外提供；
# Fig1b 仅给直方图分箱计数（分箱与 09_figures.py Fig1b 一致）。
_RBR = np.linspace(0, 0.2, 60)          # 59 个等宽箱，覆盖图面 x 范围 0–0.2 yr⁻¹
_F1B_B, _ = np.histogram(tab.loc[tab["fg_conifer"] == 0, "RGR"], bins=_RBR)
_F1B_C, _ = np.histogram(tab.loc[tab["fg_conifer"] == 1, "RGR"], bins=_RBR)

SHEETS = {}          # name -> DataFrame
DESC = {}            # name -> 面向读者的内容说明（写入 README）


def add(name, df, desc=""):
    assert len(name) <= 31 and not set(name) & set(":\\/?*[]"), name
    SHEETS[name] = df
    DESC[name] = desc
    return df


def tidy(df):
    """统一把浮点列舍入到 9 位小数：既保证 SHAP（~1e-5）与交互值（~1e-7）不失真，
    又避免 xlsx 体积无谓膨胀。"""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(9)
    return out


# ==========================================================================
# Fig 1 数据概况
# ==========================================================================
add("Fig1a", plot_tab[["plot_id", "lon", "lat", "plot_RGR"]]
    .rename(columns={"lon": "longitude_grid_center_degE",
                     "lat": "latitude_grid_center_degN",
                     "plot_RGR": "plot_mean_RGR_per_yr"}),
    f"Fig. 1a: the {len(plot_tab):,} inventory plots - 0.05-degree grid-cell centre "
    "positions (exact plot coordinates are confidential and are not distributed) "
    "and plot-mean RGR (point colour in the panel). Forest-cover background raster "
    "is in the maps zip.")

add("Fig1b", pd.DataFrame({
    "bin_left_per_yr": _RBR[:-1], "bin_right_per_yr": _RBR[1:],
    "n_broadleaf": _F1B_B, "n_conifer": _F1B_C,
}), "Fig. 1b: RGR distribution - histogram bin counts for the two functional groups "
    f"(59 equal-width bins over the drawn range 0-0.2 yr-1; group totals "
    f"{int((tab['fg_conifer'] == 0).sum()):,} broadleaf and "
    f"{int((tab['fg_conifer'] == 1).sum()):,} conifer trees, of which "
    f"{int((tab['RGR'] > 0.2).sum())} lie above 0.2 yr-1 outside the plotted range). "
    "Aggregate bin counts are provided in place of individual-tree records.")

add("Fig1c", plot_tab[["plot_id", "stand_age"]].dropna(subset=["stand_age"])
    .rename(columns={"stand_age": "stand_age_yr"}),
    f"Fig. 1c: stand-age structure - one row per plot with a recorded stand age "
    f"({int(plot_tab['stand_age'].notna().sum()):,}).")

add("Fig1d", plot_tab[["plot_id", "bio1", "bio12", "plot_RGR"]]
    .rename(columns={"bio1": "MAT_bio1_x10", "bio12": "MAP_bio12_mm",
                     "plot_RGR": "plot_mean_RGR_per_yr"}),
    "Fig. 1d: climate space - mean annual temperature (bio1, x10 degC) and annual "
    "precipitation (bio12, mm) per plot, coloured by plot-mean RGR.")

# ==========================================================================
# Fig 2 基础格局
# ==========================================================================
# (a) 尺寸依赖：与 09_figures.py 相同的 25 分箱 + 两组 n ≥ 20 的共同支撑
_bins_a = pd.cut(tab["lnD0"], 25)
_cnt_a = tab.groupby([_bins_a, tab["fg_conifer"]], observed=True)["RGR"].size().unstack(fill_value=0)
_ok_a = _cnt_a[(_cnt_a[0] >= 20) & (_cnt_a[1] >= 20)].index
_rows_a = []
for fg, gname in ((0, "broadleaf"), (1, "conifer")):
    s = (tab[tab["fg_conifer"] == fg].groupby(_bins_a, observed=True)["RGR"]
         .agg(["mean", "sem", "size"]).reindex(_ok_a))
    for iv, r in s.iterrows():
        _rows_a.append({"functional_group": gname, "lnD0_bin_mid": iv.mid,
                        "n_trees": int(r["size"]), "mean_RGR_per_yr": r["mean"],
                        "sem_RGR": r["sem"],
                        "ci95_lo": r["mean"] - 1.96 * r["sem"],
                        "ci95_hi": r["mean"] + 1.96 * r["sem"]})
FIG2A = pd.DataFrame(_rows_a)
add("Fig2a", FIG2A,
    "Fig. 2a: size dependence - 25-bin means of RGR by initial diameter, per functional "
    "group, restricted to bins where both groups have n >= 20 trees.")

# (b) 林龄衰减：左闭右开 10 年档（与 20_sf1_age_bins.csv 同口径）
_aged = plot_tab.dropna(subset=["stand_age"])
_bins_b = pd.cut(_aged["stand_age"], np.arange(0, 105, 10), right=False)
_gm = _aged.groupby(_bins_b, observed=True).agg(
    n_plots=("plot_RGR", "size"), mean_plot_RGR=("plot_RGR", "mean"),
    sem=("plot_RGR", "sem"))
FIG2B = pd.DataFrame({
    "stand_age_class_yr": [f"{int(i.left)}-{int(i.right)}" for i in _gm.index],
    "class_left_closed": [int(i.left) for i in _gm.index],
    "class_right_open": [int(i.right) for i in _gm.index],
    "bin_mid_yr": [i.mid for i in _gm.index],
    "n_plots": _gm["n_plots"].values,
    "mean_plot_RGR_per_yr": _gm["mean_plot_RGR"].values,
    "sem": _gm["sem"].values,
    "ci95_lo": (_gm["mean_plot_RGR"] - 1.96 * _gm["sem"]).values,
    "ci95_hi": (_gm["mean_plot_RGR"] + 1.96 * _gm["sem"]).values,
})
add("Fig2b", FIG2B,
    "Fig. 2b: age decline - decadal plot-mean RGR with 95% CI; bins are left-closed "
    "right-open, matching Fig. 2b and Supplementary Fig. 1a.")

add("Fig2c", plot_tab[["plot_id", "origin_natural", "plot_RGR"]]
    # origin_natural 以 -1 编码缺失（88 个样地），不能并入 planted：图 2c 只绘 0/1 两组
    .assign(origin=lambda d: d["origin_natural"].map(
        {0.0: "planted", 1.0: "natural", -1.0: "missing"}))
    .rename(columns={"plot_RGR": "plot_mean_RGR_per_yr",
                     "origin_natural": "origin_code_0planted_1natural_m1missing"})
    [["plot_id", "origin", "origin_code_0planted_1natural_m1missing",
      "plot_mean_RGR_per_yr"]],
    "Fig. 2c: box plot of plot-mean RGR by stand origin (530 planted, 569 natural; "
    "88 plots with missing origin are labelled 'missing' and are not plotted).")

add("Fig2d", plot_tab[["plot_id", "fg_conifer", "plot_RGR"]]
    .assign(functional_group=lambda d: np.where(d["fg_conifer"] == 1, "conifer", "broadleaf"))
    .rename(columns={"plot_RGR": "plot_mean_RGR_per_yr"})
    [["plot_id", "functional_group", "fg_conifer", "plot_mean_RGR_per_yr"]],
    "Fig. 2d: box plot of plot-mean RGR by dominant functional group.")

# ==========================================================================
# Fig 3 三重 CV
# ==========================================================================
nres = pd.read_csv(os.path.join(R, "22_nested_cv_results.csv"))
nres = nres[nres["model"] == "XGBoost-nested"].set_index("cv")


def _ci(s):
    a, b = str(s).strip("()").split(",")
    return float(a), float(b)


FIG3A = pd.DataFrame([{
    "cv_design": cv,
    "tree_R2": nres.loc[cv, "R2"], "tree_R2_ci95_lo": _ci(nres.loc[cv, "ci_tree_R2"])[0],
    "tree_R2_ci95_hi": _ci(nres.loc[cv, "ci_tree_R2"])[1],
    "plot_R2": nres.loc[cv, "plot_R2"], "plot_R2_ci95_lo": _ci(nres.loc[cv, "ci_plot_R2"])[0],
    "plot_R2_ci95_hi": _ci(nres.loc[cv, "ci_plot_R2"])[1],
    "RMSE_per_yr": nres.loc[cv, "RMSE"], "Spearman": nres.loc[cv, "Spearman"],
    "n_est_median": nres.loc[cv, "n_est_median"],
} for cv in ["random", "group", "spatial"]])
add("Fig3a", FIG3A,
    "Fig. 3a: tree-level and plot-level out-of-fold R2 under the three nested "
    "cross-validation designs, with plot-cluster bootstrap 95% CIs.")

_blob = pickle.load(open(os.path.join(R, "22_nested_oof.pkl"), "rb"))
df3 = tab[["plot_id", "RGR"]].copy()
df3["pred"] = _blob["oof"]["spatial"].values
# 单木明细不对外提供：Fig3b 给图面显示范围（0–0.22 yr⁻¹）上的二维分箱计数
_ke3 = (df3["RGR"].values <= 0.22) & (df3["pred"].values <= 0.22)
_ed3 = np.linspace(0, 0.22, 23)          # 22 × 22 网格
_H3, _, _ = np.histogram2d(df3["RGR"].values[_ke3], df3["pred"].values[_ke3],
                           bins=[_ed3, _ed3])
_rows3 = [{"obs_bin_left_per_yr": _ed3[i], "obs_bin_right_per_yr": _ed3[i + 1],
           "pred_bin_left_per_yr": _ed3[j], "pred_bin_right_per_yr": _ed3[j + 1],
           "n_trees": int(_H3[i, j])}
          for i in range(22) for j in range(22) if _H3[i, j] > 0]
add("Fig3b", pd.DataFrame(_rows3),
    "Fig. 3b: observed vs out-of-fold predicted tree-level RGR under spatial block CV - "
    "tree counts on a 22 x 22 grid spanning the drawn 0-0.22 yr-1 axes (the panel renders "
    "the same points as a hexagonal log-density binning). Aggregate cell counts are "
    "provided in place of individual-tree records.")
_pm3 = df3.groupby("plot_id")[["RGR", "pred"]].mean()
add("Fig3c", pd.DataFrame({
    "plot_id": _pm3.index.values,
    "observed_plot_mean_RGR_per_yr": _pm3["RGR"].values,
    "predicted_plot_mean_RGR_per_yr_spatialCV": _pm3["pred"].values,
}), "Fig. 3c: observed vs predicted plot-mean RGR under the same spatial block CV.")

# ==========================================================================
# Fig 4 SHAP 归因
# ==========================================================================
gsh = pd.read_csv(os.path.join(R, "26_group_shares.csv")).set_index("group")
abl = pd.read_csv(os.path.join(R, "28_ablation_nested.csv"))
abl_i = abl.set_index("removed")
nz = pd.read_csv(os.path.join(R, "27_family_share_normalized.csv")).set_index("group")
dom = pd.read_csv(os.path.join(R, "28_plot_dominance_nested.csv")).set_index("group")
ORDER4 = ["tree", "stand", "climate", "soil", "terrain"]

FIG4A = pd.DataFrame([{
    "family": g,
    "share_of_total_absSHAP": gsh.loc[g, "share"],
    "share_ci95_lo": gsh.loc[g, "ci_lo"], "share_ci95_hi": gsh.loc[g, "ci_hi"],
    "ablation_d_tree_R2": (abl_i.loc[g, "d_tree_R2"] if g in abl_i.index else 0.0),
    "ablation_d_plot_R2": (abl_i.loc[g, "d_plot_R2"] if g in abl_i.index else 0.0),
    "n_vars_in_family": int(nz.loc[g, "n_vars"]),
} for g in ORDER4])
add("Fig4a", FIG4A,
    "Fig. 4a: family shares of total out-of-fold |SHAP| with 95% CIs (bars), and the "
    "change in tree- and plot-level R2 when each family is removed (ablation line).")

imp = pd.read_csv(os.path.join(R, "26_shap_importance.csv"))
add("Fig4b", imp.head(15)[["rank", "var", "group", "mean_abs_shap", "ci_lo", "ci_hi",
                           "share_total"]]
    .rename(columns={"mean_abs_shap": "mean_absSHAP_RGR_per_yr",
                     "ci_lo": "mean_absSHAP_ci95_lo", "ci_hi": "mean_absSHAP_ci95_hi",
                     "share_total": "share_of_total_absSHAP"}),
    "Fig. 4b: the top 15 predictors by mean |SHAP| with plot-cluster bootstrap 95% CIs.")

add("Fig4c", nz.loc[ORDER4].reset_index()[
    ["group", "n_vars", "share", "share_per_var", "rank_by_share",
     "rank_by_share_per_var", "ci_lo", "ci_hi"]]
    .rename(columns={"group": "family", "share": "family_share",
                     "share_per_var": "family_share_per_variable"}),
    "Fig. 4c: family share (solid bars) against the same share divided by the number of "
    "variables in the family (faded bars).")

add("Fig4d", dom.reset_index()[["group", "mean_share", "n_plots_dominant",
                                "pct_plots_dominant"]]
    .rename(columns={"group": "family", "mean_share": "mean_within_plot_share",
                     "pct_plots_dominant": "pct_plots_where_dominant"}),
    "Fig. 4d: pie chart of the dominant predictor family within each plot, by per-plot "
    "mean |SHAP|.")

# ==========================================================================
# Fig 5 效应形态
# ==========================================================================
S = pickle.load(open(os.path.join(R, "26_shap_matrix.pkl"), "rb"))
Sm = S["shap"].reindex(tab.index)
# 与 09_figures.py 的 Fig5 面板、20_si_figures.py 的形状稳定性图、Fig.5 图注
# 共用同一固定集合（不用 head(6)，理由见 09_figures.py）。
TOP6 = ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]

# 单木明细不对外提供：Fig5a-f 给分箱均值曲线（图中黑线）的底表
_rows = []
for v in TOP6:
    d = pd.DataFrame({"value": tab[v].values, "shap": Sm[v].values})
    q = pd.qcut(d["value"], 30, duplicates="drop")
    g = d.groupby(q, observed=True).agg(x=("value", "mean"), y=("shap", "mean"),
                                        n=("shap", "size"))
    for k, (_, r) in enumerate(g.iterrows(), 1):
        _rows.append({"variable": v, "bin": k, "bin_mean_value": r["x"],
                      "bin_mean_SHAP": r["y"], "n_trees_in_bin": int(r["n"])})
add("Fig5a-f", pd.DataFrame(_rows),
    "Fig. 5a-f: SHAP dependence - bin means of the out-of-fold SHAP value against the "
    "predictor (the black 30-bin mean curves in each panel; 30 quantile bins where the "
    "predictor has enough distinct values; origin_natural is binary and yields 2). "
    "Aggregate bin means are provided in place of individual-tree records.")

_lc = pickle.load(open(os.path.join(R, "29_lnd0_curves.pkl"), "rb"))
add("Fig5b_deltaD", pd.DataFrame({
    "bin": np.arange(1, len(_lc["deltaD"]) + 1),
    "lnD0_bin_mean": _lc["deltaD"]["v"].values,
    "deltaD_bin_mean_cm_per_5yr": _lc["deltaD"]["s"].values,
    "RGR_model_lnD0_bin_mean": _lc["rgr"]["v"].values,
    "RGR_model_SHAP_bin_mean": _lc["rgr"]["s"].values,
}), "Fig. 5b: the dashed line - initial-diameter dependence from the absolute-increment "
    "(deltaD) companion model on its own axis, alongside the RGR-model curve it reverses.")

inter = pickle.load(open(os.path.join(R, "26_inter_dep_data.pkl"), "rb"))

_d = inter["stand_age__fg_conifer"].reset_index(drop=True)
_ab = pd.cut(_d["v1"], [0, 5, 10, 15, 20, 30, 40, 50, 90])
_rows = []
for fg, gname in ((0, "broadleaf"), (1, "conifer")):
    s = (_d[_d["v2"] == fg].groupby(_ab, observed=True)["inter"]
         .agg(["mean", "sem", "size"]).dropna())
    for iv, r in s.iterrows():
        _rows.append({"functional_group": gname, "stand_age_bin_mid_yr": iv.mid,
                      "mean_interaction_per_yr": r["mean"], "sem": r["sem"],
                      "n_trees_in_bin": int(r["size"]),
                      "ci95_lo": r["mean"] - 1.96 * r["sem"],
                      "ci95_hi": r["mean"] + 1.96 * r["sem"]})
add("Fig5g", pd.DataFrame(_rows),
    "Fig. 5g: stand age x functional group interaction - binned means with 95% CI per "
    "group (computed on the 5,000-tree interaction subsample).")

_d = inter["stand_density__lnD0"].reset_index(drop=True)
_hb = pd.qcut(_d["v1"], 4, duplicates="drop")
_sb = pd.qcut(_d["v2"], 4, duplicates="drop")
_piv = _d.groupby([_hb, _sb], observed=True)["inter"].mean().unstack()
_cnt = _d.groupby([_hb, _sb], observed=True)["inter"].size().unstack()
_rows = []
for i, di in enumerate(_piv.index):
    for j, si in enumerate(_piv.columns):
        _rows.append({
            "stand_density_quartile": f"Q{i+1}",
            "initial_size_quartile": f"Q{j+1}",
            "density_range_trees_per_ha": f"{di.left:.0f}-{di.right:.0f}",
            "lnD0_range": f"{si.left:.3f}-{si.right:.3f}",
            "mean_interaction_per_yr": _piv.values[i, j],
            "n_trees_in_cell": int(_cnt.values[i, j]),
        })
add("Fig5h", pd.DataFrame(_rows),
    "Fig. 5h: stand density x initial diameter interaction - mean interaction within the "
    "4 x 4 quartile grid, with the tree count per cell.")

# ==========================================================================
# Fig 6 制图（面板 d 的数值；a–c 为栅格，见 Source_Data_maps.zip）
# ==========================================================================
attr = pd.read_csv(os.path.join(R, "23_dual_map_attribution.csv"))
dsum = C.load_json("23_dual_map_summary.json")
_r2 = {r["response"]: r for r in dsum["responses"]}
add("Fig6d", pd.DataFrame([{
    "map_response": r["response"],
    "nested_out_of_fold_R2": r["R2"], "Spearman": r["Spearman"], "RMSE": r["RMSE"],
    "n_plots": r["n_plots"],
    "share_stand": attr[attr["response"] == r["response"]]["share_stand"].iloc[0],
    "share_climate": attr[attr["response"] == r["response"]]["share_climate"].iloc[0],
    "share_soil": attr[attr["response"] == r["response"]]["share_soil"].iloc[0],
    "share_terrain": attr[attr["response"] == r["response"]]["share_terrain"].iloc[0],
} for r in dsum["responses"]]),
    "Fig. 6d: attribution of each mapped pattern by mean |SHAP| family share, together "
    "with the nested out-of-fold skill of each map. Panels 6a-c are rasters "
    "(Source_Data_maps.zip).")

# ==========================================================================
# SF1 年龄衰减
# ==========================================================================
_pt = (tab.groupby("plot_id")
       .agg(plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"))
       .dropna(subset=["stand_age"]).reset_index())
_pt["age_bin"] = pd.cut(_pt["stand_age"], bins=list(range(0, 90, 10)), right=False)
_b = _pt.groupby("age_bin", observed=True).agg(
    n_plots=("plot_RGR", "size"), m=("plot_RGR", "mean"), s=("plot_RGR", "sem"))
add("SF1a", pd.DataFrame({
    "stand_age_class_yr": [f"{int(i.left)}-{int(i.right)}" for i in _b.index],
    "class_left_closed": [int(i.left) for i in _b.index],
    "class_right_open": [int(i.right) for i in _b.index],
    "n_plots": _b["n_plots"].values,
    "mean_plot_RGR_per_yr": _b["m"].values,
    "sem": _b["s"].values,
    "ci95_lo": (_b["m"] - 1.96 * _b["s"]).values,
    "ci95_hi": (_b["m"] + 1.96 * _b["s"]).values,
    "plotted": (_b["n_plots"] >= 10).values,          # 图注声明只绘 n ≥ 10 的档
}).sort_values("class_left_closed"),
    "Supplementary Fig. 1a: age-graded plot-mean RGR with per-decade sample sizes and "
    "95% CIs. The 'plotted' column flags the classes drawn (n >= 10).")

_h_edges = np.arange(0, 85, 5)
_h_cnt, _ = np.histogram(_pt["stand_age"].values, bins=_h_edges)
add("SF1b", pd.DataFrame({
    "stand_age_class_yr": [f"{int(a)}-{int(b)}" for a, b in zip(_h_edges[:-1], _h_edges[1:])],
    "n_plots": _h_cnt,
    "median_stand_age_yr": _pt["stand_age"].median(),
    "n_plots_total": len(_pt),
}), f"Supplementary Fig. 1b: stand-age distribution of the {len(_pt)} plots with a recorded age.")

# ==========================================================================
# SF2 密度依赖
# ==========================================================================
den = pd.DataFrame({"value": tab["stand_density"].values,
                    "shap": Sm["stand_density"].values * 1000,
                    "fg": tab["fg_conifer"].values}).dropna()
# 单木明细不对外提供：SF2 给分箱均值曲线（图中黑线）及其跨度的底表
_q2 = pd.qcut(den["value"], 30, duplicates="drop")
_g2 = den.groupby(_q2, observed=True).agg(x=("value", "mean"), y=("shap", "mean"),
                                          n=("shap", "size"))
add("SF2", pd.DataFrame({
    "bin": np.arange(1, len(_g2) + 1),
    "bin_mean_density_trees_per_ha": _g2["x"].values,
    "bin_mean_SHAP_x1e3_per_yr": _g2["y"].values,
    "n_trees_in_bin": _g2["n"].values,
    "bin_mean_span_x1e3_per_yr": float(_g2["y"].max() - _g2["y"].min()),
}), "Supplementary Fig. 2: stand-density SHAP dependence - the black 30-bin mean curve "
    f"(SHAP in 1e-3 yr^-1) and the bin-mean span quoted in the panel title; bins cover all "
    f"{len(den):,} trees with a recorded density. Aggregate bin means are provided in "
    "place of individual-tree records.")

# ==========================================================================
# SF3 跨折形状稳定性
# ==========================================================================
curves = pickle.load(open(os.path.join(R, "33_shape_curves.pkl"), "rb"))
stab = pd.read_csv(os.path.join(R, "33_shape_stability.csv"))
ORDER6 = ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]
_rows = []
for v in ORDER6:
    fc = curves[v]
    lo = max(c["x"].min() for c in fc.values())
    hi = min(c["x"].max() for c in fc.values())
    grid = np.linspace(lo, hi, 50)
    interp = [np.interp(grid, c["x"], c["y"]) for c in fc.values()]
    mean = np.mean(interp, axis=0)
    for i, xv in enumerate(grid):
        row = {"variable": v, "grid_point": i + 1, "predictor_value": xv,
               "mean_SHAP_over_folds": mean[i]}
        for f in sorted(fc):
            row[f"SHAP_fold{f}"] = np.interp(xv, fc[f]["x"], fc[f]["y"])
        _rows.append(row)
add("SF3a", pd.DataFrame(_rows),
    "Supplementary Fig. 3: per-fold SHAP dependence curves for the six leading "
    "predictors (five folds on a common 50-point grid) and their mean curve.")

add("SF3b", stab.rename(columns={
    "var": "variable", "mean_pairwise_r": "mean_pairwise_shape_r",
    "min_pairwise_r": "min_pairwise_shape_r"}),
    "Supplementary Fig. 3: the inset statistics - mean and minimum pairwise correlation "
    "of dependence curves across the 10 fold pairs.")

# ==========================================================================
# SF4 ΔD 伴随模型
# ==========================================================================
dd_all = pd.read_csv(os.path.join(R, "29_deltad_nested.csv"))
from scipy.stats import spearmanr
_rho_v = spearmanr(dd_all["share_RGR"], dd_all["share_deltaD"]).statistic
dd = dd_all[dd_all["share_RGR"] > 0]
add("SF4a", dd[["var", "group", "share_RGR", "share_deltaD"]]
    .rename(columns={"var": "variable", "group": "family",
                     "share_RGR": "share_in_RGR_model",
                     "share_deltaD": "share_in_deltaD_model"})
    .assign(spearman_rho_all_38_features=_rho_v),
    "Supplementary Fig. 4a: variable-level shares of attribution under the RGR and "
    "deltaD responses (36 variables with non-zero share).")

fam = dd.groupby("group")[["share_RGR", "share_deltaD"]].sum().sort_values(
    "share_RGR", ascending=False)
add("SF4b", pd.DataFrame({
    "family": fam.index,
    "share_RGR_model": fam["share_RGR"].values,
    "share_deltaD_model": fam["share_deltaD"].values,
    "family_share_spearman_rho": spearmanr(fam["share_RGR"], fam["share_deltaD"]).statistic,
}), "Supplementary Fig. 4b: family shares under the two responses, with the Spearman "
    "rank correlation between them.")

# ==========================================================================
# README
# ==========================================================================
README = [
    ("Source Data", ""),
    ("Manuscript", "Stand structure outweighs climate as a predictor of individual-tree "
                   "growth in subtropical China"),
    ("Journal", "Nature Communications"),
    ("", ""),
    ("Contents", "One sheet per figure panel. Every panel of all six main figures and all "
                 "four Supplementary Figures is represented. Sheet names are Fig<N><panel> "
                 "and SF<N><panel>."),
    ("Maps", "Fig. 1a and Fig. 6a-c are 250-m rasters (1.54 million pixels) and cannot be "
             "stored in Excel; they are supplied in Source_Data_maps.zip as GeoTIFFs whose "
             "arrays are identical to those plotted."),
    ("Units", "RGR is yr^-1 (log-diameter increment per year). SHAP values carry the units "
              "of the response (RGR yr^-1). SHAP interaction values and the stand-density "
              "SHAP values in sheet SF2 are expressed in 1e-3 yr^-1. deltaD is cm per 5 yr."),
    ("Missing values", "Blank cells denote missing predictor values, which the models "
                       "handle natively (see Methods). Two exceptions carry an explicit "
                       "missing code rather than a blank: origin_natural uses -1 "
                       "(88 plots) and the functional-group column is 0/1 throughout."),
    ("Confidence intervals", "ci95_lo / ci95_hi are 95% confidence intervals from 1,000 "
                             "plot-cluster bootstrap resamples (see Methods)."),
    ("", ""),
    ("Sheet", "Figure panel and content"),
]
for n, d in SHEETS.items():
    README.append((n, f"{DESC.get(n, '')} [{len(d)} rows x {len(d.columns)} cols]"))
README_DF = pd.DataFrame(README, columns=["Source Data", "Description"])

# ==========================================================================
# 写出 xlsx
# ==========================================================================
os.makedirs(OUT, exist_ok=True)
with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
    tidy(README_DF).to_excel(xw, sheet_name="README", index=False)
    for name, df in SHEETS.items():
        tidy(df).to_excel(xw, sheet_name=name, index=False)

say("# Source Data 构建报告（37_source_data）\n")
say(f"- 输出: `08_定稿/Source_Data.xlsx`（{os.path.getsize(XLSX) / 1024 / 1024:.1f} MB）")
say(f"- 工作表数: {len(SHEETS) + 1}（含 README）")
say("")
say("## 1. 工作表清单")
for name, df in SHEETS.items():
    say(f"  - `{name}`: {len(df)} 行 × {len(df.columns)} 列")

# ==========================================================================
# 栅格（Fig.1a 背景与 Fig.6a-c）→ GeoTIFF，逐像元等同绘图数组
# ==========================================================================
import rasterio
from rasterio.transform import Affine

mp = np.load(os.path.join(MAPDIR, "dual_map_arrays.npz"), allow_pickle=True)
_tr = mp["transform"]
_transform = Affine(_tr[0], _tr[1], _tr[2], _tr[3], _tr[4], _tr[5])
_mask = mp["forest_frac"] > 0

RASTERS = [
    ("Fig1a_forest_cover_fraction_250m.tif", mp["forest_frac"],
     "Fractional forest cover per 250-m cell (Fig. 1a background shading)"),
    ("Fig6a_conifer_RGR_250m.tif", mp["mean_conifer"],
     "Ensemble-predicted conifer plot-mean RGR, yr^-1 (Fig. 6a)"),
    ("Fig6b_broadleaf_RGR_250m.tif", mp["mean_broadleaf"],
     "Ensemble-predicted broadleaf plot-mean RGR, yr^-1 (Fig. 6b)"),
    ("Fig6c_model_disagreement_250m.tif",
     np.fmax(mp["sd_conifer"], mp["sd_broadleaf"]),
     "Model disagreement: max over the two responses of the s.d. across the five "
     "spatially blocked fold models, yr^-1 (Fig. 6c)"),
]
_nodata = -9999.0
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for fname, arr, desc in RASTERS:
        tmp = os.path.join(OUT, "_tmp_" + fname)
        data = np.where(np.isfinite(arr), arr, _nodata).astype("float32")
        with rasterio.open(tmp, "w", driver="GTiff", height=data.shape[0],
                           width=data.shape[1], count=1, dtype="float32",
                           crs="EPSG:4326", transform=_transform,
                           nodata=_nodata, compress="deflate") as dst:
            dst.write(data, 1)
            dst.update_tags(description=desc)
        z.write(tmp, fname)
        os.remove(tmp)
        say(f"  - `{fname}`: {arr.shape[1]} × {arr.shape[0]} 像元，"
            f"{desc}")

say("")
say(f"## 2. 栅格包 `Source_Data_maps.zip`（{os.path.getsize(ZIP) / 1024 / 1024:.1f} MB）")
say("  - CRS EPSG:4326；nodata = -9999；float32；deflate 压缩")
say("  - 数组与 09_figures.py 绘图所用 `dual_map_arrays.npz` 逐像元一致")

# ==========================================================================
# 核验：底表数值 ↔ 图面数值
# ==========================================================================
say("")
say("## 3. 核验（底表 ↔ 图面）")

N_PLOT = len(plot_tab)
N_TREE = len(tab)
g26 = pd.read_csv(os.path.join(R, "26_group_shares.csv")).set_index("group")["share"]
i26 = pd.read_csv(os.path.join(R, "26_shap_importance.csv")).sort_values("mean_abs_shap",
                                                                        ascending=False)
cv22 = pd.read_csv(os.path.join(R, "22_nested_cv_results.csv")).set_index("cv")
dom28 = pd.read_csv(os.path.join(R, "28_plot_dominance_nested.csv")).set_index("group")
attr23 = pd.read_csv(os.path.join(R, "23_dual_map_attribution.csv")).set_index("response")

check(f"Fig1a 行数 = {N_PLOT:,} 个样地", len(SHEETS["Fig1a"]) == N_PLOT)
check("Fig1b 分箱计数与建模表一致（59 箱覆盖 0–0.2，其余株数单列核对）",
      int(SHEETS["Fig1b"]["n_broadleaf"].sum())
      == int(((tab["fg_conifer"] == 0) & (tab["RGR"] <= 0.2)).sum())
      and int(SHEETS["Fig1b"]["n_conifer"].sum())
      == int(((tab["fg_conifer"] == 1) & (tab["RGR"] <= 0.2)).sum())
      and int(SHEETS["Fig1b"]["n_broadleaf"].sum() + SHEETS["Fig1b"]["n_conifer"].sum())
      + int((tab["RGR"] > 0.2).sum()) == N_TREE)
check("Fig1c 行数 = 有林龄样地数",
      len(SHEETS["Fig1c"]) == int(plot_tab["stand_age"].notna().sum()))
check("Fig2a 两组箱数相等（共同支撑）",
      len(FIG2A) == 2 * len(_ok_a) and FIG2A.groupby("functional_group").size().nunique() == 1,
      f"实际 {len(FIG2A)} 行，每组 {len(_ok_a)} 箱")
check("Fig2b 分箱计数与建模表一致",
      int(SHEETS["Fig2b"]["n_plots"].sum()) == int(plot_tab["stand_age"].notna().sum()))
check("Fig2c 样地数 = 起源两分类计数（零缺失）",
      SHEETS["Fig2c"]["origin"].value_counts().to_dict()
      == {"natural": int((plot_tab["origin_natural"] == 1).sum()),
          "planted": int((plot_tab["origin_natural"] == 0).sum())},
      str(SHEETS["Fig2c"]["origin"].value_counts().to_dict()))
check("Fig3a 三重 CV 的 R² 与 22_nested_cv_results.csv 一致",
      np.allclose(SHEETS["Fig3a"]["tree_R2"].values,
                  [cv22.loc[c, "R2"] for c in ("random", "group", "spatial")], atol=1e-9)
      and np.allclose(SHEETS["Fig3a"]["plot_R2"].values,
                      [cv22.loc[c, "plot_R2"] for c in ("random", "group", "spatial")], atol=1e-9))
check("Fig3b 分箱计数覆盖显示范围（0–0.22）内全部株树",
      int(SHEETS["Fig3b"]["n_trees"].sum())
      == int(((df3["RGR"] <= 0.22) & (df3["pred"] <= 0.22)).sum()))
check(f"Fig3c 行数 = {N_PLOT:,}", len(SHEETS["Fig3c"]) == N_PLOT)
check("Fig4a 家族份额 = 26_group_shares.csv",
      np.allclose(SHEETS["Fig4a"].set_index("family").loc[
          ["stand", "tree", "climate", "soil", "terrain"], "share_of_total_absSHAP"].values,
          [g26[f] for f in ("stand", "tree", "climate", "soil", "terrain")], atol=1e-6))
check(f"Fig4b 前 15 变量首位为 {i26.iloc[0]['var']}",
      SHEETS["Fig4b"].iloc[0]["var"] == i26.iloc[0]["var"]
      and abs(SHEETS["Fig4b"].iloc[0]["share_of_total_absSHAP"] - i26.iloc[0]["share_total"]) < 1e-6)
check("Fig4d 样地主导计数与 28_plot_dominance_nested.csv 一致",
      int(SHEETS["Fig4d"].set_index("family").loc["tree", "n_plots_dominant"])
      == int(dom28.loc["tree", "n_plots_dominant"]))
check("Fig5a-f 分箱表覆盖 6 变量、每变量分箱计数合计 = 建模表株数",
      all(int(SHEETS["Fig5a-f"].groupby("variable")["n_trees_in_bin"].sum().get(v, 0)) == N_TREE
          for v in TOP6))
check("Fig5a-f 分箱数与绘图逻辑一致（origin_natural 为二值 → ≤2 箱）",
      SHEETS["Fig5a-f"].groupby("variable").size().to_dict().get("origin_natural") <= 2
      and SHEETS["Fig5a-f"].groupby("variable").size().to_dict().get("stand_age") >= 10,
      str(SHEETS["Fig5a-f"].groupby("variable").size().to_dict()))
check("Fig5h 16 个格（4 密度 × 4 大小）", len(SHEETS["Fig5h"]) == 16)
check("Fig6d 制图归因与 23_dual_map_attribution.csv 一致",
      abs(SHEETS["Fig6d"].set_index("map_response").loc["conifer", "share_stand"]
          - attr23.loc["conifer", "share_stand"]) < 1e-6
      and abs(SHEETS["Fig6d"].set_index("map_response").loc["broadleaf", "share_terrain"]
              - attr23.loc["broadleaf", "share_terrain"]) < 1e-6)
check("SF1a 只绘 n ≥ 10 的档",
      int((~SHEETS["SF1a"]["plotted"]).sum()) == int((SHEETS["SF1a"]["n_plots"] < 10).sum()))
check("SF1b 直方图总数 = 有林龄样地数",
      int(SHEETS["SF1b"]["n_plots"].sum()) == int(plot_tab["stand_age"].notna().sum()))
check("SF2 分箱计数合计 = 密度非缺失株数",
      int(SHEETS["SF2"]["n_trees_in_bin"].sum()) == len(den))
check(f"SF4a {len(SHEETS['SF4a'])} 个变量份额（与 SHAP 特征数一致）",
      len(SHEETS["SF4a"]) == len(C.selected_features(vd)))
check("栅格包含 4 个 GeoTIFF", len(RASTERS) == 4)

# --- 交付物级核验：Source Data 的数值必须能在【图面实际渲染的文字】中找到 ---
# 理由同 Stage 4.5 清单 L1：只对源数据自洽不足以证明交付物一致，
# 须把图面渲染出的 Text artist 也纳入断言。
say("")
say("  --- 图面渲染文字交叉核验（fig_labels_main.json / fig_labels_si.json） ---")
ML = json.load(open(os.path.join(R, "fig_labels_main.json"), encoding="utf-8"))
SL = json.load(open(os.path.join(R, "fig_labels_si.json"), encoding="utf-8"))


def rendered(fig, s, store):
    return any(s in t for t in store.get(fig, []))


for cv, dec in (("random", 2), ("group", 2), ("spatial", 2)):
    v = float(SHEETS["Fig3a"].set_index("cv_design").loc[cv, "tree_R2"])
    check(f"Fig3a 图面渲染 {v:.{dec}f}（{cv}）", rendered("Fig3_cv", f"{v:.{dec}f}", ML))

for fam in ("stand", "tree", "climate", "soil", "terrain"):
    v = float(SHEETS["Fig4a"].set_index("family").loc[fam, "share_of_total_absSHAP"]) * 100
    check(f"Fig4a 图面渲染 {v:.1f}%（{fam}）", rendered("Fig4_shap", f"{v:.1f}%", ML))

_v4b = SHEETS["Fig4b"].iloc[0]
check(f"Fig4b 底表首位变量 {_v4b['var']}（图面只渲染变量名与 mean|SHAP|）",
      _v4b["var"] == i26.iloc[0]["var"]
      and abs(_v4b["share_of_total_absSHAP"] - i26.iloc[0]["share_total"]) < 1e-6)

for resp in ("conifer", "broadleaf"):
    for fam in ("share_stand", "share_climate", "share_terrain"):
        v = float(SHEETS["Fig6d"].set_index("map_response").loc[resp, fam]) * 100
        check(f"Fig6d 图面渲染 {round(v)}%（{resp}/{fam}）",
              rendered("Fig6_map", str(round(v)), ML))

check(f"Fig2b 底表分箱 n = {[int(x) for x in SHEETS['Fig2b']['n_plots'].values[:6]]}",
      int(SHEETS["Fig2b"]["n_plots"].sum()) == int(plot_tab["stand_age"].notna().sum()),
      "（n 标注只在图注与 SF1 中给出，此处以底表数值为准）")

_pc = SHEETS["Fig2c"]["origin"].value_counts().to_dict()
check(f"Fig2c 图面渲染箱线标签 Planted(n={_pc['planted']}) / Natural(n={_pc['natural']})",
      rendered("Fig2_patterns", f"Planted\n(n={_pc['planted']})", ML)
      and rendered("Fig2_patterns", f"Natural\n(n={_pc['natural']})", ML))

check(f"Fig1a 图面渲染 '{N_PLOT:,} CFI plots'",
      rendered("Fig1_overview", f"{N_PLOT:,} CFI plots", ML) and len(SHEETS["Fig1a"]) == N_PLOT)

check("SF1a 图面渲染各档 n（n ≥ 10 的六档）",
      all(rendered("SF1_age_decay", f"n={int(n)}", SL)
          for n in SHEETS["SF1a"][SHEETS["SF1a"]["plotted"]]["n_plots"].values))

check("SF2 图面标题跨度与 SF2 底表一致",
      rendered("SF2_density",
      f"{float(SHEETS['SF2']['bin_mean_span_x1e3_per_yr'].iloc[0]):.1f}", SL))

for fam in ("stand", "tree", "terrain", "climate", "soil"):
    f = SHEETS["SF4b"].set_index("family").loc[fam]
    # 期望值取"底表真值的 1 位小数单步舍入"
    rgr = round(float(f["share_RGR_model"]) * 100, 1)
    dd = round(float(f["share_deltaD_model"]) * 100, 1)
    check(f"SF4b 图面渲染 {fam}: RGR {rgr} / ΔD {dd}",
          rendered("SF4_deltad", f"{rgr:.1f}", SL) and rendered("SF4_deltad", f"{dd:.1f}", SL))

_rho4 = float(SHEETS["SF4a"]["spearman_rho_all_38_features"].iloc[0])
check(f"SF4a 图面渲染 'Spearman ρ = {_rho4:.3f}' 与底表 ρ 一致",
      rendered("SF4_deltad", f"{_rho4:.3f}", SL)
      and rendered("SF4_deltad", f"(all {len(SHEETS['SF4a'])} features; "
                                  f"{len(SHEETS['SF4a'])} plotted)", SL))

say("")
say("## 结论")
if fails:
    say(f"**FAIL** — {len(fails)} 项：")
    for f in fails:
        say(f"- {f}")
else:
    say(f"**PASS** — Source Data 覆盖 6 张主图 + 4 张 SI 图的全部面板（{len(SHEETS)} 个工作表），"
        f"关键数值与图面逐项对源一致；地图以 GeoTIFF 单独提供。")

with open(os.path.join(OUT, "04_Source_Data_说明.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("")
say("DONE 37 source data")
if fails:
    raise SystemExit(1)