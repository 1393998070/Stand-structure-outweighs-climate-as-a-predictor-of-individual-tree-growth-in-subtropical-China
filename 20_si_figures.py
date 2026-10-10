# -*- coding: utf-8 -*-
"""P1-12 SI 补充图生成（Nature Communications 风格，英文，300dpi PNG + PDF）。
SF1 年龄衰减(带箱 n) / SF2 密度依赖 / SF3 ΔD 变量级重要性 / SF4 SHAP 跨折形状稳定性
输出: 05_图表/SF1..SF4 (png+pdf) + 04_分析结果/20_sf1_age_bins.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FIG = C.FIGS
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
MM = 1 / 25.4
DOUBLE = 183 * MM
SINGLE = 89 * MM

GROUP_COLORS = {"tree": "#7f3b08", "stand": "#b35806", "climate": "#2166ac",
                "soil": "#1a9641", "terrain": "#8c510a"}
GROUP_LABEL = {"tree": "Tree attributes", "stand": "Stand structure",
               "climate": "Climate", "soil": "Soil", "terrain": "Terrain"}
NICE = {"lnD0": "ln(initial DBH)", "stand_age": "Stand age (yr)",
        "conifer_prop": "Conifer proportion", "bio12": "MAP (bio12)",
        "bio15": "Precip. seasonality (bio15)", "origin_natural": "Origin (natural)",
        "stand_density": "Stand density (trees ha$^{-1}$)"}


LABELS = {}


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"))
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    texts = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text().strip()
        if s:
            texts.append(s)
    LABELS[name] = texts
    with open(os.path.join(C.OUT, "fig_labels_si.json"), "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=1)
    plt.close(fig)
    print(f"saved {name} ({len(texts)} labels)", flush=True)


tab, vd, folds = C.load_model_data()

# ============================== SF1 年龄衰减（带箱 n） ==============================
pt = (tab.groupby("plot_id")
      .agg(plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"))
      .dropna(subset=["stand_age"]).reset_index())
bins = list(range(0, 90, 10))
pt["age_bin"] = pd.cut(pt["stand_age"], bins=bins, right=False)
b = pt.groupby("age_bin", observed=True).agg(
    n=("plot_RGR", "size"), m=("plot_RGR", "mean"), s=("plot_RGR", "sem"))
b["lo"] = b["m"] - 1.96 * b["s"]
b["hi"] = b["m"] + 1.96 * b["s"]
b.to_csv(os.path.join(C.OUT, "20_sf1_age_bins.csv"), encoding="utf-8-sig")
# 只绘 n ≥ 10 的年龄档：60–70 与 70–80 档仅 2 株与 1 株，其 SEM 无统计意义，
# 且 2 株档的误差棒（~11–55）会把纵轴拉长、压扁信息量最大的四个年代档。
# 这与 Fig. 2b 使用的六个年代档完全一致。
_b_plot = b[b["n"] >= 10]

fig = plt.figure(figsize=(DOUBLE, 52 * MM))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.32)
ax = fig.add_subplot(gs[0])
xs = np.arange(len(_b_plot))
ax.errorbar(xs, _b_plot["m"] * 1000,
            yerr=[(_b_plot["m"] - _b_plot["lo"]) * 1000, (_b_plot["hi"] - _b_plot["m"]) * 1000],
            fmt="o", ms=3.2, lw=1, color="#b35806", ecolor="#b35806",
            elinewidth=0.9, capsize=2)
for x, n, m in zip(xs, _b_plot["n"], _b_plot["m"]):
    ax.annotate(f"n={n}", (x, m * 1000), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=5.6, color="#444444")
ax.set_xticks(xs)
ax.set_xticklabels([f"{int(i.left)}–{int(i.right)}" for i in _b_plot.index],
                   rotation=30, ha="right")
ax.set_xlabel("Stand age class (yr)")
ax.set_ylabel("Plot-mean RGR (10$^{-3}$ yr$^{-1}$)")
ax.axvspan(3.5, len(_b_plot) - 0.5, color="0.92", zorder=0)
ax.text(3.5 + (len(_b_plot) - 4) / 2, ax.get_ylim()[1] * 0.97,
        "thinly sampled\n(39 plots ≥ 40 yr; the\n2- and 1-plot classes are not shown)",
        ha="center", va="top", fontsize=5.2, color="#666666")
ax.set_title("(a)  Age–growth decline with per-decade sample sizes", loc="left")

ax = fig.add_subplot(gs[1])
ax.hist(pt["stand_age"], bins=np.arange(0, 85, 5), color="#b35806",
        edgecolor="white", linewidth=0.3)
ax.set_xlabel("Stand age (yr)")
ax.set_ylabel("Number of plots")
ax.axvline(pt["stand_age"].median(), color="black", lw=0.8, ls="--")
ax.text(pt["stand_age"].median() + 1, ax.get_ylim()[1] * 0.9,
        f"median = {pt['stand_age'].median():.0f} yr", fontsize=5.8)
ax.set_title(f"(b)  Stand-age distribution (n = {pt['stand_age'].notna().sum():,} plots)", loc="left")
save(fig, "SF1_age_decay")

# ============================== SF2 密度依赖 ==============================
# P0-D-a: 改用嵌套口径 OOF SHAP 矩阵（26_*），与正文 Fig. 5 同源
with open(os.path.join(C.OUT, "26_shap_matrix.pkl"), "rb") as _f:
    import pickle as _pkl
    shap_mat = _pkl.load(_f)["shap"]
assert (shap_mat.index == tab.index).all(), "SHAP matrix rows must align with model table"
den = pd.DataFrame({"value": tab["stand_density"].values,
                    "shap": shap_mat["stand_density"].values * 1000,
                    "fg": tab["fg_conifer"].values}).dropna()

fig, ax = plt.subplots(figsize=(SINGLE, 62 * MM))
sub = den.sample(min(20000, len(den)), random_state=C.SEED)
ax.scatter(sub.loc[sub.fg == 0, "value"], sub.loc[sub.fg == 0, "shap"],
           s=1.2, alpha=0.10, color="#b2182b", edgecolors="none", rasterized=True)
ax.scatter(sub.loc[sub.fg == 1, "value"], sub.loc[sub.fg == 1, "shap"],
           s=1.2, alpha=0.10, color="#2166ac", edgecolors="none", rasterized=True)
binned = den.groupby(pd.qcut(den["value"], 30, duplicates="drop"), observed=True)["shap"].mean()
bx = den.groupby(pd.qcut(den["value"], 30, duplicates="drop"), observed=True)["value"].mean()
ax.plot(bx, binned, color="black", lw=1.3)
ax.axhline(0, color="0.6", lw=0.5, ls=":")
ax.set_xlabel("Stand density (trees ha$^{-1}$)")
ax.set_ylabel("SHAP value for stand density\n(10$^{-3}$ yr$^{-1}$)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ms=2, ls="", color="#b2182b", label="Broadleaf"),
                   Line2D([], [], marker="o", ms=2, ls="", color="#2166ac", label="Conifer"),
                   Line2D([], [], color="black", lw=1.3, label="30-bin mean")],
          loc="upper right", handletextpad=0.2, borderaxespad=0.2)
span = binned.max() - binned.min()
ax.set_title(f"Stand-density main effect (bin-mean span $\\approx$ {span:.1f} $\\times$ 10$^{{-3}}$ yr$^{{-1}}$)", loc="left")
save(fig, "SF2_density")

# ============================== SF3 ΔD 变量级重要性 ==============================
# P0-D-a: 改用嵌套口径 ΔD 伴随模型（29_*）
dd_all = pd.read_csv(os.path.join(C.OUT, "29_deltad_nested.csv"))
from scipy.stats import spearmanr
# ρ 在全 38 个特征上计算（含 2 个零归因的缺失指示变量），图上只画份额非零的变量
rho = spearmanr(dd_all["share_RGR"], dd_all["share_deltaD"]).statistic
dd = dd_all[dd_all["share_RGR"] > 0]

fig = plt.figure(figsize=(DOUBLE, 62 * MM))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.30)
ax = fig.add_subplot(gs[0])
lim = [3e-6, 1.0]
for g, sub in dd.groupby("group"):
    ax.scatter(sub["share_RGR"], sub["share_deltaD"], s=8, alpha=0.85,
               color=GROUP_COLORS[g], label=GROUP_LABEL[g], edgecolors="none")
ax.plot(lim, lim, color="0.5", lw=0.7, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lim); ax.set_ylim(lim)
for j, (_, r) in enumerate(dd.nlargest(5, "share_RGR").iterrows()):
    # 逐点指定偏移：top-5 的若干点在 (share_RGR, share_deltaD) 平面上彼此靠近，
    # 统一偏移会让标注互相压盖（实测 "Origin (natural)" 与 "Conifer proportion" 重叠）
    _off = {"stand_age": (6, 9, "left"), "lnD0": (-6, -10, "right"),
            "conifer_prop": (-6, 8, "right"), "bio12": (7, 1, "left"),
            "origin_natural": (-6, -11, "right")}
    _dx, _dy, _ha = _off.get(r["var"], (5, 4, "left"))
    ax.annotate(NICE.get(r["var"], r["var"]), (r["share_RGR"], r["share_deltaD"]),
                textcoords="offset points", xytext=(_dx, _dy), ha=_ha, fontsize=5.6,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
ax.set_xlabel("Variable share of attribution, RGR model")
ax.set_ylabel("Variable share of attribution, ΔD model")
ax.text(0.04, 0.93, f"Spearman ρ = {rho:.3f}\n(all {len(dd_all)} features; {len(dd)} plotted)",
        transform=ax.transAxes, fontsize=6.2, va="top")
ax.legend(loc="lower right", handletextpad=0.2)
ax.set_title("(a)  Variable-level shares: RGR vs ΔD response", loc="left")

ax = fig.add_subplot(gs[1])
fam = dd.groupby("group")[["share_RGR", "share_deltaD"]].sum().sort_values("share_RGR", ascending=False)
xpos = np.arange(len(fam))
ax.bar(xpos - 0.19, fam["share_RGR"] * 100, width=0.36, color="#4393c3", label="RGR model")
ax.bar(xpos + 0.19, fam["share_deltaD"] * 100, width=0.36, color="#f4a582", label="ΔD model")
for x, v in zip(xpos, fam["share_RGR"] * 100):
    ax.annotate(f"{v:.1f}", (x - 0.19, v + 0.7), ha="center", va="bottom", fontsize=5.0)
for x, v in zip(xpos, fam["share_deltaD"] * 100):
    ax.annotate(f"{v:.1f}", (x + 0.19, v + 2.6), ha="center", va="bottom", fontsize=5.0)
ax.set_xticks(xpos)
ax.set_xticklabels([GROUP_LABEL[g] for g in fam.index], rotation=20, ha="right")
ax.set_ylabel("Family share of total attribution (%)")
ax.legend(loc="upper right")
ax.set_ylim(0, fam.max().max() * 100 * 1.18)
ax.set_title("(b)  Family shares under the two responses", loc="left")
save(fig, "SF4_deltad")

# ============================== SF4 SHAP 跨折形状稳定性 ==============================
curves = pd.read_pickle(os.path.join(C.OUT, "33_shape_curves.pkl"))
stab = pd.read_csv(os.path.join(C.OUT, "33_shape_stability.csv"))
order = ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]

fig = plt.figure(figsize=(DOUBLE, 100 * MM))
gs = gridspec.GridSpec(2, 3, hspace=0.52, wspace=0.34)
for i, v in enumerate(order):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    fc = curves[v]
    # 各折分箱网格不同：在所有折共同覆盖的 x 范围内插值到公共网格后再平均
    lo = max(cur["x"].min() for cur in fc.values())
    hi = min(cur["x"].max() for cur in fc.values())
    grid = np.linspace(lo, hi, 50)
    interp = []
    for f, cur in fc.items():
        ax.plot(cur["x"], cur["y"] * 1000, lw=0.7, alpha=0.45, color="#8c510a")
        interp.append(np.interp(grid, cur["x"], cur["y"]))
    ax.plot(grid, np.mean(interp, axis=0) * 1000, lw=1.4, color="black")
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    r = stab.loc[stab["var"] == v, "mean_pairwise_r"].iloc[0]
    rmin = stab.loc[stab["var"] == v, "min_pairwise_r"].iloc[0]
    ax.set_title(f"{NICE.get(v, v)}", loc="left")
    # origin_natural 为二值变量、各折无共同支撑 → 相关系数为 NaN，图面记 n/a（与 SI 表述一致）
    _fr = lambda x: "n/a" if pd.isna(x) else f"{x:.2f}"
    ax.text(0.03, 0.05, f"mean r = {_fr(r)}\nmin r = {_fr(rmin)}",
            transform=ax.transAxes, fontsize=5.8, va="bottom",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    if i // 3 == 1:
        ax.set_xlabel(NICE.get(v, v).split(" (")[0])
    if i % 3 == 0:
        ax.set_ylabel("SHAP (10$^{-3}$ yr$^{-1}$)")
save(fig, "SF3_shape_stability")

print("DONE P1-12 SI figures")
