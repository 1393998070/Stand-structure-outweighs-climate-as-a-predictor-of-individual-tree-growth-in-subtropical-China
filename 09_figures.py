# -*- coding: utf-8 -*-
"""09 主图生成（Nature Communications 风格，英文，300dpi PNG + PDF）。
Fig1 数据概况 / Fig2 基础格局 / Fig3 三重CV / Fig4 SHAP归因 / Fig5 效应形态 / Fig6 省域制图
输出: 05_图表/Fig1..Fig6 (png+pdf) + fig 数据快照
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
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

C_CONF = "#2166ac"   # conifer
C_BRD = "#b2182b"    # broadleaf
C_SEQ = LinearSegmentedColormap.from_list("rg", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"])
GROUP_COLORS = {"tree": "#7f3b08", "stand": "#b35806", "climate": "#2166ac",
                "soil": "#1a9641", "terrain": "#8c510a"}
GROUP_LABEL = {"tree": "Tree attributes", "stand": "Stand structure",
               "climate": "Climate", "soil": "Soil", "terrain": "Terrain & hydrology"}
NICE = {"lnD0": "ln(initial DBH)", "fg_conifer": "Functional group (conifer)",
        "stand_age": "Stand age (yr)", "stand_density": "Stand density",
        "conifer_prop": "Conifer proportion", "origin_natural": "Origin (natural)",
        "forest_type": "Forest type", "bio1": "MAT (bio1)",
        "bio2": "Mean diurnal range (bio2)",
        "bio4": "Temperature seasonality (bio4)", "bio12": "MAP (bio12)",
        "bio15": "Precip. seasonality (bio15)", "bio18": "Warm-quarter precip. (bio18)",
        "bio19": "Coldest-quarter precip. (bio19)", "soil_SOC": "Soil OC",
        "soil_TN": "Soil TN", "soil_TP": "Soil TP", "soil_Tk": "Soil K",
        "soil_pH": "Soil pH", "soil_BD": "Soil bulk density", "soil_depth": "Soil depth",
        "dem": "Elevation", "slope": "Slope", "aspect_cos": "Cosine aspect",
        "aspect_sin": "Sine aspect", "curvature_total": "Total curvature",
        "dist_stream": "Distance to stream", "relief": "Relief", "flow_length": "Flow length",
        "heat_load": "Heat load", "SPI": "SPI", "TPI": "TPI", "TSI": "TSI", "TWI": "TWI"}

def nice(v):
    return NICE.get(v, v)

# Fig 4b 用的紧凑标签（长标签会侵入左侧面板）
NICE_SHORT = {
    "stand_age": "Stand age", "lnD0": "ln(DBH)", "conifer_prop": "Conifer prop.",
    "bio12": "MAP (bio12)", "origin_natural": "Origin (nat.)", "bio15": "Precip. seas.",
    "bio4": "Temp. seas.", "dem": "Elevation", "bio2": "Diurnal range",
    "relief": "Relief", "stand_density": "Stand density", "soil_SOC": "Soil OC",
    "TPI": "TPI", "aspect_sin": "Aspect (sin)", "bio18": "Warm-qtr precip.",
    "bio1": "MAT (bio1)", "soil_pH": "Soil pH", "soil_Tk": "Soil K",
}

# 图面渲染标签导出：把每张图上【真实绘制】的文字收集起来，供 34_threeway_check.py
# 做"图面 ↔ 数据源 ↔ 图注"三方核验（Round-1 的失效模式即图面与正文不一致却无自动检查）。
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
    with open(os.path.join(C.OUT, "fig_labels_main.json"), "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=1)
    plt.close(fig)
    print(f"saved {name} ({len(texts)} labels)", flush=True)


def add_cbar(fig, ax, mappable, title=None, width=0.009, gap=0.012):
    """色标精确对齐父坐标轴的【最终绘制高度】。

    plt.colorbar(ax=ax) 用坐标轴的【槽位】高度建色标，而 imshow(aspect='equal')
    的宽高调整发生在绘制时，二者不同步 → 色标会比地图高出一截（Fig.1a 实测高 2.2 倍），
    且色标标题会越过轴框压到绘图区（Fig.1d）。此处先强制绘制，再按最终位置建色标轴，
    并回读一次以确认二者确实等高（不等则在控制台报警，防止静默失准）。
    """
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + gap, pos.y0, width, pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=6)
    if title:
        cb.ax.set_title(title, fontsize=5.6, pad=3)
    fig.canvas.draw()
    a2, c2 = ax.get_position(), cax.get_position()
    if max(abs(a2.y0 - c2.y0), abs(a2.y1 - c2.y1)) > 1e-3:
        print(f"  [WARN] colorbar/axes 高度失配: axes={a2.y0:.3f}-{a2.y1:.3f} "
              f"cbar={c2.y0:.3f}-{c2.y1:.3f}", flush=True)
    return cb

def map_axes(ax, extent, lon_step=2, lat_step=1):
    """地图风格坐标轴（Fig1a / Fig6a-c）：
    · °E/°N 整数刻度替代 "Longitude (°E)" 式轴标题——地图惯例是单位内嵌刻度；
    · 四边外向刻度线（上/右仅刻度线无标签），呈现地图图廓；
    · set_aspect(1/cos(中纬))：等比绘制的 1° 经度被拉伸，广东中纬 22.9° 下
      E-W 方向偏大约 9%，按投影长度校正后省域形状不失真。
    """
    lat0 = 0.5 * (extent[2] + extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat0)))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(lon_step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(lat_step))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}°E"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}°N"))
    ax.tick_params(direction="out", top=True, right=True,
                   labeltop=False, labelright=False)


tab, vd, folds = C.load_model_data()
plot_tab = (tab.groupby("plot_id")
            .agg(lon=("lon", "first"), lat=("lat", "first"),
                 plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"),
                 origin_natural=("origin_natural", "first"),
                 conifer_prop=("conifer_prop", "first"), fg_conifer=("fg_conifer", "mean"),
                 bio1=("bio1", "first"), bio12=("bio12", "first"))
            .reset_index())
plot_tab["fg_conifer"] = (plot_tab["fg_conifer"] > 0.5).astype(int)

# 坐标粗化：国家清查规定样地精确坐标保密，公开图面只给 0.05° 格心
# （与 37_source_data.py 的 Fig1a 底表同步；正文 Methods/图 1 注有对应说明）
_COARSE = 0.05
plot_tab["lon"] = (np.floor(plot_tab["lon"] / _COARSE) + 0.5) * _COARSE
plot_tab["lat"] = (np.floor(plot_tab["lat"] / _COARSE) + 0.5) * _COARSE

# ============================== Fig 1 数据概况 ==============================
fig = plt.figure(figsize=(DOUBLE, 50 * MM))
gs = gridspec.GridSpec(1, 4, width_ratios=[1.9, 1, 1, 1], wspace=1.05)
ax = fig.add_subplot(gs[0])
mp = np.load(os.path.join(C.OUT, "map", "dual_map_arrays.npz"), allow_pickle=True)
tr = mp["transform"]
# transform 数组为 (a,b,c,d,e,f): x = a*col + b*row + c; y = d*col + e*row + f
a, b, c_, d, e, f_ = tr
fprop = mp["forest_frac"]
extent = [c_, c_ + a * fprop.shape[1], f_ + e * fprop.shape[0], f_]
# 不再叠绿色林地底纹：alpha 0.65 的 Greens 在密集样点下反而压低样点可辨度
# （用户反馈），且"样点是否落在林地上"在正文用覆盖统计表述而非图面。仅保留
# 样点 + 地图图廓（°E/°N 刻度、四边刻度线、经纬等比），坐标范围取全省栅格外包框。
sc = ax.scatter(plot_tab["lon"], plot_tab["lat"], c=plot_tab["plot_RGR"], s=3.2,
                cmap=C_SEQ, vmin=0.02, vmax=0.085, linewidths=0)
add_cbar(fig, ax, sc, title="RGR (yr$^{-1}$)")
map_axes(ax, extent)
ax.tick_params(labelsize=6)
ax.set_anchor("C")
ax.set_title(f"(a) {plot_tab['plot_id'].nunique():,} CFI plots", loc="left", fontsize=7)

ax = fig.add_subplot(gs[1])
for fg, col, lab in [(0, C_BRD, "Broadleaf"), (1, C_CONF, "Conifer")]:
    sub = tab[tab["fg_conifer"] == fg]["RGR"]
    ax.hist(sub, bins=np.linspace(0, 0.2, 60), density=True, alpha=0.5, color=col, label=lab)
ax.set_xlabel("RGR (yr$^{-1}$)")
ax.set_ylabel("Probability density (yr)")
ax.tick_params(labelsize=6)
ax.legend(fontsize=5.0, loc="upper right", handlelength=1.0, handletextpad=0.35,
          labelspacing=0.25, borderaxespad=0.25)
ax.set_title("(b) RGR distribution", loc="left", fontsize=7)

ax = fig.add_subplot(gs[2])
aged = plot_tab[plot_tab["stand_age"].notna()]
ax.hist(aged["stand_age"], bins=np.arange(0, 105, 5), color="#8073ac", alpha=0.85)
ax.set_xlabel("Stand age (yr)")
ax.set_ylabel("Number of plots")
ax.tick_params(labelsize=6)
ax.set_title("(c) Stand age structure", loc="left", fontsize=7)

ax = fig.add_subplot(gs[3])
sc = ax.scatter(plot_tab["bio1"] / 10.0, plot_tab["bio12"], c=plot_tab["plot_RGR"], s=3.2,
                cmap=C_SEQ, vmin=0.02, vmax=0.085, linewidths=0)
ax.set_xlabel("MAT (°C)")
ax.set_ylabel("MAP (mm)")
# (d) 色标位于图最右侧：水平标题会与面板标题 "(d) Climate space" 同行相抵并被画布右缘裁切，
# 改为色标右侧竖排标签；(a) 的色标在面板间隙内，保留水平标题不动
cb_d = add_cbar(fig, ax, sc)
cb_d.set_label("RGR (yr$^{-1}$)", fontsize=5.6, labelpad=1)
ax.tick_params(labelsize=6)
ax.set_title("(d) Climate space", loc="left", fontsize=7)
save(fig, "Fig1_overview")

# ============================== Fig 2 基础格局 ==============================
fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 42 * MM))
ax = axes[0]
# 仅绘制两组各自 n ≥ 20 的共同支撑区间：尾部（lnD0 > 6）单组仅 1–2 株，
# 其分箱均值纯属噪声，绘出会误导读者。
bins = pd.cut(tab["lnD0"], 25)
_cnt = tab.groupby([bins, tab["fg_conifer"]], observed=True)["RGR"].size().unstack(fill_value=0)
_ok_bins = _cnt[(_cnt[0] >= 20) & (_cnt[1] >= 20)].index   # 两组共同支撑
for fg, col, lab in [(0, C_BRD, "Broadleaf"), (1, C_CONF, "Conifer")]:
    sub = (tab[tab["fg_conifer"] == fg].groupby(bins, observed=True)["RGR"]
           .agg(["mean", "sem"]).reindex(_ok_bins))
    mid = np.array([iv.mid for iv in sub.index])
    ax.plot(mid, sub["mean"], color=col, label=lab, lw=1.2)
    ax.fill_between(mid, sub["mean"] - 1.96 * sub["sem"], sub["mean"] + 1.96 * sub["sem"],
                    color=col, alpha=0.18, lw=0)
ax.set_xlim(mid.min(), mid.max())
ax.set_xlabel("ln(initial DBH)")
ax.set_ylabel("Mean RGR (yr$^{-1}$)")
ax.legend(loc="upper right")
ax.set_title("(a) Size dependence", loc="left")

ax = axes[1]
aged = plot_tab.dropna(subset=["stand_age"])
ax.scatter(aged["stand_age"], aged["plot_RGR"], s=2.5, color="#404040", alpha=0.35, linewidths=0)
b = pd.cut(aged["stand_age"], np.arange(0, 105, 10), right=False)  # 左闭右开：与 20_sf1_age_bins.csv / 图注一致
gm = aged.groupby(b, observed=True).agg(m=("plot_RGR", "mean"), s=("plot_RGR", "sem"))
mid = np.array([iv.mid for iv in gm.index])
ax.plot(mid, gm["m"], color=C_CONF, lw=1.4)
ax.fill_between(mid, gm["m"] - 1.96 * gm["s"], gm["m"] + 1.96 * gm["s"], color=C_CONF, alpha=0.2, lw=0)
ax.set_xlabel("Stand age (yr)")
ax.set_ylabel("Plot mean RGR (yr$^{-1}$)")
ax.set_title("(b) Age decline", loc="left")
# 说明：分箱计数只在图注与 SF1 中给出（左闭右开口径）；此处不自标 n，避免图面拥挤

ax = axes[2]
data = [plot_tab.loc[plot_tab["origin_natural"] == 0, "plot_RGR"],
        plot_tab.loc[plot_tab["origin_natural"] == 1, "plot_RGR"]]
bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=0.9))
for patch, col in zip(bp["boxes"], ["#7fbf7b", "#af8dc3"]):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax.set_xticks([1, 2], [f"Planted\n(n={len(data[0])})", f"Natural\n(n={len(data[1])})"])
ax.set_ylabel("Plot mean RGR (yr$^{-1}$)")
ax.set_title("(c) Origin", loc="left")

ax = axes[3]
data = [plot_tab.loc[plot_tab["fg_conifer"] == 0, "plot_RGR"],
        plot_tab.loc[plot_tab["fg_conifer"] == 1, "plot_RGR"]]
bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=0.9))
for patch, col in zip(bp["boxes"], [C_BRD, C_CONF]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax.set_xticks([1, 2], [f"Broadleaf\n(n={len(data[0])})", f"Conifer\n(n={len(data[1])})"])
ax.set_ylabel("Plot mean RGR (yr$^{-1}$)")
ax.set_title("(d) Functional group", loc="left")
fig.subplots_adjust(wspace=0.42)
save(fig, "Fig2_patterns")

# ============================== Fig 3 三重 CV ==============================
# P0-D-a: 改用【嵌套选参】口径的结果（诚实，无选参/早停乐观偏倚）
import pickle as _pkl
nres = pd.read_csv(os.path.join(C.OUT, "22_nested_cv_results.csv"))
nres = nres[nres["model"] == "XGBoost-nested"].set_index("cv")
with open(os.path.join(C.OUT, "22_nested_oof.pkl"), "rb") as _f:
    _blob = _pkl.load(_f)
_oof_sp = _blob["oof"]["spatial"]
df3 = tab[["plot_id", "RGR"]].copy()
df3["pred"] = _oof_sp
fig = plt.figure(figsize=(DOUBLE, 48 * MM))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42)
ax = fig.add_subplot(gs[0])
schemes = ["random", "group", "spatial"]
labels = ["Random\n(tree-level)", "Grouped\n(by plot)", "Spatial block\n(0.5°)"]
x = np.arange(3); w = 0.36
r2s = [float(nres.loc[s, "R2"]) for s in schemes]
p2s = [float(nres.loc[s, "plot_R2"]) for s in schemes]
ax.bar(x - w / 2, r2s, w, color="#1b7837", alpha=0.85, label="Tree-level")
ax.bar(x + w / 2, p2s, w, color="#1b7837", alpha=0.35, label="Plot-level")
for xi, v in zip(x - w / 2, r2s):
    ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.8)
for xi, v in zip(x + w / 2, p2s):
    ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=5.8)
ax.set_xticks(x, labels)
ax.set_ylabel("R²")
ax.set_ylim(0, 0.9)
ax.legend(ncol=1, loc="upper right")
ax.set_title("(a) Cross-validation designs (nested)", loc="left")

ax = fig.add_subplot(gs[1])
sub = df3.sample(min(20000, len(df3)), random_state=1)
ax.hexbin(sub["pred"], sub["RGR"], gridsize=60, bins="log",
          cmap="Greys", mincnt=1, linewidths=0)
lims = [0, 0.22]
ax.plot(lims, lims, color=C_CONF, lw=0.8, ls="--")
r2 = float(nres.loc["spatial", "R2"])
ax.text(0.05, 0.92, f"$R^2$ = {r2:.2f} (spatial block)", transform=ax.transAxes, fontsize=6.5)
ax.set_xlim(0, 0.22); ax.set_ylim(0, 0.22)
ax.set_xlabel("Predicted RGR (yr$^{-1}$)")
ax.set_ylabel("Observed RGR (yr$^{-1}$)")
ax.set_title("(b) Tree-level, spatial CV", loc="left")

ax = fig.add_subplot(gs[2])
pm = df3.groupby("plot_id")[["RGR", "pred"]].mean()
ax.scatter(pm["pred"], pm["RGR"], s=3.5, color="#4575b4", alpha=0.5, linewidths=0)
ax.plot(lims, lims, color=C_CONF, lw=0.8, ls="--")
pr2 = float(nres.loc["spatial", "plot_R2"])
ax.text(0.05, 0.92, f"$R^2$ = {pr2:.2f} (plot mean)", transform=ax.transAxes, fontsize=6.5)
ax.set_xlim(0, 0.18); ax.set_ylim(0, 0.18)
ax.set_xlabel("Predicted plot mean RGR (yr$^{-1}$)")
ax.set_ylabel("Observed plot mean RGR")
ax.set_title("(c) Plot-level, spatial CV", loc="left")
save(fig, "Fig3_cv")

# ============================== Fig 4 SHAP 归因 ==============================
imp = pd.read_csv(os.path.join(C.OUT, "26_shap_importance.csv"))
gsh = pd.read_csv(os.path.join(C.OUT, "26_group_shares.csv"))
abl = pd.read_csv(os.path.join(C.OUT, "28_ablation_nested.csv"))
nz = pd.read_csv(os.path.join(C.OUT, "27_family_share_normalized.csv"))
dom = pd.read_csv(os.path.join(C.OUT, "28_plot_dominance_nested.csv"))
fig = plt.figure(figsize=(DOUBLE, 62 * MM))
gs = gridspec.GridSpec(1, 6, width_ratios=[1, 0.85, 1.25, 0.16, 0.9, 1.05], wspace=0.30)
ax = fig.add_subplot(gs[0])
order = ["tree", "stand", "climate", "soil", "terrain"]
sh = [gsh.set_index("group").loc[g, "share"] for g in order]
ab = [float(abl[abl["removed"] == g]["d_tree_R2"].iloc[0]) if g in set(abl["removed"]) else 0 for g in order]
x = np.arange(5)
ax.bar(x, sh, 0.6, color=[GROUP_COLORS[g] for g in order])
for xi, v, g in zip(x, sh, order):
    # 柱顶上方标注 + 白底衬：消融折线（右轴，量程已放宽至下半区）不与其交叠。
    # soil 柱顶正上方恰是折线谷底与菱形点，默认偏移会被穿过，单独收小偏移避开。
    _dy = 0.004 if g == "soil" else 0.013
    ax.text(xi, v + _dy, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=5.8,
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.9))
ax2 = ax.twinx()
ax2.plot(x, ab, "D-", color="black", ms=3, lw=0.9)
ax2.set_ylim(-0.05, 0.17)     # 右轴量程放宽：折线整体压在面板下半区
ax2.set_ylabel("ΔR²", fontsize=5.5)
ax2.tick_params(labelsize=5)
ax.set_ylim(0, 0.465)         # 顶部留白：避免 "39.4%" 压住上边框
ax.set_xticks(x, ["Tree", "Stand", "Climate", "Soil", "Terrain"], rotation=30, ha="right",
              fontsize=6)
ax.set_ylabel("SHAP share of total |SHAP|", fontsize=6)
ax.tick_params(axis="y", labelsize=6)
ax.set_title("(a) Attribution by group", loc="left", fontsize=7)

ax = fig.add_subplot(gs[2])
top = imp.head(15).iloc[::-1]
ax.barh(np.arange(len(top)), top["mean_abs_shap"], 0.65,
        color=[GROUP_COLORS[g] for g in top["group"]])
xerr = np.vstack([np.maximum(top["mean_abs_shap"] - top["ci_lo"], 0),
                  np.maximum(top["ci_hi"] - top["mean_abs_shap"], 0)])
ax.errorbar(top["mean_abs_shap"], np.arange(len(top)), xerr=xerr,
            fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.2)
ax.set_yticks(np.arange(len(top)),
              [NICE_SHORT.get(v, nice(v)) for v in top["var"]], fontsize=5.6)
ax.set_xlabel("Mean |SHAP| (RGR yr$^{-1}$)", fontsize=6)
ax.tick_params(axis="x", labelsize=6)
ax.set_title("(b) Variable importance (top 15)", loc="left", fontsize=7)

# (c) 份额 vs 按族内变量数归一后的份额 —— 说明 family share 未归一，不可作生态排序
ax = fig.add_subplot(gs[4])
nzz = nz.set_index("group").loc[order]
xx = np.arange(5); w2 = 0.38
ax.bar(xx - w2 / 2, nzz["share"], w2, color=[GROUP_COLORS[g] for g in order], alpha=0.95)
ax.bar(xx + w2 / 2, nzz["share_per_var"], w2, color=[GROUP_COLORS[g] for g in order], alpha=0.42)
ax.set_xticks(xx, ["Tree", "Stand", "Climate", "Soil", "Terrain"], rotation=30, ha="right",
              fontsize=6)
ax.set_ylabel("Share of total |SHAP|", fontsize=6)
ax.tick_params(axis="y", labelsize=6)
ax.set_ylim(0, 0.46)
ax.set_yticks([0, 0.2, 0.4])
ax.set_title("(c) Share vs per-variable share", loc="left", fontsize=7)
ax.text(0.98, 0.92, "solid: family share\nfaded: ÷ no. of variables",
        transform=ax.transAxes, fontsize=5.2, va="top", ha="right")

ax = fig.add_subplot(gs[5])
_dom = dom[dom["n_plots_dominant"] > 0]
ax.pie(_dom["n_plots_dominant"], colors=[GROUP_COLORS[g] for g in _dom["group"]],
       autopct=lambda p: f"{p:.0f}%" if p > 4 else "", startangle=90,
       center=(0, 0), radius=1.0,
       wedgeprops=dict(linewidth=0.5, edgecolor="white"),
       textprops=dict(fontsize=6))
ax.legend([Line2D([0], [0], color=GROUP_COLORS[g], lw=4) for g in _dom["group"]],
          [f"{GROUP_LABEL[g]} ({int(n)})" for g, n in zip(_dom["group"], _dom["n_plots_dominant"])],
          loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5.6)
ax.set_position([ax.get_position().x0, 0.24, ax.get_position().width * 0.80,
                 ax.get_position().height * 0.62])   # 纵向居中并放大，消除面板下方空白
ax.set_title("(d) Dominant group per plot", loc="left", fontsize=7)
save(fig, "Fig4_shap")

# ============================== Fig 5 效应形态 ==============================
_S = pd.read_pickle(os.path.join(C.OUT, "26_shap_matrix.pkl"))
Sm = _S["shap"].reindex(tab.index)
_imp5 = pd.read_csv(os.path.join(C.OUT, "26_shap_importance.csv"))
# 主效应面板固定为六个变量，与 SI 形状稳定性图（20_si_figures.py 的 order）、
# Fig.5 图注、正文三者共用同一集合。不用 head(6)：stand_density 的重要性
# （0.001518）与 bio12（0.001467）近乎并列，自动取前 6 会把林分密度拉进主图、
# 把 bio15 挤出，造成主效应图与形状稳定性图/图注/正文的变量集不一致；而密度
# 主效应已由 Supplementary Fig. 2 专门承担，重复展示亦无必要。
_F5_VARS = ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]
top_vars = _F5_VARS
dep = {v: pd.DataFrame({"value": tab[v].values, "shap": Sm[v].values,
                        "fg_conifer": tab["fg_conifer"].values}) for v in top_vars}
inter = pd.read_pickle(os.path.join(C.OUT, "26_inter_dep_data.pkl"))
_lc = pd.read_pickle(os.path.join(C.OUT, "29_lnd0_curves.pkl"))
fig = plt.figure(figsize=(DOUBLE, 100 * MM))
gs = gridspec.GridSpec(2, 4, hspace=0.62, wspace=0.78)
for i, v in enumerate(top_vars):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    d = dep[v]
    ax.scatter(d.loc[d["fg_conifer"] == 0, "value"], d.loc[d["fg_conifer"] == 0, "shap"],
               s=1.2, color=C_BRD, alpha=0.15, linewidths=0, rasterized=True)
    ax.scatter(d.loc[d["fg_conifer"] == 1, "value"], d.loc[d["fg_conifer"] == 1, "shap"],
               s=1.2, color=C_CONF, alpha=0.15, linewidths=0, rasterized=True)
    # binned mean curve
    q = pd.qcut(d["value"], 30, duplicates="drop")
    gm = d.groupby(q, observed=True)["shap"].mean()
    mid = np.array([iv.mid for iv in gm.index])
    ax.plot(mid, gm, color="black", lw=1.1)
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ylo, yhi = np.percentile(d["shap"], [0.5, 99.5])
    ax.set_ylim(ylo, yhi)
    if v == "stand_age":   # (a) 样本支撑 rug：沿轴展示样地林龄分布
        ax.plot(d["value"].values, np.full(len(d), ylo), marker="|", ls="none",
                color="#444444", ms=1.4, alpha=0.06, rasterized=True)
    if v == "lnD0":        # (b) ΔD 伴随模型的 lnD₀ 曲线（独立纵轴，单位不同）
        ax2 = ax.twinx()
        dd = _lc["deltaD"]
        ax2.plot(dd["v"].values, dd["s"].values, color="#2166ac", lw=1.0, ls="--")
        ax2.set_ylabel("ΔD\n(cm/5 yr)", fontsize=5.2, color="#2166ac")
        ax2.tick_params(labelsize=5.2, colors="#2166ac")
    ax.set_xlabel(nice(v) if v != "bio12" else "MAP (mm$\\times$10, bio12)", fontsize=6.5)
    ax.tick_params(labelsize=6)
    if i % 4 == 0:
        ax.set_ylabel("SHAP (RGR yr$^{-1}$)", fontsize=6.5)
    ax.set_title(f"({chr(97+i)}) {nice(v)}", loc="left", fontsize=7)
# 交互面板：(g) 林龄 × 功能群；(h) 密度 × 初始直径。
# 原设计用原始散点直接画交互值，但 (g) 的自变量是二值、(h) 的交互量级仅 1e-4，
# 散点图分别渲染成"空面板"与"无结构噪声"，而分箱后结构其实非常清晰
# （见 26_interactions.csv 与 26_inter_dep_data.pkl），故改为分箱汇总呈现。
_inter_keys = [k for k in ("stand_age__fg_conifer", "stand_density__lnD0") if k in inter]
# (g) 林龄 × 功能群：分箱均值 ± 95% CI 两条线 → 直接展示"两组基本平行"
if "stand_age__fg_conifer" in inter:
    ax = fig.add_subplot(gs[1, 2])
    d = inter["stand_age__fg_conifer"].reset_index(drop=True)   # 逐折拼接 → 索引重复
    ab_ = pd.cut(d["v1"], [0, 5, 10, 15, 20, 30, 40, 50, 90])
    for fg, col, lab in [(0, C_BRD, "Broadleaf"), (1, C_CONF, "Conifer")]:
        s = d[d["v2"] == fg].groupby(ab_, observed=True)["inter"].agg(["mean", "sem"]).dropna()
        mid = np.array([iv.mid for iv in s.index])
        ax.plot(mid, s["mean"] * 1000, color=col, lw=1.1, marker="o", ms=2.2, label=lab)
        ax.fill_between(mid, (s["mean"] - 1.96 * s["sem"]) * 1000,
                        (s["mean"] + 1.96 * s["sem"]) * 1000, color=col, alpha=0.18, lw=0)
    ax.axhline(0, color="#999999", lw=0.5, ls=":")
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel("Stand age (yr)", fontsize=6.5)
    ax.set_ylabel("Mean interaction\n(10$^{-3}$ yr$^{-1}$)", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.4, loc="lower right", handletextpad=0.3)
    ax.set_title("(g) Stand age × group", loc="left", fontsize=7)
    _m = inter["stand_age__fg_conifer"]["inter"].abs().mean()
    ax.text(0.03, 0.97, f"mean |interaction| = {_m*1000:.2f} × 10$^{{-3}}$",
            transform=ax.transAxes, fontsize=5.2, color="#555555", va="top")
# (h) 密度 × 初始直径：4×4 分箱均值热图 → 直接展示大小树随密度的符号反转
if "stand_density__lnD0" in inter:
    ax = fig.add_subplot(gs[1, 3])
    d = inter["stand_density__lnD0"].reset_index(drop=True)     # 同上：索引重复
    hb_ = pd.qcut(d["v1"], 4, duplicates="drop")
    sb_ = pd.qcut(d["v2"], 4, duplicates="drop")
    piv = d.groupby([hb_, sb_], observed=True)["inter"].mean().unstack() * 1000
    _lim = float(np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-_lim, vmax=_lim,
                   aspect="auto", origin="lower")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=5)
    ax.set_xticks(range(piv.shape[1]), ["Q1\nsmall", "Q2", "Q3", "Q4\nlarge"], fontsize=5.4)
    ax.set_yticks(range(piv.shape[0]), ["Q1\nsparse", "Q2", "Q3", "Q4\ndense"], fontsize=5.4)
    ax.set_xlabel("Initial size quartile (ln D$_0$)", fontsize=6.5)
    ax.set_ylabel("Stand-density quartile", fontsize=6.5)
    # 侧标注而非顶部标题：色标窄，顶部标题会横向压到 (h) 的面板标题
    cb = add_cbar(fig, ax, im, gap=0.02)
    cb.set_label("Interaction (10$^{-3}$ yr$^{-1}$)", fontsize=5.6)
    ax.set_title("(h) Density × ln(DBH)", loc="left", fontsize=7)
save(fig, "Fig5_dependence")

# ============================== Fig 6 制图 ==============================
mp = np.load(os.path.join(C.OUT, "map", "dual_map_arrays.npz"), allow_pickle=True)
_tr = mp["transform"]
_a, _b, _c, _d, _e, _f = _tr
_mg = mp["mean_conifer"]
extent = [_c, _c + _a * _mg.shape[1], _f + _e * _mg.shape[0], _f]
attr = pd.read_csv(os.path.join(C.OUT, "23_dual_map_attribution.csv"))
dsum = C.load_json("23_dual_map_summary.json")
_r2 = {r["response"]: r for r in dsum["responses"]}
fig = plt.figure(figsize=(DOUBLE, 38 * MM))
# 三张地图放进左侧内嵌 GridSpec、(d) 独占右侧：旧版为防 (c) 色标刻度压 (d) 轴题
# 而把统一 wspace 放宽到 0.85，把地图槽位压到仅 ~20 mm 宽——经度刻度标签互相
# 粘连（110°E112°E…），且 58 mm 画布高度下半空置（地图 set_aspect 后按宽定高）。
# 现内嵌 wspace=0.48 只需容纳竖排色标标签；色标标题改 set_label 竖排，
# 避免窄色标上的水平标题与顶端刻度值相撞。
gs_out = gridspec.GridSpec(1, 2, width_ratios=[3, 1.0], wspace=0.45)
gs_map = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_out[0], wspace=0.48)
_CB = dict(fraction=0.06, pad=0.04, shrink=0.60, aspect=10)

ax = fig.add_subplot(gs_map[0])
im = ax.imshow(mp["mean_conifer"], extent=extent, origin="upper", cmap=C_SEQ,
               vmin=np.nanpercentile(mp["mean_conifer"], 2),
               vmax=np.nanpercentile(mp["mean_conifer"], 98), interpolation="nearest")
cb = plt.colorbar(im, ax=ax, **_CB)
cb.set_label("RGR (yr$^{-1}$)", fontsize=5.6, labelpad=1)
cb.ax.tick_params(labelsize=5.5)
map_axes(ax, extent)          # °E/°N 刻度 + 四边图廓 + 经纬等比（地图式坐标）
ax.tick_params(labelsize=5.5)
ax.set_title("(a) Conifer", loc="left", fontsize=7)
ax.text(0.02, 0.02, f"nested $R^2$ = {_r2['conifer']['R2']:.2f}",
        transform=ax.transAxes, fontsize=6)

ax = fig.add_subplot(gs_map[1])
im = ax.imshow(mp["mean_broadleaf"], extent=extent, origin="upper", cmap=C_SEQ,
               vmin=np.nanpercentile(mp["mean_broadleaf"], 2),
               vmax=np.nanpercentile(mp["mean_broadleaf"], 98), interpolation="nearest")
cb = plt.colorbar(im, ax=ax, **_CB)
cb.set_label("RGR (yr$^{-1}$)", fontsize=5.6, labelpad=1)
cb.ax.tick_params(labelsize=5.5)
map_axes(ax, extent)
ax.tick_params(labelsize=5.5, labelleft=False)   # 与 (a) 共用纬度，刻度线保留仅去标签
ax.set_title("(b) Broadleaf", loc="left", fontsize=7)
ax.text(0.02, 0.02, f"nested $R^2$ = {_r2['broadleaf']['R2']:.2f}",
        transform=ax.transAxes, fontsize=6)

ax = fig.add_subplot(gs_map[2])
_sdg = np.fmax(mp["sd_conifer"], mp["sd_broadleaf"])  # fmax: 全 NaN 处不告警
im = ax.imshow(_sdg, extent=extent, origin="upper", cmap="viridis",
               vmin=0, vmax=np.nanpercentile(_sdg, 98), interpolation="nearest")
cb = plt.colorbar(im, ax=ax, **_CB)
cb.set_label("SD (yr$^{-1}$)", fontsize=5.6, labelpad=1)
cb.ax.tick_params(labelsize=5.5)
map_axes(ax, extent)
ax.tick_params(labelsize=5.5, labelleft=False)
ax.set_title("(c) Disagreement", loc="left", fontsize=7)

ax = fig.add_subplot(gs_out[1])
order = ["stand", "climate", "soil", "terrain"]
xx = np.arange(4); ww = 0.38
for j, (tag, col, lab) in enumerate([("conifer", C_CONF, "Conifer"),
                                     ("broadleaf", C_BRD, "Broadleaf")]):
    vals = [float(attr[attr["response"] == tag][f"share_{g}"].iloc[0]) for g in order]
    ax.bar(xx + (j - 0.5) * ww, vals, ww, color=col, alpha=0.85, label=lab)
    # 柱顶标签垂直交错（zigzag）：1 位小数后标签 4 字符（"34.4"），宽于窄面板的
    # 柱中心距（~2.6 mm），Climate（34.4/34.0）与 Soil（13.2/12.3）两对几乎等高的
    # 相邻柱标签水平必叠 → 针叶贴柱顶、阔叶整体上移一个数字高，全部标签垂直分离
    _off = 0.006 if j == 0 else 0.040
    for xi, v in zip(xx + (j - 0.5) * ww, vals):
        ax.text(xi, v + _off, f"{v*100:.1f}", ha="center", va="baseline", fontsize=5.0)
ax.set_xticks(xx, ["Stand", "Climate", "Soil", "Terrain"], rotation=40, ha="right",
              fontsize=6)
ax.set_ylabel("Share (mean |SHAP|)", fontsize=6, labelpad=1)
ax.tick_params(axis="y", labelsize=6)
ax.set_ylim(0, 0.44)    # 阔叶 terrain 标签顶 ~0.42
# 图例移出轴外置顶右（与左侧面板标题同行）：面板仅 ~29 mm 宽，图例无论悬于
# 轴内何处都会横跨到 Climate 柱标签上方（probe 实测压盖 3.5 mm）；横排双列
# 宽 ~26 mm 又与标题相撞 → 单列竖排（宽 ~12 mm）右贴，与标题各行其道
ax.legend(fontsize=5.4, loc="lower right", bbox_to_anchor=(1.01, 1.01),
          handletextpad=0.25, borderaxespad=0.0)
ax.set_title("(d) Drivers", loc="left", fontsize=7)
save(fig, "Fig6_map")

print("ALL FIGURES DONE")
