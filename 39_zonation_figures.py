# -*- coding: utf-8 -*-
"""39 广东省林地地带性分区图件（幻灯片 16:9，中文，300 dpi PNG + PDF）。

图 1 环境梯度结构      —— 全变量 PCA：碎石图、载荷双标图、载荷热图、梯度与纬度/海拔
图 2 分区数判据        —— 轮廓系数 / CH / DB / GMM BIC / Gap / 自助稳定性 / 空间连贯率
图 3 省域区划图        —— 一级区、细化口径、与纯气候对照的差异、纬度带组成
图 4 分区环境特征      —— 气候 / 土壤 / 地形水文三族标准化异常热图 + 关键因子箱线
图 5 分区判别与主导因子 —— 混淆矩阵、置换重要性、族级份额、LDA 判别空间
图 6 分区的生态含义    —— 林分 RGR、SHAP 族份额、针阔与林龄组成、气候宽裕度

输出: 05_图表/分区图/Z1..Z6 (png+pdf) + 04_分析结果/fig_labels_zonation.json
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.OUT
MAPDIR = os.path.join(R, "map")
FIGDIR = os.path.join(C.FIGS, "分区图")
os.makedirs(FIGDIR, exist_ok=True)

# ---- 中文字体：显式注册 Windows 自带字体，避免 matplotlib 缓存里没有 ----
for _p in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\Deng.ttf"):
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DengXian", "DejaVu Sans"],
    "axes.unicode_minus": False,          # SimHei 无 U+2212，用 ASCII 连字符
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 150, "savefig.dpi": 300,
    # 幻灯片图件用【确定性画布】而非 bbox_inches='tight'：
    # 一旦有对象跑到画布外，tight 会为容纳它而撑大输出，16:9 变成竖版并留下
    # 大片空白（Z6 实测 3513×5075、高比 2.23）。固定画布下越界对象直接被裁掉，
    # 输出尺寸恒为 16:9；越界情况仍由 check_layout 记录在案。
    "savefig.bbox": None,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SLIDE = (13.3333, 7.5)                    # 16:9
ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
ZCOL = ["#1b6ca8", "#e8871a", "#2e8b57", "#8e44ad", "#c0392b", "#7f8c8d",
        "#16a085", "#d35400", "#2c3e50", "#a04000"]
FCOL = {"climate": "#1b6ca8", "soil": "#2e8b57", "terrain": "#e8871a"}

with open(os.path.join(R, "38_zonation.pkl"), "rb") as f:
    Z = pickle.load(f)
G = np.load(os.path.join(MAPDIR, "zonation_grid.npz"), allow_pickle=True)

VARS, ALL27 = Z["vars"], Z["all27"]
CLIM, SOIL, TERR = Z["clim"], Z["soil"], Z["terr"]
FAM, FAM_CN, VAR_CN = Z["family"], Z["family_cn"], Z["var_cn"]
FAM_ORDER = Z["fam_order"]
K, KF = Z["k_main"], Z["k_fine"]
PLC = Z["plc"]
ZONE, ZONE_FINE = Z["zone"], Z["zone_fine"]

LABELS = {}
LAYOUT_WARN = []
GLYPH_WARN = []


def _font_cmap(font_path, index=0):
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(font_path, fontNumber=index, lazy=True)
        cm = set(f.getBestCmap().keys())
        f.close()
        return cm
    except Exception:
        return set()


def _build_cmap_union():
    """把 font.sans-serif 里所有可定位到的字体字形集合起来。

    中文字体（SimHei / DengXian）不覆盖上标字符（U+207B ⁻、U+00B9 ¹、U+00B3 ³）
    与 U+2212 减号；matplotlib 的多字体回退在部分 artist 上并不会补齐，
    结果是图上直接出现空白或方框。故此处显式核对全部渲染文字的字形覆盖。
    """
    cm = set()
    for fam in plt.rcParams["font.sans-serif"]:
        try:
            p = font_manager.findfont(font_manager.FontProperties(family=fam),
                                      fallback_to_default=False)
        except Exception:
            continue
        cm |= _font_cmap(p)
    return cm


CMAP_UNION = _build_cmap_union()


def check_glyphs(fig, name):
    """字形守卫：所有渲染文字中的字符必须被当前字体覆盖。"""
    missing = {}
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s:
            continue
        for ch in s:
            if ch in "\n\r\t ":
                continue
            if ord(ch) not in CMAP_UNION:
                missing.setdefault(ch, f"U+{ord(ch):04X}")
    if missing:
        GLYPH_WARN.append((name, missing))
        print(f"  [WARN] {name} 字形缺失（会渲染成空白/方框）: {missing}", flush=True)
    return missing


def check_layout(fig, name):
    """版式守卫：交付物级核验——渲染包围盒不得膨胀，也不得只占画布一角。

    Stage 5 教训（台账 L10/L12）：文本级断言全绿也可能存在版面缺陷，
    只有把渲染产物本身纳入检查才能发现。此处用 bbox_inches='tight' 的**实际
    英寸范围**与画布尺寸相比：
      · 膨胀（tight / canvas > 1.15）：某 artist 跑到画布外，matplotlib 为容纳
        它撑大输出，表现为图件长宽比失真（Z6 曾因此变成 3513×5075 的竖版）；
      · 欠填（tight / canvas < 0.80）：内容只占画布一角，出现大片空白。
    注意 get_tightbbox 返回**英寸**，与 fig.get_size_inches 同单位；
    早期版本在此处 except 后静默 return，导致 Z6 的膨胀被漏检——现已改为显式失败。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    try:
        # get_tightbbox 返回的已是【英寸】，与 get_size_inches 同单位；
        # 早期版本又乘了一次 dpi_scale_trans.inverted()，把尺寸缩小 100 倍，
        # 于是比值恒为 0.01、任何异常都被判为"欠填"，守卫形同失效。
        tb_in = fig.get_tightbbox(r)
    except Exception as e:
        LAYOUT_WARN.append((name, {"error": repr(e)}, []))
        print(f"  [WARN] {name} 版式守卫无法计算紧致包围盒: {e!r}", flush=True)
        return
    rw, rh = tb_in.width / W, tb_in.height / H
    bad = []
    for a in fig.findobj(matplotlib.artist.Artist):
        try:
            bb = a.get_window_extent(r)
        except Exception:
            continue
        if bb.y1 > fig.bbox.y1 + 6 or bb.x1 > fig.bbox.x1 + 6 or \
           bb.y0 < fig.bbox.y0 - 6 or bb.x0 < fig.bbox.x0 - 6:
            own = getattr(a, "get_axes", lambda: None)()
            ap = None if own is None else [round(v, 3) for v in own.get_position().bounds]
            bad.append((type(a).__name__,
                        str(getattr(a, "get_text", lambda: "")())[:36],
                        [round(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1)],
                        ap))
    if rw > 1.15 or rh > 1.15 or rw < 0.80 or rh < 0.80:
        LAYOUT_WARN.append((name, {"tight_w_in": round(tb_in.width, 2),
                                   "tight_h_in": round(tb_in.height, 2),
                                   "canvas_w_in": W, "canvas_h_in": H,
                                   "ratio_w": round(rw, 3), "ratio_h": round(rh, 3)}, bad[:8]))
        print(f"  [WARN] {name} 版式异常：紧致包围盒 "
              f"{tb_in.width:.2f}×{tb_in.height:.2f} in vs 画布 {W}×{H} in"
              f"（宽比 {rw:.2f}，高比 {rh:.2f}）", flush=True)
        print(f"         各坐标轴位置（fig 比例）: "
              + "; ".join(f"[{i}]{[round(v,3) for v in a.get_position().bounds]}"
                          for i, a in enumerate(fig.axes)), flush=True)
        for b in bad[:8]:
            print(f"         越界 {b[0]} text={b[1]!r} bbox={b[2]} axes={b[3]}", flush=True)
        for a in fig.findobj(matplotlib.artist.Artist):
            if type(a).__name__ != "Rectangle" or not hasattr(a, "get_axes"):
                continue
            try:
                bb = a.get_window_extent(r)
            except Exception:
                continue
            if bb.y1 > fig.bbox.y1 + 6:
                print(f"         [矩形明细] xy={getattr(a,'xy',None)} "
                      f"w={a.get_width()} h={a.get_height()} "
                      f"fc={a.get_facecolor()} axes={a.get_axes()}", flush=True)
        # 图级容器（legend / 手工 Rectangle / 文本）的范围——越界 artist 常出自此处
        # 各轴的图例与色标范围（轴内 legend 挂在 ax.legend_ 上，不在 fig.legends 里）
        for i, axx in enumerate(fig.axes):
            for tag, obj in (("legend", axx.get_legend()),):
                if obj is None:
                    continue
                try:
                    bb = obj.get_window_extent(r)
                    print(f"         axes[{i}].{tag} bbox="
                          f"{[round(v) for v in (bb.x0,bb.y0,bb.x1,bb.y1)]}", flush=True)
                except Exception:
                    pass
        for cont in ("legends", "artists", "texts", "patches", "lines"):
            items = getattr(fig, cont, [])
            for k, it in enumerate(items):
                try:
                    bb = it.get_window_extent(r)
                    print(f"         fig.{cont}[{k}] {type(it).__name__} "
                          f"bbox={[round(v) for v in (bb.x0,bb.y0,bb.x1,bb.y1)]}", flush=True)
                except Exception:
                    pass


def save(fig, name):
    check_layout(fig, name)
    check_glyphs(fig, name)
    # bbox_inches=None：固定 16:9 画布，越界对象被裁掉（见 rcParams 注释）
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), bbox_inches=None)
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"), bbox_inches=None)
    texts = []
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text().strip()
        if s:
            texts.append(s)
    LABELS[name] = texts
    with open(os.path.join(R, "fig_labels_zonation.json"), "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=1)
    plt.close(fig)
    print(f"saved {name} ({len(texts)} labels)", flush=True)


def ztitle(ax, txt):
    ax.set_title(txt, loc="left", fontsize=12.5, pad=6)


def panel(ax, letter, txt):
    ax.set_title(f"（{letter}）{txt}", loc="left", fontsize=12.5, pad=6)


# ==========================================================================
# 图 1 环境梯度结构
# ==========================================================================
evr, load = Z["evr"], Z["load_rot"]
S = Z["S"]
PC_ALL = Z["pca_components"]
latp, demp = Z["lat"], Z["dem_plot"]
fig = plt.figure(figsize=SLIDE)
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.25, 1.15], hspace=0.42, wspace=0.34,
                       figure=fig)

ax = fig.add_subplot(gs[0, 0])
k_show = min(12, len(evr))
ax.bar(np.arange(1, k_show + 1), evr[:k_show] * 100, color="#4a7fb5", width=0.65)
ax.plot(np.arange(1, k_show + 1), np.cumsum(evr[:k_show]) * 100, "o-",
        color="#c0392b", ms=3.5, lw=1.4)
ax.axhline(0, color="0.7", lw=0.6)
ax.set_xlabel("主成分序号"); ax.set_ylabel("解释率（%）／累计（%）")
ax.set_xticks(np.arange(1, k_show + 1))
panel(ax, "a", f"碎石图（前 {Z['n_pc']} 个 PC 累计 {evr[:Z['n_pc']].sum()*100:.1f}%）")
ax.annotate("累计解释率", (k_show, np.cumsum(evr[:k_show])[-1] * 100),
            xytext=(-8, -18), textcoords="offset points", color="#c0392b", fontsize=9.5,
            ha="right")

# (b) PC1–PC2 载荷双标图
ax = fig.add_subplot(gs[0, 1])
sc = ax.scatter(S[:, 0], S[:, 1], c=[ZCOL[z] for z in ZONE], s=13,
                alpha=0.40, linewidths=0)
scale = 2.8
lab_ord = np.argsort(-np.hypot(load[:, 0], load[:, 1]))[:8]
# 共线箭头的标签统一用径向 1.24 倍偏移会互相压盖并撞上面板标题/邻轴刻度（实测）：
# 右侧三支箭头（bio4/bio19/flow_length）夹角 < 3°，顶部三支（SPI/TWI/TSI）近乎
# 垂直。改为逐变量手工定位（数据坐标，随 38_zonation.pkl 的载荷值定标），并显式
# 扩 xlim/ylim 容纳标签；图例移至左下，为垂直向下箭头（总曲率）的标签让出右下角。
_LAB_POS = {"bio4": (2.27, -0.55, "left"), "bio19": (2.57, -0.02, "left"),
            "flow_length": (2.42, 0.50, "left"), "SPI": (1.30, 1.98, "center"),
            "TWI": (-1.30, 2.24, "center"), "TSI": (0.0, 2.60, "center"),
            "curvature_total": (0.22, -2.40, "left")}
for j, v in enumerate(VARS):
    if np.hypot(load[j, 0], load[j, 1]) < 0.45:
        continue
    x, y = load[j, 0] * scale, load[j, 1] * scale
    ax.arrow(0, 0, x, y, color=FCOL[FAM[v]], lw=1.0, head_width=0.10,
             length_includes_head=True, alpha=0.85, zorder=4)
    if j in lab_ord:
        if v in _LAB_POS:
            tx, ty, ha = _LAB_POS[v]
        else:                          # bio1 左侧独行，径向偏移即可
            tx, ty, ha = x * 1.24, y * 1.24, "center"
        ax.text(tx, ty, VAR_CN[v], color=FCOL[FAM[v]], fontsize=8.2,
                ha=ha, va="center", zorder=5,
                bbox=dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=0.80))
ax.set_xlim(-3.60, 3.75)
ax.set_ylim(-2.68, 2.95)
ax.axhline(0, color="0.8", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
ax.set_xlabel(f"PC1（{evr[0]*100:.1f}%）"); ax.set_ylabel(f"PC2（{evr[1]*100:.1f}%）")
panel(ax, "b", "PC1–PC2 载荷双标图（点按一级区着色）")
ax.legend(handles=[Patch(fc=ZCOL[z], label=f"{ROMAN[z]} 区") for z in range(K)]
          + [Patch(fc=FCOL[f], label=FAM_CN[f]) for f in FAM_ORDER],
          loc="lower left", ncol=2, fontsize=8.5, handlelength=1.1)

# (c) 载荷热图（varimax 后前 6 PC）
ax = fig.add_subplot(gs[0, 2])
npc_h = 6
M = load[:, :npc_h]
order_h = [VARS.index(v) for f in FAM_ORDER for v in
           (CLIM if f == "climate" else SOIL if f == "soil" else TERR)]
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M[order_h], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(npc_h), [f"PC{i+1}" for i in range(npc_h)])
ax.set_yticks(range(len(order_h)), [VAR_CN[VARS[j]] for j in order_h], fontsize=9)
for y, f in enumerate([FAM[VARS[j]] for j in order_h]):
    ax.add_patch(plt.Rectangle((-0.5, y - 0.5), npc_h, 1, fill=False,
                               ec=FCOL[f], lw=1.0, zorder=3))
cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("载荷", fontsize=9.5)
panel(ax, "c", "旋转后载荷（族色框）")

# (d) 梯度与纬度/海拔
ax = fig.add_subplot(gs[1, 0])
for z in range(K):
    m = ZONE == z
    ax.scatter(latp[m], S[m, 0], s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
r = np.corrcoef(latp, S[:, 0])[0, 1]
ax.set_xlabel("纬度（°N）"); ax.set_ylabel(f"PC1（{evr[0]*100:.1f}%）")
panel(ax, "d", f"PC1 与纬度（r = {r:+.2f}）")

ax = fig.add_subplot(gs[1, 1])
for z in range(K):
    m = ZONE == z
    ax.scatter(demp[m], S[m, 0], s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
r2 = np.corrcoef(demp, S[:, 0])[0, 1]
ax.set_xlabel("DEM高程（m）"); ax.set_ylabel(f"PC1（{evr[0]*100:.1f}%）")
panel(ax, "e", f"PC1 与DEM高程（r = {r2:+.2f}）")

ax = fig.add_subplot(gs[1, 2])
# 各主成分的族方差构成：把每个 PC 的载荷平方按族汇总后归一
# （27×27 的相关矩阵在全图中不可读，改用它更能说明"梯度由哪些族承载"）
sq = load ** 2
fam_share = np.vstack([sq[[VARS.index(v) for v in vs]].sum(0) for vs in
                       (CLIM, SOIL, TERR)])
fam_share = fam_share / fam_share.sum(0)
nh = 6
bot = np.zeros(nh)
for fi_, f in enumerate(FAM_ORDER):
    ax.bar(np.arange(nh), fam_share[fi_, :nh] * 100, bottom=bot,
           color=FCOL[f], width=0.66, label=FAM_CN[f])
    bot += fam_share[fi_, :nh] * 100
for i in range(nh):
    ax.text(i, 101, f"{evr[i]*100:.1f}%", ha="center", va="bottom", fontsize=9,
            color="#444444")
ax.set_xticks(range(nh), [f"PC{i+1}" for i in range(nh)])
ax.set_ylabel("方差贡献（%）"); ax.set_ylim(0, 108)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=9)
panel(ax, "f", "各主成分由哪一族承载（柱上为解释率）")

fig.suptitle("图 1  广东省林地环境梯度结构（PCA，25 个环境因子）", fontsize=15, y=0.995)
save(fig, "Z1_gradients")

# ==========================================================================
# 图 2 分区数判据
# ==========================================================================
CR = Z["criteria"].set_index("k")
fig = plt.figure(figsize=SLIDE)
gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.28)
ks = CR.index.values


def mark_k(ax):
    ax.axvline(K, color="#c0392b", ls="--", lw=1.3, alpha=0.85)
    ax.text(K, ax.get_ylim()[1], f" k={K}", color="#c0392b", fontsize=10,
            va="top", ha="left")
    ax.set_xticks(ks)


ax = fig.add_subplot(gs[0, 0])
ax.plot(ks, CR["silhouette"], "o-", color="#1b6ca8", lw=1.6, ms=5)
ax.set_xlabel("分区数 k"); ax.set_ylabel("轮廓系数")
mark_k(ax)
panel(ax, "a", "轮廓系数（整体 0.14–0.16，连续梯度的典型量级）")

ax = fig.add_subplot(gs[0, 1])
ax.plot(ks, CR["calinski"], "o-", color="#2e8b57", lw=1.6, ms=5, label="Calinski-Harabasz")
ax.set_xlabel("分区数 k"); ax.set_ylabel("Calinski-Harabasz", color="#2e8b57")
ax.tick_params(axis="y", colors="#2e8b57")
ax2 = ax.twinx()
ax2.plot(ks, CR["davies_bouldin"], "s--", color="#8e44ad", lw=1.6, ms=4.5,
         label="Davies-Bouldin")
ax2.set_ylabel("Davies-Bouldin", color="#8e44ad")
ax2.tick_params(axis="y", colors="#8e44ad")
mark_k(ax)
panel(ax, "b", "Calinski-Harabasz 与 Davies-Bouldin")

ax = fig.add_subplot(gs[1, 0])
ax.plot(ks, CR["gmm_bic"] / 1000, "o-", color="#c0392b", lw=1.6, ms=5)
ax.set_xlabel("分区数 k"); ax.set_ylabel(r"GMM BIC（$\times$10$^{3}$）", color="#c0392b")
ax.tick_params(axis="y", colors="#c0392b")
ax2 = ax.twinx()
ax2.plot(ks, CR["gap"], "s--", color="#e8871a", lw=1.6, ms=4.5)
ax2.set_ylabel("Gap 统计量", color="#e8871a")
ax2.tick_params(axis="y", colors="#e8871a")
mark_k(ax)
panel(ax, "c", "GMM BIC（越小越好）与 Gap 统计量")

ax = fig.add_subplot(gs[1, 1])
data = [Z["boot_dist"][kk] for kk in ks]
ax.boxplot(data, positions=ks, widths=0.55, showfliers=False, patch_artist=True,
           medianprops=dict(color="black", lw=1.2),
           boxprops=dict(facecolor="#9ecae1", edgecolor="#1b6ca8"))
ax.plot(ks, CR["spatial_coherence"], "o--", color="#e8871a", lw=1.5, ms=4.5)
ax.set_xlabel("分区数 k"); ax.set_ylabel("自助稳定性 ARI（箱线）")
ax.set_xticks(ks); ax.set_ylim(0, 1.02)
mark_k(ax)
ax.legend(handles=[Patch(fc="#9ecae1", ec="#1b6ca8", label="自助稳定性 ARI"),
                   Line2D([], [], color="#e8871a", ls="--", marker="o",
                          label="空间连贯率")], loc="lower left", fontsize=9)
panel(ax, "d", "自助稳定性与空间连贯率")

fig.suptitle(f"图 2  分区数判据：环境为连续梯度，k 由自助稳定性峰值确定（k = {K}）",
             fontsize=15, y=0.995)
save(fig, "Z2_criteria")

# ==========================================================================
# 图 3 省域区划图
# ==========================================================================
LAB, LABF, LABC = G["zone"], G["zone_fine"], G["zone_clim"]
MARG, DEMG = G["margin"], G["dem"]
tr = G["transform"]
ext = [tr[2], tr[2] + tr[0] * LAB.shape[1], tr[5] + tr[4] * LAB.shape[0], tr[5]]
cmap_k = ListedColormap([ZCOL[z] for z in range(K)])
cmap_kf = ListedColormap([ZCOL[z] for z in range(KF)])

fig = plt.figure(figsize=SLIDE)
# 两行：上排三张地图（广东为东西向，地图本身扁宽，故上排压扁、下排给纬度组成图），
# 避免地图在整高格子里上下留大片空白、而右侧柱图独占全高造成的失衡。
gs = gridspec.GridSpec(2, 3, height_ratios=[1.55, 1], hspace=0.42, wspace=0.22)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.where(LAB < 0, np.nan, LAB), extent=ext, origin="upper", cmap=cmap_k,
          vmin=-0.5, vmax=K - 0.5, interpolation="nearest")
ax.scatter(Z["lon"], Z["lat"], s=2.0, c="black", alpha=0.30, linewidths=0)
ax.set_xlabel("经度（°E）"); ax.set_ylabel("纬度（°N）")
ax.set_anchor("N")
panel(ax, "a", f"一级区（{K} 区）")
ax.legend(handles=[Patch(fc=ZCOL[z], label=f"{ROMAN[z]} 区") for z in range(K)],
          loc="lower left", fontsize=9)

ax = fig.add_subplot(gs[0, 1])
ax.imshow(np.where(LABF < 0, np.nan, LABF), extent=ext, origin="upper", cmap=cmap_kf,
          vmin=-0.5, vmax=KF - 0.5, interpolation="nearest")
ax.set_xlabel("经度（°E）"); ax.set_yticks([])
ax.set_anchor("N")
panel(ax, "b", f"细化口径（{KF} 亚区）")

ax = fig.add_subplot(gs[0, 2])
diff = np.where((LAB < 0) | (LABC < 0), np.nan, (LAB != LABC).astype(float))
im = ax.imshow(diff, extent=ext, origin="upper",
               cmap=ListedColormap(["#e8e8e8", "#c0392b"]),
               vmin=-0.5, vmax=1.5, interpolation="nearest")
ax.set_xlabel("经度（°E）"); ax.set_yticks([])
ax.set_anchor("N")
panel(ax, "c", f"与纯气候分区不一致（红，{Z['clim_diff']*100:.1f}%）")

ax = fig.add_subplot(gs[1, :])
zb = Z["lat_band_zone"]
xpos = np.arange(len(zb))
bot = np.zeros(len(zb))
for z in range(K):
    if z not in zb.columns:
        continue
    ax.bar(xpos, zb[z].values, bottom=bot, color=ZCOL[z], width=0.72)
    bot += zb[z].values
ax.set_xticks(xpos, [f"{iv.left:.1f}–{iv.right:.1f}" for iv in zb.index],
              rotation=0, fontsize=10)
ax.set_xlabel("纬度带（°N）"); ax.set_ylabel("像元占比（%）")
ax.set_ylim(0, 120); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlim(-0.6, len(zb) - 0.4)
za = Z["zone_area"]
# 轴外图例（bbox_to_anchor=(1.005, 0.5)）会被 16:9 固定画布右缘直接裁掉——
# 面板 d 横贯全宽，轴右缘即画布缘。改为轴内顶部单行图例，柱体限高 120 让位。
ax.legend(handles=[Patch(fc=ZCOL[z], label=f"{ROMAN[z]} 区（面积 {za[z]:.1f}%）")
                   for z in range(K)],
          loc="upper center", ncol=3, fontsize=9.5)
panel(ax, "d", "各纬度带的分区组成（全省林地面积占比见图例）")

fig.suptitle("图 3  广东省林地地带性分区（250 m；一级区与细化口径，含纯气候对照）",
             fontsize=15, y=0.975)
save(fig, "Z3_zonation_map")

# ==========================================================================
# 图 4 分区环境特征
# ==========================================================================
zan, fine_an = Z["zanom"], Z["fine_anom"]
fig = plt.figure(figsize=SLIDE)
# wspace 0.46：地形水文面板因子名最长（12 行），原 0.42 会让其 y 轴标签压进
# 左侧土壤面板的色标。
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1.25, 1.05], hspace=0.46, wspace=0.46)

for col, (fam, vs) in enumerate((("climate", CLIM), ("soil", SOIL))):
    ax = fig.add_subplot(gs[0, col])
    Mz = zan[vs].T.values
    vmax = np.nanmax(np.abs(Mz))
    im = ax.imshow(Mz, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(K), [f"{ROMAN[z]}\n{int(Z['zone_area'][z])}%" for z in range(K)],
                  fontsize=9.5)
    ax.set_yticks(range(len(vs)), [VAR_CN[v] for v in vs], fontsize=9.5)
    for i in range(len(vs)):
        for j in range(K):
            ax.text(j, i, f"{Mz[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    # 单位标签只保留 (c) 的：竖排「标准化异常」会伸入面板间隙，与右侧面板
    # 的长 y 刻度名相撞（实测 (a)↔(b)、(b)↔(c) 两处均撞）。
    panel(ax, "ab"[col], f"{FAM_CN[fam]}族（{len(vs)} 因子）")

ax = fig.add_subplot(gs[0, 2])
row = TERR
Mz = zan[row].T.values
vmax = np.nanmax(np.abs(Mz))
im = ax.imshow(Mz, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(K), [f"{ROMAN[z]}\n{int(Z['zone_area'][z])}%" for z in range(K)],
              fontsize=9.5)
# 12 行因子名最长 6 字，8.6 pt 下行距不足互相叠压（实测）：缩字号并对
# 「××指数」类长名用本面板短名；其余面板仍用全名保持可读。
_TERR_CN = {"aspect_cos": "坡向余弦", "aspect_sin": "坡向正弦", "relief": "地形起伏",
            "SPI": "水流强度", "TPI": "地形位置", "TSI": "地形形态", "TWI": "地形湿度"}
ax.set_yticks(range(len(row)), [_TERR_CN.get(v, VAR_CN[v]) for v in row], fontsize=7.5)
for i in range(len(row)):
    for j in range(K):
        ax.text(j, i, f"{Mz[i, j]:+.2f}", ha="center", va="center", fontsize=7.6)
cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("标准化异常", fontsize=9)
panel(ax, "c", f"地形水文族（{len(row)} 因子）")

ax = fig.add_subplot(gs[0, 3])
fi = Z["fam_imp"]
ax.barh([FAM_CN[f] for f in FAM_ORDER][::-1],
        [fi[f] * 100 for f in FAM_ORDER][::-1],
        color=[FCOL[f] for f in FAM_ORDER][::-1], height=0.55)
for i, f in enumerate(FAM_ORDER[::-1]):
    ax.text(fi[f] * 100 + 1.0, i, f"{fi[f]*100:.1f}%", va="center", fontsize=10.5)
ax.set_xlabel("族级判别份额（%）"); ax.set_xlim(0, 60)
panel(ax, "d", "哪一族真正决定分区")

key_vars = ["bio1", "bio12", "dem"]
key_lab = ["年均温（°C）", "年降水量（mm）", "DEM高程（m）"]
key_div = [10.0, 10.0, 1.0]
for col, (v, lab, dv) in enumerate(zip(key_vars, key_lab, key_div)):
    ax = fig.add_subplot(gs[1, col])
    data = [PLC.loc[ZONE == z, v].values / dv for z in range(K)]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2))
    for b, z in zip(bp["boxes"], range(K)):
        b.set_facecolor(ZCOL[z]); b.set_alpha(0.8); b.set_edgecolor("black")
    ax.set_xticks(range(1, K + 1), [f"{ROMAN[z]} 区" for z in range(K)])
    ax.set_ylabel(lab)
    panel(ax, "efg"[col], f"{lab}的分区分布")

ax = fig.add_subplot(gs[1, 3])
fm = fine_an[["bio1", "bio12", "dem", "soil_SOC", "TWI", "relief"]].T.values
vmax = np.nanmax(np.abs(fm))
im = ax.imshow(fm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(KF), [f"{ROMAN[z]}" for z in range(KF)])
ax.set_yticks(range(fm.shape[0]),
              ["年均温", "年降水量", "DEM高程", "土壤有机碳", "地形湿度指数", "地形起伏度"],
              fontsize=9)
for i in range(fm.shape[0]):
    for j in range(KF):
        ax.text(j, i, f"{fm[i,j]:+.2f}", ha="center", va="center", fontsize=8)
cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cb.set_label("标准化异常", fontsize=9)
panel(ax, "h", f"细化口径 {KF} 亚区（选摘 6 因子）")

fig.suptitle("图 4  各分区的环境特征（相对全省的标准化异常，红=偏高、蓝=偏低）",
             fontsize=15, y=0.995)
save(fig, "Z4_zone_profiles")

# ==========================================================================
# 图 5 分区判别与主导因子
# ==========================================================================
CM, IMP = Z["cm"], Z["imp"]
fig = plt.figure(figsize=SLIDE)
# wspace 放到 0.52：面板 (c) 的因子名较长，原 0.36 会压进面板 (b) 的柱区。
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.95, 1.5, 1.1], wspace=0.52)

ax = fig.add_subplot(gs[0])
CMn = CM / CM.sum(axis=1, keepdims=True) * 100
im = ax.imshow(CMn, cmap="Blues", vmin=0, vmax=100)
for i in range(CM.shape[0]):
    for j in range(CM.shape[1]):
        ax.text(j, i, f"{CMn[i,j]:.0f}", ha="center", va="center",
                color="white" if CMn[i, j] > 55 else "black", fontsize=11)
ax.set_xticks(range(K), [f"{ROMAN[z]}" for z in range(K)])
ax.set_yticks(range(K), [f"{ROMAN[z]} 区" for z in range(K)])
ax.set_xlabel("预测分区"); ax.set_ylabel("实际分区")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("行占比（%）", fontsize=9)
panel(ax, "a", "混淆矩阵")

ax = fig.add_subplot(gs[1])
fi = Z["fam_imp"]
bars = ax.bar([FAM_CN[f] for f in FAM_ORDER], [fi[f] * 100 for f in FAM_ORDER],
              color=[FCOL[f] for f in FAM_ORDER], width=0.6)
for b, f in zip(bars, FAM_ORDER):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
            f"{fi[f]*100:.1f}%", ha="center", fontsize=11)
ax.set_ylabel("族级判别份额（%）"); ax.set_ylim(0, 62)
panel(ax, "b", "族级归因")
ax.text(0.5, 0.90, f"判别准确率 {Z['acc_cv']*100:.1f}%\n（OOB {Z['oob']:.3f}）",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9))

ax = fig.add_subplot(gs[2])
top = IMP.head(12).iloc[::-1]
ax.barh(np.arange(len(top)), top["importance_mean"],
        color=[FCOL[f] for f in top["family"]], height=0.66,
        xerr=top["importance_sd"], error_kw=dict(ecolor="#444", lw=0.9, capsize=2))
ax.set_yticks(np.arange(len(top)), [VAR_CN[v] for v in top["var"]], fontsize=9.5)
ax.set_xlabel("置换重要性（准确率下降）")
ax.set_xlim(0, top["importance_mean"].max() * 1.22)
panel(ax, "c", "前 12 位主导因子")

ax = fig.add_subplot(gs[3])
LS = Z["lda_s"]
for z in range(K):
    m = ZONE == z
    ax.scatter(LS[m, 0], LS[m, 1], s=12, c=ZCOL[z], alpha=0.42, linewidths=0)
    mu = LS[m].mean(0)
    cov = np.cov(LS[m].T)
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    for nstd in (1, 2):
        ax.add_patch(Ellipse(mu, 2 * nstd * np.sqrt(w[-1]) * 1.6,
                             2 * nstd * np.sqrt(w[0]) * 1.6, angle=ang,
                             fill=False, ec=ZCOL[z], lw=1.3, alpha=0.9))
    ax.text(mu[0], mu[1], ROMAN[z], fontsize=13, ha="center", va="center",
            color=ZCOL[z], weight="bold")
ax.set_xlabel(f"LD1（{Z['lda_evr'][0]*100:.0f}%）")
ax.set_ylabel(f"LD2（{Z['lda_evr'][1]*100:.0f}%）")
panel(ax, "d", "判别空间（椭圆 1σ/2σ）")

fig.suptitle(f"图 5  分区的可判别性与主导因子（{len(VARS)} 个环境因子 → "
             f"判别准确率 {Z['acc_cv']*100:.1f}%）", fontsize=15, y=0.975)
save(fig, "Z5_discriminant")

# ==========================================================================
# 图 6 分区的生态含义
# ==========================================================================
fz = Z["fam_shap_zone"]
fig = plt.figure(figsize=SLIDE)
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.1, 1.15], hspace=0.45, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
data = [PLC.loc[ZONE == z, "RGR_pool"].dropna().values for z in range(K)]
bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=1.2))
for b, z in zip(bp["boxes"], range(K)):
    b.set_facecolor(ZCOL[z]); b.set_alpha(0.8); b.set_edgecolor("black")
for i, d in enumerate(data, 1):
    # 抬升 2.5pt：va="bottom" 原地绘出时标注底部与黑色中位线相切、被线穿过
    ax.annotate(f"{np.median(d):.3f}", (i, np.median(d)), xytext=(0, 2.5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9.5, color="black")
ax.set_xticks(range(1, K + 1), [f"{ROMAN[z]} 区" for z in range(K)])
ax.set_ylabel(r"样地平均 RGR（yr$^{-1}$）")
panel(ax, "a", "各分区的林木生长速率")

ax = fig.add_subplot(gs[0, 1])
bot = np.zeros(K)
for f in FAM_ORDER:
    v = fz[f].reindex(range(K)).values * 100
    ax.bar(range(K), v, bottom=bot, color=FCOL[f], width=0.62, label=FAM_CN[f])
    for z in range(K):
        if v[z] > 6:
            ax.text(z, bot[z] + v[z] / 2, f"{v[z]:.0f}", ha="center", va="center",
                    color="white", fontsize=9.5)
    bot += v
ax.set_xticks(range(K), [f"{ROMAN[z]} 区" for z in range(K)])
ax.set_ylabel("SHAP 族份额（%）"); ax.set_ylim(0, 118)
# 图例放在轴内顶部：轴外图例（bbox_to_anchor 取负值）会把 tight bbox 撑出画布，
# 使 16:9 输出变成竖版并留下大片空白（Z6 实测高比 2.23，已由版图守卫捕获）。
ax.legend(loc="upper center", ncol=3, fontsize=9.5, frameon=False,
          handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
ax.set_yticks([0, 20, 40, 60, 80, 100])
panel(ax, "b", "驱动层级是否随分区变化")

ax = fig.add_subplot(gs[0, 2])
fg = Z["fg_by_zone"].reindex(range(K)).values * 100
ag = Z["rgr_by_zone"]["mean"].reindex(range(K)).values
ax.bar(range(K), fg, color="#2166ac", width=0.55, label="针叶样地占比")
ax.set_ylabel("针叶样地占比（%）", color="#2166ac")
ax.tick_params(axis="y", colors="#2166ac"); ax.set_ylim(0, 100)
ax2 = ax.twinx()
ax2.plot(range(K), PLC.groupby("zone")["stand_age_r"].median().reindex(range(K)).values,
         "o-", color="#b35806", lw=1.8, ms=7, label="林龄中位数")
ax2.set_ylabel("林龄中位数（yr）", color="#b35806")
ax2.tick_params(axis="y", colors="#b35806")
ax.set_xticks(range(K), [f"{ROMAN[z]} 区" for z in range(K)])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
# 左下角图例压住 Ⅰ 区柱体（实测）；柱高约三成、橙线 Ⅰ 区标记低位，左上空闲
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
panel(ax, "c", "功能群与林龄组成")

ax = fig.add_subplot(gs[1, 0])
for z in range(K):
    m = ZONE == z
    ax.scatter(PLC.loc[m, "bio1"].values / 10, PLC.loc[m, "RGR_pool"].values,
               s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
ax.set_xlabel("年均温（°C）"); ax.set_ylabel(r"样地平均 RGR（yr$^{-1}$）")
panel(ax, "d", "生长对气温的响应（气候宽裕性）")

ax = fig.add_subplot(gs[1, 1])
for z in range(K):
    m = ZONE == z
    ax.scatter(PLC.loc[m, "bio12"].values / 10, PLC.loc[m, "RGR_pool"].values,
               s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
ax.set_xlabel("年降水量（mm）"); ax.set_ylabel(r"样地平均 RGR（yr$^{-1}$）")
panel(ax, "e", "生长对降水的响应")

ax = fig.add_subplot(gs[1, 2])
# 用文本手工排表，不用 matplotlib 的 ax.table：后者在本图的轴尺寸下
# 单元格 Rectangle 会被算到 y≈1887 px 高（画布仅 750 px），把 tight bbox
# 撑到 16.9 in，输出变成 3513×5075 的竖版并留下大片空白（已由版图守卫捕获）。
rows = []
for z in range(K):
    m = ZONE == z
    rows.append([f"{ROMAN[z]} 区", f"{int(m.sum())}",
                 f"{PLC.loc[m, 'bio1'].mean()/10:.1f}",
                 f"{PLC.loc[m, 'bio12'].mean()/10:.0f}",
                 f"{PLC.loc[m, 'dem'].mean():.0f}",
                 f"{PLC.loc[m, 'RGR_pool'].mean():.3f}"])
heads = ["分区", "样地数", "年均温\n(°C)", "年降水\n(mm)", "DEM高程\n(m)", "RGR\n(yr$^{-1}$)"]
xs = [0.06, 0.26, 0.44, 0.63, 0.80, 0.96]
y0, dy = 0.80, 0.155
for x, htxt in zip(xs, heads):
    ax.text(x, y0 + dy * 0.72, htxt, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="#22303f", linespacing=1.15)
ax.plot([0.02, 1.0], [y0 + dy * 0.28] * 2, transform=ax.transAxes,
        color="#8899aa", lw=1.0)
for i, r in enumerate(rows):
    yy = y0 - dy * i
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((0.02, yy - dy * 0.36), 0.98, dy * 0.72,
                                   transform=ax.transAxes, fc="#f2f5f8", ec="none",
                                   zorder=0))
    for x, cell in zip(xs, r):
        ax.text(x, yy, cell, transform=ax.transAxes, ha="center", va="center",
                fontsize=10.5, zorder=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
panel(ax, "f", "分区一览")
ax.axis("off")

fig.suptitle("图 6  分区的生态含义：生长速率与生长驱动层级的分区差异", fontsize=15, y=0.995)
save(fig, "Z6_ecology")

print("ALL ZONATION FIGURES DONE")

# ---- 交付物级核验汇总 ----
print()
print("=== 图件核验汇总 ===")
if GLYPH_WARN:
    print(f"FAIL: {len(GLYPH_WARN)} 张图存在字形缺失（会渲染成空白/方框）")
    for nm, ms in GLYPH_WARN:
        print(f"  - {nm}: {ms}")
else:
    print(f"OK: 6 张图全部文字的字形均被字体覆盖（字形集合 {len(CMAP_UNION)} 个）")
if LAYOUT_WARN:
    print(f"WARN: {len(LAYOUT_WARN)} 张图版式越界")
    for nm, ov, bad in LAYOUT_WARN:
        print(f"  - {nm}: {ov}")
else:
    print("OK: 6 张图的紧致包围盒均未越出画布")
print()
print("产出: 05_图表/分区图/Z1..Z6 (png+pdf)")
print("标签: 04_分析结果/fig_labels_zonation.json")
if GLYPH_WARN:
    raise SystemExit(1)