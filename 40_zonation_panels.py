# -*- coding: utf-8 -*-
"""40 将分区图件 Z1–Z6 的各面板拆分为独立文件（单面板版）。

复用 39_zonation_figures.py 的数据与绘图逻辑，把 32 个面板逐一重绘为
独立图片。画幅按面板类型单独设定横纵比（不再强制 16:9）：
  · 散点 / 箱线 / 柱图   ≈ 4:3（5.0×3.8 in 左右）
  · 省域地图             保持地理比例（约 1.4:1，imshow aspect='equal'）
  · 25 行载荷热图        纵向拉高（5.0×6.8）
  · 12 行地形热图        纵向拉高（5.6×5.4）
  · 纬度带组成图         横向拉宽（9.0×3.9，图例在右）
  · 分区一览表           横向宽幅（7.6×3.1）
保存用 bbox_inches='tight'（单面板无 16:9 约束，按内容收紧并留 0.12 in 边），
并沿用 39 号的两道守卫：字形守卫 + 版式守卫（tight 相对画布膨胀 >1.35× 报警）。

输出: 05_图表/分区图/单面板/Z1a_碎石图 .. Z6f_分区一览 (png+pdf，各 32 个)
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.OUT
MAPDIR = os.path.join(R, "map")
OUTDIR = os.path.join(C.FIGS, "分区图", "单面板")
os.makedirs(OUTDIR, exist_ok=True)

for _p in (r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\Deng.ttf"):
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "DengXian", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
ZCOL = ["#1b6ca8", "#e8871a", "#2e8b57", "#8e44ad", "#c0392b", "#7f8c8d",
        "#16a085", "#d35400", "#2c3e50", "#a04000"]
FCOL = {"climate": "#1b6ca8", "soil": "#2e8b57", "terrain": "#e8871a"}

with open(os.path.join(R, "38_zonation.pkl"), "rb") as f:
    Z = pickle.load(f)
G = np.load(os.path.join(MAPDIR, "zonation_grid.npz"), allow_pickle=True)

VARS = Z["vars"]
CLIM, SOIL, TERR = Z["clim"], Z["soil"], Z["terr"]
FAM, FAM_CN, VAR_CN = Z["family"], Z["family_cn"], Z["var_cn"]
FAM_ORDER = Z["fam_order"]
K, KF = Z["k_main"], Z["k_fine"]
PLC = Z["plc"]
ZONE = Z["zone"]
evr, load = Z["evr"], Z["load_rot"]
S = Z["S"]
latp, demp = Z["lat"], Z["dem_plot"]

PANEL_WARN = []


def _font_cmap(font_path):
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(font_path, lazy=True)
        cm = set(f.getBestCmap().keys())
        f.close()
        return cm
    except Exception:
        return set()


CMAP_UNION = set()
for _fam in plt.rcParams["font.sans-serif"]:
    try:
        _p = font_manager.findfont(font_manager.FontProperties(family=_fam),
                                   fallback_to_default=False)
    except Exception:
        continue
    CMAP_UNION |= _font_cmap(_p)


def panel(ax, letter, txt):
    ax.set_title(f"（{letter}）{txt}", loc="left", fontsize=12.5, pad=6)


def psave(fig, name, fixed=False):
    """保存单面板，字形 + 版式双守卫。

    fixed=False（默认）：bbox_inches='tight' 按内容收紧，单面板无 16:9 约束。
    fixed=True：确定性画布，越界对象直接裁掉——仅用于 Z6b：Ⅲ 区 SHAP 族份额
    退化（净效应≈0 → 气候 +534%、地形 -455%），负值柱会把 tight 包围盒撑到
    15.5 in（39 号整图 Z6 的已知 WARN 同源）；整图用固定画布裁掉该柱，
    此处保持一致，输出尺寸恒等于画布。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    missing = {}
    for t in fig.findobj(matplotlib.text.Text):
        for ch in t.get_text():
            if ch not in "\n\r\t " and ord(ch) not in CMAP_UNION:
                missing.setdefault(ch, f"U+{ord(ch):04X}")
    if missing:
        PANEL_WARN.append((name, "glyph", missing))
        print(f"  [WARN] {name} 字形缺失（空白/方框）: {missing}", flush=True)
    W, H = fig.get_size_inches()
    tb = None
    try:
        tb = fig.get_tightbbox(r)
    except Exception as e:
        PANEL_WARN.append((name, "tightbbox", repr(e)))
        print(f"  [WARN] {name} 紧致包围盒无法计算: {e!r}", flush=True)
    bbox = None if fixed else "tight"
    pad = {} if fixed else {"pad_inches": 0.12}
    fig.savefig(os.path.join(OUTDIR, name + ".png"), bbox_inches=bbox, **pad)
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"), bbox_inches=bbox, **pad)
    plt.close(fig)
    if tb is not None and not fixed:
        rw, rh = tb.width / W, tb.height / H
        flag = ""
        if max(rw, rh) > 1.35:
            flag = "  <-- 膨胀，需检查（可能残留画布外对象）"
            PANEL_WARN.append((name, "layout",
                               {"tight_in": [round(tb.width, 2), round(tb.height, 2)],
                                "canvas_in": [round(W, 2), round(H, 2)],
                                "ratio": [round(rw, 2), round(rh, 2)]}))
        print(f"saved {name}  目标 {W:.1f}×{H:.1f} in → tight "
              f"{tb.width:.2f}×{tb.height:.2f} in（宽比 {rw:.2f}，高比 {rh:.2f}）{flag}",
              flush=True)
    else:
        note = "（固定画布，越界柱/标签已裁掉）" if fixed else ""
        print(f"saved {name}  固定画布 {W:.1f}×{H:.1f} in{note}", flush=True)


# ==========================================================================
# 图 1 环境梯度结构（6 面板）
# ==========================================================================
# (a) 碎石图
fig, ax = plt.subplots(figsize=(5.0, 3.8))
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
psave(fig, "Z1a_碎石图")

# (b) PC1–PC2 载荷双标图
fig, ax = plt.subplots(figsize=(5.4, 4.8))
sc = ax.scatter(S[:, 0], S[:, 1], c=[ZCOL[z] for z in ZONE], s=13,
                alpha=0.40, linewidths=0)
scale = 2.8
lab_ord = np.argsort(-np.hypot(load[:, 0], load[:, 1]))[:8]
# 与 39 号 Z1(b) 同步：共线箭头（右侧三支 <3°、顶部三支近垂直）的标签
# 逐变量手工定位，避免互相压盖并撞上面板标题；图例移至左下。
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
psave(fig, "Z1b_载荷双标图")

# (c) 旋转后载荷热图
fig, ax = plt.subplots(figsize=(5.0, 6.8))
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
psave(fig, "Z1c_旋转载荷热图")

# (d) PC1 与纬度
fig, ax = plt.subplots(figsize=(5.0, 3.8))
for z in range(K):
    m = ZONE == z
    ax.scatter(latp[m], S[m, 0], s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
r = np.corrcoef(latp, S[:, 0])[0, 1]
ax.set_xlabel("纬度（°N）"); ax.set_ylabel(f"PC1（{evr[0]*100:.1f}%）")
panel(ax, "d", f"PC1 与纬度（r = {r:+.2f}）")
psave(fig, "Z1d_PC1与纬度")

# (e) PC1 与海拔
fig, ax = plt.subplots(figsize=(5.0, 3.8))
for z in range(K):
    m = ZONE == z
    ax.scatter(demp[m], S[m, 0], s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
r2 = np.corrcoef(demp, S[:, 0])[0, 1]
ax.set_xlabel("DEM高程（m）"); ax.set_ylabel(f"PC1（{evr[0]*100:.1f}%）")
panel(ax, "e", f"PC1 与DEM高程（r = {r2:+.2f}）")
psave(fig, "Z1e_PC1与DEM高程")

# (f) 各主成分由哪一族承载
fig, ax = plt.subplots(figsize=(5.0, 3.9))
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
psave(fig, "Z1f_主成分族承载")

# ==========================================================================
# 图 2 分区数判据（4 面板）
# ==========================================================================
CR = Z["criteria"].set_index("k")
ks = CR.index.values


def mark_k(ax):
    ax.axvline(K, color="#c0392b", ls="--", lw=1.3, alpha=0.85)
    ax.text(K, ax.get_ylim()[1], f" k={K}", color="#c0392b", fontsize=10,
            va="top", ha="left")
    ax.set_xticks(ks)


fig, ax = plt.subplots(figsize=(5.2, 3.9))
ax.plot(ks, CR["silhouette"], "o-", color="#1b6ca8", lw=1.6, ms=5)
ax.set_xlabel("分区数 k"); ax.set_ylabel("轮廓系数")
mark_k(ax)
panel(ax, "a", "轮廓系数（整体 0.14–0.16，连续梯度的典型量级）")
psave(fig, "Z2a_轮廓系数")

fig, ax = plt.subplots(figsize=(5.6, 3.9))
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
psave(fig, "Z2b_CH与DB")

fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.plot(ks, CR["gmm_bic"] / 1000, "o-", color="#c0392b", lw=1.6, ms=5)
ax.set_xlabel("分区数 k"); ax.set_ylabel(r"GMM BIC（$\times$10$^{3}$）", color="#c0392b")
ax.tick_params(axis="y", colors="#c0392b")
ax2 = ax.twinx()
ax2.plot(ks, CR["gap"], "s--", color="#e8871a", lw=1.6, ms=4.5)
ax2.set_ylabel("Gap 统计量", color="#e8871a")
ax2.tick_params(axis="y", colors="#e8871a")
mark_k(ax)
panel(ax, "c", "GMM BIC（越小越好）与 Gap 统计量")
psave(fig, "Z2c_BIC与Gap")

fig, ax = plt.subplots(figsize=(5.6, 3.9))
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
psave(fig, "Z2d_自助稳定性")

# ==========================================================================
# 图 3 省域区划图（4 面板）
# ==========================================================================
LAB, LABF, LABC = G["zone"], G["zone_fine"], G["zone_clim"]
tr = G["transform"]
ext = [tr[2], tr[2] + tr[0] * LAB.shape[1], tr[5] + tr[4] * LAB.shape[0], tr[5]]
cmap_k = ListedColormap([ZCOL[z] for z in range(K)])
cmap_kf = ListedColormap([ZCOL[z] for z in range(KF)])

fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.imshow(np.where(LAB < 0, np.nan, LAB), extent=ext, origin="upper", cmap=cmap_k,
          vmin=-0.5, vmax=K - 0.5, interpolation="nearest")
ax.scatter(Z["lon"], Z["lat"], s=2.0, c="black", alpha=0.30, linewidths=0)
ax.set_xlabel("经度（°E）"); ax.set_ylabel("纬度（°N）")
panel(ax, "a", f"一级区（{K} 区）")
ax.legend(handles=[Patch(fc=ZCOL[z], label=f"{ROMAN[z]} 区") for z in range(K)],
          loc="lower left", fontsize=9)
psave(fig, "Z3a_一级区地图")

fig, ax = plt.subplots(figsize=(6.4, 5.2))
ax.imshow(np.where(LABF < 0, np.nan, LABF), extent=ext, origin="upper", cmap=cmap_kf,
          vmin=-0.5, vmax=KF - 0.5, interpolation="nearest")
ax.set_xlabel("经度（°E）"); ax.set_ylabel("纬度（°N）")
panel(ax, "b", f"细化口径（{KF} 亚区）")
psave(fig, "Z3b_细化口径地图")

fig, ax = plt.subplots(figsize=(6.4, 5.2))
diff = np.where((LAB < 0) | (LABC < 0), np.nan, (LAB != LABC).astype(float))
ax.imshow(diff, extent=ext, origin="upper",
          cmap=ListedColormap(["#e8e8e8", "#c0392b"]),
          vmin=-0.5, vmax=1.5, interpolation="nearest")
ax.set_xlabel("经度（°E）"); ax.set_ylabel("纬度（°N）")
panel(ax, "c", f"与纯气候分区不一致（红，{Z['clim_diff']*100:.1f}%）")
psave(fig, "Z3c_纯气候差异地图")

fig, ax = plt.subplots(figsize=(9.0, 3.9))
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
ax.set_ylim(0, 100); ax.set_xlim(-0.6, len(zb) - 0.4)
za = Z["zone_area"]
ax.legend(handles=[Patch(fc=ZCOL[z], label=f"{ROMAN[z]} 区（面积 {za[z]:.1f}%）")
                   for z in range(K)],
          loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=10)
panel(ax, "d", "各纬度带的分区组成（全省林地面积占比见图例）")
psave(fig, "Z3d_纬度带组成")

# ==========================================================================
# 图 4 分区环境特征（8 面板）
# ==========================================================================
zan, fine_an = Z["zanom"], Z["fine_anom"]

fig, ax = plt.subplots(figsize=(5.2, 4.2))
vs = CLIM
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
cb.set_label("标准化异常", fontsize=9)
panel(ax, "a", f"气候族（{len(vs)} 因子）")
psave(fig, "Z4a_气候族热图")

fig, ax = plt.subplots(figsize=(5.4, 4.4))
vs = SOIL
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
cb.set_label("标准化异常", fontsize=9)
panel(ax, "b", f"土壤族（{len(vs)} 因子）")
psave(fig, "Z4b_土壤族热图")

fig, ax = plt.subplots(figsize=(5.6, 5.4))
row = TERR
Mz = zan[row].T.values
vmax = np.nanmax(np.abs(Mz))
im = ax.imshow(Mz, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(K), [f"{ROMAN[z]}\n{int(Z['zone_area'][z])}%" for z in range(K)],
              fontsize=9.5)
ax.set_yticks(range(len(row)), [VAR_CN[v] for v in row], fontsize=8.6)
for i in range(len(row)):
    for j in range(K):
        ax.text(j, i, f"{Mz[i, j]:+.2f}", ha="center", va="center", fontsize=7.6)
cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("标准化异常", fontsize=9)
panel(ax, "c", f"地形水文族（{len(row)} 因子）")
psave(fig, "Z4c_地形水文族热图")

fig, ax = plt.subplots(figsize=(5.0, 3.5))
fi = Z["fam_imp"]
ax.barh([FAM_CN[f] for f in FAM_ORDER][::-1],
        [fi[f] * 100 for f in FAM_ORDER][::-1],
        color=[FCOL[f] for f in FAM_ORDER][::-1], height=0.55)
for i, f in enumerate(FAM_ORDER[::-1]):
    ax.text(fi[f] * 100 + 1.0, i, f"{fi[f]*100:.1f}%", va="center", fontsize=10.5)
ax.set_xlabel("族级判别份额（%）"); ax.set_xlim(0, 60)
panel(ax, "d", "哪一族真正决定分区")
psave(fig, "Z4d_族级判别份额")

key_vars = ["bio1", "bio12", "dem"]
key_lab = ["年均温（°C）", "年降水量（mm）", "DEM高程（m）"]
key_div = [10.0, 10.0, 1.0]
key_slug = ["年均温", "年降水", "DEM高程"]
key_letter = "efg"
for v, lab, dv, slug, lt in zip(key_vars, key_lab, key_div, key_slug, key_letter):
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    data = [PLC.loc[ZONE == z, v].values / dv for z in range(K)]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2))
    for b, z in zip(bp["boxes"], range(K)):
        b.set_facecolor(ZCOL[z]); b.set_alpha(0.8); b.set_edgecolor("black")
    ax.set_xticks(range(1, K + 1), [f"{ROMAN[z]} 区" for z in range(K)])
    ax.set_ylabel(lab)
    panel(ax, lt, f"{lab}的分区分布")
    psave(fig, f"Z4{lt}_{slug}箱线")

fig, ax = plt.subplots(figsize=(5.4, 4.2))
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
psave(fig, "Z4h_细化亚区热图")

# ==========================================================================
# 图 5 分区判别与主导因子（4 面板）
# ==========================================================================
CM, IMP = Z["cm"], Z["imp"]

fig, ax = plt.subplots(figsize=(5.0, 4.5))
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
psave(fig, "Z5a_混淆矩阵")

fig, ax = plt.subplots(figsize=(5.0, 3.8))
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
psave(fig, "Z5b_族级归因")

fig, ax = plt.subplots(figsize=(5.8, 5.2))
top = IMP.head(12).iloc[::-1]
ax.barh(np.arange(len(top)), top["importance_mean"],
        color=[FCOL[f] for f in top["family"]], height=0.66,
        xerr=top["importance_sd"], error_kw=dict(ecolor="#444", lw=0.9, capsize=2))
ax.set_yticks(np.arange(len(top)), [VAR_CN[v] for v in top["var"]], fontsize=9.5)
ax.set_xlabel("置换重要性（准确率下降）")
ax.set_xlim(0, top["importance_mean"].max() * 1.22)
panel(ax, "c", "前 12 位主导因子")
psave(fig, "Z5c_主导因子")

fig, ax = plt.subplots(figsize=(5.2, 4.4))
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
psave(fig, "Z5d_LDA判别空间")

# ==========================================================================
# 图 6 分区的生态含义（6 面板）
# ==========================================================================
fz = Z["fam_shap_zone"]

fig, ax = plt.subplots(figsize=(5.0, 3.9))
data = [PLC.loc[ZONE == z, "RGR_pool"].dropna().values for z in range(K)]
bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=1.2))
for b, z in zip(bp["boxes"], range(K)):
    b.set_facecolor(ZCOL[z]); b.set_alpha(0.8); b.set_edgecolor("black")
for i, d in enumerate(data, 1):
    # 抬升 2.5pt：标注底部与黑色中位线相切会被线穿过（与 39 号同步）
    ax.annotate(f"{np.median(d):.3f}", (i, np.median(d)), xytext=(0, 2.5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9.5, color="black")
ax.set_xticks(range(1, K + 1), [f"{ROMAN[z]} 区" for z in range(K)])
ax.set_ylabel(r"样地平均 RGR（yr$^{-1}$）")
panel(ax, "a", "各分区的林木生长速率")
psave(fig, "Z6a_分区生长速率")

fig, ax = plt.subplots(figsize=(5.4, 4.3))
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
ax.legend(loc="upper center", ncol=3, fontsize=9.5, frameon=False,
          handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
ax.set_yticks([0, 20, 40, 60, 80, 100])
panel(ax, "b", "驱动层级是否随分区变化")
psave(fig, "Z6b_SHAP族份额", fixed=True)

fig, ax = plt.subplots(figsize=(5.4, 4.0))
fg = Z["fg_by_zone"].reindex(range(K)).values * 100
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
# 左下角图例压住 Ⅰ 区柱体，移到左上（与 39 号同步）
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
panel(ax, "c", "功能群与林龄组成")
psave(fig, "Z6c_功能群与林龄")

fig, ax = plt.subplots(figsize=(5.0, 3.9))
for z in range(K):
    m = ZONE == z
    ax.scatter(PLC.loc[m, "bio1"].values / 10, PLC.loc[m, "RGR_pool"].values,
               s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
ax.set_xlabel("年均温（°C）"); ax.set_ylabel(r"样地平均 RGR（yr$^{-1}$）")
panel(ax, "d", "生长对气温的响应（气候宽裕性）")
psave(fig, "Z6d_气温响应")

fig, ax = plt.subplots(figsize=(5.0, 3.9))
for z in range(K):
    m = ZONE == z
    ax.scatter(PLC.loc[m, "bio12"].values / 10, PLC.loc[m, "RGR_pool"].values,
               s=13, c=ZCOL[z], alpha=0.5, linewidths=0)
ax.set_xlabel("年降水量（mm）"); ax.set_ylabel(r"样地平均 RGR（yr$^{-1}$）")
panel(ax, "e", "生长对降水的响应")
psave(fig, "Z6e_降水响应")

# (f) 分区一览：文本手工排表（同 39 号，不用 ax.table——其单元格 Rectangle
# 会膨胀撑爆 tight 包围盒）
fig, ax = plt.subplots(figsize=(7.6, 3.1))
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
psave(fig, "Z6f_分区一览")

# ==========================================================================
# 汇总
# ==========================================================================
print()
print("=== 单面板拆分核验汇总 ===")
import glob
pngs = sorted(glob.glob(os.path.join(OUTDIR, "*.png")))
pdfs = sorted(glob.glob(os.path.join(OUTDIR, "*.pdf")))
print(f"PNG {len(pngs)} 个，PDF {len(pdfs)} 个（应为 32 + 32）")
if PANEL_WARN:
    print(f"WARN/FAIL: {len(PANEL_WARN)} 条")
    for nm, kind, info in PANEL_WARN:
        print(f"  - {nm} [{kind}]: {info}")
    if any(k == "glyph" for _, k, _ in PANEL_WARN):
        raise SystemExit(1)
else:
    print("OK: 32 个面板全部通过字形与版式守卫")
print(f"产出: {OUTDIR}")
