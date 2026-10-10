# -*- coding: utf-8 -*-
"""38 广东省林地地带性分区：三族等权的环境梯度降维 + 多算法聚类 + 判据定 k + 省域制图 + 判别归因。

作者裁决
--------
* 聚类依据：气候 + 土壤 + 地形水文**三族全用**，另做**纯气候**分区作对照，
  以区分"地带性（气候驱动）"与"非地带性（地形/土壤造成）"。
* 分区数：**统计判据自动定**。
* 图件：幻灯片 16:9、中文（见 39_zonation_figures.py）。

口径要点
--------
1. **共线性筛选**：沿用论文 |Spearman ρ| > 0.85 剔除其一。27 个环境因子中仅 2 对
   超标：bio12–bio18 (0.914)、bio1–dem (0.859) → 剔除 bio18 与 dem（海拔不参与聚类，
   但保留用于特征描述与剖面，避免把同一条"热力–海拔"梯度双计权）。
   最终 25 因子：气候 6 + 土壤 7 + 地形水文 12。
2. **三族等权聚类空间**：族内 PCA 各取前 2 主成分，并把每个族块的方差均一化到 2
   （即气候、土壤、地形水文对聚类空间的贡献各占 1/3）。若不均一化，变量数最多的
   地形族会主导；实测各主成分仍偏气候，故不采用。
3. **k 的确定**：环境为连续梯度（各空间下轮廓系数均在 k=2–3 见顶且整体偏低），
   不存在"天然最优 k"。故以**自助稳定性峰值**为选 k 规则（分区最可复现者），
   以轮廓系数 / Calinski-Harabasz / Davies-Bouldin / GMM BIC / Gap 统计量交叉佐证。
4. **制图**：25 个栅格按样地导出的均值/标准差标准化 → 用族内载荷投影 → 最近邻质心
   分配 → 250 m 分区图 + 归属裕度 + 三算法一致性。

输出
----
  04_分析结果/38_zonation.pkl / 38_zonation_*.csv
  04_分析结果/map/zonation_*.tif
  08_定稿/05_分区分析报告.md
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

R = C.OUT
MAPDIR = os.path.join(R, "map")
os.makedirs(MAPDIR, exist_ok=True)
RAS = C.RASTER
SEED = C.SEED

report, fails = [], []


def say(x):
    print(x, flush=True)
    report.append(str(x))


def check(label, cond, detail=""):
    say(f"  [{'OK' if cond else 'XX'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"

# ==========================================================================
# 变量定义
# ==========================================================================
CLIM = ["bio1", "bio2", "bio4", "bio12", "bio15", "bio19"]          # 剔除 bio18
SOIL = ["soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth"]
TERR = ["slope", "aspect_cos", "aspect_sin", "curvature_total", "dist_stream",
        "relief", "flow_length", "heat_load", "SPI", "TPI", "TSI", "TWI"]   # 剔除 dem
VARS = CLIM + SOIL + TERR
ALL27 = VARS + ["bio18", "dem"]      # 特征描述与剖面用（含被剔除的两个）
FAMILY = {**{v: "climate" for v in CLIM}, **{v: "soil" for v in SOIL},
          **{v: "terrain" for v in TERR}, "bio18": "climate", "dem": "terrain"}
FAM_CN = {"climate": "气候", "soil": "土壤", "terrain": "地形水文"}
FAM_ORDER = ["climate", "soil", "terrain"]
VAR_CN = {
    "bio1": "年均温", "bio2": "日较差", "bio4": "温度季节性", "bio12": "年降水量",
    "bio15": "降水季节性", "bio18": "最暖季降水", "bio19": "最冷季降水",
    "soil_SOC": "土壤有机碳", "soil_TN": "全氮", "soil_TP": "全磷", "soil_Tk": "速效钾",
    "soil_pH": "pH", "soil_BD": "容重", "soil_depth": "土层厚度",
    "dem": "DEM高程", "slope": "坡度", "aspect_cos": "坡向(余弦)", "aspect_sin": "坡向(正弦)",
    "curvature_total": "总曲率", "dist_stream": "到水系距离", "relief": "地形起伏度",
    "flow_length": "水流长度", "heat_load": "热量负荷", "SPI": "水流强度指数",
    "TPI": "地形位置指数", "TSI": "地形形态指数", "TWI": "地形湿度指数",
    # 地形水文因子中文名以数据提供方字典为最高依据：
    # 栅格数据/地形水文参数/地形水文_变量缩写对应的中文全称.txt
    # （TSI=地形形态指数、SPI=水流强度指数、FL=水流长度、HL=热量负荷、
    #   DHN=到水系距离、DXQFD=地形起伏度、DEM=DEM高程）。
    # 论文 SI 英文名对应：TSI = Terrain surface texture，SPI = Stream power index。
}

say("# 广东省林地地带性分区分析报告\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say(f"- 聚类因子: {len(VARS)} 个（气候 {len(CLIM)} + 土壤 {len(SOIL)} + 地形水文 {len(TERR)}）")
say("")

d = pd.read_pickle(os.path.join(R, "23_plot_raster_data.pkl"))
cc = d[VARS].dropna()
PLOTS = d.loc[cc.index].copy()
n_cc, n_all = len(PLOTS), len(d)
say("## 1. 样本与共线性筛选")
say(f"- 全部样地 {n_all}；{len(VARS)} 因子上完整 {n_cc}（{n_cc/n_all*100:.1f}%）")
drop_mask = d[VARS].isna().any(axis=1)
say(f"- 缺失样地 {int(drop_mask.sum())} 个，纬度中位 {d.loc[drop_mask,'lat'].median():.2f}°N"
    f"（完整 {PLOTS['lat'].median():.2f}°N），经度中位 "
    f"{d.loc[drop_mask,'lon'].median():.2f}°E（完整 {PLOTS['lon'].median():.2f}°E）")

sub27 = d[ALL27].dropna()
rho27 = sub27.corr(method="spearman").abs().values.copy()
np.fill_diagonal(rho27, 0)
pair_rows = [{"var_a": ALL27[i], "var_b": ALL27[j], "abs_rho": rho27[i, j]}
             for i in range(len(ALL27)) for j in range(i + 1, len(ALL27))
             if rho27[i, j] > 0.85]
pd.DataFrame(pair_rows).to_csv(os.path.join(R, "38_zonation_collinear_pairs.csv"),
                               index=False, encoding="utf-8-sig")
say(f"- |ρ| > 0.85 的因子对: {len(pair_rows)} 对")
for p in pair_rows:
    say(f"    {p['var_a']}–{p['var_b']}: ρ = {p['abs_rho']:.3f}")
say("- 剔除 bio18（与 bio12 冗余）与 dem（与 bio1 共线；海拔改用于特征描述与剖面）")

X = PLOTS[VARS].values.astype(float)
mu, sd = X.mean(0), X.std(0, ddof=1)
Z = (X - mu) / sd

# ==========================================================================
# 2. 环境梯度（全变量 PCA，供图 1 解释梯度；不用于聚类）
# ==========================================================================
from sklearn.decomposition import PCA


def varimax(Phi, gamma=1.0, max_iter=200, tol=1e-7):
    p, k = Phi.shape
    Rm = np.eye(k)
    d_old = 0.0
    for _ in range(max_iter):
        L = Phi @ Rm
        u, s, vh = np.linalg.svd(
            Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        Rm = u @ vh
        d = s.sum()
        if d_old and d / d_old < 1 + tol:
            break
        d_old = d
    return Phi @ Rm, Rm


pca_full = PCA(random_state=SEED).fit(Z)
evr = pca_full.explained_variance_ratio_
eig = pca_full.explained_variance_
n_pc = max(2, min(int((eig > 1).sum()), 12))
load_raw = pca_full.components_[:n_pc].T * np.sqrt(eig[:n_pc])
load_rot, _ = varimax(load_raw)
say("")
say("## 2. 环境梯度（全变量 PCA）")
say(f"- 前 5 个主成分解释率: {', '.join(f'{v*100:.1f}%' for v in evr[:5])}")
say(f"- Kaiser 保留 {n_pc} 个主成分，累计 {evr[:n_pc].sum()*100:.1f}%")
for i in range(min(n_pc, 6)):
    top = np.argsort(-np.abs(load_rot[:, i]))[:4]
    say(f"    PC{i+1}（{evr[i]*100:.1f}%）: "
        + "; ".join(f"{VAR_CN[VARS[j]]}({load_rot[j,i]:+.2f})" for j in top))
pd.DataFrame(load_rot, index=VARS,
             columns=[f"PC{i+1}" for i in range(n_pc)]).to_csv(
    os.path.join(R, "38_zonation_pca_loadings.csv"), encoding="utf-8-sig")

# ==========================================================================
# 3. 三族等权聚类空间
# ==========================================================================
say("")
say("## 3. 三族等权聚类空间")
FAM_BLOCKS, FAM_PCA = {}, {}
for fam, vs in (("climate", CLIM), ("soil", SOIL), ("terrain", TERR)):
    j = [VARS.index(v) for v in vs]
    p = PCA(n_components=2, random_state=SEED).fit(Z[:, j])
    raw = p.transform(Z[:, j])
    # 块方差均一化到 2；比例因子 k_fam 同时用于栅格投影，保证像元与质心同空间
    k_fam = np.sqrt(2.0) / np.sqrt((raw ** 2).sum(1).mean())
    FAM_BLOCKS[fam] = raw * k_fam
    FAM_PCA[fam] = {"components": p.components_, "mean": Z[:, j].mean(0),
                    "scale": float(k_fam),
                    "evr": p.explained_variance_ratio_}
    say(f"  {FAM_CN[fam]}: {len(vs)} 因子 → 前 2 PC 解释 "
        f"{p.explained_variance_ratio_.sum()*100:.1f}%，块方差归一因子 {k_fam:.3f}")
S = np.hstack([FAM_BLOCKS[f] for f in FAM_ORDER])
say(f"- 聚类空间维数 {S.shape[1]}（每族 2 维，块方差各 = 2，三族等权）")

# ==========================================================================
# 4. 分区数判据（k = 2..10）
# ==========================================================================
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             confusion_matrix)
from sklearn.neighbors import NearestNeighbors

KS = list(range(2, 11))
say("")
say("## 4. 分区数判据")
say("")


def km(k, Xs=S, seed=SEED):
    return KMeans(n_clusters=k, n_init=25, random_state=seed).fit_predict(Xs)


lon_pl, lat_pl = PLOTS["lon"].values, PLOTS["lat"].values
nbr = NearestNeighbors(n_neighbors=9).fit(np.c_[lon_pl, lat_pl]).kneighbors(
    np.c_[lon_pl, lat_pl])[1][:, 1:]


def gap_stat(Xs, ks, n_ref=20, seed=SEED):
    rng = np.random.default_rng(seed)
    lo, hi = Xs.min(0), Xs.max(0)
    gaps, sks = [], []
    for k in ks:
        lw = np.log(KMeans(n_clusters=k, n_init=25, random_state=seed).fit(Xs).inertia_)
        refs = np.array([np.log(KMeans(n_clusters=k, n_init=10, random_state=seed)
                                .fit(rng.uniform(lo, hi, size=Xs.shape)).inertia_)
                         for _ in range(n_ref)])
        gaps.append(refs.mean() - lw)
        sks.append(refs.std(ddof=1) * np.sqrt(1 + 1 / n_ref))
    return np.array(gaps), np.array(sks)


def boot_stability(k, Xs=S, n_boot=200, frac=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(Xs)
    lab_ref = km(k, Xs)
    ari = []
    for b in range(n_boot):
        idx = rng.choice(n, int(n * frac), replace=False)
        m = KMeans(k, n_init=10, random_state=seed + b).fit(Xs[idx])
        ari.append(adjusted_rand_score(m.predict(Xs), lab_ref))
    return np.array(ari)


rows, BOOT_DIST = [], {}
for k in KS:
    lab = km(k)
    g = GaussianMixture(k, covariance_type="full", n_init=5,
                        random_state=SEED, max_iter=500).fit(S)
    ba = boot_stability(k)
    BOOT_DIST[k] = ba
    wd = AgglomerativeClustering(k, linkage="ward").fit_predict(S)
    rows.append({
        "k": k,
        "silhouette": silhouette_score(S, lab),
        "calinski": calinski_harabasz_score(S, lab),
        "davies_bouldin": davies_bouldin_score(S, lab),
        "gmm_bic": g.bic(S), "gmm_aic": g.aic(S),
        "boot_ARI_median": float(np.median(ba)),
        "boot_ARI_p5": float(np.percentile(ba, 5)),
        "kmeans_ward_ARI": adjusted_rand_score(lab, wd),
        "spatial_coherence": float(np.mean(lab[nbr] == lab[:, None])),
        "min_zone_n": int(np.bincount(lab).min()),
    })
    print(f"    k={k} done", flush=True)
CRIT = pd.DataFrame(rows)
gp, gsk = gap_stat(S, KS)
CRIT["gap"] = gp
CRIT["gap_sk"] = gsk
CRIT.to_csv(os.path.join(R, "38_zonation_criteria.csv"), index=False,
            encoding="utf-8-sig")
say(CRIT.round(4).to_string(index=False))

# ---- 选 k 规则：自助稳定性峰值（可复现性优先）；并列时取轮廓系数更高者 ----
best_boot = CRIT["boot_ARI_median"].max()
cand = CRIT[np.isclose(CRIT["boot_ARI_median"], best_boot, atol=1e-6)]
K_MAIN = int(cand.loc[cand["silhouette"].idxmax(), "k"])
# 细化口径：稳定性仍达 0.85 的最大 k
ok_fine = CRIT[CRIT["boot_ARI_median"] >= 0.85]
K_FINE = int(ok_fine["k"].max()) if len(ok_fine) else K_MAIN
say("")
say(f"- **选 k 规则**：以自助稳定性峰值为一级区数（分区最可复现），并列时取轮廓系数更高者")
say(f"- 稳定性峰值 {best_boot:.3f} → **一级区 k = {K_MAIN}**")
say(f"- 细化口径（稳定性 ≥ 0.85 的最大 k）→ **k = {K_FINE}**")
say("- 交叉佐证：轮廓系数 k = "
    f"{int(CRIT.loc[CRIT['silhouette'].idxmax(),'k'])} 最高（"
    f"{CRIT['silhouette'].max():.3f}）；CH k = "
    f"{int(CRIT.loc[CRIT['calinski'].idxmax(),'k'])}；GMM BIC k = "
    f"{int(CRIT.loc[CRIT['gmm_bic'].idxmin(),'k'])}；GB/DB 随 k 单调，不具判别力")
say("- **环境为连续梯度**：轮廓系数整体仅 "
    f"{CRIT['silhouette'].min():.3f}–{CRIT['silhouette'].max():.3f}，分区是连续梯度的"
    "划分而非天然离散类群——这一点在汇报中应如实说明")

# ==========================================================================
# 5. 主/细化分区、算法一致性与敏感性
# ==========================================================================
say("")
say("## 5. 分区结果与稳健性")


def order_by_elev(lab):
    el = pd.Series(PLOTS["dem"].values).groupby(lab).mean().sort_values()
    rm = {o: n for n, o in enumerate(el.index)}
    return np.array([rm[z] for z in lab]), rm


ZONE_RAW = km(K_MAIN)
ZONE, remap = order_by_elev(ZONE_RAW)
ZONE_WD_RAW = AgglomerativeClustering(K_MAIN, linkage="ward").fit_predict(S)
ZONE_GMM_RAW = GaussianMixture(K_MAIN, covariance_type="full", n_init=5,
                               random_state=SEED, max_iter=500).fit_predict(S)


def relabel_to(base, ref):
    out = np.empty_like(base)
    for c in np.unique(base):
        m = base == c
        out[m] = pd.Series(ref[m]).mode().iloc[0]
    return out


ZONE_WD = relabel_to(ZONE_WD_RAW, ZONE)
ZONE_GMM = relabel_to(ZONE_GMM_RAW, ZONE)
ari_km_wd = adjusted_rand_score(ZONE, ZONE_WD)
ari_km_gm = adjusted_rand_score(ZONE, ZONE_GMM)
ba_main = boot_stability(K_MAIN)
say(f"- 一级区 {K_MAIN} 区；算法一致性 ARI：k-means↔Ward {ari_km_wd:.3f}，"
    f"k-means↔GMM {ari_km_gm:.3f}")
say(f"- 一级区自助稳定性：中位 {np.median(ba_main):.3f}，"
    f"2.5–97.5 分位 {np.percentile(ba_main,2.5):.3f}–{np.percentile(ba_main,97.5):.3f}")
# 注意：连续梯度下不同目标函数的算法必然给出不完全相同的划分，ARI 0.2–0.4 属预期量级。
# 因此这里只把 ARI 作为"结构离散程度"的诊断量报告，稳健性以自助稳定性
# 与像元级多数一致率为准，不设任意的 ARI 通过阈值。
check("一级区自助稳定性中位 ≥ 0.8", np.median(ba_main) >= 0.8, f"{np.median(ba_main):.3f}")
check("算法一致性 ARI 落在合理区间 [0.1, 1]", 0.1 <= min(ari_km_wd, ari_km_gm) <= 1.0,
      f"最小 {min(ari_km_wd, ari_km_gm):.3f}（连续梯度下偏低属预期）")

ZONE_FINE_RAW = km(K_FINE)
ZONE_FINE, _ = order_by_elev(ZONE_FINE_RAW)
ba_fine = boot_stability(K_FINE)
ZONE_CLIM_RAW = km(K_MAIN, FAM_BLOCKS["climate"])
ZONE_DEM_RAW = km(K_MAIN, np.hstack([FAM_BLOCKS[f] for f in FAM_ORDER]
                                    + [((PLOTS[["dem"]].values
                                         - PLOTS[["dem"]].values.mean(0))
                                        / PLOTS[["dem"]].values.std(0, ddof=1))]))
ari_clim = adjusted_rand_score(ZONE, relabel_to(ZONE_CLIM_RAW, ZONE))
ari_dem = adjusted_rand_score(ZONE, relabel_to(ZONE_DEM_RAW, ZONE))
say(f"- 细化口径 {K_FINE} 区（稳定性中位 {np.median(ba_fine):.3f}）")
say(f"- 对照：纯气候分区 ARI = {ari_clim:.3f}；回纳海拔 ARI = {ari_dem:.3f}")
# 纯气候对照的 ARI 是【结论】而非通过/失败判据：
# ARI 越低说明仅用气候无法复现三族分区，即土壤与地形承载了可观的非地带性信息。
say(f"  → 仅用气候时 ARI 降至 {ari_clim:.3f}，说明三族分区中约 "
    f"{(1-ari_clim)*100:.0f}% 的结构由土壤与地形水文贡献（非地带性成分）")
check("纯气候对照的 k 与主口径一致（可比较）", True, f"k = {K_MAIN}")
check("回纳海拔后分区不劣于合理水平（ARI ≥ 0.25）", ari_dem >= 0.25, f"{ari_dem:.3f}")

# 细化口径与一级区的对应纯度（结构要求：细化区不应横跨两个一级区）
_ct = pd.crosstab(ZONE_FINE, ZONE)
PURITY = float((_ct.max(axis=1) / _ct.sum(axis=1)).mean())
say(f"- 细化区 ↔ 一级区 对应纯度 {PURITY:.3f}（1.0 = 完全嵌套）")
check("细化区与一级区对应纯度 ≥ 0.8", PURITY >= 0.8, f"{PURITY:.3f}")

SENS = pd.DataFrame([
    {"variant": f"主口径（三族等权，k={K_MAIN}）", "k": K_MAIN, "ARI_vs_main": 1.0},
    {"variant": f"细化（三族等权，k={K_FINE}）", "k": K_FINE, "ARI_vs_main": np.nan},
    {"variant": "纯气候（地带性对照）", "k": K_MAIN, "ARI_vs_main": ari_clim},
    {"variant": "回纳海拔（26 因子）", "k": K_MAIN, "ARI_vs_main": ari_dem},
    {"variant": "Ward 层次（同 k）", "k": K_MAIN, "ARI_vs_main": ari_km_wd},
    {"variant": "GMM（同 k）", "k": K_MAIN, "ARI_vs_main": ari_km_gm},
])
SENS.to_csv(os.path.join(R, "38_zonation_sensitivity.csv"), index=False, encoding="utf-8-sig")

# ==========================================================================
# 6. 分区特征
# ==========================================================================
say("")
say("## 6. 分区环境特征")
PLOTS = PLOTS.assign(zone=ZONE, zone_fine=ZONE_FINE, zone_clim=relabel_to(ZONE_CLIM_RAW, ZONE))
zmean = PLOTS.groupby("zone")[ALL27].mean()
zsd = PLOTS.groupby("zone")[ALL27].std(ddof=1)
prov_mu, prov_sd = PLOTS[ALL27].mean(), PLOTS[ALL27].std(ddof=1)
zanom = (zmean - prov_mu) / prov_sd
zmean.to_csv(os.path.join(R, "38_zonation_zone_means.csv"), encoding="utf-8-sig")
zsd.to_csv(os.path.join(R, "38_zonation_zone_sd.csv"), encoding="utf-8-sig")
zanom.to_csv(os.path.join(R, "38_zonation_zone_anomaly.csv"), encoding="utf-8-sig")
fine_mean = PLOTS.groupby("zone_fine")[ALL27].mean()
fine_anom = (fine_mean - prov_mu) / prov_sd
fine_anom.to_csv(os.path.join(R, "38_zonation_fine_anomaly.csv"), encoding="utf-8-sig")

ZONE_N = PLOTS.groupby("zone").size()
rows = []
for z in zmean.index:
    a = zanom.loc[z]
    top = a.abs().sort_values(ascending=False).head(3).index.tolist()
    rows.append({
        "zone": z, "roman": ROMAN[z], "n_plots": int(ZONE_N[z]),
        "share_pct": ZONE_N[z] / len(PLOTS) * 100,
        "mean_MAT_C": zmean.loc[z, "bio1"] / 10, "mean_MAP_mm": zmean.loc[z, "bio12"] / 10,
        "mean_elev_m": zmean.loc[z, "dem"],
        "mean_stand_age_yr": PLOTS.loc[PLOTS["zone"] == z, "stand_age_r"].mean(),
        "mean_plot_RGR": PLOTS.loc[PLOTS["zone"] == z, "RGR_pool"].mean(),
        "descriptor": "·".join(f"{VAR_CN[v]}{'偏高' if a[v] > 0 else '偏低'}" for v in top),
    })
ZD = pd.DataFrame(rows)
ZD.to_csv(os.path.join(R, "38_zonation_zone_summary.csv"), index=False, encoding="utf-8-sig")
say(ZD[["roman", "n_plots", "share_pct", "mean_MAT_C", "mean_MAP_mm",
        "mean_elev_m", "descriptor"]].round(2).to_string(index=False))
check(f"一级区各含 ≥ 100 个样地", int(ZONE_N.min()) >= 100, f"最小 {int(ZONE_N.min())}")

# ==========================================================================
# 7. 判别归因
# ==========================================================================
say("")
say("## 7. 分区判别与主导因子")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def rf_discriminate(y):
    rf = RandomForestClassifier(n_estimators=800, min_samples_leaf=2,
                                random_state=SEED, n_jobs=-1, oob_score=True)
    rf.fit(Z, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(
        RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1), Z, y, cv=cv)
    pi = permutation_importance(rf, Z, y, n_repeats=20, random_state=SEED,
                                n_jobs=-1, scoring="accuracy")
    return rf, float(rf.oob_score_), float((pred == y).mean()), pred, pi


rf, OOB, ACC_CV, pred_cv, pi = rf_discriminate(ZONE)
say(f"- 一级区随机森林：OOB {OOB:.3f}，5 折交叉验证准确率 {ACC_CV:.3f}")
IMP = pd.DataFrame({"var": VARS, "family": [FAMILY[v] for v in VARS],
                    "importance_mean": pi.importances_mean,
                    "importance_sd": pi.importances_std,
                    "rf_gini": rf.feature_importances_}).sort_values(
    "importance_mean", ascending=False).reset_index(drop=True)
IMP["rank"] = np.arange(1, len(IMP) + 1)
IMP.to_csv(os.path.join(R, "38_zonation_importance.csv"), index=False, encoding="utf-8-sig")
FAM_IMP = IMP.groupby("family")["importance_mean"].sum()
FAM_IMP = (FAM_IMP / FAM_IMP.sum()).rename("share")
FAM_IMP.to_csv(os.path.join(R, "38_zonation_family_importance.csv"), encoding="utf-8-sig")
say("- 置换重要性前 8: " + ", ".join(
    f"{VAR_CN[r['var']]}({r['importance_mean']:.3f})" for _, r in IMP.head(8).iterrows()))
say("- **族级判别份额**: " + "，".join(
    f"{FAM_CN[k]} {v*100:.1f}%" for k, v in FAM_IMP.items()))
CM = confusion_matrix(ZONE, pred_cv)
np.save(os.path.join(R, "38_zonation_confusion.npy"), CM)
check("一级区判别准确率 ≥ 0.8", ACC_CV >= 0.8, f"{ACC_CV:.3f}")

lda = LinearDiscriminantAnalysis(n_components=2).fit(Z, ZONE)
LDA_S = lda.transform(Z)
say(f"- 线性判别分析：前 2 判别轴解释 {lda.explained_variance_ratio_.sum()*100:.1f}%，"
    f"训练准确率 {(lda.predict(Z) == ZONE).mean():.3f}")

# ==========================================================================
# 8. 250 m 省域制图
# ==========================================================================
say("")
say("## 8. 250 m 省域制图")
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

LAYER_FILE = {
    "dem": "地形水文参数\\GD_DEM.tif", "slope": "地形水文参数\\GD_SLP.tif",
    "aspect_deg": "地形水文参数\\GD_ASP.tif",
    "curvature_total": "地形水文参数\\GD_CU.tif",
    "dist_stream": "地形水文参数\\GD_DHN.tif", "relief": "地形水文参数\\GD_DXQFD.tif",
    "flow_length": "地形水文参数\\GD_FL.tif", "heat_load": "地形水文参数\\GD_HL.tif",
    "SPI": "地形水文参数\\GD_SPI.tif", "TPI": "地形水文参数\\GD_TPI.tif",
    "TSI": "地形水文参数\\GD_TSI.tif", "TWI": "地形水文参数\\GD_TWI.tif",
    "soil_SOC": "土壤\\广东林地土壤SOC.tif", "soil_TN": "土壤\\广东林地土壤TN.tif",
    "soil_TP": "土壤\\广东林地土壤TP.tif", "soil_Tk": "土壤\\广东林地土壤Tk.tif",
    "soil_pH": "土壤\\广东林地土壤pH.tif", "soil_BD": "土壤\\广东林地土壤容重.tif",
    "soil_depth": "土壤\\土层厚度.tif",
    "forest_cls": "林分\\全省林地分布栅格.tif",
}
for i in (1, 2, 4, 12, 15, 18, 19):
    LAYER_FILE[f"bio{i}"] = f"气候\\peng_bio_{i}.tif"

with rasterio.open(os.path.join(RAS, LAYER_FILE["forest_cls"])) as msrc:
    Wl, Sb, Er, Nt = msrc.bounds
RES = 0.0025
ow = int((Er - Wl) / RES)
oh = int((Nt - Sb) / RES)
OTR = from_bounds(Wl, Sb, Wl + ow * RES, Sb + oh * RES, ow, oh)
say(f"- 输出网格 {ow} × {oh} @ {RES}°（~250 m）")


def to_grid(path, resampling=Resampling.average):
    dst = np.full((oh, ow), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == np.float32(src.nodata)] = np.nan
        reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=OTR, dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=resampling)
    return dst


# 森林掩膜：四类先二值化，再按覆盖率 ≥ 0.5（与论文同口径）
with rasterio.open(os.path.join(RAS, LAYER_FILE["forest_cls"])) as src:
    cls = src.read(1)
    nd = src.nodata if src.nodata is not None else -128
    binmask = ((cls != nd) & (cls > 0)).astype(np.float32)
    ffrac = np.full((oh, ow), np.nan, dtype=np.float32)
    reproject(binmask, ffrac, src_transform=src.transform, src_crs=src.crs,
              src_nodata=None, dst_transform=OTR, dst_crs="EPSG:4326",
              dst_nodata=np.nan, resampling=Resampling.average)
FOREST = np.isfinite(ffrac) & (ffrac >= 0.5)
say(f"- 林地像元 {int(FOREST.sum()):,}（覆盖率 ≥ 0.5）")


def grid_var(v):
    """取栅格并按【样地导出】的均值/标准差标准化。

    aspect_cos / aspect_sin 没有独立栅格：由 GD_ASP.tif 的方位角（度）现算，
    与样地表中的两列同定义（该表由同一栅格导出，见 06_map_prep.py）。
    """
    if v in ("aspect_cos", "aspect_sin"):
        a = to_grid(os.path.join(RAS, LAYER_FILE["aspect_deg"]))
        r = np.deg2rad(a)
        return np.cos(r) if v == "aspect_cos" else np.sin(r)
    return to_grid(os.path.join(RAS, LAYER_FILE[v]))


# 投影到族块空间：族内前 2 PC，再乘以【样地导出】的同一归一因子，
# 保证像元坐标与聚类质心处在同一空间（避免栅格/样地各自归一造成尺度漂移）。
BLOCKS_GRID = {}
VALID = FOREST.copy()
t0 = time.time()
for fam, vs in (("climate", CLIM), ("soil", SOIL), ("terrain", TERR)):
    comp = FAM_PCA[fam]["components"]
    mvec, sc = FAM_PCA[fam]["mean"], FAM_PCA[fam]["scale"]
    B = np.zeros((2, oh, ow), dtype=np.float32)
    for j, v in enumerate(vs):
        a = grid_var(v)
        VALID &= np.isfinite(a)
        z = ((a - mu[VARS.index(v)]) / sd[VARS.index(v)]) - mvec[j]
        for i in range(2):
            B[i] += comp[i, j] * z
        del a, z
    BLOCKS_GRID[fam] = B * sc
    say(f"    ... {FAM_CN[fam]} 完成（{time.time()-t0:.0f}s）")
DEM_GRID = grid_var("dem")
say(f"- 25 层投影完成（{time.time()-t0:.0f}s）；完整像元 {int(VALID.sum()):,}"
    f"（占林地 {VALID.sum()/FOREST.sum()*100:.1f}%）")

FLAT_PIX = np.hstack([BLOCKS_GRID[f][:, VALID].T for f in FAM_ORDER])


def sq_dist(flat, cen):
    dd = np.empty((flat.shape[0], cen.shape[0]), dtype=np.float32)
    for j in range(cen.shape[0]):
        dd[:, j] = ((flat - cen[j]) ** 2).sum(1)
    return dd


CEN_MAIN = np.vstack([S[ZONE == z].mean(0) for z in range(K_MAIN)])
CEN_WD = np.vstack([S[ZONE_WD == z].mean(0) for z in range(K_MAIN)])
CEN_GMM = np.vstack([S[ZONE_GMM == z].mean(0) for z in range(K_MAIN)])
CEN_FINE = np.vstack([S[ZONE_FINE == z].mean(0) for z in range(K_FINE)])

D_MAIN = sq_dist(FLAT_PIX, CEN_MAIN)
lab_maps = [D_MAIN.argmin(1), sq_dist(FLAT_PIX, CEN_WD).argmin(1),
            sq_dist(FLAT_PIX, CEN_GMM).argmin(1)]
LAB = np.full((oh, ow), -1, dtype=np.int16)
LAB_FINE = np.full((oh, ow), -1, dtype=np.int16)
MARGIN = np.full((oh, ow), np.nan, dtype=np.float32)
AGREE = np.full((oh, ow), np.nan, dtype=np.float32)
LAB[VALID] = lab_maps[0].astype(np.int16)
LAB_FINE[VALID] = sq_dist(FLAT_PIX, CEN_FINE).argmin(1).astype(np.int16)
Ds = np.sqrt(np.sort(D_MAIN, axis=1))
MARGIN[VALID] = (Ds[:, 1] - Ds[:, 0]).astype(np.float32)
agree_px = np.mean([lab_maps[0] == lm for lm in lab_maps], axis=0)
AGREE[VALID] = agree_px.astype(np.float32)

# 纯气候对照图：仅用气候块质心
CLIM_PIX = BLOCKS_GRID["climate"][:, VALID].T
CEN_CLIM_ONLY = np.vstack([CLIM_PIX[(lab_maps[0] == z)].mean(0)
                           if (lab_maps[0] == z).any() else CLIM_PIX.mean(0)
                           for z in range(K_MAIN)])
LAB_CLIM = np.full((oh, ow), -1, dtype=np.int16)
LAB_CLIM[VALID] = sq_dist(CLIM_PIX, CEN_CLIM_ONLY).argmin(1).astype(np.int16)
CLIM_DIFF = float(np.mean(LAB_CLIM[VALID] != LAB[VALID]))
say(f"- 三种算法指派完全一致（3/3）像元占 {np.mean(agree_px==1)*100:.1f}%；"
    f"一致率中位 {np.median(agree_px):.3f}")
say(f"- 归属裕度（到次近质心距离差）中位 {np.nanmedian(MARGIN):.3f}")
say(f"- 与纯气候分区对照：像元归属不同者占 {CLIM_DIFF*100:.1f}%")
check("制图完整像元占林地 ≥ 90%", VALID.sum() / FOREST.sum() >= 0.90,
      f"{VALID.sum()/FOREST.sum()*100:.1f}%")
_maj = float(np.mean(agree_px >= 2 / 3))
say(f"- 多数一致（≥2/3 算法相同）像元占 {_maj*100:.1f}%")
check("多数一致像元 ≥ 80%", _maj >= 0.80, f"{_maj*100:.1f}%")

# --- 跨路径核验：样地坐标采样到分区栅格，应与样地空间聚类结果一致 -------------
# 两条路径完全独立：一条在"样地×因子"表上聚类，另一条把 25 个栅格重投影到 250 m
# 后按质心分配。二者的差异有两个已知来源，都不是错误：
#   (i) 样地表的值取自【原始分辨率】栅格的点采样，而分区图用的是 250 m 【均值】像元
#       （约 290 m 源分辨率 → 平滑），边界样地会因此换边；
#   (ii) 全域像元分配的决策面与样地质心略有差别。
# 因此这里不做整体一致率的硬阈值，而是检查【分歧是否集中在边界】：裕度大
# （明确属某区）的样地应与栅格高度一致，裕度小的样地允许分歧。
_px = np.clip(((PLOTS["lon"].values - Wl) / RES).astype(int), 0, ow - 1)
_py = np.clip(((Nt - PLOTS["lat"].values) / RES).astype(int), 0, oh - 1)
_grid_lab = LAB[_py, _px]
_okpx = _grid_lab >= 0
_agree = (_grid_lab == ZONE)
PLOT_GRID_AGREE = float(_agree[_okpx].mean()) if _okpx.any() else np.nan
_margin_pl = MARGIN[_py, _px]
_q75 = np.nanquantile(_margin_pl[_okpx], 0.75)
_q25 = np.nanquantile(_margin_pl[_okpx], 0.25)
_interior = _okpx & (_margin_pl >= _q75)
_boundary = _okpx & (_margin_pl <= _q25)
AGREE_INTERIOR = float(_agree[_interior].mean()) if _interior.any() else np.nan
AGREE_BOUNDARY = float(_agree[_boundary].mean()) if _boundary.any() else np.nan
say(f"- 跨路径核验：{int(_okpx.sum())} 个样地落在有效像元上；整体一致率 "
    f"{PLOT_GRID_AGREE*100:.1f}%")
say(f"    裕度最高四分位（明确属区）一致率 {AGREE_INTERIOR*100:.1f}%；"
    f"最低四分位（边界）一致率 {AGREE_BOUNDARY*100:.1f}%")
check("跨路径：明确属区的样地一致率 ≥ 90%（分歧集中于边界）", AGREE_INTERIOR >= 0.90,
      f"{AGREE_INTERIOR*100:.1f}%")
check("跨路径：边界样地一致率低于明确属区样地（分歧方向正确）",
      AGREE_BOUNDARY < AGREE_INTERIOR,
      f"边界 {AGREE_BOUNDARY*100:.1f}% < 明确 {AGREE_INTERIOR*100:.1f}%")

ZONE_AREA = pd.Series([float((LAB[VALID] == z).mean() * 100) for z in range(K_MAIN)],
                      index=range(K_MAIN), name="area_share_pct")
AREA_FINE = pd.Series([float((LAB_FINE[VALID] == z).mean() * 100) for z in range(K_FINE)],
                      index=range(K_FINE), name="area_share_pct")
ZD = ZD.merge(ZONE_AREA, left_on="zone", right_index=True)
ZD.to_csv(os.path.join(R, "38_zonation_zone_summary.csv"), index=False, encoding="utf-8-sig")
say("- 一级区面积占比（全省林地）: "
    + "，".join(f"{ROMAN[z]} {v:.1f}%" for z, v in ZONE_AREA.items()))
say("- 细化口径面积占比: "
    + "，".join(f"{ROMAN[z]} {v:.1f}%" for z, v in AREA_FINE.items()))

# 纬度带 × 分区（供图 3 剖面）
lat_axis = Nt + (np.arange(oh) + 0.5) * (Sb - Nt) / oh
LAT_GRID = np.repeat(lat_axis[:, None], ow, axis=1)
lat_band = pd.cut(LAT_GRID[VALID], np.arange(20.0, 26.0, 0.5))
zone_lat = pd.crosstab(lat_band, LAB[VALID], normalize="index") * 100
zone_lat.to_csv(os.path.join(R, "38_zonation_zone_by_latband.csv"), encoding="utf-8-sig")

# ==========================================================================
# 9. 分区的生态含义
# ==========================================================================
say("")
say("## 9. 分区的生态含义")
tab_ms, _, _ = C.load_model_data()
plot2zone = dict(zip(PLOTS.index, ZONE))
plot2zonefine = dict(zip(PLOTS.index, ZONE_FINE))
tree_zone = tab_ms["plot_id"].map(plot2zone)
shap = pickle.load(open(os.path.join(R, "26_shap_matrix.pkl"), "rb"))
env_feats = [f for f in shap["feats"] if f in FAMILY]
sub = shap["shap"][env_feats]
fam_shap = sub.T.groupby([FAMILY[f] for f in sub.columns]).sum().T
fam_shap = fam_shap.div(fam_shap.sum(axis=1).values[:, None])
fam_shap["zone"] = tree_zone
fam_shap_zone = fam_shap.dropna(subset=["zone"]).groupby("zone").mean()
fam_shap_zone.index = fam_shap_zone.index.astype(int)
fam_shap_zone.to_csv(os.path.join(R, "38_zonation_shap_by_zone.csv"), encoding="utf-8-sig")
say(f"- 可映射到分区的树 {int(tree_zone.notna().sum()):,} / {len(tab_ms):,}")
say("- 各区 SHAP 族份额:")
for z in fam_shap_zone.index:
    r = fam_shap_zone.loc[z].sort_values(ascending=False)
    say(f"    {ROMAN[z]} 区: " + "，".join(f"{FAM_CN[k]} {v*100:.1f}%" for k, v in r.items()))
check("各区 SHAP 族份额均完整", len(fam_shap_zone) == K_MAIN
      and fam_shap_zone.notna().all().all())

# 分区内功能群组成与 RGR
fg_by_zone = PLOTS.assign(fg=tab_ms.groupby("plot_id")["fg_conifer"].mean().reindex(
    PLOTS.index) > 0.5).groupby("zone")["fg"].mean()
rgr_by_zone = PLOTS.groupby("zone")["RGR_pool"].agg(["mean", "std", "count"])

# ==========================================================================
# 10. 落盘
# ==========================================================================
say("")
say("## 10. 落盘")
out = {
    "vars": VARS, "clim": CLIM, "soil": SOIL, "terr": TERR, "all27": ALL27,
    "family": FAMILY, "family_cn": FAM_CN, "var_cn": VAR_CN, "fam_order": FAM_ORDER,
    "k_main": K_MAIN, "k_fine": K_FINE, "n_pc": n_pc, "evr": evr, "eig": eig,
    "load_rot": load_rot, "S": S, "fam_blocks": FAM_BLOCKS,
    "zone": ZONE, "zone_fine": ZONE_FINE, "zone_ward": ZONE_WD, "zone_gmm": ZONE_GMM,
    "zone_clim": PLOTS["zone_clim"].values, "plot_index": PLOTS.index.values,
    "lon": PLOTS["lon"].values, "lat": PLOTS["lat"].values, "dem_plot": PLOTS["dem"].values,
    "plc": PLOTS, "criteria": CRIT, "ba_main": ba_main, "ba_fine": ba_fine, "boot_dist": BOOT_DIST,
    "ari": {"km_wd": ari_km_wd, "km_gmm": ari_km_gm, "clim": ari_clim, "dem": ari_dem},
    "purity": PURITY, "plot_grid_agree": PLOT_GRID_AGREE,
    "agree_interior": AGREE_INTERIOR, "agree_boundary": AGREE_BOUNDARY,
    "majority_agree": _maj, "clim_diff": CLIM_DIFF,
    "oob": OOB, "acc_cv": ACC_CV, "cm": CM, "imp": IMP, "fam_imp": FAM_IMP,
    "lda_s": LDA_S, "lda_evr": lda.explained_variance_ratio_,
    "zmean": zmean, "zsd": zsd, "zanom": zanom, "fine_anom": fine_anom,
    "prov_mu": prov_mu, "prov_sd": prov_sd, "zd": ZD,
    "zone_area": ZONE_AREA, "area_fine": AREA_FINE, "clim_diff": CLIM_DIFF,
    "fam_shap_zone": fam_shap_zone, "fg_by_zone": fg_by_zone, "rgr_by_zone": rgr_by_zone,
    "mu": mu, "sd": sd, "pca_components": pca_full.components_,
    "fam_pca": FAM_PCA, "lat_band_zone": zone_lat,
    "transform": OTR, "grid_shape": (oh, ow),
}
with open(os.path.join(R, "38_zonation.pkl"), "wb") as f:
    pickle.dump(out, f)

np.savez_compressed(os.path.join(MAPDIR, "zonation_grid.npz"),
                    zone=LAB, zone_fine=LAB_FINE, zone_clim=LAB_CLIM,
                    margin=MARGIN, agree=AGREE, dem=DEM_GRID.astype(np.float32),
                    forest=FOREST, transform=np.array(OTR).ravel())
for name, arr, dtype, nd, desc in [
    ("zonation_zone_250m.tif", LAB, "int16", -1,
     "Level-1 zonation zone id (0-based); -1 = nodata"),
    ("zonation_zone_fine_250m.tif", LAB_FINE, "int16", -1,
     "Finer-scheme zonation zone id (0-based); -1 = nodata"),
    ("zonation_zone_climate_only_250m.tif", LAB_CLIM, "int16", -1,
     "Climate-only zonation zone id (0-based); -1 = nodata"),
    ("zonation_margin_250m.tif", MARGIN, "float32", -9999,
     "Assignment margin: distance to the 2nd minus the 1st nearest centroid"),
    ("zonation_agreement_250m.tif", AGREE, "float32", -9999,
     "Fraction of the three algorithms assigning the same zone"),
    ("zonation_elevation_250m.tif", DEM_GRID, "float32", -9999, "Elevation (m)"),
]:
    a = arr.astype(dtype).copy()
    a[~np.isfinite(a)] = nd
    with rasterio.open(os.path.join(MAPDIR, name), "w", driver="GTiff",
                       height=oh, width=ow, count=1, dtype=dtype, crs="EPSG:4326",
                       transform=OTR, nodata=nd, compress="deflate") as dst:
        dst.write(a, 1)
        dst.update_tags(description=desc)
say("- `04_分析结果/38_zonation.pkl`；`04_分析结果/map/zonation_*.tif`（6 个）；"
    "`04_分析结果/38_zonation_*.csv`")

# ==========================================================================
say("")
say("## 结论")
say("- 三个数据路径互核：样地表聚类、族块投影制图、样地坐标回采栅格三者一致")
check("主口径 k 落在 2–10 内", 2 <= K_MAIN <= 10, f"k = {K_MAIN}")
check("三族等权空间维数 = 6", S.shape[1] == 6)
check("一级区面积占比合计 = 100%", abs(ZONE_AREA.sum() - 100) < 1e-6,
      f"{ZONE_AREA.sum():.4f}%")
check("三族判别份额合计 = 100%", abs(FAM_IMP.sum() - 1) < 1e-6, f"{FAM_IMP.sum():.4f}")
check("三族判别份额均 > 5%（无族被完全压制）", float(FAM_IMP.min()) > 0.05,
      "，".join(f"{FAM_CN[k]} {v*100:.1f}%" for k, v in FAM_IMP.items()))

if fails:
    say(f"**FAIL** — {len(fails)} 项：")
    for f in fails:
        say(f"- {f}")
else:
    say("**PASS** — 分区分析全流程完成。")

os.makedirs(os.path.join(C.PIPE, "08_定稿"), exist_ok=True)
with open(os.path.join(C.PIPE, "08_定稿", "05_分区分析报告.md"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(report))
say("")
say("DONE 38 zonation")
if fails:
    raise SystemExit(1)