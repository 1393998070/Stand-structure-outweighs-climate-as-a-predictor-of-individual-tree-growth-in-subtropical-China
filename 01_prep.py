# -*- coding: utf-8 -*-
"""Phase A 数据准备：变量编码 → 共线性筛选 → 五类组标注 → 折构造。
输出: 01_model_table.pkl / 01_var_dict.csv / 01_plot_folds.csv / 01_prep_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

t, p = C.load_tables()
report = []
def say(x):
    print(x)
    report.append(x)

say("# Phase A 数据准备报告\n")

# ---------- 1. 样地表编码 ----------
say("## 1. 样地编码")
say(f"- 清洗后样地: {len(p)}，单木: {len(t)}")
say(f"- 起源取值: {p['起源'].value_counts(dropna=False).to_dict()}")
say(f"- 样地类型取值: {p['样地类型'].value_counts(dropna=False).to_dict()}")

# 针叶比例（样地内针叶株数占比，来自单木表）
fg_cnt = t.groupby("plot_id")["fg"].value_counts().unstack(fill_value=0)
conif_col = [c for c in fg_cnt.columns if "针叶" in str(c)][0]
p = p.set_index("plot_id")
p["conifer_prop"] = (fg_cnt[conif_col] / fg_cnt.sum(axis=1)).reindex(p.index)
p = p.reset_index()

# 重命名为英文建模名
rename = {
    "平均年龄": "stand_age", "林分密度": "stand_density",
    "土壤_SOC": "soil_SOC", "土壤_TN": "soil_TN", "土壤_TP": "soil_TP",
    "土壤_Tk": "soil_Tk", "土壤_pH": "soil_pH", "土壤_容重": "soil_BD", "土层厚度": "soil_depth",
    "高程_DEM": "dem", "坡度": "slope", "坡向": "aspect_deg", "总曲率": "curvature_total",
    "平面曲率": "curvature_plan", "剖面曲率": "curvature_profile", "到水系距离": "dist_stream",
    "地形起伏度": "relief", "水流长度": "flow_length", "热量负荷": "heat_load",
    "水流强度指数": "SPI", "太阳辐射_春分": "solar_eq", "太阳辐射_夏至": "solar_ss",
    "太阳辐射_冬至": "solar_ws", "地形位置指数": "TPI", "地形形态指数": "TSI",
    "地形湿度指数": "TWI", "经度": "lon", "纬度": "lat",
}
for i in range(1, 20):
    rename[f"peng_BIO{i}"] = f"bio{i}"
p = p.rename(columns=rename)

# 起源二元 + 缺失指示
FT_MAP = {"针叶林": 0.0, "针阔混交林": 1.0, "阔叶林": 2.0}
p["origin_natural"] = p["起源"].map({"天然次生林": 1.0, "人工": 0.0})
assert p["origin_natural"].notna().all(), f"起源存在未映射取值: {p.loc[p['origin_natural'].isna(), '起源'].unique()}"
p["origin_missing"] = p["origin_natural"].isna().astype(int)
p["forest_type"] = p["样地类型"].map(FT_MAP)
assert p["forest_type"].notna().all(), f"样地类型存在未映射取值: {p.loc[p['forest_type'].isna(), '样地类型'].unique()}"
p = p.drop(columns=["样地类型"])
# 年龄缺失指示
p["age_missing"] = p["stand_age"].isna().astype(int)
# 坡向 → 圆形变量分解（检查量纲）
say(f"- 坡向范围: {p['aspect_deg'].min()}–{p['aspect_deg'].max()} 度")
p["aspect_cos"] = np.cos(np.deg2rad(p["aspect_deg"]))
p["aspect_sin"] = np.sin(np.deg2rad(p["aspect_deg"]))
p = p.drop(columns=["aspect_deg"])

# ---------- 2. 合并单木表 ----------
tab = t.merge(p, on="plot_id", how="inner")
say(f"\n## 2. 合并后建模主表: {tab.shape}")
say(f"- 单木 fg 取值: {tab['fg'].value_counts().to_dict()}")
tab["fg_conifer"] = (tab["fg"] == "针叶树种").astype(int)
assert tab["lon"].notna().all() and tab["lat"].notna().all(), "存在缺失坐标"

# 缺失概况
miss = tab[["stand_age", "origin_natural", "soil_SOC", "soil_TN", "soil_TP",
            "soil_Tk", "soil_pH", "soil_BD", "soil_depth", "forest_type",
            "stand_density"]].isna().mean().sort_values(ascending=False)
say("\n## 3. 预测变量缺失比例（单木级）")
say(miss.to_markdown())
soil_cols = ["soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth"]
soil_miss_same = (p[soil_cols].isna().sum(axis=1) > 0)
say(f"- 土壤任一缺失的样地: {soil_miss_same.sum()}（{soil_miss_same.mean()*100:.1f}%）")

# ---------- 3. 共线性筛选（|Spearman ρ|>0.85 层次聚类，簇内按优先级保留） ----------
say("\n## 4. 共线性筛选（|ρ|>0.85）")
num_cands = (C.TREE_VARS + C.STAND_VARS
             + C.SOIL_VARS + C.TERRAIN_VARS + C.CLIMATE_CANDIDATES)
num_cands = [v for v in dict.fromkeys(num_cands) if v in tab.columns]
sub = tab.sample(min(20000, len(tab)), random_state=C.SEED)
corr = sub[num_cands].corr(method="spearman")

D = 1 - corr.abs()
Dv = D.to_numpy(copy=True)          # 新版 pandas 的 .values 为只读视图
np.fill_diagonal(Dv, 0)
Dv = (Dv + Dv.T) / 2
Z = linkage(squareform(Dv, checks=False), method="average")
THR = 0.15  # |ρ| > 0.85 等价距离 < 0.15
labels = fcluster(Z, t=THR, criterion="distance")
clusters = {}
for v, lab in zip(num_cands, labels):
    clusters.setdefault(int(lab), []).append(v)

rgr_corr = {v: abs(spearmanr(sub[v], sub["RGR"]).statistic) for v in num_cands}
selected, dropped = [], []
for lab, members in clusters.items():
    if len(members) == 1:
        selected.append((members[0], "single"))
        continue
    ranked = sorted(members, key=lambda v: (C.PRIORITY_RANK.get(v, 999), -rgr_corr[v]))
    keep = ranked[0]
    selected.append((keep, "cluster_kept"))
    rho_keep = {m: round(corr.loc[keep, m], 3) for m in members if m != keep}
    for m in members:
        if m != keep:
            dropped.append((m, keep, rho_keep[m]))

sel_set = {v for v, _ in selected}
say(f"- 候选 {len(num_cands)} 个 → 保留 {len(sel_set)} 个，剔除 {len(dropped)} 个")
say(f"- 剔除明细（变量 ← 保留替代者, ρ）: {sorted(dropped, key=lambda x: -abs(x[2]))}")
clim_sel = sorted([v for v in sel_set if v.startswith("bio")], key=lambda x: int(x[3:]))
say(f"- 气候保留: {clim_sel}")

# ---------- 4. 变量字典 ----------
rows = []
def group_of(v):
    if v in C.TREE_VARS: return "tree"
    if v in C.STAND_VARS: return "stand"
    if v.startswith("bio"): return "climate"
    if v in C.SOIL_VARS: return "soil"
    if v in C.TERRAIN_VARS: return "terrain"
    return "other"
for v in num_cands:
    sel = v in sel_set
    reason = ""
    if not sel:
        rep = [d for d in dropped if d[0] == v][0]
        reason = f"dropped: |rho|={abs(rep[2]):.3f} with {rep[1]}"
    rows.append({"var": v, "group": group_of(v), "selected": sel, "note": reason})
var_dict = pd.DataFrame(rows)
var_dict.to_csv(os.path.join(C.OUT, "01_var_dict.csv"), index=False, encoding="utf-8-sig")
say(f"\n- 变量字典: 保留 {var_dict['selected'].sum()} 个变量")
say("- 各类组保留数: " + var_dict[var_dict['selected']].groupby("group").size().to_dict().__str__())

# ---------- 5. 折构造 ----------
say("\n## 5. 三重 CV 折构造")
# 5.1 空间块 0.5°
p["block05"] = C.block_id(p["lon"].values, p["lat"].values).values
tree_per_plot = tab.groupby("plot_id").size()
p["n_trees"] = p["plot_id"].map(tree_per_plot)
blk_size = p.groupby("block05")["n_trees"].sum().sort_values(ascending=False)
say(f"- 0.5° 空间块数: {len(blk_size)}（树数范围 {blk_size.min()}–{blk_size.max()}）")
assign = C.assign_spatial_folds(list(blk_size.items()), n_folds=5)
p["fold_spatial"] = p["block05"].map(assign)
fold_tree_counts = p.groupby("fold_spatial")["n_trees"].sum()
say(f"- 空间块折树数分布: {fold_tree_counts.to_dict()}")

# 5.2 样地分组折（GroupKFold，确定性）
gkf = GroupKFold(n_splits=5)
p = p.sort_values("plot_id").reset_index(drop=True)
fold_group = pd.Series(-1, index=p.index)
for i, (_, test_idx) in enumerate(gkf.split(p, groups=p["plot_id"])):
    fold_group.iloc[test_idx] = i
p["fold_group"] = fold_group

# 5.3 树级随机折（同一样地单木可分属训练/测试 → 刻意的天真基准）
rng = np.random.default_rng(C.SEED)
tab["fold_random"] = rng.integers(0, 5, size=len(tab))
fold_cols = ["plot_id", "lon", "lat", "block05", "fold_spatial", "fold_group", "n_trees"]
p[fold_cols].to_csv(os.path.join(C.OUT, "01_plot_folds.csv"), index=False, encoding="utf-8-sig")

# ---------- 6. 保存 ----------
keep_cols = (["plot_id", "fg", "fg_conifer", "D0", "D1", "RGR", "lnD0", "abs_inc",
              "lon", "lat", "fold_random"]
             + [v for v in num_cands if v in sel_set]
             + ["origin_natural"])
keep_cols = list(dict.fromkeys(keep_cols))
tab_out = tab[keep_cols].copy()
tab_out.to_pickle(os.path.join(C.OUT, "01_model_table.pkl"))
say(f"\n## 6. 输出")
say(f"- 01_model_table.pkl: {tab_out.shape}")
say(f"- 响应变量 RGR: mean={tab_out['RGR'].mean():.4f}, sd={tab_out['RGR'].std():.4f}, "
    f"median={tab_out['RGR'].median():.4f}, range=[{tab_out['RGR'].min():.4f}, {tab_out['RGR'].max():.4f}]")

with open(os.path.join(C.OUT, "01_prep_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\nDONE Phase A")
