# -*- coding: utf-8 -*-
"""Stage 2.5 Phase C 最终对源核验：论文剩余统计数字 vs 分析输出。"""
import re
import numpy as np
import pandas as pd

R = r'd:\3D\写论文\全省生长速率论文2\论文流水线产出\04_分析结果'
S = r'd:\3D\写论文\全省生长速率论文2\论文流水线产出\03_分析脚本'

print("=" * 70)
print("[1] SHAP 依赖端点（qcut 30 箱，与 Fig5 同口径，单位 ×10⁻³ yr⁻¹）")
dep = pd.read_pickle(R + r'\04_dep_data.pkl')
claims = {
    'stand_age':      ('+15.3 → −8.5', 15.3, -8.5),
    'lnD0':           ('+12.9 → −12.4', 12.9, -12.4),
    'conifer_prop':   ('末箱 +6.2', None, 6.2),
    'origin_natural': ('−0.5', None, -0.5),
    'bio12':          ('首正末负', None, None),
    'bio15':          ('弱负', None, None),
}
for v in ['stand_age', 'lnD0', 'conifer_prop', 'bio12', 'bio15', 'origin_natural']:
    d = dep[v]
    q = pd.qcut(d["value"], 30, duplicates="drop")
    gm = d.groupby(q, observed=True)["shap"].mean() * 1000
    mid = np.array([iv.mid for iv in gm.index])
    print(f"  {v}: 首箱 {gm.iloc[0]:+.2f} (x={mid[0]:.2f}) → 末箱 {gm.iloc[-1]:+.2f} (x={mid[-1]:.2f})")
    if v in ('stand_age', 'lnD0', 'bio15'):
        print(f"    全部箱: x={np.round(mid,1).tolist()}")
        print(f"    全部SHAP: {np.round(gm.values,1).tolist()}")
        cross = np.where(np.diff(np.sign(gm.values)) != 0)[0]
        print(f"    过零点位于 x≈{[round(mid[i],1) for i in cross]}")

print()
print("=" * 70)
print("[1b] Fig5 数据快照端点（若存在）")
import os
FIGD = r'd:\3D\写论文\全省生长速率论文2\论文流水线产出\05_图表'
if os.path.isdir(FIGD):
    for f in sorted(os.listdir(FIGD)):
        print("  ", f)

print()
print("=" * 70)
print("[2] 年龄十年分箱样地均值（论文: ≤10y 0.067 / 30–40y 0.036 / 50–60y 0.020; 60y+ 每箱<3样地）")
plot = pd.read_pickle(R + r'\00_plot_merged.pkl')
pm = plot[plot["平均年龄"].notna()].copy()
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
lab = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
g = pm.groupby(pd.cut(pm["平均年龄"], bins=bins, labels=lab, right=False), observed=True)["plot_mean_RGR"]
out = pd.DataFrame({"n_plots": g.count(), "mean_RGR": g.mean()})
print(out.to_string())
print(f"  年龄: min={pm['平均年龄'].min():.0f}, max={pm['平均年龄'].max():.0f}, "
      f"median={pm['平均年龄'].median():.0f}  (论文: 0–79, median 18)")

print()
print("=" * 70)
print("[3] 缺失率口径（树级 vs 样地级，建模样本）")
tab = pd.read_pickle(R + r'\01_model_table.pkl')
n_tree = len(tab)
for col in ["age_missing", "origin_missing"]:
    if col in tab.columns:
        print(f"  树级 {col}: {tab[col].mean()*100:.2f}%  (n={n_tree})")
p1 = tab.groupby("plot_id").first()
for col in ["age_missing", "origin_missing"]:
    if col in p1.columns:
        print(f"  样地级 {col}: {p1[col].mean()*100:.2f}%  (n_plots={len(p1)})")
soil_cols = [c for c in tab.columns if c.startswith("soil_") and not c.endswith("_missing")]
soil_miss_plot = p1[soil_cols].isna().mean(axis=0)
print("  样地级土壤缺失:", {k: f"{v*100:.1f}%" for k, v in soil_miss_plot.items()})

print()
print("=" * 70)
print("[4] 空间块统计（论文: 84 blocks, 3–2,057 trees, ~8,940/fold）")
folds = pd.read_csv(R + r'\01_plot_folds.csv')
if "block_id" in folds.columns:
    blk = folds.groupby("block_id").size()
    print(f"  块数={len(blk)}, 块内样地数 min={blk.min()}, max={blk.max()}")
if "fold" in folds.columns:
    fs = folds.groupby("fold").size()
    print(f"  折内样地数: {fs.tolist()}")

print()
print("=" * 70)
print("[5] 交互幅度（论文: age×fg ±0.4×10⁻³; density×lnD0 ±0.8×10⁻³）")
try:
    idep = pd.read_pickle(R + r'\04_int_dep_data.pkl')
    print("  可用交互数据键:", list(idep.keys())[:10])
    for k, d in idep.items():
        if isinstance(d, pd.DataFrame) and "shap_inter" in d.columns:
            amp = np.nanpercentile(d["shap_inter"], [1, 99]) * 1000
            print(f"  {k}: P1={amp[0]:+.2f}, P99={amp[1]:+.2f} (×10⁻³)")
except Exception as e:
    print("  跳过:", e)

print()
print("=" * 70)
print("[6] 样地经纬度范围（论文 Methods: 109.7–117.1°E 需改 109.8）")
print(f"  lon [{plot['经度'].min():.4f}, {plot['经度'].max():.4f}], "
      f"lat [{plot['纬度'].min():.4f}, {plot['纬度'].max():.4f}]")

print()
print("=" * 70)
print("[7] MAP_FEATS 数量（论文: 31 → 应为 29）")
src = open(S + r'\07_map_predict.py', encoding='utf-8').read()
m = re.search(r'MAP_FEATS\s*=\s*\[(.*?)\]', src, re.S)
if m:
    feats = [x.strip().strip('"\'') for x in m.group(1).split(',') if x.strip()]
    print(f"  MAP_FEATS = {len(feats)} 项: {feats}")

print()
print("=" * 70)
print("[8] 栅格提取保真度（论文: 匹配层 ρ=1.00; 年龄 ρ=0.31）")
ec = pd.read_csv(R + r'\06_extract_check.csv')
print(ec.to_string(index=False))

print()
print("=" * 70)
print("[9] RF 调参候选数（论文: 12）")
log = open(R + r'\02_run.log', encoding='utf-8', errors='ignore').read()
for line in log.splitlines():
    if 'RF' in line and ('cand' in line or '候选' in line or 'random' in line.lower()):
        print("  ", line.strip()[:120])
