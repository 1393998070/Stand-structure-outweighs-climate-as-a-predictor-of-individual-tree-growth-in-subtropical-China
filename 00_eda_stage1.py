# -*- coding: utf-8 -*-
"""Stage 1 EDA: 广东省森林单木相对生长速率数据探索性分析
输入: 样点数据/样地样木因子_整合.xlsx (Sheet「样木因子」单木两期胸径, Sheet「样地因子」样地环境变量)
输出: 论文流水线产出/04_分析结果/00_eda_summary.md 及配套 csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import os, json

BASE = r"d:\3D\写论文\全省生长速率论文2"
OUT = os.path.join(BASE, "论文流水线产出", "04_分析结果")
os.makedirs(OUT, exist_ok=True)

SRC = os.path.join(BASE, "样点数据", "样地样木因子_整合.xlsx")
tree = pd.read_excel(SRC, sheet_name="样木因子")
plot = pd.read_excel(SRC, sheet_name="样地因子")

L = []  # report lines
def p(s=""):
    print(s)
    L.append(s)

p("# Stage 1 探索性数据分析 (EDA) 报告")
p("")
p(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
p(f"- 单木记录: {len(tree):,} | 样地记录: {len(plot):,}")
p("")

# ============ 1. 数据清洗与RGR计算 ============
p("## 1. 数据清洗与 RGR 计算")
p("")
tree = tree.rename(columns={"上一期调查胸径": "D0", "本期调查胸径": "D1", "样地号": "plot_id", "树种分类": "fg"})
tree["growth"] = tree["D1"] - tree["D0"]

n0 = len(tree)
ingress = tree["D0"] == 0          # 进界木: 2012胸径为0
extreme = tree["growth"] > 200     # 5年增量>20cm, 疑似量测误差
p(f"- 进界木 (D2012=0): {ingress.sum()} 株 — 无法计算对数RGR, 剔除")
p(f"- 极端增量 (>200 mm/5yr): {extreme.sum()} 株 — 疑似量测误差, 剔除")
clean = tree[~ingress & ~extreme].copy()
p(f"- 清洗后样本: {len(clean):,} 株 (保留率 {len(clean)/n0:.1%})")
p("")

clean["RGR"] = np.log(clean["D1"] / clean["D0"]) / 5.0  # yr^-1
clean["lnD0"] = np.log(clean["D0"])
clean["abs_inc"] = clean["growth"] / 10.0  # cm/5yr

def q(x, qts=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    return {f"p{int(qt*100)}": round(float(np.quantile(x, qt)), 4) for qt in qts}

p("### 1.1 单木 RGR 分布 (yr⁻¹, 5年区间)")
p("")
p(f"- 均值 {clean['RGR'].mean():.4f} | 标准差 {clean['RGR'].std():.4f} | 偏度 {stats.skew(clean['RGR']):.2f} | 峰度 {stats.kurtosis(clean['RGR']):.2f}")
for k, v in q(clean['RGR']).items():
    p(f"  - {k}: {v}")
clean['RGR'].describe().to_frame().T.to_csv(os.path.join(OUT, "00_eda_rgr_dist.csv"), index=False)
p("")

# ============ 2. 功能群对比 ============
p("## 2. 针叶 vs 阔叶功能群对比")
p("")
grp = clean.groupby("fg").agg(
    n=("RGR", "size"), mean=("RGR", "mean"), median=("RGR", "median"),
    std=("RGR", "std"), p25=("RGR", lambda x: x.quantile(.25)), p75=("RGR", lambda x: x.quantile(.75)),
    mean_D0=("D0", "mean"), mean_inc=("abs_inc", "mean"))
p(grp.round(4).to_string())
p("")
u1, u2 = clean.loc[clean.fg == "针叶树种", "RGR"], clean.loc[clean.fg == "阔叶树种", "RGR"]
t, tp = stats.ttest_ind(u1, u2, equal_var=False)
mwu, mwp = stats.mannwhitneyu(u1, u2)
ks, ksp = stats.ks_2samp(u1, u2)
d_cohen = (u1.mean() - u2.mean()) / np.sqrt((u1.std()**2 + u2.std()**2) / 2)
p(f"- 针叶-阔叶差异: Welch t={t:.2f} (p={tp:.2e}), Mann-Whitney p={mwp:.2e}, KS={ks:.3f} (p={ksp:.2e}), Cohen's d={d_cohen:.3f}")
p("")
grp.round(4).to_csv(os.path.join(OUT, "00_eda_by_fg.csv"))

# ============ 3. RGR-初始大小关系 ============
p("## 3. RGR 与初始胸径 (大小依赖性)")
p("")
p(f"- Spearman rho(RGR, lnD0) 全体 = {stats.spearmanr(clean['RGR'], clean['lnD0']).statistic:.3f} (p<1e-300)"
  if stats.spearmanr(clean['RGR'], clean['lnD0']).pvalue < 1e-300
  else f"- Spearman rho(RGR, lnD0) 全体 = {stats.spearmanr(clean['RGR'], clean['lnD0']).statistic:.3f} (p={stats.spearmanr(clean['RGR'], clean['lnD0']).pvalue:.2e})")
for g, sub in clean.groupby("fg"):
    r = stats.spearmanr(sub['RGR'], sub['lnD0']).statistic
    p(f"  - {g}: rho = {r:.3f}")
# 分位数分箱看形态
bins = pd.qcut(clean["D0"], 10)
binned = clean.groupby(bins, observed=True).agg(mean_RGR=("RGR", "mean"), median_RGR=("RGR", "median"), n=("RGR", "size"), mean_D0=("D0", "mean"))
p("")
p("- 按初始DBH十分位分箱 (mm → RGR yr⁻¹):")
p(binned.round(4).to_string())
binned.round(4).to_csv(os.path.join(OUT, "00_eda_rgr_size_bins.csv"))
p("")

# ============ 4. 样地级聚合 ============
p("## 4. 样地级聚合")
p("")
plot_tree = clean.groupby("plot_id").agg(
    n_trees=("RGR", "size"), plot_mean_RGR=("RGR", "mean"),
    plot_median_RGR=("RGR", "median"), plot_sd_RGR=("RGR", "std"),
    mean_D0=("D0", "mean"), total_inc_cm=("abs_inc", "sum"))
p(f"- 含复测单木的样地: {len(plot_tree)} 个")
p(f"- 样地平均RGR: 均值 {plot_tree['plot_mean_RGR'].mean():.4f} | 范围 [{plot_tree['plot_mean_RGR'].min():.4f}, {plot_tree['plot_mean_RGR'].max():.4f}]")
p(f"- 样地内标准差平均 {plot_tree['plot_sd_RGR'].mean():.4f} (样地内异质性)")
icc_data = clean[["plot_id", "RGR"]].copy()
g_mean = icc_data.groupby("plot_id")["RGR"].transform("mean")
between = np.var(g_mean, ddof=1)
within = np.mean((icc_data["RGR"] - g_mean) ** 2)
n_per_plot = icc_data.groupby("plot_id")["RGR"].size().mean()
icc = between / (between + within)
p(f"- 组内相关系数 ICC(样地): {icc:.3f} — 样地解释 RGR 方差比例, 提示需分组建模/空间CV")
plot_tree.round(5).to_csv(os.path.join(OUT, "00_eda_plot_rgr.csv"))
p("")

# ============ 5. 环境变量相关结构 ============
p("## 5. 环境变量相关结构 (共线性诊断)")
p("")
env_cols = {
    "气候(BIO19)": [c for c in plot.columns if c.startswith("peng_BIO")],
    "土壤(6)": ["土壤_SOC", "土壤_TN", "土壤_TP", "土壤_Tk", "土壤_pH", "土壤_容重", "土层厚度"],
    "地形水文(17)": ["坡向", "总曲率", "高程_DEM", "到水系距离", "地形起伏度", "水流方向", "水流长度",
                 "热量负荷", "平面曲率", "剖面曲率", "坡度", "水流强度指数", "太阳辐射_春分",
                 "太阳辐射_夏至", "太阳辐射_冬至", "地形位置指数", "地形形态指数", "地形湿度指数"],
    "林分": ["平均年龄", "林分密度"],
}
env_flat = [c for cs in env_cols.values() for c in cs]
corr = plot[env_flat].corr(method="spearman")
pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if not pd.isna(corr.iloc[i, j]):
            pairs.append((abs(corr.iloc[i, j]), cols[i], cols[j], corr.iloc[i, j]))
pairs.sort(reverse=True)
p("- |Spearman ρ| ≥ 0.8 的变量对 (前20):")
for a, c1, c2, r in pairs[:20]:
    p(f"  - {c1} ~ {c2}: ρ = {r:.3f}")
p(f"- |ρ|≥0.8 的变量对总数: {sum(1 for a, *_ in pairs if a >= 0.8)}")
corr.round(3).to_csv(os.path.join(OUT, "00_eda_env_spearman.csv"))
p("")

# ============ 6. RGR 与环境变量单变量关系 ============
p("## 6. 样地平均RGR 与环境变量的单变量关系 (Spearman)")
p("")
merged = plot_tree.reset_index().merge(plot, left_on="plot_id", right_on="样地号", how="left")
rows = []
for c in [x for x in env_flat if x in merged.columns]:
    sub = merged[["plot_mean_RGR", c]].dropna()
    if len(sub) < 50:
        continue
    r, pv = stats.spearmanr(sub[c], sub["plot_mean_RGR"])
    rows.append({"变量": c, "rho": round(r, 3), "p": pv, "n": len(sub)})
uni = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
for _, row in uni.iterrows():
    p(f"  - {row['变量']:<12s} rho = {row['rho']:+.3f}  (n={row['n']}, p={'<1e-20' if row['p'] < 1e-20 else f'{row.p:.1e}'})")
uni.to_csv(os.path.join(OUT, "00_eda_univariate.csv"), index=False)
p("")
p("- 注: 起源(人工/天然次生林)与样地类型(针叶林/针阔混交林/阔叶林)为分类变量, 其 rho 仅供方向参考")

# 分类变量检验
if "起源" in merged.columns:
    sub = merged.dropna(subset=["起源", "plot_mean_RGR"])
    sub = sub[sub["起源"].isin(["人工", "天然次生林"])]
    if len(sub) > 30:
        a = sub.loc[sub.起源 == "人工", "plot_mean_RGR"]; b = sub.loc[sub.起源 == "天然次生林", "plot_mean_RGR"]
        if len(a) > 10 and len(b) > 10:
            t, tp = stats.ttest_ind(a, b, equal_var=False)
            p(f"- 起源: 人工(n={len(a)}) {a.mean():.4f} vs 天然次生林(n={len(b)}) {b.mean():.4f}, Welch t={t:.2f}, p={tp:.2e}")

# 样地类型分组检验
if "样地类型" in merged.columns:
    sub = merged.dropna(subset=["样地类型", "plot_mean_RGR"])
    grps = {k: v["plot_mean_RGR"] for k, v in sub.groupby("样地类型")}
    if len(grps) >= 2:
        f_stat, f_p = stats.f_oneway(*grps.values())
        desc = ", ".join(f"{k}(n={len(v)}) {v.mean():.4f}" for k, v in grps.items())
        p(f"- 样地类型: {desc}, one-way ANOVA F={f_stat:.2f}, p={f_p:.2e}")
p("")

# ============ 7. 空间格局 ============
p("## 7. 空间格局")
p("")
sp = merged.dropna(subset=["经度", "纬度"])
r_lat = stats.spearmanr(sp["纬度"], sp["plot_mean_RGR"]).statistic
r_lon = stats.spearmanr(sp["经度"], sp["plot_mean_RGR"]).statistic
p(f"- 纬度 ~ 样地RGR: Spearman ρ = {r_lat:+.3f} (南北梯度)")
p(f"- 经度 ~ 样地RGR: Spearman ρ = {r_lon:+.3f}")
for v in ["高程_DEM", "peng_BIO1", "peng_BIO12"]:
    if v in sp.columns:
        sub = sp.dropna(subset=[v])
        p(f"- {v} 范围: [{sub[v].min():.1f}, {sub[v].max():.1f}]")
p(f"- 样地经度范围 [{sp['经度'].min():.2f}, {sp['经度'].max():.2f}], 纬度范围 [{sp['纬度'].min():.2f}, {sp['纬度'].max():.2f}]")
# 纬度三分位对比
sp["lat_bin"] = pd.qcut(sp["纬度"], 3, labels=["南部", "中部", "北部"])
p("- 按纬度三分位的样地RGR:")
p(sp.groupby("lat_bin", observed=True)["plot_mean_RGR"].agg(["count", "mean", "median"]).round(4).to_string())
p("")

# ============ 8. 缺失值 ============
p(f"## 8. 关键变量缺失 (样地表, n={len(plot)})")
p("")
miss = plot.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) == 0:
    p(f"  - 全表无缺失：{plot.shape[1]} 列 × {len(plot)} 行均为完整观测")
for k, v in miss.items():
    p(f"  - {k}: {v} ({v/len(plot):.1%})")
p("")

# ============ 9. 关键统计快照 (供后续建模参考) ============
p("## 9. 建模参考快照")
p("")
p(f"- 单木建模样本: {len(clean):,} 株 × {len(plot.columns)} 潜在样地预测变量")
p(f"- 样地分组建模: {plot_tree.index.nunique()} 组, 组均样本量 {len(clean)/plot_tree.index.nunique():.1f}")
p(f"- 功能群样本比 针叶:阔叶 = {(clean.fg=='针叶树种').sum():,} : {(clean.fg=='阔叶树种').sum():,} ({(clean.fg=='针叶树种').mean():.1%} 针叶)")
p(f"- RGR 负值检查: min={clean['RGR'].min():.4f} (全部生长非负→RGR≥0)")
p(f"- 零生长树 (growth==0): {(tree['growth']==0).sum()} 株 — 数据方已作为调查瑕疵预先剔除, 论文中仅在数据清洗段简要说明")
if "样地类型" in merged.columns:
    p(f"- 样地类型构成: {merged['样地类型'].value_counts().to_dict()}")
if "起源" in merged.columns:
    p(f"- 起源构成: {merged['起源'].value_counts().to_dict()}")
p("")

with open(os.path.join(OUT, "00_eda_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
clean.to_pickle(os.path.join(OUT, "00_tree_clean.pkl"))
merged.to_pickle(os.path.join(OUT, "00_plot_merged.pkl"))
print("\n[DONE] 结果已保存至 04_分析结果/00_eda_summary.md")
