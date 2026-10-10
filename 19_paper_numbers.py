# -*- coding: utf-8 -*-
"""Stage 4 稿件支撑数字：极端增量分布 / 标度斜率 / 年龄分箱 n，并汇总 v4 全稿数字速查表。
输入: 样点数据/样地样木因子_整合.xlsx（样木因子）+ 00/01/04/26/27/28/29 输出
输出: 04_分析结果/19_paper_numbers.md
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"d:\3D\写论文\全省生长速率论文2"
sys.path.insert(0, os.path.join(BASE, "论文流水线产出", "03_分析脚本"))
import common as C

OUT = os.path.join(BASE, "论文流水线产出", "04_分析结果")

tree = pd.read_excel(os.path.join(BASE, "样点数据", "样地样木因子_整合.xlsx"),
                     sheet_name="样木因子")
tree = tree.rename(columns={"上一期调查胸径": "D0", "本期调查胸径": "D1",
                            "样地号": "plot_id", "树种分类": "fg"})
tree["growth"] = tree["D1"] - tree["D0"]
clean = pd.read_pickle(os.path.join(OUT, "00_tree_clean.pkl"))
plot = pd.read_pickle(os.path.join(OUT, "00_plot_merged.pkl"))

tab, vd, folds = C.load_model_data()
feats = C.selected_features(vd)
plot_tab = (tab.groupby("plot_id")
            .agg(plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"),
                 origin_natural=("origin_natural", "first"),
                 conifer_prop=("conifer_prop", "first"), lat=("lat", "first"))
            .reset_index())

L = []
def say(s):
    print(s)
    L.append(s)

def js(name):
    return C.load_json(name)

def csv(name, **kw):
    return pd.read_csv(os.path.join(OUT, name), **kw)

say("# 稿件支撑数字（v4，新数据 1,104 样地）")

# 1. 极端增量
ext = tree[tree["growth"] > 200]
ing = tree[tree["D0"] == 0]
say(f"\n## 1. 数据清洗")
say(f"- 交付配对样木: {len(tree):,} 株；进界木 (D0=0): {len(ing)} 株；极端增量 (>200 mm/5yr): {len(ext)} 株"
    f"（增量 {ext['growth'].min()}–{ext['growth'].max()} mm，占配对样本 {len(ext)/len(tree)*100:.3f}%）")
say(f"- 清洗后: {len(clean):,} 株 / {clean['plot_id'].nunique():,} 样地"
    f"；针叶 {int((clean['fg']=='针叶树种').sum()):,} / 阔叶 {int((clean['fg']=='阔叶树种').sum()):,}")
say(f"- 样地类型构成: {plot['样地类型'].value_counts().to_dict() if '样地类型' in plot.columns else 'n/a'}")
say(f"- 起源构成: {plot['起源'].value_counts().to_dict() if '起源' in plot.columns else 'n/a'}")

# 2. 标度斜率
dec = clean.groupby(pd.qcut(clean["D0"], 10, duplicates="drop")).agg(
    mRGR=("RGR", "mean"), mD0=("D0", "mean"), n=("RGR", "size"))
slope, intercept, r, pval, se = stats.linregress(np.log(dec["mD0"]), np.log(dec["mRGR"]))
sl_tree = stats.linregress(clean["lnD0"], np.log(clean["RGR"]))
say(f"\n## 2. 尺寸标度")
say(f"- 十分位均值 ln–ln OLS slope = {slope:.3f} (r²={r**2:.3f}, p={pval:.2e})")
say(f"- 树级 ln(RGR)~ln(D0) slope = {sl_tree.slope:.3f}")
say(f"- ρ(RGR, lnD0) 全体 = {stats.spearmanr(clean['RGR'], clean['lnD0']).statistic:.3f}")

# 3. 年龄分箱
b = pd.cut(plot_tab["stand_age"], np.arange(0, 105, 10), right=False)
gm = plot_tab.dropna(subset=["stand_age"]).groupby(b, observed=True).agg(
    n=("plot_RGR", "size"), mean_RGR=("plot_RGR", "mean"))
say("\n## 3. 年龄十分箱")
say(gm.round(5).to_string())
say(f"- ≥40 年: {int((plot_tab['stand_age'] >= 40).sum())} 个；≥50 年: "
    f"{int((plot_tab['stand_age'] >= 50).sum())}；≥60 年: {int((plot_tab['stand_age'] >= 60).sum())}")
say(f"- ρ(样地RGR, 林龄) = {stats.spearmanr(plot_tab.dropna(subset=['stand_age'])['stand_age'], plot_tab.dropna(subset=['stand_age'])['plot_RGR']).statistic:.3f}")
say(f"- ρ(样地RGR, 纬度) = {stats.spearmanr(plot_tab['lat'], plot_tab['plot_RGR']).statistic:+.3f}")

# 4. 模型性能 / 泄漏
cv22 = csv("22_nested_cv_results.csv")
lk = csv("22_nested_leakage.csv")
say("\n## 4. 嵌套三重 CV 与泄漏")
for _, r_ in cv22.iterrows():
    say(f"- {r_['cv']:8s}: tree R²={r_['R2']:.4f} {r_['ci_tree_R2']}, plot R²={r_['plot_R2']:.4f} {r_['ci_plot_R2']}")
for _, r_ in lk.iterrows():
    say(f"- {r_['quantity']}: {r_['value']} {r_['ci']}")
plot_lk = csv("31_leakage_plotlevel.csv")
for _, r_ in plot_lk.iterrows():
    say(f"- [plot] {r_['quantity']}: {r_['value']} {r_['ci']}")

# 5. 归因
g26 = csv("26_group_shares.csv")
i26 = csv("26_shap_importance.csv").sort_values("mean_abs_shap", ascending=False)
say("\n## 5. 嵌套 SHAP 族份额与 top 变量")
say(g26.to_string(index=False))
say(i26.head(12)[["var", "group", "mean_abs_shap", "share_total", "ci_lo", "ci_hi"]].round(6).to_string(index=False))
say(f"- 每变量归一份额: {csv('27_family_share_normalized.csv')[['group','share','n_vars','share_per_var','rank_by_share_per_var']].to_string(index=False)}")

# 6. 消融 / 方差分解 / 主导
say("\n## 6. 消融与方差分解")
say(csv("28_ablation_nested.csv").to_string(index=False))
say("\n## 6b. 样地主导族")
say(csv("28_plot_dominance_nested.csv").to_string(index=False))
vp = csv("27_varpart_fix.csv")
say("\n## 6c. 方差分解（三口径）")
say(vp.to_string(index=False))

# 7. LMM
lmm = js("02_lmm_summary.json")
say("\n## 7. LMM 基准")
say(f"- 边际 R²={lmm['marginal_R2']}, 条件 R²={lmm['conditional_R2']}, "
    f"ICC(零模型)={lmm['ICC_plot']}, 全模型隐含 ICC={lmm['ICC_model_implied']}, "
    f"n={lmm['n_complete']:,} 株 / {lmm['n_plots']:,} 样地")
co = csv("02_lmm_coefs.csv", index_col=0)
say(co.reindex(co["t"].abs().sort_values(ascending=False).index).head(10).round(4).to_string())
say("\n## 7b. LMM 同口径 OOF")
say(open(os.path.join(OUT, "13_lmm_oof_report.md"), encoding="utf-8").read().split("## 完整案例对齐比较")[1][:400])

# 8. 制图
say("\n## 8. 制图")
say(json.dumps(js("23_dual_map_summary.json")["responses"], ensure_ascii=False))
say(open(os.path.join(OUT, "18_map_rebuild_report.md"), encoding="utf-8").read().split("## 4. 年龄输入对照")[1][:600])
say(open(os.path.join(OUT, "18_map_rebuild_report.md"), encoding="utf-8").read().split("## 5. 全局内层选参")[1][:800])

# 9. 分层 / 稳定性
say("\n## 9. 气候分层与其他稳定性")
say(csv("17_climate_strata.csv").round(4).to_string(index=False))
say(csv("16_shape_stability.csv").to_string(index=False))
say(csv("33_shape_stability.csv").to_string(index=False))
say(open(os.path.join(OUT, "16_gam_stability_report.md"), encoding="utf-8").read()[:1200])

# 10. 敏感性
say("\n## 10. 敏感性套件")
say(open(os.path.join(OUT, "08_sensitivity_report.md"), encoding="utf-8").read()[:2500])

# 11. 分区
say("\n## 11. 分区分析")
say(csv("38_zonation_zone_summary.csv").to_string(index=False))
z3 = csv("38_zonation_criteria.csv")
say(f"- k=3 自助 ARI 中位 {z3.loc[z3['k']==3,'boot_ARI_median'].iloc[0]:.3f}；"
    f"轮廓系数 {z3.loc[z3['k']==3,'silhouette'].iloc[0]:.3f}")

with open(os.path.join(OUT, "19_paper_numbers.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("\nDONE 19 paper numbers")
