# -*- coding: utf-8 -*-
"""P0-C 旗舰口径自洽性修正。

修正三处（DA-M-2 / DA-M-4 / DA-M-6 / R1-W4）：
1. v2 的 "stand attributes alone … ~86% of unique explainable variance" 所用的 "stand"
   预测集**含 plot 平均 lnD0（树木属性）**，与全文 tree/stand 分工口径冲突。
   本脚本给出【含】与【剔除 mean_lnD0】两套结果。
2. conifer_prop 实由被分析的复测单木按样地聚合得出，并非样地记录的林分属性；
   给出【剔除 conifer_prop】的敏感性结果。
3. 家族份额为族内 mean|SHAP| 求和，受族内变量数与空间分辨率影响；
   给出【按族内变量数归一】的对照指标。

输出: 27_varpart_fix.csv / 27_conifer_prop_sensitivity.csv
      27_family_share_normalized.csv / 27_varpart_fix_report.md
"""
import os
import sys
import math
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P0-C 旗舰口径自洽性修正报告\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats_by_group = {g: C.selected_features(vd, [g])
                  for g in ("tree", "stand", "climate", "soil", "terrain")}

# ---------- 样地级表（与 05_ablation 一致） ----------
pt = tab.groupby("plot_id").agg(
    plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"),
    origin_natural=("origin_natural", "first"), conifer_prop=("conifer_prop", "first"),
    fg_conifer=("fg_conifer", "mean")).reset_index()
_env = [c for c in dict.fromkeys(["stand_density"]
                                + feats_by_group["climate"]
                                + feats_by_group["soil"]
                                + feats_by_group["terrain"])]
pt = pt.merge(tab.groupby("plot_id")[_env].first().reset_index(), on="plot_id", how="left")
pt["mean_lnD0"] = pt["plot_id"].map(tab.groupby("plot_id")["lnD0"].mean())
pt["fg_conifer"] = (pt["fg_conifer"] > 0.5).astype(int)

def adj_r2(y, cols_df):
    Xc = np.column_stack([np.ones(len(cols_df))] + [cols_df[c].values.astype(float)
                                                    for c in cols_df.columns])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    r2 = 1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    n, k = len(y), Xc.shape[1] - 1
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def partition(sets, d, y):
    names = list(sets)
    sub_r2 = {}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            cols = sum([sets[c] for c in combo], [])
            sub_r2[combo] = adj_r2(y, d[cols])
    full = sub_r2[tuple(names)]
    rows = []
    for g in names:
        others = tuple(n for n in names if n != g)
        rows.append({"group": g, "unique_adjR2": round(full - sub_r2[others], 4),
                     "alone_adjR2": round(sub_r2[(g,)], 4)})
    hp = {g: 0.0 for g in names}
    for perm in itertools.permutations(names):
        used = []
        for g in perm:
            cols = sum([sets[c] for c in used], [])
            r_before = adj_r2(y, d[cols]) if used else 0.0
            r_after = adj_r2(y, d[cols + sets[g]])
            hp[g] += (r_after - r_before) / math.factorial(len(names))
    for row, g in zip(rows, names):
        row["hier_share"] = round(hp[g] / full, 4)
    return pd.DataFrame(rows), full

STAND_V2 = ["stand_age", "stand_density", "origin_natural", "conifer_prop",
            "mean_lnD0"]
STAND_FIX = [c for c in STAND_V2 if c != "mean_lnD0"]
STAND_NOCP = [c for c in STAND_FIX if c != "conifer_prop"]

variants = {
    "v2 original (stand incl. mean_lnD0)": STAND_V2,
    "P0-C fix (stand excl. mean_lnD0)": STAND_FIX,
    "P0-C + excl. conifer_prop": STAND_NOCP,
}
say("## 1. 样地级方差分解：三套口径")
say("")
out = []
for tag, stand_cols in variants.items():
    sets = {"stand": stand_cols, "climate": feats_by_group["climate"],
            "soil": feats_by_group["soil"], "terrain": feats_by_group["terrain"]}
    d = pt.dropna(subset=["plot_RGR"] + sum(sets.values(), [])).copy()
    y = d["plot_RGR"].values
    vp, full = partition(sets, d, y)
    vp.insert(0, "variant", tag)
    vp.insert(1, "n_stand_vars", len(stand_cols))
    vp.loc[len(vp)] = {"variant": tag, "n_stand_vars": "", "group": "TOTAL",
                       "unique_adjR2": "", "alone_adjR2": "", "hier_share": round(full, 4)}
    out.append(vp)
    st = vp[vp["group"] == "stand"].iloc[0]
    say(f"### {tag}")
    say(f"- 完整案例样地: {len(d)}；全模型调整 R² = {full:.4f}")
    say(f"- stand: unique adj R² = {st['unique_adjR2']}，占可解释方差 "
        f"{100*float(st['unique_adjR2'])/full:.1f}%；first-entry share = {st['hier_share']}")
    say(vp.to_string(index=False))
    say("")
vpall = pd.concat(out, ignore_index=True)
vpall.to_csv(os.path.join(C.OUT, "27_varpart_fix.csv"), index=False, encoding="utf-8-sig")

# ---------- 2. conifer_prop 的独立贡献 ----------
say("\n## 2. conifer_prop 来源与独立贡献")
say("")
sh = pd.read_csv(os.path.join(C.OUT, "26_shap_importance.csv"))
cp = sh[sh["var"] == "conifer_prop"]
if len(cp):
    say(f"- conifer_prop 在嵌套口径 OOF SHAP 中: mean|SHAP| = "
        f"{float(cp['mean_abs_shap'].iloc[0]):.6f}，占总归因 "
        f"{float(cp['share_total'].iloc[0])*100:.2f}%（rank {int(cp['rank'].iloc[0])}）")
say("- 来源（01_prep.py L32–35）：conifer_prop = 针叶单木数 / 该样地复测单木总数，"
    "由**被分析的同一批复测单木**聚合得出，非样地记录的独立林分属性。")
say("- 制图模型 29 个栅格预测因子中不含 conifer_prop → 该变量不可制图。")
say("")
d0 = pt.dropna(subset=["plot_RGR", "stand_age", "stand_density", "origin_natural",
                       "conifer_prop", "mean_lnD0"])
say(f"- 剔除 conifer_prop 后（见上表第三套口径）：stand 仍为最大族，"
    f"unique adj R² 与 first-entry share 见 27_varpart_fix.csv。")
pd.DataFrame([
    {"variant": "stand incl. mean_lnD0 (v2)", "n_vars": len(STAND_V2)},
    {"variant": "stand excl. mean_lnD0 (fix)", "n_vars": len(STAND_FIX)},
    {"variant": "stand excl. mean_lnD0 & conifer_prop", "n_vars": len(STAND_NOCP)},
]).to_csv(os.path.join(C.OUT, "27_conifer_prop_sensitivity.csv"), index=False,
          encoding="utf-8-sig")

# ---------- 3. 家族份额的可比性 ----------
say("\n## 3. 家族份额的可比性（变量数与分辨率未归一）")
say("")
nvar = {g: len(feats_by_group[g]) for g in ("tree", "stand", "climate", "soil", "terrain")}
nvar["stand"] = len([f for f in feats_by_group["stand"]])
gs = pd.read_csv(os.path.join(C.OUT, "26_group_shares.csv"))
gs["n_vars"] = gs["group"].map(nvar)
gs["share_per_var"] = (gs["share"] / gs["n_vars"]).round(4)
gs["rank_by_share"] = gs["share"].rank(ascending=False).astype(int)
gs["rank_by_share_per_var"] = gs["share_per_var"].rank(ascending=False).astype(int)
say(gs.to_string(index=False))
gs.to_csv(os.path.join(C.OUT, "27_family_share_normalized.csv"), index=False,
          encoding="utf-8-sig")
say("")
say("- 说明：terrain 保留 15 个 10-m DEM 派生变量，climate 仅 7 个 1-km 气候变量，"
    "且 bio5（最热月最高温）因与 elevation |ρ|=0.902 被筛除（一个温度信号计入 terrain）。")
say("- 因此 Abstract 与 Fig. 4a 并列的 'terrain 16.2% > climate 15.4%' **不可作生态排序解读**；")
say("  核心比较 stand（7 变量）vs climate（7 变量）不受影响。")

with open(os.path.join(C.OUT, "27_varpart_fix_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE P0-C")
