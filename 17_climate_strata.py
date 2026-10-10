# -*- coding: utf-8 -*-
"""Stage 4 · P1-1 气候分层检验（"permissiveness" 命题的可检验预测）。
按降水（bio12 MAP）四分位与降水季节性（bio15）四分位分层，重算类组 SHAP 份额；
若气候份额在干燥（包络边缘）分层升高，则 "驱动因子层级是气候宽容度的函数" 从解释变为预测。
输出: 17_climate_strata.csv / 17_climate_strata_report.md
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

tab, vd, folds = C.load_model_data()
group_of = dict(zip(vd["var"], vd["group"]))
feats = C.selected_features(vd)
shap_mat = pd.read_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# P1-1 气候分层检验报告\n")
t0 = time.time()

fam_idx = {g: [i for i, f in enumerate(feats) if group_of[f] == g]
           for g in ("tree", "stand", "climate", "soil", "terrain")}
S = np.abs(shap_mat.values)
groups = tab["plot_id"].values

def shares_in_mask(mask):
    vm = S[mask].mean(axis=0)
    tot = vm.sum()
    return {g: vm[fam_idx[g]].sum() / tot for g in fam_idx}

def boot_ci_stratum(mask, unit_groups, n_boot=1000, seed=C.SEED):
    """分层内按样地 cluster bootstrap 的类组份额 CI。"""
    ug, inv = np.unique(unit_groups[mask], return_inverse=True)
    # mask 内树行索引按组
    rows = np.where(mask)[0]
    g_of_row = unit_groups[rows]
    idx_by_g = [rows[g_of_row == g] for g in ug]
    rng = np.random.default_rng(seed)
    out = {g: [] for g in fam_idx}
    for _ in range(n_boot):
        gs = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[g] for g in gs])
        vm = S[idx].mean(axis=0)
        tot = vm.sum()
        for g in fam_idx:
            out[g].append(vm[fam_idx[g]].sum() / tot)
    return {g: (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))
            for g, v in out.items()}

rows = []
for strat_var, label in [("bio12", "MAP 四分位（Q1 最干 → Q4 最湿）"),
                         ("bio15", "降水季节性四分位（Q1 最均匀 → Q4 最强季节）")]:
    plot_val = tab.groupby("plot_id")[strat_var].first()
    q = pd.qcut(plot_val, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    strat = tab["plot_id"].map(q)
    say(f"\n## 分层变量: {strat_var}（{label}）")
    say(f"- 分层切点: {[round(float(plot_val.quantile(x)), 1) for x in (.25, .5, .75)]}")
    for qlab in ["Q1", "Q2", "Q3", "Q4"]:
        mask = (strat == qlab).values
        n_t = int(mask.sum()); n_p = tab.loc[mask, "plot_id"].nunique()
        sh = shares_in_mask(mask)
        ci = boot_ci_stratum(mask, groups)
        row = {"strat_var": strat_var, "stratum": qlab, "n_trees": n_t, "n_plots": n_p,
               "cut_lo": round(float(plot_val.quantile({"Q1": 0, "Q2": .25, "Q3": .5, "Q4": .75}[qlab])), 1),
               "cut_hi": round(float(plot_val.quantile({"Q1": .25, "Q2": .5, "Q3": .75, "Q4": 1.0}[qlab])), 1)}
        for g in ("tree", "stand", "climate", "soil", "terrain"):
            row[f"share_{g}"] = round(sh[g], 4)
            row[f"ci_{g}"] = str(ci[g])
        rows.append(row)
        say(f"- {qlab} [{row['cut_lo']}, {row['cut_hi']}]: n={n_t} 树/{n_p} 样地 | "
            f"stand={sh['stand']:.3f} climate={sh['climate']:.3f} "
            f"(CI {ci['climate'][0]:.3f}–{ci['climate'][1]:.3f}) terrain={sh['terrain']:.3f} "
            f"tree={sh['tree']:.3f} soil={sh['soil']:.3f}")
    # 趋势检验：climate 份额 ~ 分层序
    sub = [r for r in rows if r["strat_var"] == strat_var]
    clim = [r["share_climate"] for r in sub]
    stnd = [r["share_stand"] for r in sub]
    from scipy.stats import spearmanr
    rho_c = spearmanr(range(4), clim).statistic
    rho_s = spearmanr(range(4), stnd).statistic
    say(f"- climate 份额随分层序 Spearman ρ = {rho_c:+.3f}；stand 份额 ρ = {rho_s:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(C.OUT, "17_climate_strata.csv"), index=False, encoding="utf-8-sig")

say(f"\n（用时 {time.time()-t0:.0f}s）")
say("\nDONE P1-1")
with open(os.path.join(C.OUT, "17_climate_strata_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
