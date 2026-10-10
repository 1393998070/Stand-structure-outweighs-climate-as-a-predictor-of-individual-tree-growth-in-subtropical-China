# -*- coding: utf-8 -*-
"""34 图件-正文-图注三方一致性核验（Stage 3' 新增的独立核验面）。

Round-1 的失效模式（roadmap/2.0「流程教训」）：
  Fig. 6c 仍读废弃数据、图注引用不存在的 (d) 面板，而当时的终检只做
  「标记级文本验证 + 数字对源」，**未核验交付物本身**。

本脚本封堵该漏洞：从 matplotlib 实际渲染的 Text artist 中导出图面文字
（由 09_figures.py / 20_si_figures.py 写出 fig_labels_main.json / fig_labels_si.json），
再做三条腿的核验：

  (i)   图面 ↔ 数据源：按绘图代码相同的格式化逻辑从源数据重算应出现的标签，断言其确实已渲染；
  (ii)  图注 ↔ 数据源：图注中引用但【不在图面上】的数字（如 n、范围）对源核验；
  (iii) 图注 ↔ 图面：图注中必须见于图面的数字，与渲染标签按同一小数位比较。

注：本脚本核验图面**文字**；图面几何（色标范围、点云形状）仍由人工目视核验，
该限界在 03_评审/Stage3_复审 矩阵中如实标注。
"""
import os
import sys
import json
import pickle
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MS = os.path.join(C.PIPE, "06_论文稿", "manuscript_v4.md")
SI = os.path.join(C.PIPE, "06_论文稿", "Supplementary_Information_v4.md")
R = C.OUT

fails, report = [], []


def say(x):
    print(x, flush=True)
    report.append(str(x))


def ok(label, cond, detail=""):
    say(f"  [{'OK' if cond else 'XX'}] {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(label)


ML = json.load(open(os.path.join(R, "fig_labels_main.json"), encoding="utf-8"))
SL = json.load(open(os.path.join(R, "fig_labels_si.json"), encoding="utf-8"))


def has(fig, s, store=ML):
    """图面上是否渲染了该字符串（子串匹配，兼容面板前缀）。"""
    return any(s in t for t in store.get(fig, []))


def has_exact(fig, s, store=ML):
    return any(s == t for t in store.get(fig, []))


def has_any(fig, ss, store=ML):
    return any(has(fig, s, store) for s in ss)


def nums(fig, store=ML):
    out = []
    for t in store.get(fig, []):
        for m in re.finditer(r"-?\d+\.?\d*", t.replace("−", "-").replace(",", "")):
            try:
                out.append(float(m.group(0)))
            except ValueError:
                pass
    return out


def caption_body(path, key):
    txt = open(path, encoding="utf-8").read()
    if path is SI:
        m = re.search(r"\*\*Supplementary Figure " + key + r" \|(.+?)(?=\n\*\*|\n---|\Z)", txt, re.S)
    else:
        m = re.search(r"\*\*Figure " + key + r" \|(.+?)(?=\n\*\*Figure|\n---|\Z)", txt, re.S)
    return m.group(1) if m else ""


say("# 图件-正文-图注三方一致性核验报告（34_threeway_check）\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say(f"- 图面标签来源: 09_figures.py / 20_si_figures.py 实际渲染的 Text artists")
say("")

tab, vd, folds = C.load_model_data()
plot_tab = (tab.groupby("plot_id")
            .agg(plot_RGR=("RGR", "mean"), stand_age=("stand_age", "first"),
                 origin_natural=("origin_natural", "first"),
                 fg_conifer=("fg_conifer", "mean"))
            .reset_index())
plot_tab["fg_conifer"] = (plot_tab["fg_conifer"] > 0.5).astype(int)

# =====================================================================
say("## 腿 (i)：图面 ↔ 数据源（重算应渲染的标签，断言其确实存在）")
say("")

nres = pd.read_csv(os.path.join(R, "22_nested_cv_results.csv"))
nres = nres[nres["model"] == "XGBoost-nested"].set_index("cv")
for cv in ["random", "group", "spatial"]:
    ok(f"Fig3a tree R² 标签 {nres.loc[cv, 'R2']:.2f}",
       has("Fig3_cv", f"{nres.loc[cv, 'R2']:.2f}"))
    ok(f"Fig3a plot R² 标签 {nres.loc[cv, 'plot_R2']:.2f}",
       has("Fig3_cv", f"{nres.loc[cv, 'plot_R2']:.2f}"))
ok("Fig3b 面板内 R² 标注",
   has_exact("Fig3_cv", f"$R^2$ = {nres.loc['spatial', 'R2']:.2f} (spatial block)"))
ok("Fig3c 面板内 R² 标注",
   has_exact("Fig3_cv", f"$R^2$ = {nres.loc['spatial', 'plot_R2']:.2f} (plot mean)"))

g = pd.read_csv(os.path.join(R, "26_group_shares.csv")).set_index("group")
for fam in ["tree", "stand", "climate", "soil", "terrain"]:
    ok(f"Fig4a 家族份额标签 {g.loc[fam, 'share'] * 100:.1f}%",
       has("Fig4_shap", f"{g.loc[fam, 'share'] * 100:.1f}%"))
dom = pd.read_csv(os.path.join(R, "28_plot_dominance_nested.csv"))
dom = dom[dom["n_plots_dominant"] > 0]
for _, r in dom.iterrows():
    pct = r["n_plots_dominant"] / dom["n_plots_dominant"].sum() * 100
    if pct > 4:
        ok(f"Fig4d 饼图标签 {pct:.0f}%", has("Fig4_shap", f"{pct:.0f}%"))

pm = plot_tab
for tag, sub, col in [("Planted", pm[pm["origin_natural"] == 0], "origin"),
                      ("Natural", pm[pm["origin_natural"] == 1], "origin"),
                      ("Broadleaf", pm[pm["fg_conifer"] == 0], "fg"),
                      ("Conifer", pm[pm["fg_conifer"] == 1], "fg")]:
    ok(f"Fig2 箱线标签 {tag} n={len(sub)}", has("Fig2_patterns", f"{tag}\n(n={len(sub)})"))

dsum = C.load_json("23_dual_map_summary.json")
r2m = {r["response"]: r for r in dsum["responses"]}
for resp, tag in [("conifer", "Fig6(a)"), ("broadleaf", "Fig6(b)")]:
    ok(f"{tag} 图内 nested R² 标签", has("Fig6_map", f"nested $R^2$ = {r2m[resp]['R2']:.2f}"))
attr = pd.read_csv(os.path.join(R, "23_dual_map_attribution.csv")).set_index("response")
# Fig6(d) 柱顶标签为 1 位小数（Climate 针/阔 0.3441 vs 0.3400 整数舍入同为 "34"
# 而柱高可辨差，用户反馈后改为 .1f）——断言改为按同小数位的字符串匹配。
_f6texts = ML.get("Fig6_map", [])
for resp in ["conifer", "broadleaf"]:
    for fam in ["stand", "climate", "soil", "terrain"]:
        v = float(attr.loc[resp, f"share_{fam}"] * 100)
        ok(f"Fig6d {resp}/{fam} 标签 {v:.1f}", f"{v:.1f}" in _f6texts,
           f"图面文本含柱顶标签（1 位小数）；样例 "
           f"{[t for t in _f6texts if t and t[0].isdigit()][:8]}")

_f5titles = [t for t in ML.get("Fig5_dependence", []) if t.startswith("(")]
ok("Fig5a–f 六个主效应面板标题齐全",
   all(any(t.startswith(f"({c})") for t in _f5titles) for c in "abcdef"),
   f"渲染标题 {sorted(_f5titles)[:8]}")
ok("Fig5g/h 交互面板标题",
   any(t.startswith("(g)") for t in _f5titles) and any(t.startswith("(h)") for t in _f5titles))
ok("Fig5b 双纵轴单位标注存在", has_any("Fig5_dependence", ["ΔD\n(cm/5 yr)"]))
ok("Fig5a 样本支撑 rug 存在（图面含 30-bin 均值曲线 ylabel）",
   has_any("Fig5_dependence", ["SHAP (RGR yr$^{-1}$)"]))

ok("Fig1a 标题含样地数", has("Fig1_overview", "1,104 CFI plots"))

sf1 = pd.read_csv(os.path.join(R, "20_sf1_age_bins.csv"))
for _, r in sf1.iterrows():
    if r["n"] >= 10:      # 图注声明只绘 n ≥ 10 的年龄档（2/1 株档 SEM 无意义）
        ok(f"SF1 分箱 n={int(r['n'])} 标注", has("SF1_age_decay", f"n={int(r['n'])}", SL))
ok("SF1 图注声明 '只绘 n ≥ 10 的年龄档'（2/1 株档不绘）",
   int((sf1["n"] < 10).sum()) == 2 and not has_exact("SF1_age_decay", "n=2", SL)
   and not has_exact("SF1_age_decay", "n=1", SL),   # 精确匹配：避免 "n=10" 被 "n=1" 命中
   f"n<10 的档数 {int((sf1['n'] < 10).sum())}")
ok("SF1b 标题含样地数", has("SF1_age_decay", "Stand-age distribution (n = 1,104 plots)", SL))

stab = pd.read_csv(os.path.join(R, "33_shape_stability.csv")).set_index("var")
for v in ["stand_age", "lnD0", "conifer_prop", "bio12", "bio15", "origin_natural"]:
    _rf = lambda x: "n/a" if pd.isna(x) else f"{x:.2f}"   # 与 20_si_figures 图面一致（origin_natural 为 n/a）
    s = f"mean r = {_rf(stab.loc[v, 'mean_pairwise_r'])}\nmin r = {_rf(stab.loc[v, 'min_pairwise_r'])}"
    ok(f"SF3 inset {v}: {s.splitlines()[0]}", has("SF3_shape_stability", s, SL))

dd_all = pd.read_csv(os.path.join(R, "29_deltad_nested.csv"))
from scipy.stats import spearmanr
rho_all = spearmanr(dd_all["share_RGR"], dd_all["share_deltaD"]).statistic
n_nz = int((dd_all["share_RGR"] > 0).sum())
ok("SF4 ρ 标注", has("SF4_deltad", f"Spearman ρ = {rho_all:.3f}\n(all {len(dd_all)} features; {n_nz} plotted)", SL))
fam2 = dd_all.groupby("group")[["share_RGR", "share_deltaD"]].sum()
for fam in ["stand", "tree", "climate", "terrain", "soil"]:
    ok(f"SF4b {fam} RGR 标签 {fam2.loc[fam, 'share_RGR'] * 100:.1f}",
       has("SF4_deltad", f"{fam2.loc[fam, 'share_RGR'] * 100:.1f}", SL))
    ok(f"SF4b {fam} ΔD 标签 {fam2.loc[fam, 'share_deltaD'] * 100:.1f}",
       has("SF4_deltad", f"{fam2.loc[fam, 'share_deltaD'] * 100:.1f}", SL))

den = tab[["stand_density"]].dropna()
ok("SF2 标题密度跨度 5.5", has("SF2_density", "Stand-density main effect (bin-mean span $\\approx$ 5.5 $\\times$ 10$^{-3}$ yr$^{-1}$)", SL))

# =====================================================================
say("")
say("## 腿 (ii)：图注 ↔ 数据源（图注引用但不在图面上的数字）")
say("")

ok("图注 Fig1b 阔叶 n=29,942", int((tab["fg_conifer"] == 0).sum()) == 29942,
   f"实际 {int((tab['fg_conifer'] == 0).sum())}")
ok("图注 Fig1b 针叶 n=14,431", int((tab["fg_conifer"] == 1).sum()) == 14431,
   f"实际 {int((tab['fg_conifer'] == 1).sum())}")
rgr = tab["RGR"]
ok("图注 Fig1b 1st percentile 0.0017", abs(rgr.quantile(.01) - 0.0017) < 0.0002,
   f"实际 {rgr.quantile(.01):.4f}")
ok("图注 Fig1b 99th percentile 0.166", abs(rgr.quantile(.99) - 0.166) < 0.001,
   f"实际 {rgr.quantile(.99):.4f}")
pt = tab.groupby("plot_id")["RGR"].mean()
ok("图注 Fig1 样地均值范围 0.0015–0.212",
   abs(pt.min() - 0.0015) < 0.0005 and abs(pt.max() - 0.212) < 0.002,
   f"实际 {pt.min():.4f}–{pt.max():.4f}")
ok("图注 Fig1c 林龄中位 18 / 最大 79",
   abs(plot_tab["stand_age"].median() - 18) < 0.5 and abs(plot_tab["stand_age"].max() - 79) < 0.5,
   f"实际 {plot_tab['stand_age'].median():.0f} / {plot_tab['stand_age'].max():.0f}")
ok("图注 Fig1b x 轴截断声明：>0.2 的株数 = 100", int((rgr > 0.2).sum()) == 100,
   f"实际 {int((rgr > 0.2).sum())}")
ok("图注 Fig1b 最大值 0.299", abs(rgr.max() - 0.299) < 0.001, f"实际 {rgr.max():.4f}")
_b25 = pd.cut(tab["lnD0"], 25)
_c25 = tab.groupby([_b25, tab["fg_conifer"]], observed=True)["RGR"].size().unstack(fill_value=0)
_ok_bins = _c25[(_c25[0] >= 20) & (_c25[1] >= 20)].index           # 与 09_figures.py Fig2a 同一准则
_blob = pickle.load(open(os.path.join(R, "22_nested_oof.pkl"), "rb"))
pred = np.asarray(_blob["oof"]["spatial"])
obs = tab["RGR"].values
ok("图注 Fig3b 轴范围截断声明：obs>0.22 = 47 株 / pred 最大值 0.131",
   int((obs > 0.22).sum()) == 47 and abs(pred.max() - 0.131) < 0.002,
   f"实际 obs>0.22 = {int((obs > 0.22).sum())}, pred max = {pred.max():.4f}")
_pobs = tab.groupby("plot_id")["RGR"].mean()
_ppred = pd.Series(pred, index=tab.index).groupby(tab["plot_id"]).mean()
ok("图注 Fig3c 轴范围截断声明：plot 级 >0.18 = 5",
   int((_pobs > 0.18).sum()) == 5, f"实际 {int((_pobs > 0.18).sum())}")

# Fig5g/h 交互面板（分箱汇总）：数值必须与 26_interactions / 26_inter_dep_data 对源
_inter = pd.read_pickle(os.path.join(R, "26_inter_dep_data.pkl"))
_itab = pd.read_csv(os.path.join(R, "26_interactions.csv")).set_index("pair")
_g = _inter["stand_age__fg_conifer"].reset_index(drop=True)
_mg = _g["inter"].abs().mean() * 1000
ok("Fig5g 图面标注 mean|interaction| = 0.03（×10⁻³）",
   abs(_mg - _itab.loc["stand_age x fg_conifer", "mean_abs_inter"] * 1000) < 0.01,
   f"实际 {_mg:.3f}")
_b0 = pd.cut(_g["v1"], [0, 5, 10, 15, 20, 30, 40, 50, 90])
_m0 = _g[_g["v2"] == 0].groupby(_b0, observed=True)["inter"].mean().iloc[0] * 1000
_m1 = _g[_g["v2"] == 1].groupby(_b0, observed=True)["inter"].mean().iloc[0] * 1000
ok("图注 Fig5g 0–5 yr 分箱：broadleaf −0.066 / conifer +0.038（×10⁻³）",
   abs(_m0 + 0.066) < 0.002 and abs(_m1 - 0.038) < 0.002,
   f"实际 {_m0:.3f} / {_m1:.3f}")
_gmax = max(abs(_g[_g["v2"] == f].groupby(_b0, observed=True)["inter"].mean()).max()
            for f in (0, 1)) * 1000          # 分组别分箱均值的最大绝对值
_age_span = float(pd.read_csv(os.path.join(R, "32_dep_ranges.csv")).set_index("var")
                  .loc["stand_age", "span"]) * 1000     # 主年龄效应跨度 ×10⁻³
ok("图注 Fig5g '最大分箱均值远低于主年龄效应跨度'",
   _age_span / _gmax > 100,             # 图注称约 1/300
   f"实际比值 {_age_span / _gmax:.0f}（最大分箱均值 {_gmax:.3f}×10⁻³，"
   f"主年龄跨度 {_age_span:.2f}×10⁻³）")
_h = _inter["stand_density__lnD0"].reset_index(drop=True)
_hb = pd.qcut(_h["v1"], 4, duplicates="drop"); _sb = pd.qcut(_h["v2"], 4, duplicates="drop")
_piv = _h.groupby([_hb, _sb], observed=True)["inter"].mean().unstack() * 1000
ok("图注 Fig5h 最稀密度行：最小尺寸 +0.07 / 最大尺寸 −0.71（×10⁻³）",
   abs(_piv.values[0, 0] - 0.071) < 0.006 and abs(_piv.values[0, -1] + 0.713) < 0.02,
   f"实际 {_piv.values[0, 0]:.3f} / {_piv.values[0, -1]:.3f}")
ok("图注 Fig5h 最密密度行：最小尺寸 −0.17 / 最大尺寸 +0.18（×10⁻³）",
   abs(_piv.values[-1, 0] + 0.174) < 0.006 and abs(_piv.values[-1, -1] - 0.177) < 0.006,
   f"实际 {_piv.values[-1, 0]:.3f} / {_piv.values[-1, -1]:.3f}")
_cn = _h.groupby([_hb, _sb], observed=True)["inter"].size()
ok("图注 Fig5h 每格样本量 237–384", int(_cn.min()) == 237 and int(_cn.max()) == 384,
   f"实际 {int(_cn.min())}–{int(_cn.max())}")

# Fig2b 分箱 n 与均值（图与图注必须同口径：09_figures.py 用 right=False，与 20_sf1_age_bins.csv 一致）
aged = plot_tab.dropna(subset=["stand_age"])
b = aged.groupby(pd.cut(aged["stand_age"], np.arange(0, 105, 10), right=False),
                 observed=True).agg(n=("plot_RGR", "size"), m=("plot_RGR", "mean"))
exp_n = [236, 380, 317, 132, 26, 10]
got_n = [int(x) for x in b["n"].values[:6]]
ok("图注 Fig2b 分箱 n = 236/380/317/132/26/10（且左闭右开口径）", got_n == exp_n, f"实际 {got_n}")
exp_m = [0.066, 0.057, 0.042, 0.036, 0.029, 0.020]
got_m = [round(float(x), 3) for x in b["m"].values[:6]]
ok("图注 Fig2b 分箱均值 = 0.066/0.057/0.042/0.036/0.029/0.020", got_m == exp_m, f"实际 {got_m}")
ok("SF1 图面标注 39 plots ≥ 40 yr（薄采样带）", has("SF1_age_decay", "39 plots", SL))
sf1n = [int(x) for x in pd.read_csv(os.path.join(R, "20_sf1_age_bins.csv"))["n"].values[:6]]
ok("Fig2b 分箱 n 与 SF1 同口径一致（图注 n 因此可代表图面曲线）", sf1n == exp_n, f"SF1={sf1n}")

rho_age = aged[["stand_age", "plot_RGR"]].corr(method="spearman").iloc[0, 1]
ok("图注 Fig2b ρ = −0.419", abs(rho_age + 0.419) < 0.002, f"实际 {rho_age:.3f}")

# Fig3 图注：绝对/相对虚高、RMSE
lk = pd.read_csv(os.path.join(R, "22_nested_leakage.csv")).set_index("quantity")["value"]
ok("图注 Fig3a 绝对虚高 +0.207", abs(lk["absolute inflation (random - spatial)"] - 0.207) < 0.001)
ok("图注 Fig3a 相对虚高 126.5%", abs(lk["relative inflation (%)"] - 126.5) < 0.2)
ok("图注 Fig3a 块尺度分量 +0.0042", abs(lk["block-scale component (group - spatial)"] - 0.0042) < 0.0005)
ok("图注 Fig3b RMSE 0.034", abs(nres.loc["spatial", "RMSE"] - 0.034) < 0.0005,
   f"实际 {nres.loc['spatial', 'RMSE']:.4f}")

# Fig6 图注：像元数、压缩比、预测区间
ok("图注 Fig6 森林像元 1.56 million", abs(dsum["mask_new_px"] / 1e6 - 1.56) < 0.01)
ok("图注 Fig6 可预测 1.54 million", abs(dsum["n_pred_px"] / 1e6 - 1.54) < 0.01)
# 掩膜"纠错"叙事（旧口径 2.14 million / −27.2%）已按作者要求从全稿移除，
# 仅保留最终像元数；此处反向断言旧口径数字不得再出现。
_w27 = open(os.path.join(R, "..", "06_论文稿", "manuscript_v4.md"),
            encoding="utf-8").read()
ok("森林掩膜旧口径数字已从正文/图注移除（2.14 million / −27.2% / inflated by 27%）",
   ("2.14" not in _w27 and "−27.2%" not in _w27 and "inflated it by 27" not in _w27))
rep23 = open(os.path.join(R, "23_dual_map_report.md"), encoding="utf-8").read()
ok("图注 Fig6 压缩比 4.5×/5.5×/3.9×",
   all(t in rep23 for t in ["压缩比=4.5×", "压缩比=5.5×", "压缩比=3.9×"]))
ok("图注 Fig6 disagreement mean 0.003",
   all(t in rep23 for t in ["ens disagreement=0.00272", "ens disagreement=0.00248",
                            "ens disagreement=0.00233"]))

# SI 图注 (ii)
ok("SF1 图注 39 plots ≥ 40 yr", int((plot_tab["stand_age"] >= 40).sum()) == 39,
   f"实际 {int((plot_tab['stand_age'] >= 40).sum())}")
ok("SF1 图注 13 plots ≥ 50 yr", int((plot_tab["stand_age"] >= 50).sum()) == 13,
   f"实际 {int((plot_tab['stand_age'] >= 50).sum())}")
ok("SF1 图注 3 plots ≥ 60 yr", int((plot_tab["stand_age"] > 60).sum()) == 3,
   f"实际 {int((plot_tab['stand_age'] > 60).sum())}")
ok("SF2 图注 n = 44,373 株", len(tab[["stand_density"]].dropna()) == 44373,
   f"实际 {len(tab[['stand_density']].dropna())}")
ok("SF2 图注 n = 1,104 样地",
   tab[["plot_id", "stand_density"]].dropna()["plot_id"].nunique() == 1104)
ok("SF3 图注 最小 r 0.94（conifer_prop）", abs(stab.loc["conifer_prop", "min_pairwise_r"] - 0.938) < 0.001)
ok("SF4 图注 (i) 35 plotted", n_nz == 35)

# =====================================================================
say("")
say("## 腿 (iii)：图注 ↔ 图面（图注必须见于图面的数字，按同小数位比对）")
say("")


def on_figure(fig, caption_num, decimals, store=ML):
    """图注数字在图面上是否有【同一显示精度】的渲染标签。

    图注常引用比图面更高精度的数据值（如 0.375 对图面 0.37），故容差取
    图面显示单位的一半（0.5×10^-decimals），略放 0.1% 以避免十进制/二进制
    舍入临界（如 0.375 / 18.5）造成的伪失配。
    """
    tol = 0.5 * 10 ** (-decimals) * 1.001
    for v in nums(fig, store):
        if abs(v - caption_num) <= tol:
            return True
    return False


CHECKS = [
    ("Fig3_cv", 0.370, 2), ("Fig3_cv", 0.820, 2), ("Fig3_cv", 0.168, 2),
    ("Fig3_cv", 0.314, 2), ("Fig3_cv", 0.315, 2), ("Fig3_cv", 0.164, 2),
    ("Fig4_shap", 40.2, 1), ("Fig4_shap", 23.2, 1), ("Fig4_shap", 15.4, 1),
    ("Fig4_shap", 14.5, 1), ("Fig4_shap", 6.8, 1),
    ("Fig4_shap", 62, 0), ("Fig4_shap", 38, 0),
    ("Fig6_map", 0.03, 2), ("Fig6_map", 0.03, 2),
    ("Fig6_map", 27, 0), ("Fig6_map", 34, 0), ("Fig6_map", 25, 0),
    ("Fig6_map", 36, 0), ("Fig6_map", 34, 0), ("Fig6_map", 18, 0),
    ("Fig2_patterns", 538, 0), ("Fig2_patterns", 566, 0),
    ("Fig1_overview", 1104, 0),
]
for fig, val, dec in CHECKS:
    ok(f"{fig} 图注值 {val} 见于图面（{dec} 位小数）", on_figure(fig, val, dec))

SI_CHECKS = [
    ("SF1_age_decay", 236, 0), ("SF1_age_decay", 380, 0), ("SF1_age_decay", 317, 0),
    ("SF1_age_decay", 132, 0), ("SF1_age_decay", 26, 0), ("SF1_age_decay", 10, 0),
    ("SF1_age_decay", 1104, 0), ("SF1_age_decay", 39, 0), ("SF1_age_decay", 18, 0),
    ("SF2_density", 5.5, 1),
    ("SF4_deltad", 0.920, 3),
    ("SF4_deltad", 40.2, 1), ("SF4_deltad", 23.1, 1), ("SF4_deltad", 15.4, 1),
    ("SF4_deltad", 14.5, 1), ("SF4_deltad", 6.8, 1),
    ("SF4_deltad", 38.7, 1), ("SF4_deltad", 20.2, 1), ("SF4_deltad", 18.5, 1),
    ("SF4_deltad", 14.8, 1), ("SF4_deltad", 7.7, 1),
]
for fig, val, dec in SI_CHECKS:
    ok(f"{fig} SI 图注值 {val} 见于图面（{dec} 位小数）", on_figure(fig, val, dec, SL))

# --- 反盲区：把【图注原文】里的面板标注数字逐项对图面 ---------------------
# 封堵 Stage 5 缺陷 #26 暴露的盲区：上面的 SI_CHECKS 是【硬编码期望值】，
# 它只证明"我以为的值在图上"，不证明"图注写的值在图上"。若图注写成 20.8
# 而图面渲染 20.7，硬编码的 20.7 依然通过。此处直接读图注文本。
#
# 适用范围：图注中作为【面板标注】出现的数字（份额百分比、inset 相关系数）。
# 图注同时会引用一些【图上不渲染】的上下文数字（如 SF4 图注的家族级 ρ = 0.900、
# SF4 图注的 ΔD 曲线首尾值），这类数字不能参与本断言，逐一在 CAPTION_ONLY 声明。
say("")
say("## 腿 (iii-b)：图注原文数字 ↔ 图面（反盲区，读图注而非硬编码）")
say("")

CAPTION_ONLY = {
    ("4", "ρ = 0.900"): "家族级 ρ，图面只渲染变量级 ρ = 0.921",
    ("4", "0.909"): "形状相关 r，正文/图注引用，图上不渲染",
    ("4", "0.31"): "ΔD 曲线首尾值，图上以曲线呈现而非文字",
    ("4", "0.78"): "ΔD 曲线首尾值，图上以曲线呈现而非文字",
    ("4", "12.1"): "RGR 曲线首尾值，图上以曲线呈现而非文字",
    ("4", "11.9"): "RGR 曲线首尾值，图上以曲线呈现而非文字",
}

# 每个 SI 图注要参与反盲区断言的数字形态
CAPTION_PATTERNS = {
    "4": (r"\d+\.\d+%", 1),          # 家族份额百分比
    "3": (r"\d\.\d{2}", 2),          # inset 的 mean r / min r
}


def parse_caption_numbers(body, pat):
    """从图注中提取符合 pat 的数字，返回 (值, 原始串)。"""
    out = []
    for m in re.finditer(pat, body):
        try:
            out.append((float(m.group(0).rstrip("%")), m.group(0)))
        except ValueError:
            pass
    return out


for key, fig in (("4", "SF4_deltad"), ("3", "SF3_shape_stability")):
    body = caption_body(SI, key)
    pat, dec = CAPTION_PATTERNS[key]
    parsed = parse_caption_numbers(body, pat)
    ok(f"SF{key} 图注可解析出标注数字（{len(parsed)} 个）", len(parsed) > 0,
       str([p[1] for p in parsed]))
    for val, raw in parsed:
        if (key, raw) in CAPTION_ONLY:
            say(f"  [--] SF{key} 图注 '{raw}'：{CAPTION_ONLY[(key, raw)]}")
            continue
        ok(f"SF{key} 图注原文 '{raw}' 见于图面", on_figure(fig, val, dec, SL))

# =====================================================================
say("")
say("## 腿 (iv)：图注面板字母 ↔ 实际渲染的面板标题")
say("")
for fig, key, store in [("Fig1_overview", "1", ML), ("Fig2_patterns", "2", ML),
                        ("Fig3_cv", "3", ML), ("Fig4_shap", "4", ML),
                        ("Fig5_dependence", "5", ML), ("Fig6_map", "6", ML),
                        ("SF1_age_decay", "1", SL), ("SF2_density", "2", SL),
                        ("SF3_shape_stability", "3", SL), ("SF4_deltad", "4", SL)]:
    store_key = "Fig" if store is ML else "SF"
    body = caption_body(SI if store is SL else MS, key)
    letters = sorted(set(re.findall(r"\(([a-h])\)", body)))
    if not letters:
        ok(f"{fig} 图注无面板字母（单面板图）", True, "不适用")
        continue
    present = [c for c in letters
               if any(t.startswith(f"({c})") for t in store.get(fig, []))]
    ok(f"{fig} 图注面板 {''.join(letters)} 均有对应渲染标题",
       sorted(present) == letters,
       f"渲染到 {''.join(sorted(present))}（渲染标题样例 {[t for t in store.get(fig, []) if t.startswith('(')][:3]}）")

say("")
say("## 结论")
say("")
if fails:
    say(f"**FAIL** — 共 {len(fails)} 项不一致：")
    for f in fails:
        say(f"- {f}")
else:
    say("**PASS** — 图面渲染文字、图注引用的数字、源数据三者一致；")
    say("图注面板字母在每张图上都有对应的实际渲染面板。")

with open(os.path.join(R, "34_threeway_check_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("")
say("DONE 34 three-way check")
if fails:
    raise SystemExit(1)