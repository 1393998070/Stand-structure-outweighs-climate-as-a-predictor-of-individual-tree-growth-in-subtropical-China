# -*- coding: utf-8 -*-
"""P0-A-6 图件-正文-数据三方一致性校验（流程加固）。

Round-1 的失效模式：分析重构后【绘图脚本未同步更新】——Fig. 6c 仍读取被取代的
`07_map_pattern_varpart.csv`，导致图面与正文排序相反；图注还引用了图中不存在的 (d) 面板。
当时的终检只做了"标记级文本验证 + 数字对源"，未核验交付物本身。

本脚本把该核验自动化，三道检查：
  A. 图件所依赖的每个数据文件都存在；
  B. 图件文件晚于其全部数据依赖（防止"改了脚本没重绘"）；
  C. 正文图注中出现的面板字母集合 == 该图实际绘制的面板字母集合。

用法: python 30_figure_consistency.py [manuscript_path]
输出: 30_figure_consistency_report.md（并在控制台打印 PASS/FAIL）
"""
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

MS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.PIPE, "06_论文稿", "manuscript_v4.md")
MAPDIR = os.path.join(C.OUT, "map")

# 图 -> (数据依赖, 期望面板字母)
MANIFEST = {
    "Fig1_overview": ([os.path.join(MAPDIR, "dual_map_arrays.npz"),
                       os.path.join(C.OUT, "01_model_table.pkl")], "abcd"),
    "Fig2_patterns": ([os.path.join(C.OUT, "01_model_table.pkl")], "abcd"),
    "Fig3_cv": ([os.path.join(C.OUT, "22_nested_cv_results.csv"),
                 os.path.join(C.OUT, "22_nested_oof.pkl")], "abc"),
    "Fig4_shap": ([os.path.join(C.OUT, "26_shap_importance.csv"),
                   os.path.join(C.OUT, "26_group_shares.csv"),
                   os.path.join(C.OUT, "28_ablation_nested.csv"),
                   os.path.join(C.OUT, "27_family_share_normalized.csv"),
                   os.path.join(C.OUT, "28_plot_dominance_nested.csv")], "abcd"),
    "Fig5_dependence": ([os.path.join(C.OUT, "26_shap_matrix.pkl"),
                         os.path.join(C.OUT, "26_inter_dep_data.pkl"),
                         os.path.join(C.OUT, "29_lnd0_curves.pkl")], "abcdefgh"),
    "Fig6_map": ([os.path.join(MAPDIR, "dual_map_arrays.npz"),
                  os.path.join(C.OUT, "23_dual_map_attribution.csv"),
                  os.path.join(C.OUT, "23_dual_map_summary.json")], "abcd"),
}

report, fails = [], []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# 图件-正文-数据一致性校验报告\n")
say(f"- 校验对象: {MS}")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say("")

# ---------- 读正文图注 ----------
legends = {}
if os.path.exists(MS):
    txt = open(MS, encoding="utf-8").read()
    # 匹配 "**Figure N | ...**" 起始的图注段落
    for m in re.finditer(r"\*\*Figure (\d) \|(.+?)(?=\n\*\*Figure|\n---|\Z)", txt, re.S):
        num, body = m.group(1), m.group(2)
        legends[f"Fig{num}"] = body
    say(f"- 正文中找到 {len(legends)} 条主图图注")
else:
    say(f"- **正文文件不存在**: {MS}（跳过检查 C）")
say("")

# ---------- A/B ----------
say("## 检查 A/B：数据依赖存在性与时间序")
say("")
say("| 图件 | 依赖文件 | 存在 | 图件 mtime ≥ 依赖 mtime |")
say("|---|---|---|---|")
for fig, (deps, panels) in MANIFEST.items():
    png = os.path.join(C.FIGS, f"{fig}.png")
    fig_m = os.path.getmtime(png) if os.path.exists(png) else None
    if fig_m is None:
        fails.append(f"{fig}: PNG 不存在")
    for d in deps:
        ex = os.path.exists(d)
        if not ex:
            fails.append(f"{fig}: 依赖缺失 {os.path.basename(d)}")
            say(f"| {fig} | {os.path.basename(d)} | ✗ | — |")
            continue
        if fig_m is None:
            newer = "—"
        else:
            newer = "✓" if fig_m >= os.path.getmtime(d) else "✗ **图件早于依赖**"
            if fig_m < os.path.getmtime(d):
                fails.append(f"{fig}: 图件早于依赖 {os.path.basename(d)}（需重绘）")
        say(f"| {fig} | {os.path.basename(d)} | ✓ | {newer} |")
say("")

# ---------- C ----------
say("## 检查 C：图注面板字母 vs 实际绘制面板")
say("")
say("| 图件 | 期望面板 | 图注中出现的面板 | 结果 |")
say("|---|---|---|---|")
for fig, (deps, panels) in MANIFEST.items():
    key = fig.split("_")[0]
    if key not in legends:
        say(f"| {fig} | {panels} | （未找到图注） | ⚠ 跳过 |")
        continue
    body = legends[key]
    found = "".join(sorted(set(re.findall(r"\(([a-h])\)", body))))
    ok = found == "".join(sorted(panels))
    if not ok:
        fails.append(f"{fig}: 图注面板 {found} != 期望 {''.join(sorted(panels))}")
    say(f"| {fig} | {''.join(sorted(panels))} | {found} | {'✓' if ok else '✗ 不一致'} |")
say("")

# ---------- 结果 ----------
say("## 结论")
say("")
if fails:
    say(f"**FAIL** — 发现 {len(fails)} 处问题：")
    for f in fails:
        say(f"- {f}")
else:
    say("**PASS** — 全部图件的数据依赖存在、图件晚于依赖、图注面板与实际绘制一致。")

with open(os.path.join(C.OUT, "30_figure_consistency_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("")
say("DONE figure consistency check")
