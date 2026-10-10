# -*- coding: utf-8 -*-
"""P1-12 附：极端增量阈值敏感性修正。
问题: 08_sensitivity.py 的 threshold 250/300 在已剔除 >200 mm 的清洗表上运行，
      两档与 200 完全相同（伪对照）。原始数据中 (200,250] 有 13 株、(250,300] 有 2 株。
修复: 从原始样木表恢复 19 株被剔除极端树，重建模型表行，真实重算 150/200/250/300。
输出: 08_sensitivity_results.csv 中 threshold_* 行更新 + 21_threshold_fix_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

report = []
def say(x):
    print(x, flush=True)
    report.append(str(x))

say("# 极端增量阈值敏感性修正（P1-12 附）\n")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "block05", "fold_spatial", "fold_group"]],
                on="plot_id", how="left")
params = C.load_json("02_best_params.json")["xgb_best"]
feats = C.selected_features(vd)
group_of = dict(zip(vd["var"], vd["group"]))
groups = ["tree", "stand", "climate", "soil", "terrain"]

# ---------- 1. 从原始数据恢复被剔除的极端树 ----------
raw = pd.read_excel(os.path.join(C.ROOT, "样点数据", "样地样木因子_整合.xlsx"),
                    sheet_name="样木因子")
raw = raw.rename(columns={"上一期调查胸径": "D0", "本期调查胸径": "D1",
                          "样地号": "plot_id", "树种分类": "fg"})
raw = raw[raw["D0"] > 0].copy()
raw["growth"] = raw["D1"] - raw["D0"]
ext = raw[raw["growth"] > 200].copy()
say(f"- 原始配对样本中被剔除的极端树: {len(ext)} 株"
    f"（增量 {ext['growth'].min()}–{ext['growth'].max()} mm/5yr）")

plot_tpl = tab.drop_duplicates("plot_id").set_index("plot_id")
ext = ext[ext["plot_id"].isin(plot_tpl.index)].copy()
say(f"- 全部位于保留样地，可重建模型行: {len(ext)} 株")

ext["RGR"] = np.log(ext["D1"] / ext["D0"]) / 5.0
ext["lnD0"] = np.log(ext["D0"])
ext["fg_conifer"] = (ext["fg"] == "针叶树种").astype(int)
tree_cols = ["plot_id", "D0", "D1", "RGR", "lnD0", "fg_conifer", "growth"]
plot_cols = [c for c in tab.columns if c not in tree_cols]
extm = ext[tree_cols].merge(plot_tpl[plot_cols].reset_index(), on="plot_id", how="left")
# 特征列允许 NaN（土壤等原生缺失，与基准表同样由 XGBoost 处理）；仅折列与响应列必须完整
assert extm["fold_spatial"].notna().all() and extm["RGR"].notna().all(), "极端树行折列/响应缺失"
say(f"- 重建行: {len(extm)}，折列与响应完整（特征 NaN 与基准表同口径: "
    f"{int(extm[feats].isna().sum().sum())} 个）")

base = tab.dropna(subset=["fold_spatial"]).copy()
say(f"- 基准表（growth ≤ 200）: {len(base)} 株\n")

def run_variant(sub, tag):
    oof, models = C.fit_xgb_folds(sub, feats, params, "fold_spatial", return_models=True)
    m = C.metrics(sub["RGR"], oof)
    pm = sub.assign(pred=oof).groupby("plot_id")[["RGR", "pred"]].mean()
    m["plot_R2"] = C.metrics(pm["RGR"], pm["pred"])["R2"]
    row = {"variant": tag, "n": len(sub), "tree_R2": m["R2"], "plot_R2": m["plot_R2"],
           "RMSE": m["RMSE"], "Spearman": m["Spearman"]}
    parts = []
    for k, mm in enumerate(models):
        te = sub[sub["fold_spatial"] == k]
        dm = xgb.DMatrix(te[feats])
        sv = mm.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        parts.append(pd.DataFrame(sv, index=te.index, columns=feats))
    sm = pd.concat(parts)
    gs = sm.abs().mean()
    for g in groups:
        share = gs[[f for f in feats if group_of.get(f) == g]].sum() / gs.sum()
        row[f"share_{g}"] = round(float(share), 4)
    return row

say("## 修正后的四档阈值（真实样本）")
rows = []
for thr in (150, 200, 250, 300):
    if thr == 150:
        sub = base[base["D1"] - base["D0"] <= 150]
    elif thr == 200:
        sub = base
    else:
        add = extm[extm["growth"] <= thr]
        sub = pd.concat([base, add[[c for c in base.columns]]], ignore_index=True)
    row = run_variant(sub, f"threshold_{thr}mm")
    rows.append(row)
    say(f"- ≤{thr}mm: n={row['n']}, tree R2={row['tree_R2']:.4f}, plot R2={row['plot_R2']:.4f}, "
        + ", ".join(f"{g}={row[f'share_{g}']:.3f}" for g in groups))

# ---------- 2. 更新敏感性总表 ----------
df = pd.read_csv(os.path.join(C.OUT, "08_sensitivity_results.csv"))
df = df[~df["variant"].str.startswith("threshold_")]
new = pd.DataFrame(rows)
cols = list(df.columns)
new = new[cols]
df = pd.concat([new, df], ignore_index=True)
df = df.sort_values("variant", key=lambda s: s.map(
    {r["variant"]: i for i, r in enumerate(rows)}).fillna(99))
df.to_csv(os.path.join(C.OUT, "08_sensitivity_results.csv"), index=False, encoding="utf-8-sig")

r150 = rows[0]["tree_R2"]; r250 = rows[2]["tree_R2"]; r300 = rows[3]["tree_R2"]
say(f"\n- 阈值间 tree R2 极差: {max(r['tree_R2'] for r in rows) - min(r['tree_R2'] for r in rows):.4f}")
say("- 结论: 主文 '阈值 150–300 mm 改变 tree R² < 0.008、类组份额基本不变' 的表述在真实样本下"
    + ("成立" if max(r['tree_R2'] for r in rows) - min(r['tree_R2'] for r in rows) < 0.008 else "需更新"))

with open(os.path.join(C.OUT, "21_threshold_fix_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\nDONE threshold fix")
