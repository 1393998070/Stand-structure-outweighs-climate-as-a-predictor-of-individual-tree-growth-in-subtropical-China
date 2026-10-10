# -*- coding: utf-8 -*-
"""P0-B-a 气候宽容度命题的方差匹配检验。

问题（R1-W2 / R2-W2 / DA-M-3）：按降水四分位分层后，层内降水方差被压缩到约 1/4，
climate 份额被机械压低，因此"层内 climate 份额仍低"不能检验"气候是否在更干处更重要"。

本检验改为【等宽滑动窗口】：沿降水梯度取固定宽度（400 mm）的重叠窗口，使各窗口的
降水方差在构造上可比（并报告实测层内 SD 以验证匹配），再在每个窗口内重跑
空间块 CV + OOF SHAP，比较 climate 份额随窗口位置（干旱度）的变化。

为比较的内部一致性，所有窗口使用同一套固定超参数与固定迭代数（不选参、不早停），
故窗口间的差异只反映数据本身，不含选参偏倚。

输出:
  24_var_matched_climate.csv / 24_var_matched_report.md
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

say("# P0-B-a 气候宽容度命题：方差匹配检验\n")
say(f"- 生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
say("- 设计: 沿降水梯度取固定宽度 300 mm 的重叠滑动窗口（方差匹配），窗口内重跑")
say("  空间块 CV + OOF SHAP；所有窗口共用同一固定超参数与迭代数（无选参偏倚）")
say("- 同时给出【全局模型子集】对照估计量（脚本 17 的口径），两种估计量互相验证")
say("")

tab, vd, folds = C.load_model_data()
tab = tab.merge(folds[["plot_id", "fold_spatial"]], on="plot_id", how="left")
feats = C.selected_features(vd)
grp_of = dict(zip(vd["var"], vd["group"]))

# 降水单位：模型表中 bio12 为 mm×10（样地表 peng_BIO12 同口径）
b = tab["bio12"].dropna()
lo, hi = float(b.min()), float(b.max())
say(f"- bio12 全距: {lo:.0f} – {hi:.0f}（mm×10，即 {lo/10:.0f} – {hi/10:.0f} mm）")
say(f"- 样地数 {tab['plot_id'].nunique()}，单木 {len(tab)}")
say("")

FIXED = {"max_depth": 5, "learning_rate": 0.02, "subsample": 1.0,
         "colsample_bytree": 0.9, "min_child_weight": 10, "reg_alpha": 0.5,
         "reg_lambda": 10.0}
N_EST = 250
WIDTH = 3000.0   # 300 mm（窄窗换取更多窗口，提升趋势检验的功效）
STEP = 1500.0    # 150 mm

def spatial_folds_for(df, n_folds=5, seed=C.SEED):
    b = C.block_id(df["lon"].values, df["lat"].values).values
    blk = pd.Series(b, index=df.index).value_counts().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blk))
    assign = C.assign_spatial_folds([list(blk.items())[i] for i in order], n_folds=n_folds)
    return pd.Series(b, index=df.index).map(assign)

def oof_shap_shares(df):
    """窗口内空间块 CV + OOF SHAP；返回 (family shares, oof R2, n_plots, n_trees)。"""
    f = spatial_folds_for(df)
    oof = pd.Series(np.nan, index=df.index)
    parts = []
    for k in range(5):
        tr = df[f != k]; te = df[f == k]
        if len(te) < 20 or len(tr) < 100:
            continue
        m = xgb.XGBRegressor(n_estimators=N_EST, tree_method="hist",
                             random_state=C.SEED + k, n_jobs=-1, **FIXED)
        m.fit(tr[feats], tr["RGR"], verbose=False)
        oof.loc[te.index] = m.predict(te[feats])
        dm = xgb.DMatrix(te[feats], feature_names=feats)
        sv = m.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        parts.append(pd.DataFrame(np.abs(sv), columns=feats))
    if not parts:
        return None
    mas = pd.concat(parts).mean(axis=0)
    fam = mas.groupby(pd.Series(grp_of).reindex(mas.index)).sum()
    share = fam / fam.sum()
    ok = oof.notna()
    r2 = float(1 - np.sum((df.loc[ok, "RGR"] - oof[ok]) ** 2)
               / np.sum((df.loc[ok, "RGR"] - df.loc[ok, "RGR"].mean()) ** 2))
    return share, r2, int(df.loc[ok, "plot_id"].nunique()), int(ok.sum())

# ---------- 等宽滑动窗口 ----------
centers = np.arange(lo + WIDTH / 2, hi - WIDTH / 2 + 1, STEP)
rows = []
say("## 1. 等宽滑动窗口（宽 300 mm，步长 150 mm）— 分层内重拟合计")
say("")
for c in centers:
    a0, a1 = c - WIDTH / 2, c + WIDTH / 2
    sub = tab[(tab["bio12"] >= a0) & (tab["bio12"] <= a1)].dropna(subset=feats + ["RGR", "lon", "lat"])
    if len(sub) < 2000 or sub["plot_id"].nunique() < 60:
        say(f"- 窗口 {a0/10:.0f}–{a1/10:.0f} mm: 样本不足（n={len(sub)}）跳过")
        continue
    out = oof_shap_shares(sub)
    if out is None:
        continue
    share, r2, npl, ntr = out
    rows.append({"center_mm": round(c / 10, 0), "lo_mm": round(a0 / 10, 0),
                 "hi_mm": round(a1 / 10, 0),
                 "within_SD_mm": round(float(sub["bio12"].std()) / 10, 1),
                 "n_trees": ntr, "n_plots": npl, "oof_R2": round(r2, 4),
                 **{f"share_{g}": round(float(share.get(g, np.nan)), 4)
                    for g in ("tree", "stand", "climate", "soil", "terrain")}})
    say(f"- 窗口 {a0/10:.0f}–{a1/10:.0f} mm（中心 {c/10:.0f}）：n={ntr} 株 / {npl} 样地，"
        f"层内 SD={sub['bio12'].std()/10:.1f} mm，OOF R²={r2:.3f}，"
        f"climate={share.get('climate', np.nan):.3f}，stand={share.get('stand', np.nan):.3f}")

wm = pd.DataFrame(rows)
wm.to_csv(os.path.join(C.OUT, "24_var_matched_climate.csv"), index=False,
          encoding="utf-8-sig")

# ---------- 1b. 同一窗口的【全局模型子集】口径（脚本 17 的估计量，作对照） ----------
say("\n## 1b. 同一窗口的【全局模型子集】口径（对照估计量）")
shap_path = os.path.join(C.OUT, "26_shap_matrix.pkl")
if os.path.exists(shap_path):
    Sall = pd.read_pickle(shap_path)["shap"]
    src = "26_shap_matrix.pkl（嵌套选参口径）"
else:
    Sall = pd.read_pickle(os.path.join(C.OUT, "04_shap_matrix.pkl"))
    src = "04_shap_matrix.pkl（同折选参口径，临时）"
say(f"- 全局 OOF SHAP 矩阵来源: {src}，形状 {Sall.shape}")
fam_idx = {g: [i for i, f in enumerate(feats) if grp_of[f] == g]
           for g in ("tree", "stand", "climate", "soil", "terrain")}
gsub = []
for c in centers:
    a0, a1 = c - WIDTH / 2, c + WIDTH / 2
    msk = ((tab["bio12"] >= a0) & (tab["bio12"] <= a1)).values
    common = Sall.index.intersection(tab.index[msk])
    vm = Sall.loc[common].abs().values.mean(axis=0)
    tot = vm.sum()
    sh = {g: float(vm[fam_idx[g]].sum() / tot) for g in fam_idx}
    gsub.append({"center_mm": round(c / 10, 0), "n_trees": int(len(common)),
                 **{f"share_{g}": round(sh[g], 4) for g in fam_idx}})
    say(f"- 窗口中心 {c/10:.0f} mm: n={len(common)}, climate={sh['climate']:.3f}, "
        f"stand={sh['stand']:.3f}")
gsd = pd.DataFrame(gsub)
gsd.to_csv(os.path.join(C.OUT, "24_global_subset_windows.csv"), index=False,
           encoding="utf-8-sig")

# ---------- 对照：原四分位分层（复现，展示方差压缩） ----------
say("\n## 2. 对照：原四分位分层（层内方差被压缩）")
q = pd.qcut(tab["bio12"], 4, labels=["Q1(driest)", "Q2", "Q3", "Q4(wettest)"])
qrows = []
for lab in ["Q1(driest)", "Q2", "Q3", "Q4(wettest)"]:
    sub = tab[q == lab].dropna(subset=feats + ["RGR", "lon", "lat"])
    out = oof_shap_shares(sub)
    if out is None:
        continue
    share, r2, npl, ntr = out
    qrows.append({"stratum": lab, "within_SD_mm": round(float(sub["bio12"].std()) / 10, 1),
                  "n_trees": ntr, "n_plots": npl, "oof_R2": round(r2, 4),
                  "share_climate": round(float(share.get("climate", np.nan)), 4),
                  "share_stand": round(float(share.get("stand", np.nan)), 4)})
    say(f"- {lab}: 层内 SD={sub['bio12'].std()/10:.1f} mm, n={ntr}, "
        f"climate={share.get('climate', np.nan):.3f}")
qd = pd.DataFrame(qrows)
qd.to_csv(os.path.join(C.OUT, "24_quartile_control.csv"), index=False, encoding="utf-8-sig")

# ---------- 统计 ----------
say("\n## 3. 检验结果")
if len(wm) >= 3:
    from scipy.stats import spearmanr, linregress
    rho, pv = spearmanr(wm["center_mm"], wm["share_climate"])
    lr = linregress(wm["center_mm"], wm["share_climate"])
    say(f"- 等宽窗口（n={len(wm)}）: Spearman(窗口中心, climate 份额) = {rho:+.3f} (P={pv:.3f})")
    say(f"- 线性趋势: slope = {lr.slope*100:.4f} %/mm (P={lr.pvalue:.3f}, "
        f"R²={lr.rvalue**2:.3f})")
    say(f"- 层内 SD 范围: {wm['within_SD_mm'].min():.1f} – {wm['within_SD_mm'].max():.1f} mm "
        f"（匹配良好则各窗口可比）")
    say(f"- 最干窗口 climate 份额 = {wm['share_climate'].iloc[0]:.3f}；"
        f"最湿窗口 = {wm['share_climate'].iloc[-1]:.3f}")
    say(f"- 全样本 climate 份额（14_shap_boot_report 口径）= 0.1542")
say("")
if len(gsd) >= 3:
    from scipy.stats import spearmanr as sr3
    rho3, pv3 = sr3(gsd["center_mm"], gsd["share_climate"])
    say(f"- 全局子集口径（n={len(gsd)}）: Spearman(窗口中心, climate 份额) = "
        f"{rho3:+.3f} (P={pv3:.3f})；最干窗口={gsd['share_climate'].iloc[0]:.3f}，"
        f"最湿窗口={gsd['share_climate'].iloc[-1]:.3f}")
    say("")
if len(qd) >= 3:
    from scipy.stats import spearmanr as sr2
    rho2, pv2 = sr2(range(len(qd)), qd["share_climate"])
    say(f"- 四分位对照: Spearman(分位序, climate 份额) = {rho2:+.3f} (P={pv2:.3f})；"
        f"层内 SD 范围 {qd['within_SD_mm'].min():.1f} – {qd['within_SD_mm'].max():.1f} mm")
    say(f"- 说明: 四分位分层的层内 SD 远小于全样本 SD "
        f"({float(tab['bio12'].std())/10:.1f} mm)，故其层内 climate 份额被机械压低，"
        f"不能用于检验命题方向。")

with open(os.path.join(C.OUT, "24_var_matched_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
say("\nDONE P0-B-a")
