# -*- coding: utf-8 -*-
"""共享工具：路径、变量类组、空间分块、折构造、评估指标。全部脚本 import 本模块。"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = r"d:\3D\写论文\全省生长速率论文2"
PIPE = os.path.join(ROOT, "论文流水线产出")
OUT = os.path.join(PIPE, "04_分析结果")
FIGS = os.path.join(PIPE, "05_图表")
RASTER = os.path.join(ROOT, "栅格数据")
SEED = 42

# ---------- 变量类组（编码后的英文名；climate 由共线性筛选确定，写入 var_dict） ----------
TREE_VARS = ["lnD0", "fg_conifer"]
# v4 (2026-09): 新数据样地表以「样地类型」三分类（针叶林/针阔混交林/阔叶林）取代旧「树种编码」，
# 且全表零缺失 → 缺失指示变量 age_missing/origin_missing 不再作为特征
STAND_VARS = ["stand_age", "stand_density", "origin_natural",
              "conifer_prop", "forest_type"]
SOIL_VARS = ["soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth"]
TERRAIN_VARS = ["dem", "slope", "aspect_cos", "aspect_sin", "curvature_total",
                "curvature_plan", "curvature_profile", "dist_stream", "relief",
                "flow_length", "heat_load", "SPI", "solar_eq", "solar_ss",
                "solar_ws", "TPI", "TSI", "TWI"]
CLIMATE_CANDIDATES = [f"bio{i}" for i in range(1, 20)]

# 共线性簇内保留优先级（生态解释优先，其次与 RGR 相关强度兜底）
PRIORITY = [
    "lnD0", "stand_age", "stand_density", "conifer_prop", "origin_natural",
    "fg_conifer", "forest_type",
    "bio1", "bio4", "bio12", "bio19", "bio15",
    "soil_SOC", "soil_TN", "soil_TP", "soil_Tk", "soil_pH", "soil_BD", "soil_depth",
    "dem", "slope", "aspect_cos", "aspect_sin", "TWI", "TPI", "heat_load", "SPI",
    "relief", "dist_stream", "solar_ss", "solar_ws", "solar_eq", "flow_length",
    "curvature_total", "curvature_plan", "curvature_profile", "TSI",
]
PRIORITY_RANK = {v: i for i, v in enumerate(PRIORITY)}


def load_tables():
    t = pd.read_pickle(os.path.join(OUT, "00_tree_clean.pkl"))
    p = pd.read_pickle(os.path.join(OUT, "00_plot_merged.pkl"))
    return t, p


def load_model_data():
    """返回 (model_table, var_dict, plot_folds)。Phase A 之后所有脚本从这里取数。"""
    tab = pd.read_pickle(os.path.join(OUT, "01_model_table.pkl"))
    var_dict = pd.read_csv(os.path.join(OUT, "01_var_dict.csv"))
    folds = pd.read_csv(os.path.join(OUT, "01_plot_folds.csv"))
    return tab, var_dict, folds


def selected_features(var_dict, groups=None):
    vd = var_dict[var_dict["selected"]]
    if groups is not None:
        vd = vd[vd["group"].isin(groups)]
    return vd["var"].tolist()


def block_id(lon, lat, size=0.5):
    """空间块编号（保留浮点精度，避免细块被截断合并）。"""
    bx = np.floor(np.asarray(lon, float) / size) * size
    by = np.floor(np.asarray(lat, float) / size) * size
    return pd.Series([f"{x:.3f}_{y:.3f}" for x, y in zip(bx, by)])


def assign_spatial_folds(blocks_by_size, n_folds=5):
    """blocks_by_size: [(block_id, n_trees), ...] 按树数降序；贪心装袋平衡各折树数。"""
    fold_load = [0] * n_folds
    assign = {}
    for b, n in blocks_by_size:
        f = int(np.argmin(fold_load))
        assign[b] = f
        fold_load[f] += n
    return assign


def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - yhat) ** 2) / ss if ss > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    rho = float(spearmanr(y, yhat).statistic) if len(y) > 2 else np.nan
    return {"R2": round(float(r2), 4), "RMSE": round(rmse, 5),
            "MAE": round(mae, 5), "Spearman": round(rho, 4)}


def cluster_bootstrap_ci(y, yhat, groups, n_boot=1000, seed=SEED):
    """按样地聚类 bootstrap（numpy 向量化），返回 tree 级 R2/rho 与 plot 级 R2 的 95% CI。"""
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == i)[0] for i in range(len(ug))]
    cnt = np.bincount(inv, minlength=len(ug)).astype(float)
    pm_y = np.bincount(inv, weights=y, minlength=len(ug)) / cnt
    pm_h = np.bincount(inv, weights=yhat, minlength=len(ug)) / cnt
    rng = np.random.default_rng(seed)
    tr_r2, tr_rho, pl_r2 = [], [], []
    for _ in range(n_boot):
        gs = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by_g[g] for g in gs])
        yy, hh = y[idx], yhat[idx]
        r2 = 1 - np.sum((yy - hh) ** 2) / np.sum((yy - yy.mean()) ** 2)
        from scipy.stats import rankdata
        ry, rh = rankdata(yy), rankdata(hh)
        rho = np.corrcoef(ry, rh)[0, 1]
        o, pr = pm_y[gs], pm_h[gs]
        p2 = 1 - np.sum((o - pr) ** 2) / np.sum((o - o.mean()) ** 2)
        tr_r2.append(r2); tr_rho.append(rho); pl_r2.append(p2)
    q = lambda a: (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))
    return {"tree_R2": q(tr_r2), "tree_Spearman": q(tr_rho), "plot_R2": q(pl_r2)}


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


# ---------- 共享 XGB 训练器（03/04/05/08 复用，保证 OOF 一致） ----------
def fit_xgb_folds(tab, feats, params, fold_col, n_folds=5, seed=SEED, y_col="RGR",
                  return_models=False):
    """按给定折列训练 XGB 并返回 OOF 预测。早停以各折验证集为准。"""
    import xgboost as xgb
    oof = pd.Series(np.nan, index=tab.index)
    models = []
    for k in range(n_folds):
        tr = tab[tab[fold_col] != k]
        te = tab[tab[fold_col] == k]
        Xtr, ytr = tr[feats], tr[y_col]
        Xte, yte = te[feats], te[y_col]
        m = xgb.XGBRegressor(
            n_estimators=4000, early_stopping_rounds=80, eval_metric="rmse",
            tree_method="hist", random_state=seed + k, n_jobs=-1, **params)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        oof.loc[te.index] = m.predict(Xte)
        models.append(m)
    if return_models:
        return oof, models
    return oof


def fit_rf_folds(tab, feats, params, fold_col, n_folds=5, seed=SEED, y_col="RGR",
                 return_models=False):
    """Random Forest 按折训练，返回 OOF 预测（供 03/04 复用）。"""
    from sklearn.ensemble import RandomForestRegressor
    oof = pd.Series(np.nan, index=tab.index)
    models = []
    for k in range(n_folds):
        tr = tab[tab[fold_col] != k]
        te = tab[tab[fold_col] == k]
        m = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
        m.fit(tr[feats], tr[y_col])
        oof.loc[te.index] = m.predict(te[feats])
        models.append(m)
    if return_models:
        return oof, models
    return oof


# ---------- 嵌套 CV：诚实选参（P0-D-a） ----------
XGB_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.9],
    "min_child_weight": [1, 5, 10, 20],
    "reg_alpha": [0, 0.5, 2, 5],
    "reg_lambda": [1, 2, 5, 10],
}
RF_SPACE = [{"n_estimators": 600, "max_features": mf, "min_samples_leaf": msl,
             "max_depth": md}
            for mf in (0.3, 0.5, "sqrt") for msl in (1, 5) for md in (None, 20)]


def draw_xgb_candidates(n_cand, seed=SEED, space=None):
    space = space or XGB_SPACE
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_cand):
        p = {k: rng.choice(v) for k, v in space.items()}
        out.append({k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
                    for k, v in p.items()})
    return out


def _fit_es(tr, va, feats, params, seed, y_col="RGR", n_cap=1200, patience=50):
    """内层拟合：以 va 早停（va 必属外层训练集内部）。"""
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=n_cap, early_stopping_rounds=patience,
                         eval_metric="rmse", tree_method="hist", random_state=seed,
                         n_jobs=-1, **params)
    m.fit(tr[feats], tr[y_col], eval_set=[(va[feats], va[y_col])], verbose=False)
    return m


def nested_xgb_oof(tab, feats, fold_col, n_folds=5, n_cand=24, seed=SEED, y_col="RGR",
                   space=None, verbose=True):
    """嵌套 CV 的 XGBoost 诚实评估。

    外层每折：以其余折作为内层验证完成【选参 + 早停迭代数选择】，
    再用选中参数与迭代数在整份外层训练集上重训（不再早停），预测外层留出折。
    因此外层留出折从未参与任何选择，OOF 预测不含选参/早停乐观偏倚。
    返回 (oof, models, per_fold_params, tuning_log)。
    """
    import xgboost as xgb
    cands = draw_xgb_candidates(n_cand, seed, space)
    folds_all = sorted(tab[fold_col].unique())
    oof = pd.Series(np.nan, index=tab.index)
    models, chosen, log = [], [], []
    for k in folds_all:
        outer_tr = tab[tab[fold_col] != k]
        outer_te = tab[tab[fold_col] == k]
        inner = [f for f in folds_all if f != k]
        best_rmse, best_p, best_its = np.inf, None, [200]
        for ci, p in enumerate(cands):
            rms, its = [], []
            for f in inner:
                itr = outer_tr[outer_tr[fold_col] != f]
                iva = outer_tr[outer_tr[fold_col] == f]
                m = _fit_es(itr, iva, feats, p, seed)
                pred = m.predict(iva[feats])
                rms.append(float(np.sqrt(np.mean((iva[y_col].values - pred) ** 2))))
                its.append(int(m.best_iteration) + 1)
            mr = float(np.mean(rms))
            if mr < best_rmse:
                best_rmse, best_p, best_its = mr, p, its
            if verbose and (ci + 1) % 8 == 0:
                print(f"    fold {k}: candidate {ci+1}/{len(cands)}, "
                      f"best inner RMSE so far = {best_rmse:.5f}", flush=True)
        n_est = int(max(1, np.median(best_its)))
        m = xgb.XGBRegressor(n_estimators=n_est, tree_method="hist",
                             random_state=seed + int(k), n_jobs=-1, **best_p)
        m.fit(outer_tr[feats], outer_tr[y_col], verbose=False)
        oof.loc[outer_te.index] = m.predict(outer_te[feats])
        models.append(m)
        chosen.append({**best_p, "n_estimators": n_est})
        log.append({"fold": int(k), "inner_rmse": round(best_rmse, 5), "n_est": n_est, **best_p})
        if verbose:
            print(f"  outer fold {k}: inner RMSE={best_rmse:.5f} n_est={n_est} {best_p}",
                  flush=True)
    return oof, models, chosen, pd.DataFrame(log)
