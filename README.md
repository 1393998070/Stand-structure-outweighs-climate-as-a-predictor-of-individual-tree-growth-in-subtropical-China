# Analysis code deposit — ScienceDB / GitHub

Manuscript: *Stand structure outweighs climate as a predictor of individual-tree  
growth in subtropical China* (v4.0).

The complete analysis is scripted end-to-end with fixed random seeds: data  
preparation, modelling, attribution, ablation, mapping, sensitivity analysis,  
figure generation, verification checks and the submission build. Scripts share  
paths and loaders through `common.py` (edit the `ROOT` constant at its top to  
point at your own copy of the working directory).

## Environment

Python 3.12.9 with pandas 3.0.5, numpy 2.4.6, scipy 1.18.0, scikit-learn 1.6.1,  
statsmodels 0.15.0, xgboost 3.4.1, shap 0.52.0, matplotlib 3.10.0;  
geospatial processing uses rasterio 1.4.3 and pyproj 3.6.1.

## Run order

| Stage                    | Scripts                                                                                                                                                                        | Purpose                                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Data & EDA            | `00_eda_stage1.py`, `01_prep.py`                                                                                                                                               | Cleaning, RGR computation, predictor collinearity screen (51 → 35 features), fold construction                                                               |
| 2. Models                | `02_main_model.py`, `02b_lmm.py`, `03_triple_cv.py`                                                                                                                            | XGBoost/random forest fits, LMM benchmark, random vs plot-grouped vs spatial-block CV                                                                        |
| 3. Attribution           | `04_shap_oof.py`, `05_ablation.py`, `14_shap_cluster_boot.py`, `14b_perm_check.py`, `15_deltad_shap.py`, `16_gam_stability.py`                                                 | Out-of-fold SHAP, group ablation, cluster bootstrap, permutation-SHAP check, ΔD companion, GAM check                                                         |
| 4. Nested protocol       | `22_nested_cv.py`, `26_nested_shap.py`, `28_nested_ablation.py`, `29_deltad_nested.py`, `31_leakage_plotlevel.py`, `33_shape_stability_nested.py`                              | Headline nested cross-validation, nested SHAP/ablation, leakage decomposition, shape stability                                                               |
| 5. Mapping               | `06_map_prep.py`, `07_map_predict.py`, `18_map_rebuild.py`, `23_map_dual.py`                                                                                                   | 250-m raster prediction, dual functional-group maps, map attribution                                                                                         |
| 6. Sensitivity           | `08_sensitivity.py`, `11_soil_depth.py`, `12_repeated_cv.py`, `13_lmm_oof.py`, `17_climate_strata.py`, `21_threshold_fix.py`, `24_var_matched_climate.py`, `27_varpart_fix.py` | Thresholds, block sizes, seeds, soil depth, climate-stratum and variance-partition checks                                                                    |
| 7. Figures & source data | `09_figures.py`, `20_si_figures.py`, `37_source_data.py`                                                                                                                       | Figs. 1–6, Supplementary Figs. 1–4, `Source_Data.xlsx` + maps zip (Fig. 1a positions are snapped to 0.05° grid centres — exact coordinates are confidential) |
| 8. Verification          | `30_figure_consistency.py`, `34_threeway_check.py`, `35_nc_format_check.py`, `99_integrity_check.py`                                                                           | Figure↔caption↔data-source three-way checks and journal-format checks                                                                                        |
| 9. Build & export        | `36_build_submission_docs.py`, `42_data_deposit.py`                                                                                                                            | DOCX/PDF submission build (via pandoc); export of the coordinate-free aggregate derived products for the public data deposit                                 |
| Supporting               | `38_zonation.py`, `39_zonation_figures.py`, `40_zonation_panels.py`                                                                                                            | Climate-zone (zonation) supporting analysis — not part of the submitted figures                                                                              |

## Data access

The raw Continuous Forest Inventory data are not distributed here: they are  
restricted by national regulations governing confidential forest inventory plot  
coordinates and by the data-licence agreement under which they were used; in  
accordance with those regulations the raw records cannot be shared with third  
parties, including reviewers and editors. The coordinate-free aggregate derived  
products needed to inspect every reported result accompany this deposit in  
`04_数据归档/` (see its `README.md`).

**DOI 占位：【投稿/发表时回填 ScienceDB 归档 DOI】；GitHub 仓库 URL 【待作者填写】**
