#!/usr/bin/env python3
"""
Statistical Analysis Suite for Evo2 Yarrowia Language Modeling Project
Calculates:
1. Chi-Square Test of Independence for Known vs. Predicted EC classes
2. Distributional similarity metrics (Cosine, Spearman, Pearson, JSD, Total Variation Distance)
3. Two-proportion z-tests and paired t-tests for Mean vs. Last-Token Pooling
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon, cosine, braycurtis

def analyze_yarrowia_ec_distribution():
    print("=" * 80)
    print("1. YARROWIA LIPOLYTICA: KNOWN VS. PREDICTED EC CLASSES (LEVEL 1)")
    print("=" * 80)

    classes = ["EC 1 (Oxidoreductases)", "EC 2 (Transferases)", "EC 3 (Hydrolases)",
               "EC 4 (Lyases)", "EC 5 (Isomerases)", "EC 6 (Ligases)", "EC 7 (Translocases)"]

    known_counts = np.array([168, 418, 341, 68, 57, 65, 29])
    pred_counts  = np.array([1294, 2302, 1519, 90, 66, 33, 4])

    n_known = known_counts.sum()
    n_pred  = pred_counts.sum()

    p_known = known_counts / n_known
    p_pred  = pred_counts / n_pred

    contingency = np.array([known_counts, pred_counts])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    cos_sim = np.dot(p_known, p_pred) / (np.linalg.norm(p_known) * np.linalg.norm(p_pred))
    rho, p_rho = stats.spearmanr(p_known, p_pred)
    r_val, p_r = stats.pearsonr(p_known, p_pred)
    js_dist = jensenshannon(p_known, p_pred, base=2)
    js_div  = js_dist ** 2
    tvd = 0.5 * np.sum(np.abs(p_known - p_pred))
    overlap = 1.0 - tvd

    print(f"Sample Sizes: Known N = {n_known}, Predicted N = {n_pred}")
    print(f"Chi-Square Test of Independence: chi2 = {chi2:.4f}, df = {dof}, p-value = {p_val:.4e}")
    print(f"Cosine Similarity:               {cos_sim:.4f} ({cos_sim*100:.2f}%)")
    print(f"Distribution Overlap (1 - TVD):  {overlap:.4f} ({overlap*100:.2f}%)")
    print(f"Spearman Rank Correlation:       rho = {rho:.4f} (p = {p_rho:.4e})")
    print(f"Pearson Linear Correlation:      r = {r_val:.4f} (p = {p_r:.4e})")
    print(f"Jensen-Shannon Distance:         {js_dist:.4f}")
    print(f"Jensen-Shannon Divergence:       {js_div:.4f}")
    print()

def analyze_mean_vs_last_token():
    print("=" * 80)
    print("2. MEAN VS. LAST-TOKEN POOLING SIGNIFICANCE (BENCHMARK 4)")
    print("=" * 80)

    n_val = 18420
    n_tercile = 6140

    def two_prop(p1, p2, n):
        p_pool = (p1 + p2) / 2
        se = np.sqrt(2 * p_pool * (1 - p_pool) / n)
        z = (p1 - p2) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p

    z_mlp, p_mlp = two_prop(0.7332, 0.6355, n_val)
    z_xgb, p_xgb = two_prop(0.6208, 0.3897, n_val)

    print(f"MLP Overall (N={n_val}):     Mean 73.32% vs Last 63.55% | z = {z_mlp:.2f}, p = {p_mlp:.2e}")
    print(f"XGBoost Overall (N={n_val}): Mean 62.08% vs Last 38.97% | z = {z_xgb:.2f}, p = {p_xgb:.2e}")

    # Paired t-tests across length cohorts
    mlp_mean = np.array([0.7332, 0.7451, 0.7466, 0.7080])
    mlp_last = np.array([0.6355, 0.6713, 0.6406, 0.5945])
    t_mlp, p_t_mlp = stats.ttest_rel(mlp_mean, mlp_last)
    print(f"Paired t-test (MLP length cohorts, n=4):     t = {t_mlp:.3f}, p = {p_t_mlp:.4e}")

    xgb_mean = np.array([0.6208, 0.6526, 0.6151, 0.5946])
    xgb_last = np.array([0.3897, 0.4221, 0.4008, 0.3461])
    t_xgb, p_t_xgb = stats.ttest_rel(xgb_mean, xgb_last)
    print(f"Paired t-test (XGBoost length cohorts, n=4): t = {t_xgb:.3f}, p = {p_t_xgb:.4e}")

    all_mean = np.concatenate([mlp_mean, xgb_mean])
    all_last = np.concatenate([mlp_last, xgb_last])
    t_all, p_t_all = stats.ttest_rel(all_mean, all_last)
    w_all, p_w_all = stats.wilcoxon(all_mean, all_last)
    print(f"Paired t-test (All 8 cohorts combined):      t = {t_all:.3f}, p = {p_t_all:.4e}")
    print(f"Wilcoxon signed-rank test (All 8 cohorts):   W = {w_all:.1f}, p = {p_w_all:.4e}")
    print()

if __name__ == "__main__":
    analyze_yarrowia_ec_distribution()
    analyze_mean_vs_last_token()
