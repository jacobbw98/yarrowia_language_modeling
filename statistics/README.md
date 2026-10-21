# Statistical Analysis & Benchmark Documentation

This directory contains the complete statistical analysis suite, tabular data, and publication-ready figures for the **Evo2 Genome Language Model Functional EC Prediction Pipeline** and its application to ***Yarrowia lipolytica***.

---

## Directory Contents

| File / Folder | Description |
| :--- | :--- |
| [`run_statistical_tests.py`](run_statistical_tests.py) | Standalone executable Python script reproducing all hypothesis tests, z-scores, and similarity metrics. |
| [`yarrowia_ec_class_breakdown.csv`](yarrowia_ec_class_breakdown.csv) | Full quantitative breakdown (counts, percentages, differences, ratios, standardized residuals) for Level-1 EC classes. |
| [`mean_vs_last_pooling_stats.csv`](mean_vs_last_pooling_stats.csv) | Z-scores, p-values, and accuracy differences for Mean vs. Last-Token pooling across sequence length brackets. |
| [`benchmark_comparison_summary.csv`](benchmark_comparison_summary.csv) | Master benchmark performance table across models, pooling strategies, and layers. |
| [`figures/`](figures/) | High-resolution (300 DPI) publication/presentation table assets. |

---

## 1. Yarrowia lipolytica: Known vs. Predicted Functional EC Classes

A comparison of the macro-level functional distribution between previously known *Y. lipolytica* enzymes ($N = 1,146$) and newly predicted unannotated genes ($N = 5,308$, Top-1 predictions):

### Similarity Metrics

* **Cosine Similarity:** **`0.9731` (97.31%)** — Indicates an extremely high degree of global alignment in the functional proportions.
* **Spearman Rank Correlation ($\rho$):** **`0.9643` ($p = 4.54 \times 10^{-4}$)** — The rank order of enzyme abundance is almost perfectly conserved between known and predicted sets.
* **Pearson Linear Correlation ($r$):** **`0.9633` ($p = 4.85 \times 10^{-4}$)** — Strong linear relationship across class proportions.
* **Distribution Overlap ($1 - \text{Total Variation Distance}$):** **`0.8339` (83.39%)** — $83.4\%$ of probability mass is shared.
* **Jensen-Shannon Divergence (JSD):** **`0.0561`** *(Distance: $0.2368$)*.

### Hypothesis Testing ($\chi^2$ Test of Independence)

* **$\chi^2$ Statistic:** **`457.75`** ($df = 6$)
* **$p$-value:** **`1.05 × 10⁻⁹⁵` ($p < 0.0001$)**
* **Interpretation:** While the overall macro-distribution is preserved ($\sim 97\%$ cosine similarity), the massive sample size ($N = 6,454$) confirms statistically significant enrichment in dominant classes (EC 1 and EC 2) and conservative under-prediction of minor classes (EC 4–7).

### Class-by-Class Breakdown

| EC Level-1 Class | Known Count ($N=1146$) | Known % | Pred Count ($N=5308$) | Pred % | Difference ($\Delta$) | Pred / Known Ratio | Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EC 1 (Oxidoreductases)** | 168 | 14.66% | 1,294 | 24.38% | $+9.72\%$ | **$1.66\times$** | **Enriched** |
| **EC 2 (Transferases)** | 418 | 36.47% | 2,302 | 43.37% | $+6.89\%$ | **$1.19\times$** | **Slightly Enriched** |
| **EC 3 (Hydrolases)** | 341 | 29.76% | 1,519 | 28.62% | $-1.14\%$ | **$0.96\times$** | **Near Perfect Match** |
| **EC 4 (Lyases)** | 68 | 5.93% | 90 | 1.70% | $-4.24\%$ | **$0.29\times$** | **Depleted** |
| **EC 5 (Isomerases)** | 57 | 4.97% | 66 | 1.24% | $-3.73\%$ | **$0.25\times$** | **Depleted** |
| **EC 6 (Ligases)** | 65 | 5.67% | 33 | 0.62% | $-5.05\%$ | **$0.11\times$** | **Depleted** |
| **EC 7 (Translocases)** | 29 | 2.53% | 4 | 0.08% | $-2.46\%$ | **$0.03\times$** | **Depleted** |
| **Total** | **1,146** | **100.00%** | **5,308** | **100.00%** | **—** | **—** | **—** |

---

## 2. Mean Pooling vs. Last-Token Pooling Significance (Benchmark 4)

Tested on the Benchmark 4 validation set ($N = 18,420$ validation sequences; 2,228 EC classes):

* **MLP (Best Layer 17):**
  * Mean Pooling: **`73.32%`** vs. Last-Token: **`63.55%`** ($\Delta = +9.77\%$)
  * Two-proportion test: $z = 20.17$, **$p = 1.86 \times 10^{-90}$ ($p < 0.0001$)**
  * Paired $t$-test across length terciles ($n=4$): $t = 11.35$, **$p = 0.0015$**
* **XGBoost (Best Layer 19):**
  * Mean Pooling: **`62.08%`** vs. Last-Token: **`38.97%`** ($\Delta = +23.11\%$)
  * Two-proportion test: $z = 44.36$, **$p < 10^{-300}$ ($p < 0.0001$)**
  * Paired $t$-test across length terciles ($n=4$): $t = 33.09$, **$p = 6.07 \times 10^{-5}$**
* **Combined Cohorts ($n = 8$ conditions):**
  * Paired $t$-test: $t = 6.39$, **$p = 3.70 \times 10^{-4}$ ($p < 0.001$)**
  * Wilcoxon signed-rank test: $W = 0.0$, **$p = 0.0078$** (Mean pooling outperformed Last-Token in 100% of cases)

---

## 3. Benchmark 4 Sequence Length Group Definitions

Validation sequences were partitioned into three equal terciles based on nucleotide sequence length:

* **Short:** **$< 500$ bases** ($6,140$ validation sequences)
* **Medium:** **$501 - 1,000$ bases** ($6,140$ validation sequences)
* **Long:** **$\ge 1,001$ bases** ($6,140$ validation sequences)

---

## 4. Random Forest Multi-Layer Concatenation Baseline

* **RF Combined (L9+24+26):**
  * Overall Validation EC Accuracy: **`0.1331` (13.31%)**
  * Short Acc: **`0.1334` (13.34%)**
  * Medium Acc: **`0.1358` (13.58%)** *(Peak across length brackets)*
  * Long Acc: **`0.1300` (13.00%)**
* Outperformed all single-layer Random Forest baselines: RF Block (`0.1191`), RF Post Norm (`0.1162`), and RF MLP (`0.1007`).

---

## How to Reproduce

Run the reproduction script from within this directory:

```bash
python run_statistical_tests.py
```
