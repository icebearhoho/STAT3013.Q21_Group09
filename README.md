# Vietnam Socioeconomic Analysis, 1989-2019

Research-grade Python project for studying long-run socioeconomic change in Vietnam using IPUMS-style census microdata from 1989, 1999, 2009, and 2019. In this version of the project, machine learning and deep learning are the central research focus: the core question is how well predictive and representation-learning methods capture household socioeconomic status, and what they add relative to classical descriptive and econometric approaches.

## Research Scope

The project studies:

1. ML and DL prediction of household wealth class
2. latent socioeconomic representation learning
3. household wealth and housing inequality
4. urban-rural disparities
5. education and labor structure
6. gender inequality
7. proxy education mobility
8. regional convergence and divergence

## Project Structure

```text
data/
  raw/
  interim/
  processed/

metadata/
  missing_rules.csv
  harmonization_rules.csv
  variable_dictionary.csv

outputs/
  tables/
  figures/
  reports/
  logs/
  models/

src/
  main.py
  validate_data.py
  econometrics.py
  gender_analysis.py
  education_mobility.py
  regional_analysis.py
  deep_learning.py
  robustness_checks.py
  visualize.py
```

## Methodology

### 1. Streaming data pipeline

- reads the raw master CSV in chunks
- standardizes column names
- applies variable-year-specific missing value rules when metadata exists
- applies variable-year-specific harmonization rules when metadata exists
- preserves the existing chunk-based build logic for household and person outputs

### 2. Data credibility and validation

The pipeline does not silently invent metadata mappings. Instead it produces explicit validation artifacts:

- metadata schema audits
- metadata usage audit
- missingness summary by variable and year
- category consistency report across years
- harmonization coverage report with unresolved raw codes
- outlier summaries for key continuous variables
- structural break flags for suspicious distribution shifts
- variable documentation artifact combining metadata and year coverage

### 3. Core ML/DL analyses

- Logistic Regression baseline
- Random Forest baseline
- SVM baseline
- XGBoost baseline when available
- LightGBM baseline when available
- Bayesian Network as an interpretable probabilistic benchmark
- MLP classifier for wealth-class prediction
- autoencoder-based latent household representation
- comparative model evaluation using accuracy and macro F1
- temporal validation across census waves
- feature-importance analysis for interpretability
- subgroup performance analysis across urban-rural, year, and region slices
- latent feature evaluation for downstream prediction and clustering
- K-Means clustering for latent household grouping
- Isolation Forest for anomaly and outlier structure analysis

### 4. Supporting empirical analyses

- descriptive household statistics
- weighted yearly wealth trends
- yearly Gini coefficients
- urban-rural wealth comparisons
- household fixed-effects style regressions with year and region fixed effects, robust standard errors, and urban-by-year interactions
- person-level education-gender returns regressions with education-by-gender interactions
- dedicated gender inequality module
- education mobility proxy module based on repeated cross-sections, cohorts, and changing education-wealth associations
- regional sigma and beta convergence analysis

### 5. Research presentation outputs

- PCA wealth index vs simple asset score
- alternative PCA specifications
- subgroup sensitivity by urban-rural status
- subgroup sensitivity by early, middle, and late periods
- static HTML research dashboard summarizing the main figures and result tables

## Running the Project

### Install

```bash
pip install -r requirements.txt
```

### Run full pipeline

```bash
python src/main.py
```

### Input expectations

- Preferred raw file: `data/raw/ipumsi_00002.csv`
- If the raw file is missing, the project can still run downstream analysis from the existing processed datasets and harmonized master file, as long as these are present:
  - `data/interim/harmonized_master.csv`
  - `data/processed/household_analysis_ready.csv`
  - `data/processed/person_analysis_ready.csv`

## Key Outputs

### Tables

- `outputs/tables/household_summary_statistics.csv`
- `outputs/tables/yearly_weighted_wealth_mean.csv`
- `outputs/tables/yearly_weighted_gini.csv`
- `outputs/tables/urban_rural_wealth_by_year.csv`
- `outputs/tables/household_regression_coefficients.csv`
- `outputs/tables/person_education_gender_returns_coefficients.csv`
- `outputs/tables/gender_gaps_by_year.csv`
- `outputs/tables/education_wealth_relationship_over_time.csv`
- `outputs/tables/region_year_wealth.csv`
- `outputs/tables/regional_sigma_convergence.csv`
- `outputs/tables/traditional_model_comparison.csv`
- `outputs/tables/deep_learning_model_comparison.csv`
- `outputs/tables/bayesian_network_prediction.csv`
- `outputs/tables/bayesian_network_structure_edges.csv`
- `outputs/tables/bayesian_network_cpd_summary.csv`
- `outputs/tables/bayesian_network_scenario_analysis.csv`
- `outputs/tables/model_comparison_all.csv`
- `outputs/tables/temporal_model_validation.csv`
- `outputs/tables/model_feature_importance.csv`
- `outputs/tables/subgroup_model_performance.csv`
- `outputs/tables/kmeans_cluster_metrics.csv`
- `outputs/tables/kmeans_cluster_summary.csv`
- `outputs/tables/isolation_forest_summary.csv`
- `outputs/tables/isolation_forest_top_records.csv`
- `outputs/tables/autoencoder_evaluation.csv`
- `outputs/tables/autoencoder_latent_features.csv`
- `outputs/tables/representation_benchmark_pca_vs_autoencoder.csv`
- `outputs/tables/robustness_wealth_measurement.csv`

### Figures

- `outputs/figures/wealth_trend_over_time.png`
- `outputs/figures/gini_evolution.png`
- `outputs/figures/urban_rural_wealth_comparison.png`
- `outputs/figures/gender_gap_trends.png`
- `outputs/figures/education_wealth_relationship.png`
- `outputs/figures/regional_sigma_convergence.png`
- `outputs/figures/model_comparison_chart.png`

### Reports

- `outputs/reports/metadata_usage_audit.csv`
- `outputs/reports/missingness_summary_by_variable_year.csv`
- `outputs/reports/category_consistency_by_year.csv`
- `outputs/reports/harmonization_coverage_report.csv`
- `outputs/reports/structural_break_flags.csv`
- `outputs/reports/outlier_summary.csv`
- `outputs/reports/variable_documentation.csv`
- `outputs/reports/education_mobility_scope_note.csv`
- `outputs/reports/regional_convergence_interpretation.csv`
- `outputs/reports/research_dashboard.html`

## Important Interpretation Notes

- `missing_rules.csv` and `harmonization_rules.csv` are treated as incomplete unless fully documented. The code uses them when present and generates reports where coverage is missing.
- `variable_dictionary.csv` is parsed defensively because metadata files can contain formatting noise.
- The education mobility module is explicitly a repeated cross-section proxy analysis, not true intergenerational mobility unless additional linkage data is available.
- ML and DL are the central contribution of this version of the project. Descriptive and econometric analyses are used to interpret, validate, and contextualize the predictive findings.
- Autoencoder training requires PyTorch. If PyTorch is unavailable, the project writes a status report and continues without crashing.

## Limitations

- Harmonization quality remains bounded by the supplied metadata.
- Some variables are unavailable in early census waves, especially labor variables.
- Person-level regressions use memory-aware sampled analysis files rather than the entire 2GB person extract.
- Region consistency assumes `GEO1_VN` is comparable enough for broad convergence analysis; users should review any coding changes across census rounds.

## Suggested Thesis Narrative

This repository is structured to support a final-year thesis or research portfolio chapter:

1. document data cleaning and harmonization challenges
2. show transparency about incomplete metadata
3. present descriptive inequality trends
4. estimate robust regression models
5. extend into gender, education, and regional development
6. compare interpretable ML baselines with deep learning
7. interpret the winning models using the descriptive and econometric evidence
8. close with sensitivity and robustness checks
