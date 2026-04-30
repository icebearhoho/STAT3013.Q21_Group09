# Vietnam Socioeconomic Analysis, 1989-2019
# Machine Learning, Deep Learning, and Socioeconomic Structure in Vietnam, 1989-2019

Research-grade Python project for studying long-run socioeconomic change in Vietnam using IPUMS-style census microdata from 1989, 1999, 2009, and 2019. The current project version is ML/DL-centered: the main question is how well predictive and representation-learning models capture household socioeconomic status and what they add relative to classical descriptive and econometric analysis.

## Quick Info

### Project title

- **Machine Learning, Deep Learning, and Socioeconomic Structure in Vietnam, 1989-2019**

### Repository purpose

- build a reproducible census microdata pipeline
- construct household socioeconomic indicators
- benchmark ML, DL, Bayesian Network, clustering, and anomaly models
- generate research tables, figures, and a static dashboard

### Link manifest

- [Demo video](https://docs.google.com/document/d/16SBqCxc2bjI3q8NuLhUzL6De8Hg0-O7gfuDEzlTahRA/edit?usp=sharing)
- [Dataset](https://docs.google.com/document/d/16SBqCxc2bjI3q8NuLhUzL6De8Hg0-O7gfuDEzlTahRA/edit?usp=sharing)

### Dataset link

- Local raw dataset expected at: `data/raw/ipumsi_00002.csv`

### License

- No standalone `LICENSE` file is currently included in this repository.

## Environment Requirements

### Recommended environment

- Windows with PowerShell
- Python `3.11` recommended
- `pip` available in PATH

### Python packages

Install from:

Research-grade Python project for studying long-run socioeconomic change in Vietnam using IPUMS-style census microdata from 1989, 1999, 2009, and 2019. In this version of the project, machine learning and deep learning are the central research focus: the core question is how well predictive and representation-learning methods capture household socioeconomic status, and what they add relative to classical descriptive and econometric approaches.
```bash
pip install -r requirements.txt
```

Current dependency list includes:

- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `matplotlib`
- `scipy`
- `joblib`
- `xgboost`
- `lightgbm`
- `torch`
- `pgmpy`

### Notes

- `torch` is required for the PyTorch-based deep learning path.
- `pgmpy` is required for Bayesian Network structure learning and inference.
- `lightgbm` and `xgboost` are optional in the sense that they are benchmark models, but they should be installed if you want the full model comparison.

## Run Instructions

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the full pipeline

```powershell
python src/main.py
```

### 4. Main outputs to check

- `outputs/tables/`
- `outputs/figures/`
- `outputs/reports/research_dashboard.html`

## Research Scope

The project studies:
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
## Input Expectations

- Preferred raw file: `data/raw/ipumsi_00002.csv`
- If the raw file is missing, the project can still run downstream analysis from the existing processed datasets and harmonized master file, as long as these are present:
