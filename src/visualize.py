from __future__ import annotations

from pathlib import Path

# NumPy is used for numerical operations and creating smooth line data
import numpy as np
# Matplotlib is the core engine that draws the graphs and shapes
import matplotlib.pyplot as plt
# Pandas is used to interact with CSV data tables
import pandas as pd
# make_interp_spline takes jagged data points and calculates a smooth, curved line between them
from scipy.interpolate import make_interp_spline
# PCA (Principal Component Analysis) compresses multiple variables down into "Wealth" score
from sklearn.decomposition import PCA
# Scalers adjust our data so big numbers (income) don't overpower small numbers (0/1 categories)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import directory paths from config file
from config import OUTPUT_FIGURES_DIR, OUTPUT_REPORTS_DIR, OUTPUT_TABLES_DIR, RAW_DIR

# Set a basic visual style for the plots
plt.style.use("ggplot")

def _save_macro_micro_plot(df: pd.DataFrame, x_col: str, y_col: str, output_path, title: str):
    """Draws the main chart comparing National GDP (Bars) vs. Household Wealth (Line)"""
    
    # Calculate the average wealth score for every single year
    df_agg = df.groupby(x_col, as_index=False)[y_col].mean().sort_values(x_col)

    # Load the official World Bank GDP data (in data/raw) to act as reality check
    try:
        csv_path = RAW_DIR / "vietnamGDP.csv"
        if not csv_path.exists():
            csv_path = Path("vietnamGDP.csv")
            
        # Skip the first 4 rows because World Bank CSVs have a text header
        gdp_df = pd.read_csv(csv_path, skiprows=4)
        
        # Find the specific row for Vietnam
        vn_row = gdp_df[gdp_df["Country Name"] == "Viet Nam"].iloc[0]
        indicator_name = "National GDP (Billions of US$)"
        
        # Convert the raw GDP numbers into 'Billions'
        df_agg["Macro_Metric"] = df_agg[x_col].apply(
            lambda y: float(vn_row.get(str(int(y)), 0)) / 1_000_000_000
        )
    except Exception as e:
        print(f"Could not load vietnamGDP.csv: {e}")
        return
        
    # Set up the blank canvas (figure) and the primary axis (ax1) for the GDP bars
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(11, 6), facecolor='white')
    ax1.set_facecolor('white')
    ax1.grid(axis='y', color='#e0e0e0', linestyle='-', linewidth=0.5, zorder=0)

    # Draw the blue GDP bars
    color_bar = '#8fbbe8' 
    ax1.set_xlabel('Census Year', fontsize=12, fontweight='bold', color='#333333')
    ax1.set_ylabel(indicator_name, color='#2c3e50', fontsize=12, fontweight='bold')
    bars = ax1.bar(df_agg[x_col], df_agg["Macro_Metric"], width=1.8, color=color_bar, 
                   edgecolor='#4a90e2', linewidth=1.5, alpha=0.6, label="National GDP", zorder=2)
    
    # Format the primary axis ticks
    ax1.tick_params(axis='y', labelcolor='#2c3e50', labelsize=11)
    ax1.set_xticks(df_agg[x_col])
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_ylim(0, 400)
    ax1.set_yticks([0, 100, 200, 300, 400])

    # Put a text label (like '$...B') on top of every GDP bar
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'${height:.1f}B',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     color='#2980b9', fontweight='bold', fontsize=10)

    # Create a secondary axis (ax2) that shares the same X-axis, used for the orange Line graph
    ax2 = ax1.twinx()

    # Draw the orange PCA Wealth line
    color_line = '#d35400' 
    ax2.set_ylabel('Wealth Progress Index', color=color_line, fontsize=12, fontweight='bold')
    ax2.plot(df_agg[x_col], df_agg[y_col], color=color_line, marker='o', 
             linewidth=3.5, markersize=10, markeredgecolor='white', markeredgewidth=2, 
             label="Household Wealth", zorder=3)
    ax2.tick_params(axis='y', labelcolor=color_line, labelsize=11)

    # Dynamically adjust the Y-limits so the line has breathing room at the top and bottom
    y_min, y_max = df_agg[y_col].min(), df_agg[y_col].max()
    y_range = y_max - y_min
    if y_range == 0: y_range = 1  
    ax2.set_ylim(y_min - (y_range * 0.2), y_max + (y_range * 0.3))

    # Add text labels to the line points
    for i in range(len(df_agg)):
        x_val = df_agg[x_col].iloc[i]
        y_val = df_agg[y_col].iloc[i]
        
        # Ensure the last point will not be overlapped
        if i == len(df_agg) - 1:
            offset = (0, -22)
            va = 'top'
        else:
            offset = (0, 15)
            va = 'bottom'
            
        ax2.annotate(f"{y_val:.1f}", 
                     xy=(x_val, y_val), 
                     xytext=offset, 
                     textcoords='offset points', 
                     ha='center', va=va, 
                     color=color_line, fontweight='bold', fontsize=12,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # Clean up the chart by hiding the top and side borders (spines)
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')

    # Save the final image to the output folder
    plt.title(title, fontsize=16, pad=20, fontweight='bold', color='#222222')
    fig.tight_layout()
    png_path = str(output_path).replace('.html', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

def _save_line_plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str, hue: str | None = None) -> None:
    """Draws a standard line plot, used for things like Gini, Urban vs Rural, and Education ROI"""
    if df.empty:
        return
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.grid(axis='y', color='#e0e0e0', linestyle='-', linewidth=0.5, zorder=0)
    
    # A palette of professional colors to use for different lines
    colors = ['#2c3e50', '#d35400', '#16a085', '#f39c12', '#8e44ad', '#c0392b', '#2980b9']

    def plot_smooth(x_data, y_data, label=None, color_idx=0):
        """Helper function to convert jagged data points into a smooth curved line"""
        x_data_np = x_data.to_numpy()
        y_data_np = y_data.to_numpy()
        c = colors[color_idx % len(colors)]
        
        # We need at least 3 points to draw a mathematical curve (spline)
        if len(x_data_np) >= 3:
            # Generate 300 micro-points between our real points to create the smooth curve
            x_smooth = np.linspace(x_data_np.min(), x_data_np.max(), 300)
            spline = make_interp_spline(x_data_np, y_data_np, k=2) 
            y_smooth = spline(x_smooth)
            
            ax.plot(x_smooth, y_smooth, label=label, color=c, linewidth=3, alpha=0.9, zorder=3)
            ax.scatter(x_data_np, y_data_np, s=80, color=c, edgecolor='white', linewidth=2, zorder=4)
        else:
            # Fallback for small datasets: just connect the dots with straight lines
            ax.plot(x_data_np, y_data_np, marker="o", label=label, color=c, linewidth=3, 
                    markersize=9, markeredgecolor='white', markeredgewidth=2, zorder=3)

        # --- DYNAMIC PRESENTATION LABELS: Annotate 1989 and 2019 exactly ---
        if len(x_data_np) > 0:
            show_labels = True
            # Specific rule: If this is the Education chart, hide the labels for the middle tiers
            if title == "Education-Wealth Relationship" and label in ["Primary", "Secondary"]:
                show_labels = False
                
            if show_labels:
                def fmt(val):
                    # Format as 1 decimal if over 10 (e.g., 85.4), or 2 decimals if under 10 (e.g., 1.85)
                    return f"{val:.1f}" if val > 10 else f"{val:.2f}"
                    
                # Alternate the label position (up or down) based on the line color to prevent overlapping text
                y_offset = -15 if color_idx % 2 == 0 else 12
                v_align = 'top' if color_idx % 2 == 0 else 'bottom'
                    
                # Draw the label on the first point (e.g., 1989)
                ax.annotate(fmt(y_data_np[0]), xy=(x_data_np[0], y_data_np[0]),
                            xytext=(0, y_offset), textcoords='offset points', ha='center', va=v_align,
                            color=c, fontweight='bold', fontsize=11,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))
                
                # Draw the label on the last point (e.g., 2019)
                ax.annotate(fmt(y_data_np[-1]), xy=(x_data_np[-1], y_data_np[-1]),
                            xytext=(0, y_offset), textcoords='offset points', ha='center', va=v_align,
                            color=c, fontweight='bold', fontsize=11,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # If 'hue' is provided (like "Male/Female"), split the data and draw multiple lines
    if hue and hue in df.columns:
        keys = sorted(df[hue].unique())
        for idx, key in enumerate(keys):
            grp = df[df[hue] == key]
            grp_agg = grp.groupby(x, as_index=False)[y].mean().sort_values(x)
            plot_smooth(grp_agg[x], grp_agg[y], label=str(key), color_idx=idx)
        
        # Place the legend outside the chart to the right
        ax.legend(title=hue.replace('_', ' ').title(), title_fontsize='11', 
                  fontsize='11', frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        # Otherwise, just draw one main line
        df_agg = df.groupby(x, as_index=False)[y].mean().sort_values(x)
        plot_smooth(df_agg[x], df_agg[y], color_idx=0)

    # Standard styling and cleanup
    ax.set_xticks(df[x].unique())
    ax.set_xlabel('Census Year', fontsize=11, fontweight='bold', color='#333333')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#333333')
    ax.tick_params(axis='both', colors='#555555', labelsize=10)

    # Add 15% padding to the top AND bottom of the graph so labels don't get chopped off
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax - ymin) * 0.15, ymax + (ymax - ymin) * 0.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')

    plt.title(title, fontsize=15, pad=20, fontweight='bold', color='#222222')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _save_bar_plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str, xlabel: str) -> None:
    """Draws horizontal bar charts, primarily used for comparing Machine Learning models"""
    if df.empty:
        return
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')
    ax.set_facecolor('white')

    # Grab the top 15 rows so the chart doesn't get overwhelmingly tall
    df_sorted = df.sort_values(by=y, ascending=True).tail(15) 
    ax.grid(axis='x', color='#e0e0e0', linestyle='-', linewidth=0.5, zorder=0)

    color_bar = '#3498db'
    bars = ax.barh(df_sorted[x], df_sorted[y], color=color_bar, 
                   edgecolor='#2980b9', linewidth=1, height=0.6, zorder=3)

    # Add the exact numerical score (e.g., 0.632) directly next to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max(df_sorted[y]) * 0.01),  
                bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center', color='#333333', fontsize=10, fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', color='#333333')
    
    # Format the names of the models (e.g., change "random_forest" to "Random Forest")
    cleaned_labels = [str(label).replace('_', ' ').title() for label in df_sorted[x]]
    ax.set_yticks(range(len(cleaned_labels)))
    ax.set_yticklabels(cleaned_labels, fontsize=11, color='#2c3e50', fontweight='bold')
    
    ax.tick_params(axis='x', colors='#555555', labelsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')

    plt.title(title, fontsize=15, pad=20, fontweight='bold', color='#222222')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _load_if_exists(path: Path) -> pd.DataFrame | None:
    """Helper tool to only load CSV files if they actually exist, preventing crash errors"""
    return pd.read_csv(path) if path.exists() else None

def _format_top_rows(df: pd.DataFrame | None, max_rows: int = 8) -> str:
    """Converts a Pandas DataFrame into a clean HTML table for the web dashboard"""
    if df is None or df.empty:
        return "<p>No table available.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="table table-sm")

def generate_figures(output_dir: Path = OUTPUT_FIGURES_DIR, tables_dir: Path = OUTPUT_TABLES_DIR) -> None:
    """The master function that reads the data and commands the drawing of all charts"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Look for the processed data file
    processed_data_path = Path("data/processed/household_analysis.csv")
    if not processed_data_path.exists():
        processed_data_path = Path("data/processed/household_analysis_ready.csv")
        
    if processed_data_path.exists():
        df_full = pd.read_csv(processed_data_path)
        
        # Identify the modern infrastructure variables we want to use for our Wealth Index
        pca_features = ["ELECTRIC_H", "WATSUP_H", "URBAN_H"]
        pca_features = [f for f in pca_features if f in df_full.columns]
        
        if pca_features:
            df_pca = df_full[pca_features].copy()
            
            # Clean the data: extract just the numbers from any strings
            for col in pca_features:
                df_pca[col] = df_pca[col].astype(str).str.extract(r'(\d+)').astype(float)
            
            df_pca = df_pca.fillna(df_pca.median())
            
            # Scale the data so all variables have equal mathematical weight
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_pca)
            
            # Run the Global PCA to compress the 3 variables into 1 "Wealth" score
            pca = PCA(n_components=1)
            df_full["global_pca_wealth"] = pca.fit_transform(scaled_data)
            
            # TEMPORAL ANCHOR: Check if the PCA algorithm accidentally mapped wealth backwards
            # If 1989 scored higher than 2019, flip the axis so 'Up' equals 'Wealthier'
            try:
                mean_1989 = df_full[df_full["YEAR"] == 1989]["global_pca_wealth"].mean()
                mean_2019 = df_full[df_full["YEAR"] == 2019]["global_pca_wealth"].mean()
                if mean_1989 > mean_2019:
                    df_full["global_pca_wealth"] *= -1
            except Exception:
                pass 
                
            # Convert the raw PCA math (-1 to 1) into a clean, intuitive 0 to 100 Index
            min_max_scaler = MinMaxScaler(feature_range=(0, 100))
            df_full["wealth_progress_index"] = min_max_scaler.fit_transform(df_full[["global_pca_wealth"]])
            
            # Draw the main GDP vs Wealth chart
            _save_macro_micro_plot(
                df_full, "YEAR", "wealth_progress_index", 
                output_dir / "wealth_trend_over_time.png", 
                "Vietnam Economic Miracle: Macro GDP vs. Wealth Progress Index"
            )

            # RURAL/URBAN LEGEND FIX: Calculate the progress index specifically for Rural vs Urban
            if "URBAN_H" in df_full.columns:
                df_ur = df_full.groupby(["YEAR", "URBAN_H"], as_index=False)["wealth_progress_index"].mean()
                # Catch IPUMS codes (like '0' or '2') and map them to readable English labels
                df_ur["URBAN_LABEL"] = df_ur["URBAN_H"].astype(str).replace({
                    "2": "Urban", "2.0": "Urban", "1": "Rural", "1.0": "Rural",
                    "0": "Urban", "0.0": "Urban"
                })
                _save_line_plot(df_ur, "YEAR", "wealth_progress_index", output_dir / "urban_rural_wealth_comparison.png", "Urban vs Rural Wealth Progress", "Wealth Progress Index (0-100)", hue="URBAN_LABEL")

            # EDUCATION ROI: We must load the interim master file because Education was dropped from the household file
            master_path = Path("data/interim/harmonized_master.csv")
            if not master_path.exists():
                master_path = Path("data/interim/cleaned_master.csv")
                
            if master_path.exists():
                # Load only the 5 required columns to save memory and process instantly
                df_master = pd.read_csv(master_path, usecols=lambda c: c in ["YEAR", "ELECTRIC_H", "WATSUP_H", "URBAN_H", "EDATTAIN_H"])
                
                if "EDATTAIN_H" in df_master.columns:
                    # Filter out '0' (children too young for school) and '9' (unknown/missing data)
                    df_edu = df_master[~df_master["EDATTAIN_H"].isin([0, 0.0, "0", "0.0", 9, 9.0, "9", "9.0"])].copy()
                    
                    # Re-run the exact same PCA and Scaling math on this specific education dataset
                    pca_cols = ["ELECTRIC_H", "WATSUP_H", "URBAN_H"]
                    pca_cols = [c for c in pca_cols if c in df_edu.columns]
                    
                    if pca_cols:
                        df_edu_pca = df_edu[pca_cols].copy()
                        for col in pca_cols:
                            df_edu_pca[col] = df_edu_pca[col].astype(str).str.extract(r'(\d+)').astype(float)
                        
                        df_edu_pca = df_edu_pca.fillna(df_edu_pca.median())
                        scaled_edu = StandardScaler().fit_transform(df_edu_pca)
                        df_edu["global_pca_wealth"] = PCA(n_components=1).fit_transform(scaled_edu)
                        
                        try:
                            if df_edu[df_edu["YEAR"] == 1989]["global_pca_wealth"].mean() > df_edu[df_edu["YEAR"] == 2019]["global_pca_wealth"].mean():
                                df_edu["global_pca_wealth"] *= -1
                        except Exception: pass
                        
                        df_edu["wealth_progress_index"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df_edu[["global_pca_wealth"]])
                        
                        # Group by Year and Education tier, relabel the codes, and draw the plot
                        df_edu_agg = df_edu.groupby(["YEAR", "EDATTAIN_H"], as_index=False)["wealth_progress_index"].mean()
                        df_edu_agg["EDATTAIN_H"] = df_edu_agg["EDATTAIN_H"].astype(str).replace({
                            "1": "Less than Primary", "1.0": "Less than Primary",
                            "2": "Primary", "2.0": "Primary",
                            "3": "Secondary", "3.0": "Secondary",
                            "4": "University", "4.0": "University"
                        })
                        
                        _save_line_plot(df_edu_agg, "YEAR", "wealth_progress_index", output_dir / "education_wealth_relationship.png", "Education-Wealth Relationship", "Wealth Progress Index (0-100)", hue="EDATTAIN_H")
                        
    # For the rest of the charts, safely check if the CSV exists, then draw it
    gini_path = tables_dir / "yearly_weighted_gini.csv"
    if gini_path.exists():
        df = pd.read_csv(gini_path)
        _save_line_plot(df, "YEAR", "GINI", output_dir / "gini_evolution.png", "Gini Evolution", "Gini coefficient")

    gender_gap_path = tables_dir / "gender_gaps_by_year.csv"
    if gender_gap_path.exists():
        df = pd.read_csv(gender_gap_path)
        df["SEX_H"] = df["SEX_H"].astype(str).replace({"1": "Male", "1.0": "Male", "2": "Female", "2.0": "Female"})
        _save_line_plot(df, "YEAR", "MEAN_EDUCATION_ATTAINMENT", output_dir / "gender_gap_trends.png", "Gender Gap in Education Over Time", "Mean education attainment", hue="SEX_H")

    sigma_path = tables_dir / "regional_sigma_convergence.csv"
    if sigma_path.exists():
        df = pd.read_csv(sigma_path)
        _save_line_plot(df, "YEAR", "sigma_std", output_dir / "regional_sigma_convergence.png", "Regional Sigma Convergence", "Std. dev. of regional mean wealth")

    model_comp_path = tables_dir / "model_comparison_all.csv"
    if model_comp_path.exists():
        df = pd.read_csv(model_comp_path)
        _save_bar_plot(df.sort_values("macro_f1", ascending=False), "model", "macro_f1", output_dir / "model_comparison_chart.png", "Traditional ML vs Deep Learning", "Macro F1")

def build_research_dashboard(
    figures_dir: Path = OUTPUT_FIGURES_DIR,
    tables_dir: Path = OUTPUT_TABLES_DIR,
    reports_dir: Path = OUTPUT_REPORTS_DIR,
) -> Path:
    """Builds the final HTML webpage, compiling all generated images and CSV tables together"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = reports_dir / "research_dashboard.html"

    # Load all available machine learning and statistical result tables
    model_comp = _load_if_exists(tables_dir / "model_comparison_all.csv")
    temporal = _load_if_exists(tables_dir / "temporal_model_validation.csv")
    importance = _load_if_exists(tables_dir / "model_feature_importance.csv")
    subgroup = _load_if_exists(tables_dir / "subgroup_model_performance.csv")
    repr_benchmark = _load_if_exists(tables_dir / "representation_benchmark_pca_vs_autoencoder.csv")
    reg = _load_if_exists(tables_dir / "household_regression_coefficients.csv")
    gender = _load_if_exists(tables_dir / "gender_gaps_by_year.csv")
    dl_temporal = _load_if_exists(tables_dir / "deep_learning_temporal_validation.csv")
    dl_importance = _load_if_exists(tables_dir / "mlp_feature_importance.csv")
    dl_subgroup = _load_if_exists(tables_dir / "mlp_subgroup_performance.csv")
    dl_ablation = _load_if_exists(tables_dir / "deep_learning_ablation.csv")
    dl_latent_sweep = _load_if_exists(tables_dir / "autoencoder_latent_sweep.csv")
    bn_edges = _load_if_exists(tables_dir / "bayesian_network_structure_edges.csv")
    bn_scenarios = _load_if_exists(tables_dir / "bayesian_network_scenario_analysis.csv")
    kmeans_metrics = _load_if_exists(tables_dir / "kmeans_cluster_metrics.csv")
    kmeans_summary = _load_if_exists(tables_dir / "kmeans_cluster_summary.csv")
    anomaly_summary = _load_if_exists(tables_dir / "isolation_forest_summary.csv")

    # Generate the top summary bullet point comparing the best vs worst ML models
    key_points = []
    if model_comp is not None and not model_comp.empty:
        ranked = model_comp.sort_values("macro_f1", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        baseline = ranked.iloc[-1]
        key_points.append(
            f"The central result of this study is predictive: {best['model']} achieves the strongest wealth-class performance with macro F1 = {best['macro_f1']:.3f}, compared with {baseline['model']} at {baseline['macro_f1']:.3f}."
        )
    if not key_points:
        key_points.append("Run the full pipeline to populate empirical findings and figures.")

    # Create HTML image tags for every visualization graph we saved
    figure_blocks = [
        ("Wealth Trend", "wealth_trend_over_time.png"),
        ("Gini Evolution", "gini_evolution.png"),
        ("Urban-Rural Wealth", "urban_rural_wealth_comparison.png"),
        ("Gender Gap Trends", "gender_gap_trends.png"),
        ("Education and Wealth", "education_wealth_relationship.png"),
        ("Regional Convergence", "regional_sigma_convergence.png"),
        ("Model Comparison", "model_comparison_chart.png"),
    ]
    figure_html = []
    for title, filename in figure_blocks:
        path = figures_dir / filename
        if path.exists():
            figure_html.append(
                f"""
                <section class="card">
                  <h3>{title}</h3>
                  <img src="../figures/{filename}" alt="{title}">
                </section>
                """
            )

    # Compose the massive HTML file block by block (CSS Styling, Images, Tables, JS scripts)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vietnam SES Research Dashboard</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 0;
      padding: 0;
      background: #f6f1e8;
      color: #1f1a17;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2, h3 {{
      font-weight: 600;
      margin-bottom: 12px;
    }}
    .lead {{
      font-size: 18px;
      line-height: 1.6;
      max-width: 860px;
    }}
    .grid-figures {{
      display: flex;
      flex-direction: column;
      gap: 32px;
      margin-top: 32px;
      margin-bottom: 32px;
    }}
    
    /* --- THE ULTIMATE SINGLE COLUMN TABLE FIX --- */
    /* Forces the tables to stack vertically across the full width of the screen */
    .grid-tables {{
      display: flex !important;
      flex-direction: column !important;
      gap: 32px !important;
      width: 100% !important;
    }}
    
    .card {{
      background: #fffdf9;
      border: 1px solid #d8cfc1;
      border-radius: 14px;
      padding: 24px;
      box-shadow: 0 8px 24px rgba(60, 45, 30, 0.06);
      /* Initial state for the fade-in animation */
      opacity: 0;
      transform: translateY(30px);
      transition: opacity 0.6s ease-out, transform 0.6s ease-out;
      width: 100% !important; 
      box-sizing: border-box !important;
      overflow-x: auto !important; 
    }}
    
    /* Triggered by JavaScript when the user scrolls down to this card */
    .card.is-visible {{
      opacity: 1;
      transform: translateY(0);
    }}
    
    img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #e3d8ca;
      background: #fffdf9;
    }}
    ul {{ line-height: 1.7; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; min-width: 800px; }}
    th, td {{ border-bottom: 1px solid #eadfce; padding: 12px 8px; text-align: left; }}
    th {{ background-color: #fcfaf5; }}
    .note {{ font-size: 14px; color: #5e544c; margin-top: 24px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Vietnam Socioeconomic Change, 1989-2019</h1>
    <p class="lead">
      This static dashboard summarizes the empirical outputs of the Vietnam census microdata project.
      It is designed as a research presentation artifact rather than a product interface, with machine learning and deep learning placed at the center of the analysis.
    </p>

    <section class="card">
      <h2>Main Findings Snapshot</h2>
      <ul>
        {''.join(f'<li>{point}</li>' for point in key_points)}
      </ul>
    </section>

    <!-- Insert all Graph Images here -->
    <div class="grid-figures">
      {''.join(figure_html)}
    </div>

    <!-- Insert all generated HTML Tables here -->
    <div class="grid-tables">
      <section class="card">
        <h3>Model Comparison</h3>
        {_format_top_rows(model_comp)}
      </section>
      <section class="card">
        <h3>Temporal Validation</h3>
        {_format_top_rows(temporal)}
      </section>
      <section class="card">
        <h3>Feature Importance</h3>
        {_format_top_rows(importance)}
      </section>
      <section class="card">
        <h3>Subgroup Performance</h3>
        {_format_top_rows(subgroup)}
      </section>
      <section class="card">
        <h3>PCA vs Autoencoder</h3>
        {_format_top_rows(repr_benchmark)}
      </section>
      <section class="card">
        <h3>Household Regression Coefficients</h3>
        {_format_top_rows(reg)}
      </section>
      <section class="card">
        <h3>Gender Gap Table</h3>
        {_format_top_rows(gender)}
      </section>
      <section class="card">
        <h3>MLP Temporal Validation</h3>
        {_format_top_rows(dl_temporal)}
      </section>
      <section class="card">
        <h3>MLP Feature Importance</h3>
        {_format_top_rows(dl_importance)}
      </section>
      <section class="card">
        <h3>MLP Subgroup Performance</h3>
        {_format_top_rows(dl_subgroup)}
      </section>
      <section class="card">
        <h3>DL Ablation</h3>
        {_format_top_rows(dl_ablation)}
      </section>
      <section class="card">
        <h3>Autoencoder Latent Sweep</h3>
        {_format_top_rows(dl_latent_sweep)}
      </section>
      <section class="card">
        <h3>Bayesian Network Edges</h3>
        {_format_top_rows(bn_edges)}
      </section>
      <section class="card">
        <h3>Bayesian Network Scenarios</h3>
        {_format_top_rows(bn_scenarios)}
      </section>
      <section class="card">
        <h3>K-Means Metrics</h3>
        {_format_top_rows(kmeans_metrics)}
      </section>
      <section class="card">
        <h3>K-Means Cluster Summary</h3>
        {_format_top_rows(kmeans_summary)}
      </section>
      <section class="card">
        <h3>Isolation Forest Summary</h3>
        {_format_top_rows(anomaly_summary)}
      </section>
    </div>

    <p class="note">
      ML and deep learning are the primary analytical core of this version of the project.
      Descriptive distributional evidence and econometric models are retained as interpretation and validation layers around the predictive results.
    </p>
  </div>
  
  <script>
    // This script runs when the webpage finishes loading
    document.addEventListener("DOMContentLoaded", () => {{
      const cards = document.querySelectorAll('.card');
      
      // IntersectionObserver acts as a motion sensor, waiting for the user to scroll down
      const observerOptions = {{
        root: null,
        rootMargin: '0px 0px -50px 0px',
        threshold: 0.02
      }};
      
      const observerCallback = (entries, observer) => {{
        // Filter out cards that are completely on the screen
        const intersectingEntries = entries.filter(entry => entry.isIntersecting);
        
        // Add a 'staggered delay' (150ms per item) so things pop onto the screen one-by-one in a domino effect
        intersectingEntries.forEach((entry, index) => {{
          setTimeout(() => {{
            entry.target.classList.add('is-visible');
          }}, index * 150); 
          observer.unobserve(entry.target);
        }});
      }};
      
      const observer = new IntersectionObserver(observerCallback, observerOptions);
      cards.forEach(card => observer.observe(card));
    }});
  </script>
</body>
</html>
"""
    # Write the compiled string to the final HTML file
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path

# When this python script is run directly from the terminal, execute these steps
if __name__ == "__main__":
    print("Redrawing all charts with smooth curves...")
    generate_figures()  
    
    print("Stitching pictures into the dashboard...")
    build_research_dashboard()
    
    print("Done! Go refresh your web browser.")