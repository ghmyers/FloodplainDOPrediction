"""
TITLE: [Module Name]
AUTHOR: [Your Name]
DATE: 2025-04-03
DESCRIPTION: [Brief description of what this module does]
"""

#!/usr/bin/env python3
import os
import sys

# Get project directory dynamically
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)

# Ensure PROJECT_DIR is in sys.path for imports
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Append src/ to Python's module search path
sys.path.append(os.path.join(PROJECT_DIR, "src"))

# Define subdirectories
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')

# Postprocessing functions
import tensorflow as tf
def extract_true_values(generator, steps):
    """
    Takes a data generator and its steps (number of batches to generate) and returns an array of the targets in produced. 

    Args:
        generator (Generator): The data generator that produces batches of data.
        steps (int): The number of batches to generate.

    Returns:
        y_true (tf.Array): A tensorflow array of true values
    """
    y_batches = []

    for step, (X_batch, y_batch) in enumerate(generator):
        if step >= steps:  # Stop after val_steps steps
            break
        y_batches.append(y_batch)

    # Combine all batches into a single array or tensor
    y_true = tf.concat(y_batches, axis=0)
    return y_true


import matplotlib.pyplot as plt
import hydroeval as he
from sklearn.metrics import mean_squared_error, r2_score
import math
import numpy as np
def evaluate_river_predictions(y_true, y_preds, scaler_y):
    """
    Evaluates river LSTM model performance on the river testing data.

    Args:
        y_true (np.Array): Observed DO values from the river testing data.
        y_preds (np.Array): Predicted DO values from the river LSTM model.
        scaler_y (MinMaxScaler): Scaler to rescale data to original scale.

    Returns:
        performance_metrics (dict): Dictionary of rmse, r2, kge, and bias.
    """
    # Convert to 1-D arrays and inverse tranform
    y_true -= 1e-7
    y_preds -= 1e-7
    
    # Rescale data to original scale
    actual = scaler_y.inverse_transform(np.array(y_true).flatten().reshape(-1, 1))
    predictions = scaler_y.inverse_transform(np.array(y_preds).flatten().reshape(-1, 1))
    
    plt.hist(actual)
    plt.title("Distribution of DO in Testing Data")
    plt.show()
    plt.hist(predictions)
    plt.title("Distribution of DO Predictions")
    plt.show()
    
    rmse = math.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    kge = he.evaluator(he.kge, actual, predictions)
    bias = he.evaluator(he.pbias, actual, predictions)
    performance_metrics = {"RMSE": round(rmse, 3),
                            'KGE': round(kge[0][0], 3),
                            f"R{chr(0x00B2)}": round(r2, 3),
                            'Bias': round(bias[0], 3)}
    return performance_metrics

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def plot_result_sites(y_true, y_preds, test_start, test_end, scaler, site_code, FT=False, train=False, FP=False):
    # store dates
    dates = y_true.index
    test_start = dates[test_start]
    test_end = dates[test_end]
    
    # Convert to 1-D arrays and inverse transform
    y_true -= 1e-7
    y_preds -= 1e-7
    
    actual = scaler.inverse_transform(np.array(y_true).flatten().reshape(-1, 1))
    predictions = scaler.inverse_transform(np.array(y_preds).flatten().reshape(-1, 1))
    
    # convert back to pandas series and index by date
    actual = pd.Series(actual.flatten(), index=dates, name='Actual')
    predictions = pd.Series(predictions.flatten(), index=dates, name='Predictions')
    actual = actual.asfreq('D')
    predictions = predictions.asfreq('D')
    
    # Plot predictions
    fig, ax1 = plt.subplots(figsize=(15, 6))
    plt.rcParams.update({'font.size': 16})
    colors = plt.cm.Set2(np.linspace(0, 1, 7))
    ax1.set_xlabel('Days')
    ax1.set_ylabel('DO (mg/L)')
    ax1.plot(actual, label='Actual', color=colors[0])
    ax1.plot(predictions, label='Predictions', color=colors[1])
    if train:
        plt.axvline(x=test_start, color='black', linestyle="--", label="Train/Test Split")
        plt.axvline(x=test_end, color='black', linestyle="--")

        
        actual, predictions = actual.dropna(), predictions.dropna()
        
        # Performance Metrics (evaluated in testing data)
        rmse = math.sqrt(mean_squared_error(actual[test_start:], predictions[test_start:]))
        r2 = r2_score(actual[test_start:], predictions[test_start:])
        kge = he.evaluator(he.kge, actual[test_start:], predictions[test_start:])
        bias = he.evaluator(he.pbias, actual[test_start:], predictions[test_start:])
        performance_metrics = {"RMSE": round(rmse, 2),
                              'KGE': round(kge[0][0], 2),
                              f"R{chr(0x00B2)}": round(r2, 2),
                              'Bias': round(bias[0], 2)}
        metrics_text = '\n'.join([f'{key}: {value}' for key, value in performance_metrics.items()])
        fig.text(0.91, 0.62, metrics_text, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.15))
        fig.text(0.91, 0.67, 'Testing Data', fontsize=14, verticalalignment='top')
        
        rmse_whole = math.sqrt(mean_squared_error(actual, predictions))
        r2_whole = r2_score(actual, predictions)
        kge_whole = he.evaluator(he.kge, actual, predictions)
        bias_whole = he.evaluator(he.pbias, actual, predictions)
        performance_metrics_whole = {"RMSE": round(rmse_whole, 2),
                                     'KGE': round(kge_whole[0][0], 2),
                                     f"R{chr(0x00B2)}": round(r2_whole, 2),
                                     'Bias': round(bias_whole[0], 2)}
        metrics_text_whole = '\n'.join([f'{key}: {value}' for key, value in performance_metrics_whole.items()])
        fig.text(0.91, 0.35, metrics_text_whole, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.15))
        fig.text(0.91, 0.4, 'All Data', fontsize=14, verticalalignment='top')
        PM_df_allData = pd.DataFrame.from_dict(performance_metrics_whole, 'index')
    else:
        actual, predictions = actual.dropna(), predictions.dropna()
        rmse = math.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)
        kge = he.evaluator(he.kge, actual, predictions)
        bias = he.evaluator(he.pbias, actual, predictions)
        performance_metrics = {"RMSE": round(rmse, 2),
                                'KGE': round(kge[0][0], 2),
                                f"R{chr(0x00B2)}": round(r2, 2),
                                'Bias': round(bias[0], 2)}
        metrics_text = '\n'.join([f'{key}: {value}' for key, value in performance_metrics.items()])
        fig.text(0.91, 0.7, metrics_text, fontsize=14, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.15))
            
        fig.legend(loc="upper left", bbox_to_anchor=(0.9, 0.85), fontsize=12)
        plt.title(f'{site_code} Actual and Predicted DO Values', pad=10)
        PM_df = pd.DataFrame.from_dict(performance_metrics, 'index')
        
# Analyze model results
def get_prediction_dataframe(y, preds_PT, preds_FT, preds_FT_FRZ, preds_FT_FRZ_KD, preds_FT_FRZ_RD, preds_FP, scaler_y, site):
    dates = y.index

    y -= 1e-7
    preds_PT -= 1e-7
    preds_FT -= 1e-7
    preds_FT_FRZ -= 1e-7
    preds_FT_FRZ_KD -= 1e-7
    preds_FT_FRZ_RD -= 1e-7
    preds_FP -= 1e-7
 
    # actual = scaler.inverse_transform(np.array(y_true).flatten().reshape(-1, 1))
    y_series = scaler_y.inverse_transform(np.array(y).flatten().reshape(-1, 1))
    preds_PT_series = scaler_y.inverse_transform(np.array(preds_PT).flatten().reshape(-1, 1))
    preds_FT_series = scaler_y.inverse_transform(np.array(preds_FT).flatten().reshape(-1, 1))
    preds_FT_FRZ_series = scaler_y.inverse_transform(np.array(preds_FT_FRZ).flatten().reshape(-1, 1))
    preds_FT_FRZ_KD_series = scaler_y.inverse_transform(np.array(preds_FT_FRZ_KD).flatten().reshape(-1, 1))
    preds_FT_FRZ_RD_series = scaler_y.inverse_transform(np.array(preds_FT_FRZ_RD).flatten().reshape(-1, 1))
    preds_FP_series = scaler_y.inverse_transform(np.array(preds_FP).flatten().reshape(-1, 1))

    

    y_series = pd.Series(y_series.flatten(), index=dates, name='DO')
    preds_PT_series = pd.Series(preds_PT_series.flatten(), index=dates, name='DO_preds_PT')
    preds_FT_series = pd.Series(preds_FT_series.flatten(), index=dates, name='DO_preds_FT')
    preds_FT_FRZ_series = pd.Series(preds_FT_FRZ_series.flatten(), index=dates, name='DO_preds_FT_FRZ')
    preds_FT_FRZ_KD_series = pd.Series(preds_FT_FRZ_KD_series.flatten(), index=dates, name='DO_preds_FT_FRZ_K')
    preds_FT_FRZ_RD_series = pd.Series(preds_FT_FRZ_RD_series.flatten(), index=dates, name='DO_preds_FT_FRZ_R')
    preds_FP_series = pd.Series(preds_FP_series.flatten(), index=dates, name='DO_preds_FP')
    


    pred_list = [y_series, preds_PT_series, preds_FT_series, preds_FT_FRZ_series, preds_FT_FRZ_KD_series, preds_FT_FRZ_RD_series, preds_FP_series]
    pred_df = pd.DataFrame(pred_list).T
    
    return pred_df


def analyze_model_results(df_dict):
    results = []
    results_low = []
    for site, df in df_dict.items():
        actual = df['DO']
        predictions_PT = df['DO_preds_PT']
        predictions_FT = df['DO_preds_FT']
        predictions_FT_FRZ = df['DO_preds_FT_FRZ']
        predictions_FT_FRZ_KD = df['DO_preds_FT_FRZ_K']
        predictions_FT_FRZ_RD = df['DO_preds_FT_FRZ_R']
        predictions_FP = df['DO_preds_FP']
        pred_list = [predictions_PT, predictions_FT, predictions_FT_FRZ, predictions_FT_FRZ_KD, predictions_FT_FRZ_RD, predictions_FP]
        model_names = ["River LSTM", "TL LSTM", "TL-FRZ LSTM", "TL-FRZ-K LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
        performance_metrics = {}
        for i, predictions in enumerate(pred_list):
            rmse = math.sqrt(mean_squared_error(actual, predictions))
            r2 = r2_score(actual, predictions)
            kge = he.evaluator(he.kge, actual, predictions)
            bias = he.evaluator(he.pbias, actual, predictions)
            performance_metrics[f"{model_names[i]} RMSE"] = round(rmse, 2)
            performance_metrics[f'{model_names[i]} KGE'] = round(kge[0][0], 2)
            performance_metrics[f"{model_names[i]} R{chr(0x00B2)}"] = round(r2, 2)
            performance_metrics[f'{model_names[i]} Bias'] = round(bias[0], 2)
        
        performance_metrics['site'] = site
        results.append(performance_metrics)
    
    master_df = pd.DataFrame(results).set_index('site')
    
    # Perform same analysis for low DO timeperiods
    for site, df in df_dict.items():
        print(site)
        actual = df['DO']
        predictions_PT = df['DO_preds_PT']
        predictions_FT = df['DO_preds_FT']
        predictions_FT_FRZ = df['DO_preds_FT_FRZ']
        predictions_FT_FRZ_KD = df['DO_preds_FT_FRZ_K']
        predictions_FT_FRZ_RD = df['DO_preds_FT_FRZ_R']
        predictions_FP = df['DO_preds_FP']
        model_names = ["River LSTM", "TL LSTM", "TL-FRZ LSTM", "TL-FRZ-K LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
        performance_metrics_low = {}
    
        low_inds = actual[actual < 5].index.tolist()
        actual_low = actual[low_inds]
        predictions_PT_low = predictions_PT[low_inds]
        predictions_FT_low = predictions_FT[low_inds]
        predictions_FT_FRZ_low = predictions_FT_FRZ[low_inds]
        predictions_FT_FRZ_KD_low = predictions_FT_FRZ_KD[low_inds]
        predictions_FT_FRZ_RD_low = predictions_FT_FRZ_RD[low_inds]
        predictions_FP_low = predictions_FP[low_inds]
        pred_list_low = [predictions_PT_low, predictions_FT_low, predictions_FT_FRZ_low, predictions_FT_FRZ_KD_low, predictions_FT_FRZ_RD_low, predictions_FP_low]
        
        for i, predictions in enumerate(pred_list_low):
            rmse = math.sqrt(mean_squared_error(actual_low, predictions))
            r2 = r2_score(actual_low, predictions)
            kge = he.evaluator(he.kge, actual_low, predictions)
            bias = he.evaluator(he.pbias, actual_low, predictions)
            performance_metrics_low[f"{model_names[i]} RMSE"] = round(rmse, 2)
            performance_metrics_low[f'{model_names[i]} KGE'] = round(kge[0][0], 2)
            performance_metrics_low[f"{model_names[i]} R{chr(0x00B2)}"] = round(r2, 2)
            performance_metrics_low[f'{model_names[i]} Bias'] = round(bias[0], 2)
    
        performance_metrics_low['site'] = site
        results_low.append(performance_metrics_low)
    
    master_df_low = pd.DataFrame(results_low).set_index('site')

    return master_df, master_df_low

def analyze_model_results_noKDFRZ(df_dict):
    results = []
    for site, df in df_dict.items():
        actual = df['DO']
        predictions_PT = df['DO_preds_PT']
        predictions_FT = df['DO_preds_FT']
        predictions_FT_FRZ_RD = df['DO_preds_FT_FRZ_R']
        predictions_FP = df['DO_preds_FP']
        pred_list = [predictions_PT, predictions_FT, predictions_FT_FRZ_RD, predictions_FP]
        model_names = ["River LSTM", "TL LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
        performance_metrics = {}
        for i, predictions in enumerate(pred_list):
            rmse = math.sqrt(mean_squared_error(actual, predictions))
            r2 = r2_score(actual, predictions)
            kge = he.evaluator(he.kge, actual, predictions)
            bias = he.evaluator(he.pbias, actual, predictions)
            performance_metrics[f"{model_names[i]} RMSE"] = round(rmse, 2)
            performance_metrics[f'{model_names[i]} KGE'] = round(kge[0][0], 2)
            performance_metrics[f"{model_names[i]} R{chr(0x00B2)}"] = round(r2, 2)
            performance_metrics[f'{model_names[i]} Bias'] = round(bias[0], 2)
        
        performance_metrics['site'] = site
        results.append(performance_metrics)
    
    master_df = pd.DataFrame(results).set_index('site')

    return master_df

import scipy.stats as stats
import scikit_posthocs as sp
import itertools

def run_posthoc_tests(results_df):
    metrics = ['RMSE', 'KGE', f'R{chr(0x00B2)}', 'Bias']
    model_names = ["River LSTM", "TL LSTM", "TL-FRZ LSTM", 
                   "TL-FRZ-K LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
    
    results_list = []

    for metric in metrics:
        valid_models = []
        data = []

        # Filter models that have non-empty data
        for model in model_names:
            values = results_df[f"{model} {metric}"].dropna()
            if len(values) > 0:
                valid_models.append(model)
                data.append(values.values)

        # Skip metric if fewer than 2 models have data
        if len(valid_models) < 2:
            continue

        # Kruskal-Wallis test
        H_stat, p_kw = stats.kruskal(*data)
        results_list.append({
            'Metric': metric,
            'Test': 'Kruskal-Wallis',
            'Group1': 'All models',
            'Group2': '',
            'p-value': p_kw,
            'Significant': p_kw < 0.05
        })
        

        # If significant, perform Dunn’s test
        if p_kw < 0.05:
            # Prepare long-form DataFrame
            melted = pd.DataFrame({
                'Score': list(itertools.chain.from_iterable(data)),
                'Model': list(itertools.chain.from_iterable([[model]*len(group) for model, group in zip(valid_models, data)]))
            })

            dunn = sp.posthoc_dunn(melted, val_col='Score', group_col='Model', p_adjust='bonferroni')

            for i, j in itertools.combinations(valid_models, 2):
                p_dunn = dunn.loc[i, j]
                results_list.append({
                    'Metric': metric,
                    'Test': 'Dunn-Bonferroni',
                    'Group1': i,
                    'Group2': j,
                    'p-value': p_dunn,
                    'Significant': p_dunn < 0.05
                })

    return pd.DataFrame(results_list)

def run_posthoc_tests_noKDFRZ(results_df):
    metrics = ['RMSE', 'KGE', f'R{chr(0x00B2)}', 'Bias']
    model_names = ["River LSTM", "TL LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
    
    results_list = []

    for metric in metrics:
        valid_models = []
        data = []

        # Filter models that have non-empty data
        for model in model_names:
            values = results_df[f"{model} {metric}"].dropna()
            if len(values) > 0:
                valid_models.append(model)
                data.append(values.values)

        # Skip metric if fewer than 2 models have data
        if len(valid_models) < 2:
            continue

        # Kruskal-Wallis test
        H_stat, p_kw = stats.kruskal(*data)
        results_list.append({
            'Metric': metric,
            'Test': 'Kruskal-Wallis',
            'Group1': 'All models',
            'Group2': '',
            'p-value': p_kw,
            'Significant': p_kw < 0.05
        })
        

        # If significant, perform Dunn’s test
        if p_kw < 0.05:
            # Prepare long-form DataFrame
            melted = pd.DataFrame({
                'Score': list(itertools.chain.from_iterable(data)),
                'Model': list(itertools.chain.from_iterable([[model]*len(group) for model, group in zip(valid_models, data)]))
            })

            dunn = sp.posthoc_dunn(melted, val_col='Score', group_col='Model', p_adjust='bonferroni')

            for i, j in itertools.combinations(valid_models, 2):
                p_dunn = dunn.loc[i, j]
                results_list.append({
                    'Metric': metric,
                    'Test': 'Dunn-Bonferroni',
                    'Group1': i,
                    'Group2': j,
                    'p-value': p_dunn,
                    'Significant': p_dunn < 0.05
                })

    return pd.DataFrame(results_list)
# Boxplots
import seaborn as sns
def boxplot_model_performance(summary_df, figpath):
    metrics = ['RMSE', 'KGE', 'R²', 'Bias']
    model_names = ["River LSTM", "TL LSTM", "TL-FRZ LSTM", "TL-FRZ-K LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
    colors = sns.color_palette('icefire')
    color_list = [colors[0], colors[2], colors[3], colors[4], colors[5], colors[1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        # Collect values for each model across all sites
        data = [summary_df[f"{model} {metric}"] for model in model_names]

        # Create boxplot
        bp = ax.boxplot(data, labels=model_names, patch_artist=True)

        # Apply custom box colors
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')

        # Make median lines black
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1)

        # Remove top/right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # No grid lines
        ax.grid(False)

        ax.set_title(f"{metric} Distribution Across Sites")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(figpath, dpi=300)
    plt.show()
    
    

def boxplot_model_performance_noKDFRZ(summary_df, figpath):
    metrics = ['RMSE', 'KGE', 'R²', 'Bias']
    model_names = ["River LSTM", "TL LSTM","TL-FRZ-R LSTM", "Floodplain LSTM"]
    colors = sns.color_palette('icefire')
    color_list = [colors[0], colors[2], colors[5], colors[1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        # Collect values for each model across all sites
        data = [summary_df[f"{model} {metric}"] for model in model_names]

        # Create boxplot
        bp = ax.boxplot(data, labels=model_names, patch_artist=True)

        # Apply custom box colors
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')

        # Make median lines black
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1)

        # Remove top/right spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # No grid lines
        ax.grid(False)

        ax.set_title(f"{metric} Distribution Across Sites")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(figpath, dpi=300)
    plt.show()
    
    

def add_significance_brackets(ax, pairs,
                              y_offset_frac=0.1,      # vertical gap between rows of brackets
                              bracket_height_frac=0.015, # height of each bracket
                              text_pad_frac=0.0001,       # extra pad for asterisks
                              fontsize=10):
    """
    Draw significance brackets whose spacing is a *fraction* of the axis height,
    so appearance is consistent regardless of data scale.
    """
    ylim_orig = ax.get_ylim()
    y_range   = ylim_orig[1] - ylim_orig[0]

    for i, (idx1, idx2, label) in enumerate(pairs):
        # Convert box index (0‑based) ➜ x‑position (1‑based as mpl draws them)
        x1, x2 = idx1 + 1, idx2 + 1

        # Vertical position of this row’s bracket (in data coords)
        y_base = ylim_orig[1] + (i + 1) * y_offset_frac * y_range
        h      = bracket_height_frac * y_range

        # Bracket
        ax.plot([x1, x1, x2, x2],
                [y_base, y_base + h, y_base + h, y_base],
                lw=1.5, c='black', clip_on=False)

        # Text (asterisks / p‑value)
        ax.text((x1 + x2) * 0.5,
                y_base + h + text_pad_frac * y_range,
                label, ha='center', va='bottom',
                fontsize=fontsize, clip_on=False)

    # Expand the y‑axis so everything is visible
    extra_space = (len(pairs) + 2) * y_offset_frac * y_range
    ax.set_ylim(ylim_orig[0], ylim_orig[1] + extra_space)
# -----------------------------------------------------------------------------
def boxplot_model_performance_with_significance(summary_df, stats_df, figpath):
    metrics      = ['RMSE', 'KGE', 'R²', 'Bias']
    model_names  = ["River LSTM", "TL LSTM", "TL-FRZ LSTM",
                    "TL-FRZ-K LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
    panel_labels = ["a)", "b)", "c)", "d)"]

    # colors       = plt.cm.tab10(np.linspace(0, 1, 8))
    colors = sns.color_palette('icefire')
    color_list   = [colors[0], colors[2], colors[3],
                    colors[4], colors[5], colors[1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        # ---------- Box‑and‑whisker -------------------------------------------------
        data = [summary_df[f"{model} {metric}"].dropna()
                for model in model_names]

        bp = ax.boxplot(data, labels=model_names, patch_artist=True)

        # color & style
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        for med in bp['medians']:
            med.set_color('black'); med.set_linewidth(2)

        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(False)
        ax.set_title(f"{panel_labels[i]} {metric} Distribution Across Models", fontsize=14)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30, labelsize=12)

        # ----------- Median labels -------------------------------------------------
        y_range     = ax.get_ylim()[1] - ax.get_ylim()[0]
        text_offset = 0.01 * y_range                       # 1 % of axis height

        for i, d in enumerate(data):
            y_val = np.percentile(d, 100)
            med_val = np.mean(d)
            ax.text(i + 1,                               # x‑position (1‑based)
                    y_val + text_offset,               # just above median line
                    f"{med_val:.2f}",
                    ha='center', va='bottom',
                    fontsize=10, color='black')
        # ---------- Significance annotations ---------------------------------------
        dunn_results = (
            stats_df.loc[
                (stats_df['Metric'] == metric) &
                (stats_df['Test']   == 'Dunn-Bonferroni') &
                (stats_df['Significant'])
            ]
        )

        if not dunn_results.empty:
            pairs = []
            for _, row in dunn_results.iterrows():
                try:
                    idx1 = model_names.index(row['Group1'])
                    idx2 = model_names.index(row['Group2'])
                except ValueError:
                    continue 

                # Add p-valu * label
                p = row['p-value']
                label = "***" if p < 0.005 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                pairs.append((min(idx1, idx2), max(idx1, idx2), label))

            if pairs:
                add_significance_brackets(ax, pairs)

    plt.tight_layout()
    plt.savefig(figpath, dpi=600)
    plt.show()
    
def boxplot_model_performance_with_significance_noKDFRZ(summary_df, stats_df, figpath):
    metrics      = ['RMSE', 'KGE', 'R²', 'Bias']
    model_names  = ["River LSTM", "TL LSTM", "TL-FRZ-R LSTM", "Floodplain LSTM"]
    panel_labels = ["a)", "b)", "c)", "d)"]

    # colors       = plt.cm.tab10(np.linspace(0, 1, 8))
    colors = sns.color_palette('icefire')
    color_list   = [colors[0], colors[2], colors[5], colors[1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        # ---------- Box‑and‑whisker -------------------------------------------------
        data = [summary_df[f"{model} {metric}"].dropna()
                for model in model_names]

        bp = ax.boxplot(data, labels=model_names, patch_artist=True)

        # color & style
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        for med in bp['medians']:
            med.set_color('black'); med.set_linewidth(2)

        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(False)
        ax.set_title(f"{panel_labels[i]} {metric} Distribution Across Models", fontsize=14)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30, labelsize=12)

        # ----------- Median labels -------------------------------------------------
        y_range     = ax.get_ylim()[1] - ax.get_ylim()[0]
        text_offset = 0.01 * y_range                       # 1 % of axis height

        for i, d in enumerate(data):
            y_val = np.percentile(d, 100)
            med_val = np.mean(d)
            ax.text(i + 1,                               # x‑position (1‑based)
                    y_val + text_offset,               # just above median line
                    f"{med_val:.2f}",
                    ha='center', va='bottom',
                    fontsize=10, color='black')
        # ---------- Significance annotations ---------------------------------------
        dunn_results = (
            stats_df.loc[
                (stats_df['Metric'] == metric) &
                (stats_df['Test']   == 'Dunn-Bonferroni') &
                (stats_df['Significant'])
            ]
        )

        if not dunn_results.empty:
            pairs = []
            for _, row in dunn_results.iterrows():
                try:
                    idx1 = model_names.index(row['Group1'])
                    idx2 = model_names.index(row['Group2'])
                except ValueError:
                    continue 

                # Add p-valu * label
                p = row['p-value']
                label = "***" if p < 0.005 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                pairs.append((min(idx1, idx2), max(idx1, idx2), label))

            if pairs:
                add_significance_brackets(ax, pairs)

    plt.tight_layout()
    plt.savefig(figpath, dpi=600)
    plt.show()
    

# Site prediction time series plotting

def plot_all_result_sites(y_true, y_preds_PT, y_preds_FT, y_preds_FT_FRZ, y_preds_FT_FRZ_KD, y_preds_FT_FRZ_RD, y_preds_FP, test_start, test_end, scaler, site_code, train=False, TL_only=False):
    # store dates
    dates = y_true.index
    test_start = dates[test_start]
    test_end = dates[test_end]
    
    # Convert to 1-D arrays and inverse tranform
    y_true -= 1e-7
    y_preds_PT -= 1e-7
    y_preds_FT -= 1e-7
    y_preds_FT_FRZ -= 1e-7
    y_preds_FT_FRZ_KD -= 1e-7
    y_preds_FT_FRZ_RD -= 1e-7
    y_preds_FP -= 1e-7
    
    actual = scaler.inverse_transform(np.array(y_true).flatten().reshape(-1, 1))
    predictions_PT = scaler.inverse_transform(np.array(y_preds_PT).flatten().reshape(-1, 1))
    predictions_FT = scaler.inverse_transform(np.array(y_preds_FT).flatten().reshape(-1, 1))
    predictions_FT_FRZ = scaler.inverse_transform(np.array(y_preds_FT_FRZ).flatten().reshape(-1, 1))
    predictions_FT_FRZ_KD = scaler.inverse_transform(np.array(y_preds_FT_FRZ_KD).flatten().reshape(-1, 1))
    predictions_FT_FRZ_RD = scaler.inverse_transform(np.array(y_preds_FT_FRZ_RD).flatten().reshape(-1, 1))
    predictions_FP = scaler.inverse_transform(np.array(y_preds_FP).flatten().reshape(-1, 1))
    
    # convert back to pandas series and index by date
    actual = pd.Series(actual.flatten(), index=dates, name='Actual')
    predictions_PT = pd.Series(predictions_PT.flatten(), index=dates, name='River LSTM Predictions')
    predictions_FT = pd.Series(predictions_FT.flatten(), index=dates, name='TL LSTM Predictions')
    predictions_FT_FRZ = pd.Series(predictions_FT_FRZ.flatten(), index=dates, name='TL-FRZ LSTM Predictions')
    predictions_FT_FRZ_KD = pd.Series(predictions_FT_FRZ_KD.flatten(), index=dates, name='TL-FRZ-K LSTM Predictions')
    predictions_FT_FRZ_RD = pd.Series(predictions_FT_FRZ_RD.flatten(), index=dates, name='TL-FRZ-R LSTM Predictions')
    predictions_FP = pd.Series(predictions_FP.flatten(), index=dates, name='Floodplain LSTM Predictions')
    actual = actual.asfreq('D')
    predictions_PT = predictions_PT.asfreq('D')
    predictions_FT = predictions_FT.asfreq('D')
    predictions_FT_FRZ = predictions_FT_FRZ.asfreq('D')
    predictions_FT_FRZ_KD = predictions_FT_FRZ_KD.asfreq('D')
    predictions_FT_FRZ_RD = predictions_FT_FRZ_RD.asfreq('D')
    predictions_FP = predictions_FP.asfreq('D')
    
    # Plot predictions
    fig, ax1 = plt.subplots(figsize=(15, 6))
    plt.rcParams.update({'font.size': 16})
    colors = sns.color_palette("icefire")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('DO (mg/L)')
    if TL_only:
        ax1.plot(actual, label='Actual', color='#7f7f7f')
        ax1.plot(predictions_FT, label='TL LSTM Predictions', color=colors[2])
        ax1.plot(predictions_FT_FRZ, label='TL-FRZ LSTM Predictions', color=colors[3])
        ax1.plot(predictions_FT_FRZ_KD, label='TL-FRZ-KD LSTM Predictions', color=colors[4])
        ax1.plot(predictions_FT_FRZ_RD, label='TL-FRZ-RD LSTM Predictions', color=colors[5])
        
        if train:
            plt.axvline(x=test_start, color='black', linestyle="--", label="Train/Test Split")
            plt.axvline(x=test_end, color='black', linestyle="--")
            
        fig.subplots_adjust(right=0.80)      # values <1 move the right border left
        legend = fig.legend(loc="upper left", bbox_to_anchor=(0.8, 0.88), fontsize=12, borderaxespad=0)
        plt.title(f'{site_code} Actual and Predicted DO Values', pad=10)
        # plt.tight_layout()
        # plt.savefig(FIGURE_DIR + f'/predictions/TL_only/{site_code}_TL_only_model_predictions.png', dpi=600,
        #             bbox_inches='tight', bbox_extra_artists=[legend])
        plt.show()
    else:
        ax1.plot(actual, label='Actual', color='#7f7f7f')
        ax1.plot(predictions_PT, label='River LSTM Predictions', color=colors[0])
        ax1.plot(predictions_FP, label='Floodplain LSTM Predictions', color=colors[1])
        ax1.plot(predictions_FT, label='TL LSTM Predictions', color=colors[2])
        # ax1.plot(predictions_FT_FRZ, label='TL-FRZ LSTM Predictions', color=colors[4])
        # ax1.plot(predictions_FT_FRZ_KD, label='TL-FRZ-KD LSTM Predictions', color=colors[5])
        ax1.plot(predictions_FT_FRZ_RD, label='TL-FRZ-R LSTM Predictions', color=colors[5])
        if train:
            plt.axvline(x=test_start, color='black', linestyle="--", label="Train/Test Split")
            plt.axvline(x=test_end, color='black', linestyle="--")
            
        fig.subplots_adjust(right=0.80)      # values <1 move the right border left
        legend = fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.-15), fontsize=12, borderaxespad=0, ncol=3)
        plt.title(f'{site_code} Actual and Predicted DO Values', pad=10)
        # plt.tight_layout()
        
        # plt.savefig(FIGURE_DIR + f'/predictions/{site_code}_model_predictions.png', dpi=600,
        #             bbox_inches="tight", bbox_extra_artists=[legend])
        plt.show()
