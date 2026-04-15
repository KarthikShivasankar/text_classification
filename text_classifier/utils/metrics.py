"""Utility functions for metrics calculation and visualization."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from scipy.special import softmax
import scipy.stats
from sklearn.preprocessing import label_binarize
from itertools import cycle
from typing import Dict, List, Optional, Union, Any, Tuple


def plot_detailed_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Plot a detailed confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(14, 12))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Detailed Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Increase the size of axis labels for better visibility
    ax.set_xticklabels(class_names, fontsize=14)
    ax.set_yticklabels(class_names, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title("Detailed Confusion Matrix", fontsize=16)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.savefig(os.path.join(output_dir, "detailed_confusion_matrix.png"))
    plt.close(fig)


def calculate_confidence_intervals(metrics, confidence=0.95):
    """
    Calculate confidence intervals for metrics.

    Args:
        metrics: Dictionary of metrics
        confidence: Confidence level

    Returns:
        Dictionary of confidence intervals
    """
    metrics_mean = {key: np.mean(value) for key, value in metrics.items()}
    metrics_std = {key: np.std(value, ddof=1) for key, value in metrics.items()}

    rounds = len(next(iter(metrics.values())))

    metrics_ci_bounds = {}
    for key in metrics_mean:
        mean = metrics_mean[key]
        std = metrics_std[key]
        t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=rounds - 1)
        ci_lower = mean - t_value * (std / np.sqrt(rounds))
        ci_upper = mean + t_value * (std / np.sqrt(rounds))
        metrics_ci_bounds[key] = {"ci_lower": ci_lower, "ci_upper": ci_upper}

    return metrics_mean, metrics_std, metrics_ci_bounds


def save_metrics(metrics, metrics_mean, metrics_std, metrics_ci_bounds, output_dir):
    """
    Save metrics to JSON files.

    Args:
        metrics: Dictionary of metrics
        metrics_mean: Dictionary of mean metrics
        metrics_std: Dictionary of standard deviations
        metrics_ci_bounds: Dictionary of confidence intervals
        output_dir: Directory to save the files
    """
    with open(os.path.join(output_dir, "metrics_all_fold.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(output_dir, "metrics_mean.json"), "w") as f:
        json.dump(metrics_mean, f, indent=4)

    with open(os.path.join(output_dir, "metrics_std.json"), "w") as f:
        json.dump(metrics_std, f, indent=4)

    with open(os.path.join(output_dir, "metrics_ci_bounds.json"), "w") as f:
        json.dump(metrics_ci_bounds, f, indent=4)


def plot_metrics_visualization(
    metrics_mean, metrics_std, metrics_ci_bounds, output_dir
):
    """
    Plot a visualization of metrics.

    Args:
        metrics_mean: Dictionary of mean metrics
        metrics_std: Dictionary of standard deviations
        metrics_ci_bounds: Dictionary of confidence intervals
        output_dir: Directory to save the plot
    """
    metrics_names = list(metrics_mean.keys())
    means = [metrics_mean[name] for name in metrics_names]
    stds = [metrics_std[name] for name in metrics_names]
    ci_lowers = [metrics_ci_bounds[name]["ci_lower"] for name in metrics_names]
    ci_uppers = [metrics_ci_bounds[name]["ci_upper"] for name in metrics_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = range(len(metrics_names))

    # Mean values
    ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.6,
        color="skyblue",
        label="Mean",
    )

    # CI bounds
    for i in x_positions:
        ax.plot(
            [i, i],
            [ci_lowers[i], ci_uppers[i]],
            color="orange",
            marker="o",
            label="95% CI" if i == 0 else "",
        )

    # Improving the visual
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics_names, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Mean, and 95% CI of Metrics")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_visualization.png"))
    plt.close()


def log_system_usage():
    """
    Log system usage.

    Returns:
        Tuple of (gpu_load, gpu_memory_usage, cpu_usage, memory_usage)
    """
    try:
        import GPUtil
        import psutil

        gpu_load = [gpu.load for gpu in GPUtil.getGPUs()]
        gpu_memory_usage = [gpu.memoryUsed for gpu in GPUtil.getGPUs()]
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().used / (1024**3)  # Convert to GB

        return gpu_load, gpu_memory_usage, cpu_usage, memory_usage
    except ImportError:
        print("GPUtil or psutil not installed. Cannot log system usage.")
        return None, None, None, None
