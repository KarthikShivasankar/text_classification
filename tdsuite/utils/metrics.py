"""Metrics for evaluating model performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef
)
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns


def compute_metrics(df, output_dir=None, save_plots=True):
    """
    Compute and save classification metrics.
    
    Args:
        df: DataFrame containing predictions and ground truth
        output_dir: Directory to save metrics and plots
        save_plots: Whether to save plots
    
    Returns:
        Dictionary of metrics
    """
    # Extract predictions and ground truth
    y_true = df["label"]
    y_pred = df["predicted_class"]
    
    # Ensure labels are numeric
    if y_true.dtype == 'object' and y_pred.dtype == 'object':
        # Convert string labels to numeric
        unique_labels = sorted(y_true.unique())
        if len(unique_labels) != 2:
            raise ValueError("Binary classification requires exactly 2 unique labels")
        
        # Find the positive category (the one that doesn't start with "non_")
        positive_label = None
        non_label = None
        for label in unique_labels:
            if str(label).startswith("non_"):
                non_label = label
            else:
                positive_label = label
        
        # If we couldn't identify the labels by prefix, use the first one as non_ and second as positive
        if non_label is None or positive_label is None:
            non_label = unique_labels[0]
            positive_label = unique_labels[1]
        
        # Create mapping: non_ -> 0, positive -> 1
        label_map = {non_label: 0, positive_label: 1}
        y_true = y_true.map(label_map)
        y_pred = y_pred.map(label_map)
    elif y_true.dtype != 'object' and y_pred.dtype == 'object':
        # Convert string predictions to numeric
        unique_labels = y_pred.unique()
        if len(unique_labels) != 2:
            raise ValueError("Binary classification requires exactly 2 unique labels")
        
        # Find the positive category (the one that doesn't start with "non_")
        positive_label = None
        non_label = None
        for label in unique_labels:
            if str(label).startswith("non_"):
                non_label = label
            else:
                positive_label = label
        
        # If we couldn't identify the labels by prefix, use the first one as non_ and second as positive
        if non_label is None or positive_label is None:
            sorted_labels = sorted(unique_labels)
            non_label = sorted_labels[0]
            positive_label = sorted_labels[1]
        
        # Create mapping: non_ -> 0, positive -> 1
        label_map = {non_label: 0, positive_label: 1}
        y_pred = y_pred.map(label_map)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Try to compute ROC AUC if probabilities are available
    roc_auc = None
    if "class_probabilities" in df.columns:
        # Extract positive class probability if it's a list
        if isinstance(df["class_probabilities"].iloc[0], list):
            positive_probs = df["class_probabilities"].apply(lambda x: x[1])
            roc_auc = roc_auc_score(y_true, positive_probs)
    elif "predicted_probability" in df.columns:
        roc_auc = roc_auc_score(y_true, df["predicted_probability"])
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "confusion_matrix": conf_matrix.tolist()
    }
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    
    # Save metrics to file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save plots
        if save_plots:
            # Metrics summary bar chart
            plt.figure(figsize=(10, 6))
            metrics_names = ["Accuracy", "Precision", "Recall", "F1", "MCC"]
            metrics_values = [accuracy, precision, recall, f1, mcc]
            colors = ["blue", "green", "red", "purple", "orange"]
            
            if roc_auc is not None:
                metrics_names.append("ROC AUC")
                metrics_values.append(roc_auc)
                colors.append("brown")
            
            bars = plt.bar(metrics_names, metrics_values, color=colors)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.1)
            plt.ylabel("Score")
            plt.title("Binary Classification Metrics")
            plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
            plt.close()
            
            # Confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
            plt.close()
            
            # ROC curve
            if roc_auc is not None:
                probs_for_curve = None
                if "class_probabilities" in df.columns:
                    if isinstance(df["class_probabilities"].iloc[0], list):
                        probs_for_curve = df["class_probabilities"].apply(lambda x: x[1])
                elif "predicted_probability" in df.columns:
                    probs_for_curve = df["predicted_probability"]
                
                if probs_for_curve is not None:
                    fpr, tpr, _ = roc_curve(y_true, probs_for_curve)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
                    plt.close()
    
    return metrics 