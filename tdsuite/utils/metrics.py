"""Metrics for evaluating model performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os


def compute_metrics(predictions_df, output_dir=None, save_plots=True):
    """
    Compute and display metrics for model predictions.
    
    Args:
        predictions_df: DataFrame containing predictions and ground truth
        output_dir: Directory to save plots (optional)
        save_plots: Whether to save plots to disk (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Extract ground truth and predictions
    y_true = predictions_df["label"]
    y_pred = predictions_df["predicted_class"]
    
    # Ensure both labels are of the same type
    if y_true.dtype != y_pred.dtype:
        # If one is numeric and the other is string, convert both to strings
        if y_true.dtype == 'object' and y_pred.dtype != 'object':
            # Convert numeric predictions to strings using the same mapping as in inference
            unique_labels = y_true.unique()
            if len(unique_labels) == 2:  # Binary classification
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
                
                # Create mapping: 0 -> non_, 1 -> positive
                label_map = {0: non_label, 1: positive_label}
                y_pred = y_pred.map(label_map)
            else:  # Multi-class classification
                # For multi-class, we'll use the sorted labels
                sorted_labels = sorted(unique_labels)
                label_map = {i: label for i, label in enumerate(sorted_labels)}
                y_pred = y_pred.map(label_map)
        elif y_true.dtype != 'object' and y_pred.dtype == 'object':
            # Convert string predictions to numeric
            unique_labels = y_pred.unique()
            if len(unique_labels) == 2:  # Binary classification
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
            else:  # Multi-class classification
                # For multi-class, we'll use the sorted labels
                sorted_labels = sorted(unique_labels)
                label_map = {label: i for i, label in enumerate(sorted_labels)}
                y_pred = y_pred.map(label_map)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Try to compute ROC AUC if probabilities are available
    roc_auc = None
    if "probability" in predictions_df.columns:
        try:
            roc_auc = roc_auc_score(y_true, predictions_df["probability"])
        except:
            pass
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Create metrics dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    
    # Save metrics to file if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            import json
            json.dump(metrics, f, indent=4)
    
    # Create and save plots if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true)))
        plt.xticks(tick_marks, np.unique(y_true), rotation=45)
        plt.yticks(tick_marks, np.unique(y_true))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Plot ROC curve if probabilities are available
        if "probability" in predictions_df.columns and roc_auc is not None:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, predictions_df["probability"])
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, "roc_curve.png"))
            plt.close()
    
    return metrics 