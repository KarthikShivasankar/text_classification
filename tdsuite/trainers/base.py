"""Base trainer classes for technical debt classification."""

import os
import json
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments
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
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import cycle
from sklearn.preprocessing import label_binarize
from codecarbon import EmissionsTracker
import datetime


class WeightedLossTrainer(Trainer):
    """Trainer with weighted loss for imbalanced datasets."""

    def __init__(self, class_weights=None, *args, **kwargs):
        """
        Initialize the trainer.

        Args:
            class_weights: Weights for each class
            *args: Arguments for the base Trainer
            **kwargs: Keyword arguments for the base Trainer
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

        if self.class_weights is not None:
            device = self.model.device
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(
                device
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the weighted loss.

        Args:
            model: The model
            inputs: The inputs
            return_outputs: Whether to return the outputs
            num_items_in_batch: Number of items in the batch (ignored)

        Returns:
            Loss value or tuple of (loss, outputs)
        """
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if self.class_weights is not None:
            loss_func = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss


class BaseTrainer:
    """Base trainer for technical debt classification tasks."""

    def __init__(
        self,
        model,
        tokenizer,
        training_args,
        compute_metrics=None,
        class_weights=None,
        output_dir=None,
        track_emissions=True,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            tokenizer: The tokenizer
            training_args: Training arguments
            compute_metrics: Function to compute metrics
            class_weights: Weights for each class
            output_dir: Directory to save outputs
            track_emissions: Whether to track carbon emissions
        """
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.compute_metrics = compute_metrics
        self.class_weights = class_weights
        self.output_dir = output_dir or training_args.output_dir
        self.track_emissions = track_emissions

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize emissions tracker
        if self.track_emissions:
            self.tracker = EmissionsTracker()

    def train(self, train_dataset, eval_dataset=None):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset

        Returns:
            Training results
        """
        # Convert TrainingConfig to TrainingArguments
        from transformers import TrainingArguments
        
        # Get the list of valid arguments for TrainingArguments
        valid_args = TrainingArguments.__init__.__code__.co_varnames
        
        # Filter out any unexpected arguments
        training_args_dict = self.training_args.to_dict()
        
        # Handle wandb_project and related parameters
        wandb_project = training_args_dict.pop("wandb_project", None)
        wandb_run_name = training_args_dict.pop("wandb_run_name", None)
        
        # Filter out any other unexpected arguments
        filtered_args = {k: v for k, v in training_args_dict.items() if k in valid_args}
        
        # Add batch_eval_metrics attribute
        filtered_args["batch_eval_metrics"] = False
        
        # Create TrainingArguments object with only valid arguments
        training_args = TrainingArguments(**filtered_args)
        
        # Initialize trainer
        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            class_weights=self.class_weights,
        )

        # Start emissions tracking
        if self.track_emissions:
            self.tracker.start()

        # Train the model
        train_result = trainer.train()

        # Save training metrics
        train_metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "epoch": train_result.metrics.get("epoch", 0),
        }
        
        # Save the model
        trainer.save_model(self.output_dir)

        # Evaluate the model
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            
            # Combine training and evaluation metrics
            all_metrics = {
                "training": train_metrics,
                "evaluation": eval_metrics
            }

            # Save all metrics
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=4)

        # Stop emissions tracking
        if self.track_emissions:
            emissions_data = self.tracker.stop()

            # Save emissions data
            tracker_path = os.path.join(self.output_dir, "carbon_emissions.json")
            if hasattr(emissions_data, 'toJSON'):
                emissions_json = json.loads(emissions_data.toJSON())
            else:
                # If emissions_data is a float or other type, create a simple JSON object
                emissions_json = {
                    "emissions": float(emissions_data),
                    "unit": "kgCO2e",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            with open(tracker_path, "w") as f:
                json.dump(emissions_json, f, indent=4)

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

        return train_result

    def evaluate(self, test_dataset):
        """
        Evaluate the model.

        Args:
            test_dataset: Test dataset

        Returns:
            Evaluation results
        """
        # Initialize trainer
        trainer = WeightedLossTrainer(
            model=self.model,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            class_weights=self.class_weights,
        )

        # Evaluate the model
        predictions = trainer.predict(test_dataset)

        return predictions

    def predict_with_probabilities(self, test_dataset, test_data, idx2cat, top_n=1):
        """
        Make predictions with probabilities.

        Args:
            test_dataset: Test dataset
            test_data: Test data DataFrame
            idx2cat: Mapping from indices to categories
            top_n: Number of top predictions to return

        Returns:
            DataFrame with predictions and probabilities
        """
        # Get predictions
        predictions = self.evaluate(test_dataset)

        # Convert logits to probabilities
        probs = softmax(predictions.predictions, axis=1)
        probs = np.round(probs, 3)

        # Convert probabilities to DataFrame
        probs_df = pd.DataFrame(
            probs, columns=[idx2cat[i] for i in range(probs.shape[1])]
        )

        # Find top probabilities and indices
        top_probs = np.partition(-probs, top_n)[:, :top_n] * -1
        top_indices = np.argpartition(-probs, top_n)[:, :top_n]

        # Map indices to class names
        top_class_names = np.vectorize(idx2cat.get)(top_indices)

        # Convert to DataFrames
        top_probs_df = pd.DataFrame(
            top_probs, columns=[f"Top_{i+1}_Prob" for i in range(top_n)]
        )
        top_class_names_df = pd.DataFrame(
            top_class_names, columns=[f"Top_{i+1}_Class" for i in range(top_n)]
        )

        # Concatenate with original data
        result_df = pd.concat(
            [
                test_data.reset_index(drop=True),
                probs_df.reset_index(drop=True),
                top_probs_df.reset_index(drop=True),
                top_class_names_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return result_df

    @staticmethod
    def create_compute_metrics(output_dir, cat2idx, idx2cat, is_binary=False):
        """
        Create a function to compute metrics.

        Args:
            output_dir: Directory to save metrics
            cat2idx: Mapping from categories to indices
            idx2cat: Mapping from indices to categories
            is_binary: Whether this is a binary classification task

        Returns:
            Function to compute metrics
        """
        def compute_metrics(eval_pred):
            """
            Compute metrics for evaluation.

            Args:
                eval_pred: Evaluation predictions

            Returns:
                Dictionary of metrics
            """
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            # Compute metrics
            results = {}

            if is_binary:
                # Binary classification metrics
                results["accuracy"] = (predictions == labels).mean()
                results["roc_auc"] = roc_auc_score(labels, predictions)
                results["average_precision"] = average_precision_score(labels, predictions)

                # Classification report
                report = classification_report(
                    labels,
                    predictions,
                    target_names=[idx2cat[0], idx2cat[1]],
                    output_dict=True,
                )

                # Add metrics from classification report
                for metric_name, metric_value in report.items():
                    if isinstance(metric_value, dict):
                        for sub_metric, value in metric_value.items():
                            results[f"{metric_name}_{sub_metric}"] = value
                    else:
                        results[metric_name] = metric_value

            else:
                # Multi-class classification metrics
                results["accuracy"] = (predictions == labels).mean()

                # Classification report
                report = classification_report(
                    labels,
                    predictions,
                    target_names=[idx2cat[i] for i in range(len(cat2idx))],
                    output_dict=True,
                )

                # Add metrics from classification report
                for metric_name, metric_value in report.items():
                    if isinstance(metric_value, dict):
                        for sub_metric, value in metric_value.items():
                            results[f"{metric_name}_{sub_metric}"] = value
                    else:
                        results[metric_name] = metric_value

            return results

        return compute_metrics

    @staticmethod
    def plot_binary_metrics(labels, logits, cat2idx, idx2cat, output_dir):
        """
        Plot metrics for binary classification.

        Args:
            labels: Ground truth labels
            logits: Model output logits
            cat2idx: Mapping from categories to indices
            idx2cat: Mapping from indices to categories
            output_dir: Directory to save plots
        """
        # Convert logits to probabilities
        probs = softmax(logits, axis=1)[:, 1]

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()

        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

        # Plot confusion matrix
        cm = confusion_matrix(labels, np.argmax(logits, axis=1))
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(cat2idx))
        plt.xticks(tick_marks, [idx2cat[i] for i in range(len(cat2idx))], rotation=45)
        plt.yticks(tick_marks, [idx2cat[i] for i in range(len(cat2idx))])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

    @staticmethod
    def plot_multi_class_metrics(labels, logits, cat2idx, idx2cat, output_dir):
        """
        Plot metrics for multi-class classification.

        Args:
            labels: Ground truth labels
            logits: Model output logits
            cat2idx: Mapping from categories to indices
            idx2cat: Mapping from indices to categories
            output_dir: Directory to save plots
        """
        # Convert logits to probabilities
        probs = softmax(logits, axis=1)

        # Binarize labels
        labels_bin = label_binarize(labels, classes=range(len(cat2idx)))

        # Plot ROC curve for each class
        plt.figure(figsize=(10, 8))
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red"])
        for i, color in zip(range(len(cat2idx)), colors):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=color,
                lw=2,
                label=f"ROC curve for {idx2cat[i]} (AUC = {roc_auc:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curves")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curves.png"))
        plt.close()

        # Plot confusion matrix
        cm = confusion_matrix(labels, np.argmax(logits, axis=1))
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        tick_marks = np.arange(len(cat2idx))
        plt.xticks(tick_marks, [idx2cat[i] for i in range(len(cat2idx))], rotation=45)
        plt.yticks(tick_marks, [idx2cat[i] for i in range(len(cat2idx))])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

    @staticmethod
    def visualize_and_save_metrics_and_results(
        fold_results, output_dir, confidence=0.95
    ):
        """
        Visualize and save metrics and results from k-fold cross-validation.

        Args:
            fold_results: List of results from each fold
            output_dir: Directory to save results
            confidence: Confidence level for confidence intervals
        """
        # Extract metrics from fold results
        metrics = {}
        for fold_idx, fold_result in enumerate(fold_results):
            for metric_name, metric_value in fold_result.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(metric_value)

        # Compute mean and confidence intervals
        results = {}
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            ci = scipy.stats.t.interval(
                confidence, len(values) - 1, loc=mean_value, scale=std_value / np.sqrt(len(values))
            )

            results[metric_name] = {
                "mean": mean_value,
                "std": std_value,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }

        # Save results
        with open(os.path.join(output_dir, "k_fold_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # Plot metrics
        plt.figure(figsize=(12, 6))
        metric_names = list(results.keys())
        mean_values = [results[m]["mean"] for m in metric_names]
        ci_lower = [results[m]["ci_lower"] for m in metric_names]
        ci_upper = [results[m]["ci_upper"] for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        plt.bar(x, mean_values, width, label="Mean")
        plt.errorbar(
            x,
            mean_values,
            yerr=[np.array(mean_values) - np.array(ci_lower), np.array(ci_upper) - np.array(mean_values)],
            fmt="none",
            color="black",
            capsize=5,
            label=f"{confidence*100}% Confidence Interval",
        )

        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.title("K-Fold Cross-Validation Results")
        plt.xticks(x, metric_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "k_fold_results.png"))
        plt.close() 