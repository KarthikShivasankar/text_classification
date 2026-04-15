"""Trainer classes for text classification."""

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
from datasets import load_metric
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import cycle
from sklearn.preprocessing import label_binarize
from codecarbon import EmissionsTracker


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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the weighted loss.

        Args:
            model: The model
            inputs: The inputs
            return_outputs: Whether to return the outputs

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


class TextClassificationTrainer:
    """Base trainer for text classification tasks."""

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
        # Initialize trainer
        trainer = WeightedLossTrainer(
            model=self.model,
            args=self.training_args,
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

        # Save the model
        trainer.save_model(self.output_dir)

        # Evaluate the model
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()

            # Save evaluation metrics
            with open(os.path.join(self.output_dir, "eval_metrics.json"), "w") as f:
                json.dump(eval_metrics, f, indent=4)

        # Stop emissions tracking
        if self.track_emissions:
            emissions_data = self.tracker.stop()

            # Save emissions data
            tracker_path = os.path.join(self.output_dir, "carbon_emissions.json")
            emissions_json = json.loads(emissions_data.toJSON())
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
            output_dir: Directory to save outputs
            cat2idx: Mapping from categories to indices
            idx2cat: Mapping from indices to categories
            is_binary: Whether the task is binary classification

        Returns:
            Function to compute metrics
        """

        def compute_metrics(eval_pred):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
            predictions = np.argmax(logits, axis=-1)

            # Compute classification report
            report = classification_report(labels, predictions, output_dict=True)
            with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
                json.dump(report, f, indent=4)

            # Determine number of classes
            num_classes = len(np.unique(labels))

            # Set average type based on classification type
            average_type = "binary" if num_classes == 2 else "macro"

            # Compute basic metrics
            precision = load_metric("precision").compute(
                predictions=predictions, references=labels, average=average_type
            )["precision"]
            recall = load_metric("recall").compute(
                predictions=predictions, references=labels, average=average_type
            )["recall"]
            f1 = load_metric("f1").compute(
                predictions=predictions, references=labels, average=average_type
            )["f1"]
            acc = load_metric("accuracy").compute(
                predictions=predictions, references=labels
            )["accuracy"]
            mcc = load_metric("matthews_correlation").compute(
                predictions=predictions, references=labels
            )["matthews_correlation"]

            # Calculate AUC conditionally
            if num_classes == 2:
                y_score = softmax(logits, axis=1)[:, 1]
                auc_value = roc_auc_score(labels, y_score)
            elif num_classes > 2:
                try:
                    auc_value = roc_auc_score(
                        labels,
                        softmax(logits, axis=1),
                        multi_class="ovr",
                        average="macro",
                    )
                except:
                    auc_value = "N/A"
            else:
                auc_value = "N/A"

            metrics = {
                "precision": precision,
                "recall": recall,
                "acc": acc,
                "mcc": mcc,
                "f1": f1,
                "auc": auc_value,
            }

            # Save metrics
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            # Plot metrics
            if is_binary:
                TextClassificationTrainer.plot_binary_metrics(
                    labels, logits, cat2idx, idx2cat, output_dir
                )
            else:
                TextClassificationTrainer.plot_multi_class_metrics(
                    labels, logits, cat2idx, idx2cat, output_dir
                )

            return metrics

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
            output_dir: Directory to save outputs
        """
        # For ROC Curve
        fpr, tpr, _ = roc_curve(labels, softmax(logits, axis=1)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(12, 12))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve for {idx2cat[1]} (AUC = {roc_auc:.2f})",
        )
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(
            f"Receiver Operating Characteristic Curve for {idx2cat[0]} vs {idx2cat[1]}",
            fontsize=16,
        )
        plt.legend(loc="lower right", fontsize=10)
        plt.savefig(f"{output_dir}/roc_curve.png", bbox_inches="tight")
        plt.close()

        # For Confusion Matrix
        plt.figure(figsize=(8, 8))
        cm = confusion_matrix(labels, logits.argmax(axis=1))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=16)
        plt.colorbar()
        plt.ylabel(f"True label ({idx2cat[0]} or {idx2cat[1]})", fontsize=14)
        plt.xlabel(f"Predicted label ({idx2cat[0]} or {idx2cat[1]})", fontsize=14)
        plt.savefig(f"{output_dir}/confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # For Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(
            labels, softmax(logits, axis=1)[:, 1]
        )

        plt.figure(figsize=(12, 12))
        plt.plot(
            recall,
            precision,
            color="b",
            lw=2,
            label=f"Precision-Recall curve for {idx2cat[1]}",
        )
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.title(
            f"Precision-Recall Curve for {idx2cat[0]} vs {idx2cat[1]}", fontsize=16
        )
        plt.legend(loc="lower right", fontsize=10)
        plt.savefig(f"{output_dir}/precision_recall_curve.png", bbox_inches="tight")
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
            output_dir: Directory to save outputs
        """
        # Define a larger set of colours
        colours = cycle(
            [
                "b",
                "g",
                "r",
                "c",
                "m",
                "y",
                "k",
                "orange",
                "purple",
                "pink",
                "brown",
                "violet",
            ]
        )

        # For ROC Curve
        y_bin_labels = label_binarize(labels, classes=[i for i in range(len(cat2idx))])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        plt.figure(figsize=(12, 12))
        for i, colour in zip(range(len(cat2idx)), colours):
            fpr[i], tpr[i], _ = roc_curve(
                y_bin_labels[:, i], softmax(logits, axis=1)[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i],
                tpr[i],
                color=colour,
                lw=2,
                label=f"Class {idx2cat[i]} (AUC = {roc_auc[i]:.2f})",
            )
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("Receiver Operating Characteristic Curve", fontsize=16)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
        plt.savefig(f"{output_dir}/roc_curve.png", bbox_inches="tight")
        plt.close()

        # For Confusion Matrix
        plt.figure(figsize=(14, 12))
        cm = confusion_matrix(labels, np.argmax(logits, axis=-1))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=16)
        plt.colorbar()
        tick_marks = np.arange(len(idx2cat))
        plt.xticks(
            tick_marks,
            [idx2cat[i] for i in range(len(idx2cat))],
            rotation=45,
            fontsize=12,
        )
        plt.yticks(tick_marks, [idx2cat[i] for i in range(len(idx2cat))], fontsize=12)
        plt.ylabel("True label", fontsize=14)
        plt.xlabel("Predicted label", fontsize=14)
        plt.savefig(f"{output_dir}/confusion_matrix.png", bbox_inches="tight")
        plt.close()

        # For Precision-Recall Curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        plt.figure(figsize=(12, 12))
        for i, colour in zip(range(len(cat2idx)), colours):
            precision[i], recall[i], _ = precision_recall_curve(
                y_bin_labels[:, i], softmax(logits, axis=1)[:, i]
            )
            average_precision[i] = average_precision_score(
                y_bin_labels[:, i], softmax(logits, axis=1)[:, i]
            )
            plt.plot(
                recall[i],
                precision[i],
                color=colour,
                lw=2,
                label=f"Class {idx2cat[i]} (AP = {average_precision[i]:.2f})",
            )
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.title("Precision-Recall Curve", fontsize=16)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
        plt.savefig(f"{output_dir}/precision_recall_curve.png", bbox_inches="tight")
        plt.close()

    @staticmethod
    def visualize_and_save_metrics_and_results(
        fold_results, output_dir, confidence=0.95
    ):
        """
        Visualize and save metrics and results.

        Args:
            fold_results: Results from each fold
            output_dir: Directory to save outputs
            confidence: Confidence level for intervals
        """
        # Calculate metrics
        rounds = len(fold_results)

        indexed_data = {index: value for index, value in enumerate(fold_results)}

        results_path = os.path.join(output_dir, "fold_results.json")

        with open(results_path, "w") as f:
            json.dump(indexed_data, f, indent=4)

        # Extracting the specific metrics from the indexed data
        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
            "acc": [],
            "mcc": [],
        }

        for item in indexed_data.values():
            metrics["precision"].append(item["eval_precision"])
            metrics["recall"].append(item["eval_recall"])
            metrics["f1"].append(item["eval_f1"])
            metrics["auc"].append(item["eval_auc"])
            metrics["acc"].append(item["eval_acc"])
            metrics["mcc"].append(item["eval_mcc"])

        # Calculating mean and standard deviation for each metric
        metrics_mean = {key: np.mean(value) for key, value in metrics.items()}
        metrics_std = {key: np.std(value, ddof=1) for key, value in metrics.items()}

        # Calculating confidence intervals using t-distribution
        metrics_ci_bounds = {}
        for key in metrics_mean:
            mean = metrics_mean[key]
            std = metrics_std[key]
            t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=rounds - 1)
            ci_lower = mean - t_value * (std / np.sqrt(rounds))
            ci_upper = mean + t_value * (std / np.sqrt(rounds))
            metrics_ci_bounds[key] = {"ci_lower": ci_lower, "ci_upper": ci_upper}

        # Save metrics
        with open(os.path.join(output_dir, "metrics_all_fold.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(os.path.join(output_dir, "metrics_mean.json"), "w") as f:
            json.dump(metrics_mean, f, indent=4)

        with open(os.path.join(output_dir, "metrics_std.json"), "w") as f:
            json.dump(metrics_std, f, indent=4)

        with open(os.path.join(output_dir, "metrics_ci_bounds.json"), "w") as f:
            json.dump(metrics_ci_bounds, f, indent=4)

        # Plotting
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


class BinaryClassificationTrainer(TextClassificationTrainer):
    """Trainer for binary classification tasks."""

    def __init__(self, *args, **kwargs):
        """Initialize the trainer."""
        super().__init__(*args, **kwargs)

    def train_with_k_fold(self, data_processor, data, text_col, label_col, n_splits=5):
        """
        Train with k-fold cross-validation.

        Args:
            data_processor: Data processor
            data: Data DataFrame
            text_col: Name of the column containing the text
            label_col: Name of the column containing the labels
            n_splits: Number of folds

        Returns:
            List of evaluation metrics for each fold
        """
        fold_results = []

        for fold, (train_data, val_data) in enumerate(
            data_processor.prepare_k_fold(data, text_col, label_col, n_splits=n_splits),
            1,
        ):
            print(f"Training on Fold {fold}")

            # Create datasets
            train_dataset = data_processor.create_dataset(
                train_data, text_col, label_col
            )
            val_dataset = data_processor.create_dataset(val_data, text_col, label_col)

            # Train the model
            self.train(train_dataset, val_dataset)

            # Evaluate the model
            eval_metrics = self.evaluate(val_dataset).metrics

            # Store metrics
            fold_results.append(eval_metrics)

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

        # Visualize and save metrics
        self.visualize_and_save_metrics_and_results(fold_results, self.output_dir)

        return fold_results


class MultiClassificationTrainer(TextClassificationTrainer):
    """Trainer for multi-class classification tasks."""

    def __init__(self, *args, **kwargs):
        """Initialize the trainer."""
        super().__init__(*args, **kwargs)

    def train_with_k_fold(self, data_processor, data, text_col, label_col, n_splits=5):
        """
        Train with k-fold cross-validation.

        Args:
            data_processor: Data processor
            data: Data DataFrame
            text_col: Name of the column containing the text
            label_col: Name of the column containing the labels
            n_splits: Number of folds

        Returns:
            List of evaluation metrics for each fold
        """
        fold_results = []

        for fold, (train_data, val_data) in enumerate(
            data_processor.prepare_k_fold(data, text_col, label_col, n_splits=n_splits),
            1,
        ):
            print(f"Training on Fold {fold}")

            # Create datasets
            train_dataset = data_processor.create_dataset(
                train_data, text_col, label_col
            )
            val_dataset = data_processor.create_dataset(val_data, text_col, label_col)

            # Train the model
            self.train(train_dataset, val_dataset)

            # Evaluate the model
            eval_metrics = self.evaluate(val_dataset).metrics

            # Store metrics
            fold_results.append(eval_metrics)

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

        # Visualize and save metrics
        self.visualize_and_save_metrics_and_results(fold_results, self.output_dir)

        return fold_results
