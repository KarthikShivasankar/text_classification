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
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
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
        
        # Set evaluation_strategy to "no" if no eval_dataset is provided
        if eval_dataset is None and filtered_args.get("evaluation_strategy") != "no":
            filtered_args["evaluation_strategy"] = "no"
        
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
            # Make sure output directory exists
            emissions_dir = os.path.join(self.output_dir, "emissions")
            os.makedirs(emissions_dir, exist_ok=True)
            
            # Configure tracker
            self.tracker = EmissionsTracker(
                output_dir=emissions_dir,
                project_name="training",
                output_file="training_emissions.csv",
                allow_multiple_runs=True
            )
            
            # Start tracking
            self.tracker.start()
            print("🌱 Carbon emissions tracking started for training")

        # Train the model
        print(f"🚀 Starting training for {training_args.num_train_epochs} epochs")
        train_result = trainer.train()
        print("✅ Training completed!")

        # Save training metrics
        train_metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            "epoch": train_result.metrics.get("epoch", 0),
        }
        
        # Save the model
        print(f"💾 Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)

        # Evaluate the model
        if eval_dataset is not None:
            print("📊 Evaluating model on validation set")
            eval_metrics = trainer.evaluate()
            
            # Combine training and evaluation metrics
            all_metrics = {
                "training": train_metrics,
                "evaluation": eval_metrics
            }

            # Save all metrics
            metrics_file = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(all_metrics, f, indent=4)
            print(f"📊 Metrics saved to {metrics_file}")

        # Stop emissions tracking
        if self.track_emissions:
            try:
                emissions_data = self.tracker.stop()
                
                # Save emissions data
                emissions_dir = os.path.join(self.output_dir, "emissions")
                os.makedirs(emissions_dir, exist_ok=True)
                tracker_path = os.path.join(emissions_dir, "training_emissions.json")
                
                if emissions_data is None:
                    # Handle case where emissions data is None
                    emissions_json = {
                        "emissions": 0.0,
                        "unit": "kgCO2e",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "error": "No emissions data was recorded"
                    }
                    print("\n⚠️ Warning: No emissions data was recorded")
                elif hasattr(emissions_data, 'toJSON'):
                    try:
                        emissions_json = json.loads(emissions_data.toJSON())
                    except (TypeError, json.JSONDecodeError):
                        # Handle invalid JSON
                        emissions_json = {
                            "emissions": 0.0,
                            "unit": "kgCO2e",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "error": "Invalid emissions data format"
                        }
                        print("\n⚠️ Warning: Invalid emissions data format")
                else:
                    # If emissions_data is a float or other type, create a simple JSON object
                    try:
                        emissions_value = float(emissions_data)
                        emissions_json = {
                            "emissions": emissions_value,
                            "unit": "kgCO2e",
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    except (TypeError, ValueError):
                        # Handle case where emissions_data can't be converted to float
                        emissions_json = {
                            "emissions": 0.0,
                            "unit": "kgCO2e",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "error": "Emissions data couldn't be converted to float"
                        }
                        print("\n⚠️ Warning: Emissions data couldn't be converted to float")
                
                # Save the JSON file
                with open(tracker_path, "w") as f:
                    json.dump(emissions_json, f, indent=4)
                
                # Safely extract emissions value for display
                emissions_value = emissions_json.get("emissions", 0.0)
                if not isinstance(emissions_value, (int, float)):
                    emissions_value = 0.0
                
                print(f"\n🌱 Carbon emissions: {emissions_value:.6f} kgCO2e")
                print(f"🌱 Emissions data saved to {tracker_path}")
            
            except Exception as e:
                # Catch any unexpected errors during emissions tracking
                print(f"\n⚠️ Warning: Error in emissions tracking: {str(e)}")
                # Ensure this doesn't break the training process

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
        # Get a copy of training arguments with evaluation_strategy set to "no"
        training_args_dict = self.training_args.to_dict()
        # Set evaluation_strategy to "no" since we're only making predictions
        training_args_dict["evaluation_strategy"] = "no"
        # When evaluation_strategy is "no", we must set load_best_model_at_end to False
        training_args_dict["load_best_model_at_end"] = False
        # Also set save_strategy to "no" to match evaluation_strategy
        training_args_dict["save_strategy"] = "no"
        # Filter out any unexpected arguments
        valid_args = TrainingArguments.__init__.__code__.co_varnames
        filtered_args = {k: v for k, v in training_args_dict.items() if k in valid_args}
        # Create new training arguments
        eval_training_args = TrainingArguments(**filtered_args)

        # Initialize trainer with adjusted training arguments
        trainer = WeightedLossTrainer(
            model=self.model,
            args=eval_training_args,
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
                eval_pred: Tuple of (predictions, labels)
            
            Returns:
                Dictionary of metrics
            """
            from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
            
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            
            # Compute metrics
            results = {}
            results["accuracy"] = (predictions == labels).mean()
            
            # Binary classification metrics
            results["precision"] = precision_score(labels, predictions, average='binary')
            results["recall"] = recall_score(labels, predictions, average='binary')
            results["f1"] = f1_score(labels, predictions, average='binary')
            
            # Add Matthews correlation coefficient
            results["mcc"] = matthews_corrcoef(labels, predictions)
            
            # Add ROC AUC when appropriate
            if is_binary:
                # Get probabilities for the positive class
                probs = softmax(logits, axis=1)[:, 1]
                try:
                    results["roc_auc"] = roc_auc_score(labels, probs)
                except ValueError:
                    # In case there's only one class in the labels
                    results["roc_auc"] = 0.0
            
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
        # Calculate overall metrics
        from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        
        predictions = np.argmax(logits, axis=1)
        
        # Convert logits to probabilities
        probs = softmax(logits, axis=1)[:, 1]
        
        # Calculate key metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        f1 = f1_score(labels, predictions, average='binary')
        mcc = matthews_corrcoef(labels, predictions)
        try:
            roc_auc = roc_auc_score(labels, probs)
        except ValueError:
            roc_auc = 0.0

        # Save metrics to file
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mcc": float(mcc),
            "roc_auc": float(roc_auc)
        }
        with open(os.path.join(output_dir, "binary_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Create a metrics summary plot
        plt.figure(figsize=(10, 6))
        metrics_data = [accuracy, precision, recall, f1, mcc, roc_auc]
        metrics_names = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC AUC"]
        colors = ["blue", "green", "red", "purple", "orange", "brown"]
        
        bars = plt.bar(metrics_names, metrics_data, color=colors)
        
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

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
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
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

        # Plot confusion matrix
        cm = confusion_matrix(labels, predictions)
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