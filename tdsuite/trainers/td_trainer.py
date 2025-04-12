"""Specialized trainer for technical debt classification."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.model_selection import KFold
from transformers import TrainingArguments
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score

from .base import BaseTrainer
from ..data.dataset import TDDataset


class TDTrainer(BaseTrainer):
    """Trainer for technical debt classification tasks."""

    def __init__(
        self,
        model,
        tokenizer,
        training_args,
        compute_metrics=None,
        class_weights=None,
        output_dir=None,
        track_emissions=True,
        n_splits=5,
        random_state=42,
        data_config=None,
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
            n_splits: Number of folds for cross-validation
            random_state: Random state for reproducibility
            data_config: Data configuration
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            output_dir=output_dir,
            track_emissions=track_emissions,
        )
        self.n_splits = n_splits
        self.random_state = random_state
        self.data_config = data_config

    def train_with_cross_validation(
        self,
        train_dataset,
        eval_dataset=None,
        cat2idx=None,
        idx2cat=None,
        is_binary=False,
    ):
        """
        Train the model with k-fold cross-validation.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            cat2idx: Mapping from categories to indices
            idx2cat: Mapping from indices to categories
            is_binary: Whether this is a binary classification task

        Returns:
            List of evaluation results for each fold
        """
        import datetime
        from sklearn.model_selection import KFold
        
        # Create compute metrics function if not provided
        if self.compute_metrics is None and cat2idx is not None and idx2cat is not None:
            self.compute_metrics = self.create_compute_metrics(
                self.output_dir, cat2idx, idx2cat, is_binary
            )

        # Initialize emissions tracker for the entire cross-validation process
        emissions_tracker = None
        if self.track_emissions:
            try:
                # Create emissions directory
                emissions_dir = os.path.join(self.output_dir, "emissions")
                os.makedirs(emissions_dir, exist_ok=True)
                
                # Configure tracker
                from codecarbon import EmissionsTracker
                emissions_tracker = EmissionsTracker(
                    output_dir=emissions_dir,
                    project_name="cross_validation",
                    output_file="cross_validation_emissions.csv",
                    allow_multiple_runs=True
                )
                emissions_tracker.start()
                print("🌱 Carbon emissions tracking started for cross-validation")
            except Exception as e:
                print(f"\n⚠️ Warning: Failed to initialize emissions tracking: {str(e)}")
                emissions_tracker = None

        # Initialize k-fold cross-validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Initialize list to store metrics for each fold
        fold_results = []

        # Loop over folds
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_dataset.labels)):
            print(f"\n🔄 Training fold {fold_idx + 1}/{self.n_splits}")
            
            # Create output directory for this fold
            fold_output_dir = os.path.join(self.output_dir, f"fold_{fold_idx + 1}")
            os.makedirs(fold_output_dir, exist_ok=True)

            # Create fold training arguments
            fold_training_args = TrainingArguments(
                output_dir=fold_output_dir,
                num_train_epochs=self.training_args.num_train_epochs,
                per_device_train_batch_size=self.training_args.per_device_train_batch_size,
                per_device_eval_batch_size=self.training_args.per_device_eval_batch_size,
                learning_rate=self.training_args.learning_rate,
                weight_decay=self.training_args.weight_decay,
                warmup_steps=self.training_args.warmup_steps,
                gradient_accumulation_steps=self.training_args.gradient_accumulation_steps,
                evaluation_strategy="no" if eval_dataset is None else self.training_args.evaluation_strategy,
                save_strategy=self.training_args.save_strategy,
                load_best_model_at_end=self.training_args.load_best_model_at_end,
                metric_for_best_model=self.training_args.metric_for_best_model,
                greater_is_better=self.training_args.greater_is_better,
                logging_steps=self.training_args.logging_steps,
                save_total_limit=self.training_args.save_total_limit,
                remove_unused_columns=self.training_args.remove_unused_columns,
                report_to=self.training_args.report_to,
                disable_tqdm=self.training_args.disable_tqdm,
                no_cuda=self.training_args.no_cuda,
            )

            # Create fold trainer
            fold_trainer = TDTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                training_args=fold_training_args,
                compute_metrics=self.compute_metrics,
                class_weights=self.class_weights,
                output_dir=fold_output_dir,
                track_emissions=False,  # Disable emissions tracking for individual folds
                n_splits=self.n_splits,
                random_state=self.random_state,
                data_config=self.data_config,
            )

            # Create train and validation datasets for this fold
            train_encodings = {}
            for key, val in train_dataset.encodings.items():
                if isinstance(val, torch.Tensor):
                    train_encodings[key] = val[train_idx]
                else:
                    train_encodings[key] = torch.tensor([val[i] for i in train_idx])
            train_labels = torch.tensor([train_dataset.labels[i] for i in train_idx])
            train_fold_dataset = TDDataset(train_encodings, train_labels)

            val_encodings = {}
            for key, val in train_dataset.encodings.items():
                if isinstance(val, torch.Tensor):
                    val_encodings[key] = val[val_idx]
                else:
                    val_encodings[key] = torch.tensor([val[i] for i in val_idx])
            val_labels = torch.tensor([train_dataset.labels[i] for i in val_idx])
            val_fold_dataset = TDDataset(val_encodings, val_labels)

            # Train the model
            train_result = fold_trainer.train(
                train_fold_dataset,
                eval_dataset=val_fold_dataset,
            )

            # Save training metrics for this fold
            train_metrics = {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
                "epoch": train_result.metrics.get("epoch", 0),
            }

            # Evaluate the model
            eval_result = fold_trainer.evaluate(val_fold_dataset)

            # Save fold results
            fold_metrics = {
                "training": train_metrics,
                "evaluation": eval_result.metrics
            }
            
            # Save metrics for this fold
            with open(os.path.join(fold_output_dir, "metrics.json"), "w") as f:
                json.dump(fold_metrics, f, indent=4)

            # Add evaluation metrics to fold results
            fold_results.append(eval_result.metrics)

            # Plot metrics
            if cat2idx is not None and idx2cat is not None:
                if is_binary:
                    self.plot_metrics(
                        eval_result.label_ids,
                        eval_result.predictions,
                        fold_output_dir,
                    )

        # Visualize and save k-fold results
        self.visualize_and_save_metrics_and_results(fold_results, self.output_dir)

        # Stop emissions tracking
        if emissions_tracker is not None:
            try:
                emissions = emissions_tracker.stop()
                
                # Save emissions data
                emissions_dir = os.path.join(self.output_dir, "emissions")
                tracker_path = os.path.join(emissions_dir, "cross_validation_emissions.json")
                
                if emissions is None:
                    # Handle case where emissions data is None
                    emissions_json = {
                        "emissions": 0.0,
                        "unit": "kgCO2e",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "error": "No emissions data was recorded"
                    }
                    print("\n⚠️ Warning: No emissions data was recorded")
                else:
                    try:
                        emissions_value = float(emissions)
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
                
                print(f"\n🌱 Total carbon emissions for cross-validation: {emissions_value:.6f} kgCO2e")
                print(f"🌱 Emissions data saved to {tracker_path}")
            
            except Exception as e:
                # Catch any unexpected errors during emissions tracking
                print(f"\n⚠️ Warning: Error in emissions tracking: {str(e)}")
        
        return fold_results

    def train_with_early_stopping(
        self,
        train_dataset,
        eval_dataset=None,
        cat2idx=None,
        idx2cat=None,
        is_binary=False,
        patience=3,
        min_delta=0.01,
    ):
        """
        Train the model with early stopping.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            cat2idx: Mapping from categories to indices
            idx2cat: Mapping from indices to categories
            is_binary: Whether this is a binary classification task
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in monitored quantity to qualify as an improvement

        Returns:
            Training results
        """
        # Create compute metrics function if not provided
        if self.compute_metrics is None and cat2idx is not None and idx2cat is not None:
            self.compute_metrics = self.create_compute_metrics(
                self.output_dir, cat2idx, idx2cat, is_binary
            )

        # Update training arguments for early stopping
        self.training_args.load_best_model_at_end = True
        self.training_args.metric_for_best_model = "eval_loss"
        self.training_args.greater_is_better = False
        self.training_args.eval_strategy = "epoch"
        self.training_args.save_strategy = "epoch"
        self.training_args.save_total_limit = patience + 1

        # Train the model
        train_result = self.train(train_dataset, eval_dataset)

        # Plot metrics
        if eval_dataset is not None and cat2idx is not None and idx2cat is not None:
            eval_result = self.evaluate(eval_dataset)
            if is_binary:
                self.plot_metrics(
                    eval_result.label_ids,
                    eval_result.predictions,
                    self.output_dir,
                )

        return train_result

    def predict_with_confidence(
        self,
        test_dataset,
        test_data,
        idx2cat,
        confidence_threshold=0.5,
        top_n=1,
    ):
        """
        Make predictions with confidence thresholds.

        Args:
            test_dataset: Test dataset
            test_data: Test data DataFrame
            idx2cat: Mapping from indices to categories
            confidence_threshold: Confidence threshold for predictions
            top_n: Number of top predictions to return

        Returns:
            DataFrame with predictions and probabilities
        """
        # Get predictions with probabilities
        result_df = self.predict_with_probabilities(
            test_dataset, test_data, idx2cat, top_n
        )

        # Add confidence column
        result_df["confidence"] = result_df[f"Top_{1}_Prob"]
        result_df["is_confident"] = result_df["confidence"] >= confidence_threshold

        return result_df

    def predict_with_ensemble(
        self,
        test_dataset,
        test_data,
        idx2cat,
        model_paths,
        top_n=1,
    ):
        """
        Make predictions using an ensemble of models.

        Args:
            test_dataset: Test dataset
            test_data: Test data DataFrame
            idx2cat: Mapping from indices to categories
            model_paths: List of paths to saved models
            top_n: Number of top predictions to return

        Returns:
            DataFrame with ensemble predictions and probabilities
        """
        # Initialize list to store predictions from each model
        all_predictions = []

        # Make predictions with each model
        for model_path in model_paths:
            # Load model
            self.model = self.model.from_pretrained(model_path)

            # Get predictions
            predictions = self.evaluate(test_dataset)
            all_predictions.append(predictions.predictions)

        # Average predictions
        ensemble_predictions = np.mean(all_predictions, axis=0)

        # Convert logits to probabilities
        probs = self.softmax(ensemble_predictions, axis=1)
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

    def plot_metrics(self, labels, logits, output_dir):
        """
        Plot metrics for binary classification.
        
        Args:
            labels: Ground truth labels
            logits: Model logits
            output_dir: Directory to save plots
        """
        # Convert logits to probabilities
        probabilities = torch.softmax(torch.tensor(logits), dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # Compute metrics
        accuracy = (predictions == labels).float().mean()
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = confusion_matrix(labels, predictions)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
        
        # Save metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        }
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2) 