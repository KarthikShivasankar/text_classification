"""Specialized trainer for technical debt classification."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.model_selection import KFold
from transformers import TrainingArguments

from .base import BaseTrainer


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
            List of results from each fold
        """
        # Initialize k-fold cross-validation
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Create compute metrics function if not provided
        if self.compute_metrics is None and cat2idx is not None and idx2cat is not None:
            self.compute_metrics = self.create_compute_metrics(
                self.output_dir, cat2idx, idx2cat, is_binary
            )

        # Initialize results list
        fold_results = []

        # Perform k-fold cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            # Create fold output directory
            fold_output_dir = os.path.join(self.output_dir, f"fold_{fold_idx}")
            os.makedirs(fold_output_dir, exist_ok=True)

            # Create fold training arguments
            fold_training_args = TrainingArguments(
                **{
                    **self.training_args.__dict__,
                    "output_dir": fold_output_dir,
                }
            )

            # Create fold trainer
            fold_trainer = TDTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                training_args=fold_training_args,
                compute_metrics=self.compute_metrics,
                class_weights=self.class_weights,
                output_dir=fold_output_dir,
                track_emissions=self.track_emissions,
            )

            # Train the model
            train_result = fold_trainer.train(
                train_dataset.select(train_idx),
                eval_dataset=train_dataset.select(val_idx),
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
            eval_result = fold_trainer.evaluate(train_dataset.select(val_idx))

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
                    self.plot_binary_metrics(
                        eval_result.label_ids,
                        eval_result.predictions,
                        cat2idx,
                        idx2cat,
                        fold_output_dir,
                    )
                else:
                    self.plot_multi_class_metrics(
                        eval_result.label_ids,
                        eval_result.predictions,
                        cat2idx,
                        idx2cat,
                        fold_output_dir,
                    )

        # Visualize and save k-fold results
        self.visualize_and_save_metrics_and_results(fold_results, self.output_dir)

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
                self.plot_binary_metrics(
                    eval_result.label_ids,
                    eval_result.predictions,
                    cat2idx,
                    idx2cat,
                    self.output_dir,
                )
            else:
                self.plot_multi_class_metrics(
                    eval_result.label_ids,
                    eval_result.predictions,
                    cat2idx,
                    idx2cat,
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