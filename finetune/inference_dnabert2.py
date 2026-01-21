#!/usr/bin/env python3
"""
Inference Script for DNABERT-2

This script performs inference on a CSV file using a trained DNABERT-2 classifier.
It outputs predictions with probability scores for threshold analysis.

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for comparison)

Output CSV format:
    - sequence: Original sequence
    - label: Original label (if present)
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label (argmax or thresholded)

Usage:
    python inference_dnabert2.py \
        --input_csv /path/to/test.csv \
        --model_path zhihan1996/DNABERT-2-117M \
        --classifier_path /path/to/classifier.pt \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
import pickle
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on CSV file with DNABERT-2 model and classifier"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="zhihan1996/DNABERT-2-117M",
        help="Path to DNABERT-2 model (HuggingFace model name or local path)",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        required=True,
        help="Path to trained classifier (.pt for neural, .pkl for logistic)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with _predictions suffix)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length (BPE tokens, roughly 0.25 * sequence bp length)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="Pooling strategy for embeddings",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for prob_1 (default: 0.5)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    return parser.parse_args()


class ThreeLayerNN(nn.Module):
    """Simple 3-layer neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.network(x)


def extract_embeddings(
    model,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    pooling: str,
    device: torch.device,
) -> np.ndarray:
    """Extract embeddings from DNABERT-2 for given sequences."""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]

            # Apply pooling
            if pooling == "mean":
                # Mean pooling over sequence length (excluding padding)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            elif pooling == "max":
                # Max pooling over sequence length
                embeddings = hidden_states.max(dim=1)[0]
            elif pooling == "cls":
                # CLS token embedding (first token)
                embeddings = hidden_states[:, 0, :]

            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def run_inference(
    embeddings: np.ndarray,
    classifier,
    classifier_type: str,
    scaler: StandardScaler,
    device: torch.device,
) -> tuple:
    """
    Run inference using classifier on embeddings.

    Args:
        embeddings: Extracted embeddings
        classifier: Trained classifier (neural network or logistic regression)
        classifier_type: 'neural' or 'logistic'
        scaler: StandardScaler for normalizing embeddings
        device: Device to run on

    Returns:
        Tuple of (probabilities array shape (n, 2), predictions array)
    """
    # Scale embeddings
    scaled_embeddings = scaler.transform(embeddings)

    if classifier_type == 'neural':
        classifier.eval()
        with torch.no_grad():
            X = torch.FloatTensor(scaled_embeddings).to(device)
            outputs = classifier(X)
            probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
    else:  # logistic
        preds = classifier.predict(scaled_embeddings)
        probs = classifier.predict_proba(scaled_embeddings)

    return probs, preds


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    # AUC
    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Confusion matrix values
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("DNABERT-2 Inference")
    print("=" * 60)

    start_time = time.time()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Load DNABERT-2 model
    print(f"\nLoading DNABERT-2 model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    # Load classifier
    print(f"Loading classifier from: {args.classifier_path}")
    if args.classifier_path.endswith('.pkl'):
        # Logistic regression
        with open(args.classifier_path, 'rb') as f:
            classifier_data = pickle.load(f)
        classifier = classifier_data['classifier']
        scaler = classifier_data['scaler']
        classifier_type = 'logistic'
    else:
        # Neural network
        checkpoint = torch.load(args.classifier_path, map_location=device)
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint.get('hidden_dim', 256)
        classifier = ThreeLayerNN(input_dim, hidden_dim).to(device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        # Load scaler
        scaler_path = args.classifier_path.replace('.pt', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print("  Warning: Scaler not found, using StandardScaler with default params")
            scaler = StandardScaler()
        classifier_type = 'neural'

    # Extract embeddings
    sequences = df["sequence"].tolist()
    print(f"\nExtracting embeddings (pooling={args.pooling})...")
    embeddings = extract_embeddings(
        model, tokenizer, sequences,
        args.batch_size, args.max_length, args.pooling, device,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # Fit scaler if needed (for neural network without saved scaler)
    if classifier_type == 'neural' and not hasattr(scaler, 'mean_'):
        print("  Fitting scaler on input data (not recommended for production)")
        scaler.fit(embeddings)

    # Run inference
    print("\nRunning inference...")
    probs, preds = run_inference(
        embeddings, classifier, classifier_type, scaler, device,
    )

    # Apply custom threshold if specified
    if args.threshold != 0.5:
        print(f"Applying custom threshold: {args.threshold}")
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds_thresholded

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and save metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds_thresholded, probs)

        # Add metadata
        metrics["model_path"] = args.model_path
        metrics["classifier_path"] = args.classifier_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["num_samples"] = len(df)
        metrics["pooling"] = args.pooling

        # Save metrics
        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        # Print metrics
        print("\n" + "=" * 60)
        print("METRICS (threshold = {:.2f})".format(args.threshold))
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print("=" * 60)

    elif has_labels:
        # Just print basic accuracy even if not saving
        labels = df["label"].values
        acc = accuracy_score(labels, preds_thresholded)
        print(f"\nAccuracy: {acc:.4f}")

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
