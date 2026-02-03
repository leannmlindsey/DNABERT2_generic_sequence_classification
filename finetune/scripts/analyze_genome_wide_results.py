import json
import glob
import os
import math
import pandas as pd
import numpy as np
from pathlib import Path

def collapse_overlapping_windows(predictions_df, window_size=2000, step_size=1000):
    """
    Collapse overlapping window predictions using majority voting
    
    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns: start, end, prediction, score (optional), label
    window_size : int
        Size of each window (default: 2000)
    step_size : int
        Step size between windows (default: 1000)
    
    Returns:
    --------
    collapsed_df : DataFrame
        Non-overlapping segments with majority vote predictions
    """
    
    # Determine the genome range
    min_pos = predictions_df['start'].min()
    max_pos = predictions_df['end'].max()
    
    # Create non-overlapping segments of step_size
    segments = []
    
    for pos in range(min_pos, max_pos, step_size):
        seg_start = pos
        seg_end = min(pos + step_size, max_pos)
        
        # Find all windows that cover this segment
        overlapping = predictions_df[
            (predictions_df['start'] <= seg_start) & 
            (predictions_df['end'] >= seg_end)
        ]
        
        if len(overlapping) == 0:
            continue
        
        # Majority vote for prediction
        votes = overlapping['prediction'].sum()
        total_votes = len(overlapping)
        majority_pred = 1 if votes > total_votes / 2 else 0
        
        # Average the scores if available
        avg_score = overlapping['score'].mean() if 'score' in overlapping.columns else 0.5
        
        # Get the true label (should be same for all overlapping windows)
        true_label = overlapping['label'].iloc[0] if 'label' in overlapping.columns else 0
        
        segments.append({
            'start': seg_start,
            'end': seg_end,
            'prediction': majority_pred,
            'score': avg_score,
            'label': true_label,
            'num_votes': total_votes,
            'votes_for_phage': votes
        })
    
    return pd.DataFrame(segments)

def apply_phage_clustering_filter(predictions_df, merge_gap=3000, min_cluster_size=1000, window_size=5):
    """
    Apply clustering filter to reduce false positives and merge nearby phage predictions
    
    Parameters:
    -----------
    predictions_df : DataFrame
        DataFrame with columns: start, end, prediction (0 or 1), score (optional)
    merge_gap : int
        Maximum gap (nt) between segments to merge into same cluster (default: 3000)
    min_cluster_size : int
        Minimum total size (nt) for a phage cluster to be kept (default: 1000)
    window_size : int
        Window size for bidirectional smoothing (default: 5)
    verbose : bool
        Print detailed processing information (default: False)
    
    Returns:
    --------
    filtered_df : DataFrame
        DataFrame with filtered predictions
    """
    
    # Sort by start position
    df = predictions_df.sort_values('start').copy()
    
    # Apply bidirectional smoothing to predictions if we have scores
    if 'score' in df.columns:
        # Forward pass (left to right)
        forward_smooth = df['score'].ewm(span=window_size, adjust=False).mean()
        
        # Backward pass (right to left) - reverse, smooth, reverse back
        backward_smooth = df['score'][::-1].ewm(span=window_size, adjust=False).mean()[::-1]
        
        # Average both directions for bidirectional smoothing
        df['smoothed_score'] = (forward_smooth + backward_smooth) / 2
        
        # Update predictions based on smoothed scores (threshold 0.5)
        df['prediction'] = (df['smoothed_score'] >= 0.5).astype(int)
    
    # Filter to only phage predictions
    phage_df = df[df['prediction'] == 1].copy()
    
    if len(phage_df) == 0:
        return df  # No phage predicted, return original
    
    # Cluster nearby phage segments
    clusters = []
    current_cluster = [phage_df.iloc[0]]
    
    for idx in range(1, len(phage_df)):
        prev_segment = current_cluster[-1]
        curr_segment = phage_df.iloc[idx]
        
        # Check if gap between segments is less than merge_gap
        gap = curr_segment['start'] - prev_segment['end']
        
        if gap <= merge_gap:
            current_cluster.append(curr_segment)
        else:
            # Save current cluster and start new one
            clusters.append(current_cluster)
            current_cluster = [curr_segment]
    
    # Don't forget the last cluster
    clusters.append(current_cluster)
    
    # Filter clusters by minimum size and mark segments
    valid_indices = set()
    
    for cluster in clusters:
        cluster_start = cluster[0]['start']
        cluster_end = cluster[-1]['end']
        cluster_size = cluster_end - cluster_start
        
        if cluster_size >= min_cluster_size:
            # Keep all segments in this cluster
            for segment in cluster:
                valid_indices.add(segment.name)
    
    # Update predictions: set to 0 if not in valid cluster
    df['filtered_prediction'] = df['prediction'].copy()
    df.loc[~df.index.isin(valid_indices), 'filtered_prediction'] = 0
    
    return df

def calculate_mcc(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient"""
    numerator = (tp * tn) - (fp * fn)
    
    # Use float to avoid overflow
    denominator_val = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    
    if denominator_val <= 0:
        return 0.0
    
    denominator = math.sqrt(denominator_val)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_metrics(tp, tn, fp, fn):
    """Calculate all metrics from confusion matrix"""
    total = tp + tn + fp + fn
    
    metrics = {}
    metrics['accuracy'] = (tp + tn) / total if total > 0 else 0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['mcc'] = calculate_mcc(tp, tn, fp, fn)
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0
    
    return metrics

def summarize_genome_predictions(directory_path, model_name, output_dir='.',
                                output_individual=None,
                                output_summary=None,
                                apply_filter=False,
                                merge_gap=3000,
                                min_cluster_size=1000,
                                window_size=5,
                                verbose=False):
    """
    Read all JSON prediction files from a directory and summarize results
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing JSON files
    model_name : str
        Name/identifier for this model (for comparing multiple models)
    output_dir : str
        Directory to save output CSV files (default: current directory)
    output_individual : str
        Output CSV filename for individual genome results (default: {model_name}_individual.csv)
    output_summary : str
        Output CSV filename for summary metrics (default: {model_name}_summary.csv)
    apply_filter : bool
        Apply phage clustering filter to reduce false positives (default: False)
    merge_gap : int
        Maximum gap (nt) between segments to merge (default: 3000)
    min_cluster_size : int
        Minimum cluster size (nt) to keep (default: 1000)
    window_size : int
        Window size for bidirectional smoothing (default: 5)
    """
    
    # Set default output filenames with model name if not provided
    if output_individual is None:
        suffix = "_filtered_individual.csv" if apply_filter else "_individual.csv"
        output_individual = f"{model_name}{suffix}"
    if output_summary is None:
        suffix = "_filtered_summary.csv" if apply_filter else "_summary.csv"
        output_summary = f"{model_name}{suffix}"
    
    # Add output directory to paths
    output_individual = os.path.join(output_dir, output_individual)
    output_summary = os.path.join(output_dir, output_summary)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return None, None
    
    print(f"Found {len(json_files)} JSON files")
    
    # Store individual file results
    results = []
    
    # Accumulators for aggregate calculation
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_samples = 0
    
    # Lists for averaging
    mcc_values = []
    fnr_values = []
    fpr_values = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract filename
            filename = os.path.basename(json_file)
            
            # If filtering is requested, look for corresponding CSV file
            if apply_filter:
                # Find corresponding CSV file (replace _metrics.json with .csv)
                csv_file = json_file.replace('_metrics.json', '.csv')
                
                if os.path.exists(csv_file):
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"Filename: {filename}")
                    
                    # Read the CSV file with predictions
                    pred_df = pd.read_csv(csv_file)
                    
                    if verbose:
                        print(f"Total predictions: {len(pred_df)}")
                    
                    if 'pred_label' in pred_df.columns:
                        # Calculate metrics before filtering
                        original_tp = data.get('true_positives', 0)
                        original_tn = data.get('true_negatives', 0)
                        original_fp = data.get('false_positives', 0)
                        original_fn = data.get('false_negatives', 0)
                        
                        original_mcc = calculate_mcc(original_tp, original_tn, original_fp, original_fn)
                        original_recall = original_tp / (original_tp + original_fn) if (original_tp + original_fn) > 0 else 0
                        
                        phage_before = pred_df['pred_label'].sum()
                        if verbose:
                            print(f"Phage predictions before filtering and merge: {phage_before}")
                        
                        # Rename columns to match expected format
                        pred_df = pred_df.rename(columns={'pred_label': 'prediction'})
                        if 'prob_1' in pred_df.columns:
                            pred_df = pred_df.rename(columns={'prob_1': 'score'})
                        
                        # First, collapse overlapping windows with majority voting
                        collapsed_df = collapse_overlapping_windows(pred_df, window_size=2000, step_size=1000)
                        
                        # Apply clustering filter
                        filtered_df = apply_phage_clustering_filter(
                            collapsed_df, 
                            merge_gap=merge_gap,
                            min_cluster_size=min_cluster_size,
                            window_size=window_size
                        )
                        
                        phage_after = filtered_df['filtered_prediction'].sum() if 'filtered_prediction' in filtered_df.columns else 0
                        if verbose:
                            print(f"Phage predictions after filtering and merge: {phage_after}")
                        
                        # Recalculate confusion matrix from filtered predictions
                        if 'label' in filtered_df.columns and 'filtered_prediction' in filtered_df.columns:
                            tp = ((filtered_df['label'] == 1) & (filtered_df['filtered_prediction'] == 1)).sum()
                            tn = ((filtered_df['label'] == 0) & (filtered_df['filtered_prediction'] == 0)).sum()
                            fp = ((filtered_df['label'] == 0) & (filtered_df['filtered_prediction'] == 1)).sum()
                            fn = ((filtered_df['label'] == 1) & (filtered_df['filtered_prediction'] == 0)).sum()
                            
                            new_mcc = calculate_mcc(tp, tn, fp, fn)
                            new_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                            
                            if verbose:
                                print(f"MCC before: {original_mcc:.4f}")
                                print(f"MCC after: {new_mcc:.4f}")
                                print(f"Recall before: {original_recall:.4f} ({original_recall*100:.2f}%)")
                                print(f"Recall after: {new_recall:.4f} ({new_recall*100:.2f}%)")
                        else:
                            if verbose:
                                print(f"WARNING: Could not find 'label' column")
                            # Fall back to original values
                            tp = data.get('true_positives', 0)
                            tn = data.get('true_negatives', 0)
                            fp = data.get('false_positives', 0)
                            fn = data.get('false_negatives', 0)
                    else:
                        if verbose:
                            print(f"WARNING: CSV file missing 'pred_label' column")
                        tp = data.get('true_positives', 0)
                        tn = data.get('true_negatives', 0)
                        fp = data.get('false_positives', 0)
                        fn = data.get('false_negatives', 0)
                else:
                    if verbose:
                        print(f"WARNING: CSV file not found: {csv_file}")
                    # Fall back to original values
                    tp = data.get('true_positives', 0)
                    tn = data.get('true_negatives', 0)
                    fp = data.get('false_positives', 0)
                    fn = data.get('false_negatives', 0)
            else:
                # Use original confusion matrix values
                tp = data.get('true_positives', 0)
                tn = data.get('true_negatives', 0)
                fp = data.get('false_positives', 0)
                fn = data.get('false_negatives', 0)
            
            # Calculate metrics for this file
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            mcc = data.get('mcc', calculate_mcc(tp, tn, fp, fn))
            
            # Store individual result
            results.append({
                'filename': filename,
                'samples': data.get('num_samples', tp + tn + fp + fn),
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'fnr': fnr,
                'fpr': fpr,
                'mcc': mcc,
                'accuracy': data.get('accuracy', 0),
                'precision': data.get('precision', 0),
                'recall': data.get('recall', 0),
                'specificity': data.get('specificity', 0)
            })
            
            # Accumulate totals
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_samples += data.get('num_samples', tp + tn + fp + fn)
            
            # Store for averaging
            mcc_values.append(mcc)
            fnr_values.append(fnr)
            fpr_values.append(fpr)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Create DataFrame with individual results
    df_individual = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    aggregate_metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
    
    # Calculate averaged metrics
    avg_mcc = np.mean(mcc_values)
    avg_fnr = np.mean(fnr_values)
    avg_fpr = np.mean(fpr_values)
    
    # Create summary DataFrame with model identifier
    summary_data = [
        {
            'model': model_name,
            'method': 'aggregate',
            'num_files': len(results),
            'total_samples': total_samples,
            'total_tp': total_tp,
            'total_tn': total_tn,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'fnr': aggregate_metrics['fnr'],
            'fpr': aggregate_metrics['fpr'],
            'mcc': aggregate_metrics['mcc'],
            'accuracy': aggregate_metrics['accuracy'],
            'precision': aggregate_metrics['precision'],
            'recall': aggregate_metrics['recall'],
            'specificity': aggregate_metrics['specificity'],
            'f1': aggregate_metrics['f1']
        },
        {
            'model': model_name,
            'method': 'average',
            'num_files': len(results),
            'total_samples': total_samples,
            'total_tp': '',
            'total_tn': '',
            'total_fp': '',
            'total_fn': '',
            'fnr': avg_fnr,
            'fpr': avg_fpr,
            'mcc': avg_mcc,
            'accuracy': df_individual['accuracy'].mean(),
            'precision': df_individual['precision'].mean(),
            'recall': df_individual['recall'].mean(),
            'specificity': df_individual['specificity'].mean(),
            'f1': df_individual['f1'].mean() if 'f1' in df_individual.columns else 0
        }
    ]
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to separate CSV files
    df_individual.to_csv(output_individual, index=False)
    df_summary.to_csv(output_summary, index=False)
    
    print(f"\nIndividual results saved to {output_individual}")
    print(f"Summary metrics saved to {output_summary}")
    print(f"\nProcessed {len(results)} files")
    print(f"Total samples: {total_samples}")
    print(f"\nAggregate Metrics (from summed confusion matrix):")
    print(f"  MCC: {aggregate_metrics['mcc']:.4f}")
    print(f"  FNR: {aggregate_metrics['fnr']:.4f} ({aggregate_metrics['fnr']*100:.2f}%)")
    print(f"  FPR: {aggregate_metrics['fpr']:.4f} ({aggregate_metrics['fpr']*100:.2f}%)")
    print(f"  Recall: {aggregate_metrics['recall']:.4f} ({aggregate_metrics['recall']*100:.2f}%)")
    print(f"\nAveraged Metrics (mean across files):")
    print(f"  MCC: {avg_mcc:.4f}")
    print(f"  FNR: {avg_fnr:.4f} ({avg_fnr*100:.2f}%)")
    print(f"  FPR: {avg_fpr:.4f} ({avg_fpr*100:.2f}%)")
    print(f"  Recall: {df_individual['recall'].mean():.4f} ({df_individual['recall'].mean()*100:.2f}%)")
    
    return df_individual, df_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Summarize genome-wide prediction metrics from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single model
  python analyze_genome_wide_results.py -d /path/to/json/files -m model1 -o results
  
  # Compare multiple models
  python analyze_genome_wide_results.py -d /path/to/model1 -m model1 -s model1_summary.csv
  python analyze_genome_wide_results.py -d /path/to/model2 -m model2 -s model2_summary.csv
  
  # Then combine:
  python -c "import pandas as pd; pd.concat([pd.read_csv('model1_summary.csv'), pd.read_csv('model2_summary.csv')]).to_csv('comparison.csv', index=False)"
        """
    )
    
    parser.add_argument('-d', '--directory', required=True,
                        help='Directory containing JSON prediction files')
    parser.add_argument('-m', '--model-name', required=True,
                        help='Model identifier/name for this run')
    parser.add_argument('-r', '--output-dir', default='.',
                        help='Directory to save output CSV files (default: current directory)')
    parser.add_argument('-i', '--output-individual', default=None,
                        help='Output CSV filename for individual genome results (default: {model_name}_individual.csv)')
    parser.add_argument('-s', '--output-summary', default=None,
                        help='Output CSV filename for summary metrics (default: {model_name}_summary.csv)')
    parser.add_argument('-o', '--output-prefix', default=None,
                        help='Prefix for both output files (overrides -i and -s, but not model name in filename)')
    parser.add_argument('--filter', action='store_true',
                        help='Apply phage clustering filter to reduce false positives')
    parser.add_argument('--merge-gap', type=int, default=3000,
                        help='Maximum gap (nt) between segments to merge into cluster (default: 3000)')
    parser.add_argument('--min-cluster-size', type=int, default=1000,
                        help='Minimum cluster size (nt) to keep (default: 1000)')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Window size for bidirectional smoothing (default: 5)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed processing information for each file')
    
    args = parser.parse_args()
    
    # Handle output prefix if provided
    if args.output_prefix:
        output_individual = f"{args.output_prefix}_individual.csv"
        output_summary = f"{args.output_prefix}_summary.csv"
    else:
        output_individual = args.output_individual
        output_summary = args.output_summary
    
    # Run the analysis
    df_individual, df_summary = summarize_genome_predictions(
        directory_path=args.directory,
        model_name=args.model_name,
        output_dir=args.output_dir,
        output_individual=output_individual,
        output_summary=output_summary,
        apply_filter=args.filter,
        merge_gap=args.merge_gap,
        min_cluster_size=args.min_cluster_size,
        window_size=args.window_size,
        verbose=args.verbose
    )
    
    if df_individual is not None and df_summary is not None:
        # Display first few rows
        print("\nFirst 5 individual files:")
        print(df_individual.head())
        print("\nSummary metrics:")
        print(df_summary)
