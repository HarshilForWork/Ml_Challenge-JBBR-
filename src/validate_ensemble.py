"""
Validate ensemble predictions by comparing predictions across models
and analyzing the quality of the weighted ensemble
"""

import numpy as np
import pandas as pd
from pathlib import Path

def validate_ensemble():
    """Compare predictions across all models and ensemble"""
    
    output_dir = Path("data/output")
    
    # Load all predictions
    ensemble_df = pd.read_csv(output_dir / "ensemble_kfold_predictions.csv")
    fold1_df = pd.read_csv(output_dir / "fold1_predictions.csv")
    fold2_df = pd.read_csv(output_dir / "fold2_predictions.csv")
    fold3_df = pd.read_csv(output_dir / "fold3_predictions.csv")
    
    ensemble_preds = ensemble_df['label'].values
    fold1_preds = fold1_df['label'].values
    fold2_preds = fold2_df['label'].values
    fold3_preds = fold3_df['label'].values
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE VALIDATION REPORT")
    print(f"{'='*80}\n")
    
    # Validation MAE scores for each fold
    mae_scores = {1: 0.1564, 2: 0.1568, 3: 0.1548}
    inverse_weights = {fold: 1.0 / mae for fold, mae in mae_scores.items()}
    weight_sum = sum(inverse_weights.values())
    normalized_weights = {fold: w / weight_sum for fold, w in inverse_weights.items()}
    
    print("Model Validation Scores and Ensemble Weights:")
    print("-" * 80)
    print(f"{'Fold':<8} {'Val MAE':<15} {'Inverse MAE':<20} {'Normalized Weight':<20}")
    print("-" * 80)
    for fold in [1, 2, 3]:
        mae = mae_scores[fold]
        inv_mae = inverse_weights[fold]
        norm_w = normalized_weights[fold]
        print(f"{fold:<8} {mae:<15.4f} {inv_mae:<20.4f} {norm_w:<20.4f}")
    print("-" * 80)
    
    # Prediction statistics
    print(f"\nPrediction Statistics:")
    print("-" * 80)
    print(f"{'Model':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 80)
    
    models = {
        'Fold 1': fold1_preds,
        'Fold 2': fold2_preds,
        'Fold 3': fold3_preds,
        'Ensemble': ensemble_preds
    }
    
    for name, preds in models.items():
        print(f"{name:<20} {preds.mean():<15.4f} {preds.std():<15.4f} {preds.min():<15.4f} {preds.max():<15.4f}")
    
    print("-" * 80)
    
    # Prediction agreement analysis
    print(f"\nPrediction Agreement Analysis:")
    print("-" * 80)
    
    # Calculate pairwise correlations
    corr_1_2 = np.corrcoef(fold1_preds, fold2_preds)[0, 1]
    corr_1_3 = np.corrcoef(fold1_preds, fold3_preds)[0, 1]
    corr_2_3 = np.corrcoef(fold2_preds, fold3_preds)[0, 1]
    corr_ens_1 = np.corrcoef(ensemble_preds, fold1_preds)[0, 1]
    corr_ens_2 = np.corrcoef(ensemble_preds, fold2_preds)[0, 1]
    corr_ens_3 = np.corrcoef(ensemble_preds, fold3_preds)[0, 1]
    
    print(f"Pairwise Prediction Correlations:")
    print(f"  Fold 1 vs Fold 2: {corr_1_2:.4f}")
    print(f"  Fold 1 vs Fold 3: {corr_1_3:.4f}")
    print(f"  Fold 2 vs Fold 3: {corr_2_3:.4f}")
    print(f"\nEnsemble Correlations:")
    print(f"  Ensemble vs Fold 1: {corr_ens_1:.4f}")
    print(f"  Ensemble vs Fold 2: {corr_ens_2:.4f}")
    print(f"  Ensemble vs Fold 3: {corr_ens_3:.4f}")
    
    # Mean absolute differences
    print(f"\nMean Absolute Differences Between Predictions:")
    print(f"  |Fold1 - Fold2|: {np.mean(np.abs(fold1_preds - fold2_preds)):.4f}")
    print(f"  |Fold1 - Fold3|: {np.mean(np.abs(fold1_preds - fold3_preds)):.4f}")
    print(f"  |Fold2 - Fold3|: {np.mean(np.abs(fold2_preds - fold3_preds)):.4f}")
    
    print(f"\nDifference Between Ensemble and Individual Models:")
    print(f"  |Ensemble - Fold1|: {np.mean(np.abs(ensemble_preds - fold1_preds)):.4f}")
    print(f"  |Ensemble - Fold2|: {np.mean(np.abs(ensemble_preds - fold2_preds)):.4f}")
    print(f"  |Ensemble - Fold3|: {np.mean(np.abs(ensemble_preds - fold3_preds)):.4f}")
    
    print("-" * 80)
    
    # Prediction distribution analysis
    print(f"\nPrediction Distribution by Ranges:")
    print("-" * 80)
    
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    print(f"{'Range':<15} {'Fold1':<12} {'Fold2':<12} {'Fold3':<12} {'Ensemble':<12}")
    print("-" * 80)
    
    for low, high in ranges:
        count_f1 = np.sum((fold1_preds >= low) & (fold1_preds < high))
        count_f2 = np.sum((fold2_preds >= low) & (fold2_preds < high))
        count_f3 = np.sum((fold3_preds >= low) & (fold3_preds < high))
        count_ens = np.sum((ensemble_preds >= low) & (ensemble_preds < high))
        
        pct_f1 = 100 * count_f1 / len(fold1_preds)
        pct_f2 = 100 * count_f2 / len(fold2_preds)
        pct_f3 = 100 * count_f3 / len(fold3_preds)
        pct_ens = 100 * count_ens / len(ensemble_preds)
        
        print(f"[{low:.2f}-{high:.2f}) {pct_f1:>10.1f}% {pct_f2:>10.1f}% {pct_f3:>10.1f}% {pct_ens:>10.1f}%")
    
    print("-" * 80)
    
    # Consistency analysis - how much do predictions agree?
    print(f"\nConsistency Analysis:")
    print("-" * 80)
    
    # Calculate std of predictions for each sample across models
    sample_stds = np.std(np.array([fold1_preds, fold2_preds, fold3_preds]), axis=0)
    
    print(f"Across-Model Std for Each Sample:")
    print(f"  Mean: {sample_stds.mean():.4f}")
    print(f"  Min:  {sample_stds.min():.4f}")
    print(f"  Max:  {sample_stds.max():.4f}")
    print(f"  Median: {np.median(sample_stds):.4f}")
    
    # Percentage of samples where all models agree within certain tolerance
    tolerances = [0.01, 0.05, 0.10]
    print(f"\nPercentage of Samples Where Models Agree (within tolerance):")
    for tol in tolerances:
        pct_agree = 100 * np.sum(sample_stds < tol) / len(sample_stds)
        print(f"  Within ±{tol}: {pct_agree:.1f}%")
    
    print("-" * 80)
    
    print(f"\n{'='*80}")
    print(f"✅ VALIDATION COMPLETE")
    print(f"{'='*80}\n")
    
    # Summary
    print(f"Summary:")
    print(f"- Ensemble uses inverse-MAE weighting: Fold1={normalized_weights[1]:.4f}, Fold2={normalized_weights[2]:.4f}, Fold3={normalized_weights[3]:.4f}")
    print(f"- Model agreement (correlation): {(corr_1_2 + corr_1_3 + corr_2_3) / 3:.4f} avg")
    print(f"- Prediction consistency (mean std across models): {sample_stds.mean():.4f}")
    print(f"- Predictions are {'consistent' if sample_stds.mean() < 0.05 else 'diverse'} across models")

if __name__ == "__main__":
    validate_ensemble()
