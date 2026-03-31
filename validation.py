import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve
import json
import os
import warnings
import shap

def run_validation(df, outdir='artifacts/validation'):
    """
    IFRS9 model validation with metrics and plots
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Determine column names flexibly
    target_cols = [col for col in df.columns if col in ['target', 'target_default']]
    pred_cols = [col for col in df.columns if col in ['pd_est', 'pd_calibrated']]
    
    if not target_cols or not pred_cols:
        print(f"Warning: Could not find proper target/prediction columns")
        print(f"Available columns: {df.columns.tolist()}")
        return {}
    
    target_col = target_cols[0]
    pred_col = pred_cols[0]
    
    y_true = df[target_col]
    y_pred = df[pred_col]
    
    print(f"Using target: {target_col}, predictions: {pred_col}")
    print(f"Target distribution: {y_true.value_counts().to_dict()}")
    print(f"Prediction stats: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred)
        
        # Kolmogorov-Smirnov statistic
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            # Calculate KS statistic manually
            sorted_scores = np.sort(np.concatenate([pos_scores, neg_scores]))
            tpr_vals = []
            fpr_vals = []
            
            for threshold in sorted_scores:
                tp = np.sum((y_pred >= threshold) & (y_true == 1))
                fp = np.sum((y_pred >= threshold) & (y_true == 0))
                fn = np.sum((y_pred < threshold) & (y_true == 1))
                tn = np.sum((y_pred < threshold) & (y_true == 0))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tpr_vals.append(tpr)
                fpr_vals.append(fpr)
            
            ks_stat = np.max(np.abs(np.array(tpr_vals) - np.array(fpr_vals)))
        else:
            ks_stat = 0.0
        
        metrics = {
            'roc_auc': auc,
            'ks_statistic': ks_stat,
            'brier_score': brier,
            'n_samples': len(df),
            'default_rate': y_true.mean()
        }
        
        print(f"ROC AUC: {auc:.4f}")
        print(f"KS Statistic: {ks_stat:.4f}")
        print(f"Brier Score: {brier:.4f}")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {
            'error': str(e),
            'n_samples': len(df),
            'default_rate': y_true.mean()
        }
    
    # Save metrics
    with open(os.path.join(outdir, 'pd_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots if possible
    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - PD Model')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'pd_roc_best.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - PD Model')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(outdir, 'pd_pr_best.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calibration Plot
        if len(np.unique(y_true)) > 1:  # Only if we have both classes
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=10, strategy='uniform'
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Plot - PD Model')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(outdir, 'pd_calibration_best.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Validation plots saved to {outdir}")
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    return metrics