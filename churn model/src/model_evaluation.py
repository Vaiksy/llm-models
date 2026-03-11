import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
import shap

class ChurnModelEvaluator:
    """
    Evaluate and visualize churn prediction model performance
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Comprehensive model evaluation
        """
        ## Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        ## Metrics
        print("\n" + "="*50)
        print("MODEL EVALUATION REPORT")
        print("="*50)
        
        print(f"\nThreshold: {threshold}")
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
        
        ## Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'])
        plt.title('Confusion Matrix', fontsize=15, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                  fontsize=15, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba, save_path=None):
        """
        Plot Precision-Recall curve
        """
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=15, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if self.feature_names is not None:
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
            else:
                feature_imp = pd.DataFrame({
                    'feature': [f'Feature {i}' for i in range(len(importances))],
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_imp, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances', fontsize=15, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_imp
        else:
            print("Model doesn't have feature_importances_ attribute")
            return None
    
    def plot_shap_summary(self, X_test, sample_size=1000, save_path=None):
        """
        Plot SHAP summary for model interpretability
        """
        try:
            # Sample data if too large
            if len(X_test) > sample_size:
                X_sample = X_test.sample(sample_size, random_state=42)
            else:
                X_sample = X_test
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, use positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, 
                             feature_names=self.feature_names,
                             show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating SHAP plot: {e}")
    
    def analyze_threshold(self, y_test, y_pred_proba, thresholds=None):
        """
        Analyze different threshold values
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        
        results_df = pd.DataFrame(results)
        
        print("\nThreshold Analysis:")
        print(results_df.to_string(index=False))
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(results_df['threshold'], results_df['precision'], 
                    marker='o', label='Precision')
        axes[0].plot(results_df['threshold'], results_df['recall'], 
                    marker='s', label='Recall')
        axes[0].plot(results_df['threshold'], results_df['f1_score'], 
                    marker='^', label='F1-Score')
        axes[0].set_xlabel('Threshold', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(results_df['threshold'], results_df['true_positives'], 
                    marker='o', label='True Positives')
        axes[1].plot(results_df['threshold'], results_df['false_positives'], 
                    marker='s', label='False Positives')
        axes[1].plot(results_df['threshold'], results_df['false_negatives'], 
                    marker='^', label='False Negatives')
        axes[1].set_xlabel('Threshold', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Predictions vs Threshold', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        

        return results_df



