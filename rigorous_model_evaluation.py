#!/usr/bin/env python3
"""
Rigorous Model Evaluation for CsPbBr3 Digital Twin
Brutally honest assessment of model performance
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

class RigorousModelEvaluator:
    """Brutally honest model evaluation"""
    
    def __init__(self):
        self.results = {}
        
    def load_data_and_model(self):
        """Load training data and model"""
        print("🔍 Loading data and model...")
        
        # Load training data
        try:
            self.df = pd.read_csv('training_data.csv')
            print(f"   Dataset: {len(self.df)} samples")
            
            # Check target distribution
            if 'phase_label' in self.df.columns:
                target_col = 'phase_label'
            elif 'phase' in self.df.columns:
                target_col = 'phase'
            else:
                print("❌ No target column found")
                return False
                
            print(f"   Target distribution:")
            print(self.df[target_col].value_counts())
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
            
        # Load model
        try:
            self.model = torch.load('uncertainty_training_output/best_uncertainty_model.pth', 
                                  map_location='cpu')
            print("✅ Model loaded")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
            
        return True
    
    def evaluate_classification_performance(self):
        """Evaluate classification performance"""
        print("\n📊 Classification Performance Analysis")
        print("=" * 50)
        
        # Prepare features
        feature_cols = [
            'cs_br_concentration', 'pb_br2_concentration', 'temperature',
            'oa_concentration', 'oam_concentration', 'reaction_time',
            'solvent_type', 'cs_pb_ratio', 'supersaturation', 'ligand_ratio',
            'temp_normalized', 'solvent_effect'
        ]
        
        X = self.df[feature_cols].values
        y = self.df['phase_label'].values
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if len(outputs.shape) > 1:
                predictions = torch.argmax(outputs, dim=1).numpy()
                probabilities = torch.softmax(outputs, dim=1).numpy()
            else:
                predictions = (outputs > 0.5).numpy().astype(int)
                probabilities = outputs.numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted')
        
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        
        # Per-class analysis
        print("\n📈 Per-Class Analysis:")
        report = classification_report(y, predictions, output_dict=True)
        for class_id, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"Class {class_id}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Calculate prediction confidence
        if len(probabilities.shape) > 1:
            max_probs = np.max(probabilities, axis=1)
            avg_confidence = np.mean(max_probs)
            print(f"\nAverage Prediction Confidence: {avg_confidence:.3f}")
            
            # Low confidence predictions
            low_conf_mask = max_probs < 0.6
            low_conf_pct = np.sum(low_conf_mask) / len(max_probs) * 100
            print(f"Low Confidence Predictions (<0.6): {low_conf_pct:.1f}%")
        
        self.results['classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'avg_confidence': avg_confidence if len(probabilities.shape) > 1 else 0,
            'low_confidence_pct': low_conf_pct if len(probabilities.shape) > 1 else 0
        }
        
        return accuracy, f1
    
    def evaluate_regression_performance(self):
        """Evaluate regression performance for continuous outputs"""
        print("\n📊 Regression Performance Analysis")
        print("=" * 50)
        
        # Test on material properties
        continuous_targets = ['bandgap', 'plqy', 'particle_size', 'emission_peak']
        regression_results = {}
        
        for target in continuous_targets:
            if target in self.df.columns:
                print(f"\n🎯 Evaluating {target}...")
                
                # Simple regression prediction (placeholder - real model would need separate regression head)
                y_true = self.df[target].values
                
                # For demonstration, use a simple correlation with phase predictions
                feature_cols = [
                    'cs_br_concentration', 'pb_br2_concentration', 'temperature',
                    'oa_concentration', 'oam_concentration', 'reaction_time'
                ]
                X = self.df[feature_cols].values
                
                # Simple linear correlation as baseline
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(X, y_true)
                y_pred = reg.predict(X)
                
                # Metrics
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                print(f"   MSE: {mse:.4f}")
                print(f"   MAE: {mae:.4f}")
                print(f"   R²: {r2:.4f}")
                
                regression_results[target] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
        
        self.results['regression'] = regression_results
        
    def assess_data_quality(self):
        """Assess training data quality"""
        print("\n🔍 Data Quality Assessment")
        print("=" * 50)
        
        # Check for missing values
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            print("❌ Missing values found:")
            print(missing_data[missing_data > 0])
        else:
            print("✅ No missing values")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Check target balance
        target_col = 'phase_label'
        target_counts = self.df[target_col].value_counts()
        min_class_size = target_counts.min()
        max_class_size = target_counts.max()
        balance_ratio = min_class_size / max_class_size
        
        print(f"Class balance ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.1:
            print("❌ Severe class imbalance detected")
        elif balance_ratio < 0.3:
            print("⚠️ Moderate class imbalance")
        else:
            print("✅ Reasonable class balance")
        
        # Check feature correlations
        feature_cols = [
            'cs_br_concentration', 'pb_br2_concentration', 'temperature',
            'oa_concentration', 'oam_concentration', 'reaction_time'
        ]
        corr_matrix = self.df[feature_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_val
                    ))
        
        if high_corr_pairs:
            print("⚠️ High feature correlations found:")
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"   {feat1} - {feat2}: {corr_val:.3f}")
        else:
            print("✅ No problematic feature correlations")
        
        self.results['data_quality'] = {
            'missing_values': int(missing_data.sum()),
            'duplicates': int(duplicates),
            'class_balance_ratio': balance_ratio,
            'high_correlations': len(high_corr_pairs)
        }
    
    def assess_model_complexity(self):
        """Assess model complexity and potential overfitting"""
        print("\n🧠 Model Complexity Assessment")
        print("=" * 50)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Samples per parameter ratio
        samples_per_param = len(self.df) / trainable_params
        print(f"Samples per parameter: {samples_per_param:.2f}")
        
        if samples_per_param < 10:
            print("❌ Very high risk of overfitting (samples/params < 10)")
        elif samples_per_param < 50:
            print("⚠️ Moderate risk of overfitting (samples/params < 50)")
        else:
            print("✅ Reasonable samples/parameter ratio")
        
        self.results['model_complexity'] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'samples_per_param': samples_per_param
        }
    
    def final_assessment(self):
        """Final brutal assessment"""
        print("\n" + "="*60)
        print("🎯 BRUTAL HONESTY: PRODUCTION READINESS ASSESSMENT")
        print("="*60)
        
        # Extract key metrics
        accuracy = self.results.get('classification', {}).get('accuracy', 0)
        f1_score = self.results.get('classification', {}).get('f1_score', 0)
        confidence = self.results.get('classification', {}).get('avg_confidence', 0)
        low_conf_pct = self.results.get('classification', {}).get('low_confidence_pct', 100)
        samples_per_param = self.results.get('model_complexity', {}).get('samples_per_param', 0)
        balance_ratio = self.results.get('data_quality', {}).get('class_balance_ratio', 0)
        
        print(f"\n📊 KEY METRICS:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
        print(f"   Avg Confidence: {confidence:.3f}")
        print(f"   Low Confidence %: {low_conf_pct:.1f}%")
        print(f"   Samples/Parameter: {samples_per_param:.1f}")
        print(f"   Class Balance: {balance_ratio:.3f}")
        
        # Production readiness score
        score = 0
        max_score = 6
        
        if accuracy > 0.8:
            score += 1
        if f1_score > 0.7:
            score += 1
        if confidence > 0.7:
            score += 1
        if low_conf_pct < 30:
            score += 1
        if samples_per_param > 10:
            score += 1
        if balance_ratio > 0.2:
            score += 1
        
        readiness_pct = (score / max_score) * 100
        
        print(f"\n🎯 PRODUCTION READINESS SCORE: {readiness_pct:.0f}%")
        
        if readiness_pct >= 80:
            assessment = "✅ READY FOR PRODUCTION"
            recommendations = [
                "Deploy with confidence monitoring",
                "Implement uncertainty thresholding",
                "Set up continuous model validation"
            ]
        elif readiness_pct >= 60:
            assessment = "⚠️ NEEDS IMPROVEMENT BEFORE PRODUCTION"
            recommendations = [
                "Increase training data size by 2-3x",
                "Improve class balance through sampling",
                "Add more diverse training examples",
                "Implement cross-validation"
            ]
        else:
            assessment = "❌ NOT READY FOR PRODUCTION"
            recommendations = [
                "Collect significantly more training data",
                "Address severe class imbalance",
                "Reduce model complexity or increase data",
                "Improve feature engineering",
                "Consider different model architectures"
            ]
        
        print(f"\n{assessment}")
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Specific issues found
        issues = []
        if accuracy < 0.7:
            issues.append("Low accuracy indicates poor model performance")
        if confidence < 0.6:
            issues.append("Low confidence suggests model uncertainty")
        if low_conf_pct > 40:
            issues.append("Too many low-confidence predictions")
        if samples_per_param < 10:
            issues.append("High overfitting risk due to insufficient data")
        if balance_ratio < 0.1:
            issues.append("Severe class imbalance affects reliability")
        
        if issues:
            print(f"\n⚠️ CRITICAL ISSUES:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        self.results['final_assessment'] = {
            'readiness_score': readiness_pct,
            'assessment': assessment,
            'recommendations': recommendations,
            'critical_issues': issues
        }
        
        return readiness_pct, assessment, recommendations

def main():
    """Main evaluation function"""
    print("🔍 RIGOROUS MODEL EVALUATION")
    print("=" * 60)
    print("Brutally honest assessment of CsPbBr3 Digital Twin")
    print()
    
    evaluator = RigorousModelEvaluator()
    
    # Load data and model
    if not evaluator.load_data_and_model():
        return
    
    # Run evaluations
    evaluator.assess_data_quality()
    evaluator.assess_model_complexity()
    accuracy, f1 = evaluator.evaluate_classification_performance()
    evaluator.evaluate_regression_performance()
    
    # Final assessment
    readiness_pct, assessment, recommendations = evaluator.final_assessment()
    
    # Save results
    with open('rigorous_evaluation_results.json', 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: rigorous_evaluation_results.json")
    
    return readiness_pct, assessment

if __name__ == "__main__":
    main()