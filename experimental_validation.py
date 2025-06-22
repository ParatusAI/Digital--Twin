#!/usr/bin/env python3
"""
Experimental Validation System for CsPbBr₃ Digital Twin
Tracks experimental results vs model predictions
"""

import pandas as pd
import numpy as np
import json
import torch
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentalConditions:
    """Data structure for experimental synthesis conditions"""
    experiment_id: str
    cs_br_concentration: float  # mol/L
    pb_br2_concentration: float  # mol/L
    temperature: float  # °C
    oa_concentration: float  # mol/L (oleic acid)
    oam_concentration: float  # mol/L (oleylamine)
    reaction_time: float  # minutes
    solvent_type: int  # 0=DMSO, 1=DMF, 2=Toluene, 3=Octadecene, 4=Mixed
    date_conducted: str
    researcher: str
    notes: str = ""

@dataclass
class ExperimentalResults:
    """Data structure for experimental synthesis results"""
    experiment_id: str
    # Phase characterization
    dominant_phase: str  # CsPbBr3_3D, Cs4PbBr6_0D, CsPb2Br5_2D, Mixed, Failed
    phase_purity: float  # 0-1 (from XRD analysis)
    secondary_phases: List[str]
    
    # Material properties
    bandgap: Optional[float]  # eV (from UV-vis)
    plqy: Optional[float]  # Photoluminescence quantum yield
    particle_size: Optional[float]  # nm (from TEM/DLS)
    emission_peak: Optional[float]  # nm (from PL spectroscopy)
    
    # Visual observations
    solution_color: str
    precipitate_observed: bool
    
    # Characterization methods used
    characterization_methods: List[str]  # ['XRD', 'UV-vis', 'PL', 'TEM', etc.]
    
    # Quality metrics
    synthesis_success: bool
    yield_percentage: Optional[float]
    
    # Additional measurements
    additional_properties: Dict[str, float]
    analysis_date: str
    notes: str = ""

@dataclass
class ModelPrediction:
    """Data structure for model predictions"""
    experiment_id: str
    predicted_phase: str
    phase_probabilities: Dict[str, float]
    confidence: float
    predicted_properties: Dict[str, float]
    prediction_date: str

class ExperimentalValidator:
    """Main class for experimental validation tracking"""
    
    def __init__(self, data_dir: str = "experimental_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # File paths
        self.experiments_file = self.data_dir / "experiments.csv"
        self.results_file = self.data_dir / "results.csv"
        self.predictions_file = self.data_dir / "predictions.csv"
        self.validation_file = self.data_dir / "validation_analysis.json"
        
        # Load model components
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self._load_model()
        
        # Phase mapping
        self.phase_names = {
            0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 
            3: 'Mixed', 4: 'Failed'
        }
        self.phase_to_num = {v: k for k, v in self.phase_names.items()}
    
    def _load_model(self):
        """Load the trained model and preprocessing components"""
        try:
            # Import model class
            import sys
            sys.path.append('.')
            from test_trained_model import SimpleNeuralNetwork, load_trained_model, load_scaler
            
            # Load model
            self.model, config = load_trained_model()
            self.scaler = load_scaler()
            self.feature_cols = config['feature_columns']
            
            logger.info("✅ Model loaded successfully for validation")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Model predictions will not be available")
    
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"EXP_{timestamp}"
    
    def record_experimental_conditions(self, conditions: ExperimentalConditions) -> str:
        """Record experimental conditions before synthesis"""
        
        # Convert to DataFrame row
        df_row = pd.DataFrame([asdict(conditions)])
        
        # Append to experiments file
        if self.experiments_file.exists():
            existing_df = pd.read_csv(self.experiments_file)
            df_combined = pd.concat([existing_df, df_row], ignore_index=True)
        else:
            df_combined = df_row
        
        df_combined.to_csv(self.experiments_file, index=False)
        
        logger.info(f"📝 Recorded experimental conditions for {conditions.experiment_id}")
        return conditions.experiment_id
    
    def get_model_prediction(self, conditions: ExperimentalConditions) -> Optional[ModelPrediction]:
        """Generate model prediction for given conditions"""
        
        if self.model is None or self.scaler is None:
            logger.warning("Model not available for predictions")
            return None
        
        try:
            # Convert conditions to model input format
            synthesis_params = {
                'cs_br_concentration': conditions.cs_br_concentration,
                'pb_br2_concentration': conditions.pb_br2_concentration,
                'temperature': conditions.temperature,
                'oa_concentration': conditions.oa_concentration,
                'oam_concentration': conditions.oam_concentration,
                'reaction_time': conditions.reaction_time,
                'solvent_type': conditions.solvent_type
            }
            
            # Generate prediction using existing function
            from test_trained_model import predict_synthesis
            prediction = predict_synthesis(self.model, self.scaler, synthesis_params, self.feature_cols)
            
            # Convert to our format
            predicted_phase = self.phase_names[prediction['phase']['predicted_class']]
            phase_probs = {
                self.phase_names[i]: prob 
                for i, prob in enumerate(prediction['phase']['probabilities'])
            }
            
            model_prediction = ModelPrediction(
                experiment_id=conditions.experiment_id,
                predicted_phase=predicted_phase,
                phase_probabilities=phase_probs,
                confidence=prediction['phase']['confidence'],
                predicted_properties=prediction['properties'],
                prediction_date=datetime.now().isoformat()
            )
            
            # Save prediction
            self._save_prediction(model_prediction)
            
            logger.info(f"🔮 Generated prediction for {conditions.experiment_id}: {predicted_phase} ({prediction['phase']['confidence']:.3f})")
            return model_prediction
            
        except Exception as e:
            logger.error(f"Failed to generate prediction: {e}")
            return None
    
    def _save_prediction(self, prediction: ModelPrediction):
        """Save model prediction to file"""
        
        # Convert to DataFrame row
        pred_data = asdict(prediction)
        # Flatten nested dictionaries
        pred_data['phase_prob_CsPbBr3_3D'] = pred_data['phase_probabilities']['CsPbBr3_3D']
        pred_data['phase_prob_Cs4PbBr6_0D'] = pred_data['phase_probabilities']['Cs4PbBr6_0D']
        pred_data['phase_prob_CsPb2Br5_2D'] = pred_data['phase_probabilities']['CsPb2Br5_2D']
        pred_data['phase_prob_Mixed'] = pred_data['phase_probabilities']['Mixed']
        pred_data['phase_prob_Failed'] = pred_data['phase_probabilities']['Failed']
        
        pred_data['pred_bandgap'] = pred_data['predicted_properties']['bandgap']
        pred_data['pred_plqy'] = pred_data['predicted_properties']['plqy']
        pred_data['pred_particle_size'] = pred_data['predicted_properties']['particle_size']
        pred_data['pred_emission_peak'] = pred_data['predicted_properties']['emission_peak']
        
        # Remove nested dictionaries
        del pred_data['phase_probabilities']
        del pred_data['predicted_properties']
        
        df_row = pd.DataFrame([pred_data])
        
        # Append to predictions file
        if self.predictions_file.exists():
            existing_df = pd.read_csv(self.predictions_file)
            df_combined = pd.concat([existing_df, df_row], ignore_index=True)
        else:
            df_combined = df_row
        
        df_combined.to_csv(self.predictions_file, index=False)
    
    def record_experimental_results(self, results: ExperimentalResults):
        """Record experimental results after synthesis and characterization"""
        
        # Convert to DataFrame row
        results_data = asdict(results)
        # Convert list fields to strings
        results_data['secondary_phases'] = ','.join(results.secondary_phases)
        results_data['characterization_methods'] = ','.join(results.characterization_methods)
        results_data['additional_properties'] = json.dumps(results.additional_properties)
        
        df_row = pd.DataFrame([results_data])
        
        # Append to results file
        if self.results_file.exists():
            existing_df = pd.read_csv(self.results_file)
            df_combined = pd.concat([existing_df, df_row], ignore_index=True)
        else:
            df_combined = df_row
        
        df_combined.to_csv(self.results_file, index=False)
        
        logger.info(f"📊 Recorded experimental results for {results.experiment_id}")
        
        # Trigger validation analysis
        self.analyze_prediction_accuracy(results.experiment_id)
    
    def analyze_prediction_accuracy(self, experiment_id: str):
        """Analyze prediction accuracy for a specific experiment"""
        
        try:
            # Load data
            predictions_df = pd.read_csv(self.predictions_file)
            results_df = pd.read_csv(self.results_file)
            
            # Find matching records
            pred_row = predictions_df[predictions_df['experiment_id'] == experiment_id]
            result_row = results_df[results_df['experiment_id'] == experiment_id]
            
            if pred_row.empty or result_row.empty:
                logger.warning(f"Missing prediction or result data for {experiment_id}")
                return
            
            pred = pred_row.iloc[0]
            result = result_row.iloc[0]
            
            # Phase prediction accuracy
            phase_correct = pred['predicted_phase'] == result['dominant_phase']
            confidence = pred['confidence']
            
            # Property prediction accuracy (if measured)
            property_accuracies = {}
            for prop in ['bandgap', 'plqy', 'particle_size', 'emission_peak']:
                if pd.notna(result[prop]) and pd.notna(pred[f'pred_{prop}']):
                    predicted = pred[f'pred_{prop}']
                    actual = result[prop]
                    relative_error = abs(predicted - actual) / actual * 100
                    property_accuracies[prop] = {
                        'predicted': float(predicted),
                        'actual': float(actual),
                        'relative_error_percent': float(relative_error)
                    }
            
            # Create analysis summary
            analysis = {
                'experiment_id': experiment_id,
                'phase_prediction_correct': bool(phase_correct),
                'predicted_phase': pred['predicted_phase'],
                'actual_phase': result['dominant_phase'],
                'prediction_confidence': float(confidence),
                'actual_phase_purity': float(result['phase_purity']) if pd.notna(result['phase_purity']) else None,
                'property_accuracies': property_accuracies,
                'analysis_date': datetime.now().isoformat()
            }
            
            # Save analysis
            self._save_validation_analysis(analysis)
            
            # Log results
            logger.info(f"📈 Validation Analysis for {experiment_id}:")
            logger.info(f"   Phase Prediction: {'✅ Correct' if phase_correct else '❌ Incorrect'}")
            logger.info(f"   Predicted: {pred['predicted_phase']} (conf: {confidence:.3f})")
            logger.info(f"   Actual: {result['dominant_phase']} (purity: {result['phase_purity']:.3f})")
            
            for prop, acc in property_accuracies.items():
                logger.info(f"   {prop}: {acc['relative_error_percent']:.1f}% error")
                
        except Exception as e:
            logger.error(f"Failed to analyze prediction accuracy: {e}")
    
    def _save_validation_analysis(self, analysis: Dict[str, Any]):
        """Save validation analysis to file"""
        
        # Load existing analyses
        if self.validation_file.exists():
            with open(self.validation_file, 'r') as f:
                all_analyses = json.load(f)
        else:
            all_analyses = []
        
        # Add new analysis
        all_analyses.append(analysis)
        
        # Save updated analyses
        with open(self.validation_file, 'w') as f:
            json.dump(all_analyses, f, indent=2)
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        try:
            # Load all validation analyses
            if not self.validation_file.exists():
                logger.warning("No validation data available")
                return {}
            
            with open(self.validation_file, 'r') as f:
                analyses = json.load(f)
            
            if not analyses:
                logger.warning("No validation analyses found")
                return {}
            
            # Calculate overall statistics
            total_experiments = len(analyses)
            phase_correct = sum(1 for a in analyses if a['phase_prediction_correct'])
            phase_accuracy = phase_correct / total_experiments * 100
            
            # Property accuracy statistics
            property_stats = {}
            for prop in ['bandgap', 'plqy', 'particle_size', 'emission_peak']:
                errors = []
                for analysis in analyses:
                    if prop in analysis['property_accuracies']:
                        errors.append(analysis['property_accuracies'][prop]['relative_error_percent'])
                
                if errors:
                    property_stats[prop] = {
                        'mean_error_percent': float(np.mean(errors)),
                        'median_error_percent': float(np.median(errors)),
                        'std_error_percent': float(np.std(errors)),
                        'n_measurements': len(errors)
                    }
            
            # Confidence vs accuracy analysis
            confidence_bins = []
            accuracy_by_confidence = []
            
            for analysis in analyses:
                conf = analysis['prediction_confidence']
                correct = analysis['phase_prediction_correct']
                confidence_bins.append(conf)
                accuracy_by_confidence.append(correct)
            
            # Generate report
            report = {
                'report_date': datetime.now().isoformat(),
                'total_experiments': total_experiments,
                'phase_prediction_accuracy_percent': phase_accuracy,
                'experiments_with_correct_phase': phase_correct,
                'property_accuracy_statistics': property_stats,
                'confidence_analysis': {
                    'mean_confidence': float(np.mean(confidence_bins)),
                    'confidence_vs_accuracy_correlation': float(np.corrcoef(confidence_bins, accuracy_by_confidence)[0, 1]) if len(confidence_bins) > 1 else 0.0
                },
                'individual_experiments': analyses
            }
            
            # Save report
            report_file = self.data_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📋 Validation Report Generated:")
            logger.info(f"   Total Experiments: {total_experiments}")
            logger.info(f"   Phase Accuracy: {phase_accuracy:.1f}%")
            logger.info(f"   Mean Confidence: {report['confidence_analysis']['mean_confidence']:.3f}")
            logger.info(f"   Report saved to: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return {}
    
    def plot_validation_results(self, save_plots: bool = True):
        """Generate validation plots"""
        
        try:
            if not self.validation_file.exists():
                logger.warning("No validation data for plotting")
                return
            
            with open(self.validation_file, 'r') as f:
                analyses = json.load(f)
            
            if not analyses:
                logger.warning("No validation analyses for plotting")
                return
            
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('CsPbBr₃ Digital Twin - Experimental Validation Results', fontsize=16)
            
            # 1. Phase Prediction Accuracy
            ax1 = axes[0, 0]
            correct = [a['phase_prediction_correct'] for a in analyses]
            phases = [a['predicted_phase'] for a in analyses]
            
            phase_counts = {}
            phase_correct = {}
            for phase, is_correct in zip(phases, correct):
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if is_correct:
                    phase_correct[phase] = phase_correct.get(phase, 0) + 1
            
            phase_names = list(phase_counts.keys())
            accuracies = [phase_correct.get(p, 0) / phase_counts[p] * 100 for p in phase_names]
            
            bars = ax1.bar(phase_names, accuracies, color=['green' if acc > 50 else 'red' for acc in accuracies])
            ax1.set_title('Phase Prediction Accuracy by Phase Type')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xticklabels(phase_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.1f}%', ha='center', va='bottom')
            
            # 2. Confidence vs Accuracy
            ax2 = axes[0, 1]
            confidences = [a['prediction_confidence'] for a in analyses]
            correct_numeric = [1 if c else 0 for c in correct]
            
            ax2.scatter(confidences, correct_numeric, alpha=0.7, s=50)
            ax2.set_xlabel('Prediction Confidence')
            ax2.set_ylabel('Correct Prediction (1=Yes, 0=No)')
            ax2.set_title('Confidence vs Accuracy')
            
            # Add trend line
            z = np.polyfit(confidences, correct_numeric, 1)
            p = np.poly1d(z)
            ax2.plot(confidences, p(confidences), "r--", alpha=0.8)
            
            # 3. Property Prediction Errors
            ax3 = axes[1, 0]
            property_errors = {'bandgap': [], 'plqy': [], 'particle_size': [], 'emission_peak': []}
            
            for analysis in analyses:
                for prop in property_errors.keys():
                    if prop in analysis['property_accuracies']:
                        property_errors[prop].append(analysis['property_accuracies'][prop]['relative_error_percent'])
            
            # Box plot of errors
            error_data = [errors for errors in property_errors.values() if errors]
            error_labels = [prop for prop, errors in property_errors.items() if errors]
            
            if error_data:
                ax3.boxplot(error_data, labels=error_labels)
                ax3.set_title('Property Prediction Errors')
                ax3.set_ylabel('Relative Error (%)')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No Property Data Available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Property Prediction Errors')
            
            # 4. Experiment Timeline
            ax4 = axes[1, 1]
            experiment_dates = []
            cumulative_accuracy = []
            
            for i, analysis in enumerate(analyses):
                experiment_dates.append(i + 1)  # Sequential experiment number
                current_correct = sum(1 for a in analyses[:i+1] if a['phase_prediction_correct'])
                cumulative_accuracy.append(current_correct / (i + 1) * 100)
            
            ax4.plot(experiment_dates, cumulative_accuracy, 'b-o', linewidth=2, markersize=4)
            ax4.set_xlabel('Experiment Number')
            ax4.set_ylabel('Cumulative Accuracy (%)')
            ax4.set_title('Model Performance Over Time')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_file = self.data_dir / f"validation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Validation plots saved to: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate validation plots: {e}")

def main():
    """Example usage of the experimental validation system"""
    
    # Initialize validator
    validator = ExperimentalValidator()
    
    print("🧪 CsPbBr₃ Experimental Validation System")
    print("=" * 50)
    
    # Example: Record experimental conditions
    conditions = ExperimentalConditions(
        experiment_id=validator.generate_experiment_id(),
        cs_br_concentration=1.1,
        pb_br2_concentration=1.1,
        temperature=190.0,
        oa_concentration=0.4,
        oam_concentration=0.2,
        reaction_time=75.0,
        solvent_type=0,  # DMSO
        date_conducted=datetime.now().strftime("%Y-%m-%d"),
        researcher="Test User",
        notes="Testing optimal conditions from digital twin"
    )
    
    # Record conditions and get prediction
    exp_id = validator.record_experimental_conditions(conditions)
    prediction = validator.get_model_prediction(conditions)
    
    print(f"\n📝 Experiment {exp_id} recorded")
    if prediction:
        print(f"🔮 Model predicts: {prediction.predicted_phase} (confidence: {prediction.confidence:.3f})")
    
    print("\n💡 Next steps:")
    print("1. Conduct synthesis experiment")
    print("2. Characterize materials (XRD, UV-vis, PL, TEM)")
    print("3. Record results using validator.record_experimental_results()")
    print("4. Generate validation report")

if __name__ == "__main__":
    main()