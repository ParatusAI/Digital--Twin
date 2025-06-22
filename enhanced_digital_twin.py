#!/usr/bin/env python3
"""
Enhanced CsPbBr₃ Digital Twin System
Integrates uncertainty quantification, active learning, and web interface
"""

import torch
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import argparse

from uncertainty_models import BayesianNeuralNetwork, MCDropoutNeuralNetwork, EnsembleNeuralNetwork
from active_learning import ActiveLearningOrchestrator
from validation_pipeline import ValidationPipeline
from experimental_validation import ExperimentalValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDigitalTwin:
    """Enhanced Digital Twin with uncertainty quantification and active learning"""
    
    def __init__(self, model_dir: str = "uncertainty_training_output", 
                 data_dir: str = "experimental_data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        
        # Model components
        self.uncertainty_model = None
        self.baseline_model = None
        self.scaler = None
        self.feature_columns = None
        self.model_type = None
        
        # System components
        self.active_learner = None
        self.validation_pipeline = None
        self.validator = None
        
        # Phase mapping
        self.phase_names = {
            0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 
            3: 'Mixed', 4: 'Failed'
        }
        
        # Initialize system
        self._load_models()
        self._initialize_components()
    
    def _load_models(self):
        """Load trained models and components"""
        
        try:
            # Load uncertainty model if available
            uncertainty_config_file = self.model_dir / "best_model_info.json"
            if uncertainty_config_file.exists():
                with open(uncertainty_config_file, 'r') as f:
                    config = json.load(f)
                
                self.model_type = config['model_type']
                self.feature_columns = config['feature_columns']
                
                # Load uncertainty scaler
                scaler_path = Path(config['scaler_path'])
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                
                # Load uncertainty model
                input_dim = len(self.feature_columns)
                
                if self.model_type == 'bayesian':
                    self.uncertainty_model = BayesianNeuralNetwork(input_dim)
                    model_path = self.model_dir / "bayesian_model.pth"
                elif self.model_type == 'mcdropout':
                    self.uncertainty_model = MCDropoutNeuralNetwork(input_dim)
                    model_path = self.model_dir / "mcdropout_model.pth"
                elif self.model_type == 'ensemble':
                    self.uncertainty_model = EnsembleNeuralNetwork(input_dim)
                    model_path = self.model_dir / "ensemble_model.pth"
                
                if model_path.exists():
                    self.uncertainty_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    self.uncertainty_model.eval()
                    logger.info(f"✅ Loaded {self.model_type} uncertainty model")
                else:
                    logger.warning(f"Uncertainty model file not found: {model_path}")
            
            else:
                logger.info("No uncertainty model found, trying baseline model")
        
        except Exception as e:
            logger.error(f"Error loading uncertainty model: {e}")
        
        # Fallback to baseline model
        if self.uncertainty_model is None:
            try:
                from test_trained_model import load_trained_model, load_scaler
                self.baseline_model, config = load_trained_model()
                self.scaler = load_scaler()
                self.feature_columns = config['feature_columns']
                logger.info("✅ Loaded baseline model")
            except Exception as e:
                logger.error(f"Error loading baseline model: {e}")
                logger.warning("No trained model available")
    
    def _initialize_components(self):
        """Initialize system components"""
        
        if self.uncertainty_model is not None or self.baseline_model is not None:
            # Initialize active learning
            model = self.uncertainty_model if self.uncertainty_model is not None else self.baseline_model
            self.active_learner = ActiveLearningOrchestrator(model, self.feature_columns, self.scaler)
            
            # Initialize validation pipeline
            self.validation_pipeline = ValidationPipeline(str(self.data_dir))
            self.validator = ExperimentalValidator(str(self.data_dir))
            
            logger.info("✅ System components initialized")
        else:
            logger.error("Cannot initialize components without a trained model")
    
    def predict_with_uncertainty(self, conditions: Dict[str, float], 
                                num_samples: int = 100) -> Dict[str, Any]:
        """Make predictions with uncertainty quantification"""
        
        if self.uncertainty_model is None:
            return self._baseline_prediction(conditions)
        
        try:
            # Convert conditions to features
            features = self._conditions_to_features(conditions)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Get uncertainty prediction
            self.uncertainty_model.eval()
            outputs = self.uncertainty_model.predict_with_uncertainty(features_tensor, num_samples)
            
            # Process outputs
            phase_probs = outputs['phase_probabilities']['mean'].squeeze()
            phase_std = outputs['phase_probabilities']['std'].squeeze()
            predicted_phase = torch.argmax(phase_probs).item()
            
            # Property predictions with uncertainty
            properties = {}
            for prop_name, stats in outputs['properties'].items():
                properties[prop_name] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'confidence_interval': [
                        float(stats['quantile_025']),
                        float(stats['quantile_975'])
                    ]
                }
            
            return {
                'predicted_phase': self.phase_names[predicted_phase],
                'phase_probabilities': [float(p) for p in phase_probs],
                'phase_uncertainties': [float(u) for u in phase_std],
                'confidence': float(torch.max(phase_probs)),
                'uncertainty_score': float(torch.mean(phase_std)),
                'properties': properties,
                'has_uncertainty': True,
                'model_type': self.model_type,
                'prediction_timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Uncertainty prediction failed: {e}")
            return self._baseline_prediction(conditions)
    
    def _baseline_prediction(self, conditions: Dict[str, float]) -> Dict[str, Any]:
        """Fallback to baseline prediction"""
        
        if self.baseline_model is None:
            return {
                'predicted_phase': 'Unknown',
                'phase_probabilities': [0.2] * 5,
                'phase_uncertainties': [0.0] * 5,
                'confidence': 0.2,
                'uncertainty_score': 0.0,
                'properties': {},
                'has_uncertainty': False,
                'model_type': 'baseline',
                'prediction_timestamp': datetime.now().isoformat()
            }
        
        try:
            from test_trained_model import predict_synthesis
            
            synthesis_params = {
                'cs_br_concentration': conditions['cs_br_concentration'],
                'pb_br2_concentration': conditions['pb_br2_concentration'],
                'temperature': conditions['temperature'],
                'oa_concentration': conditions['oa_concentration'],
                'oam_concentration': conditions['oam_concentration'],
                'reaction_time': conditions['reaction_time'],
                'solvent_type': conditions['solvent_type']
            }
            
            prediction = predict_synthesis(self.baseline_model, self.scaler, synthesis_params, self.feature_columns)
            
            return {
                'predicted_phase': self.phase_names[prediction['phase']['predicted_class']],
                'phase_probabilities': prediction['phase']['probabilities'],
                'phase_uncertainties': [0.0] * 5,
                'confidence': prediction['phase']['confidence'],
                'uncertainty_score': 0.0,
                'properties': {
                    prop_name: {
                        'mean': float(value),
                        'std': 0.0,
                        'confidence_interval': [float(value), float(value)]
                    }
                    for prop_name, value in prediction['properties'].items()
                },
                'has_uncertainty': False,
                'model_type': 'baseline',
                'prediction_timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Baseline prediction failed: {e}")
            return {
                'predicted_phase': 'Error',
                'phase_probabilities': [0.0] * 5,
                'phase_uncertainties': [0.0] * 5,
                'confidence': 0.0,
                'uncertainty_score': 1.0,
                'properties': {},
                'has_uncertainty': False,
                'model_type': 'error',
                'prediction_timestamp': datetime.now().isoformat()
            }
    
    def _conditions_to_features(self, conditions: Dict[str, float]) -> np.ndarray:
        """Convert synthesis conditions to model features"""
        
        features = []
        for col in self.feature_columns:
            if col in conditions:
                features.append(conditions[col])
            elif col == 'cs_pb_ratio':
                features.append(conditions['cs_br_concentration'] / conditions['pb_br2_concentration'])
            elif col == 'supersaturation':
                features.append(np.log((conditions['cs_br_concentration'] * conditions['pb_br2_concentration']) / 
                                     (0.1 + conditions['temperature'] / 1000)))
            elif col == 'ligand_ratio':
                ligand_total = conditions['oa_concentration'] + conditions['oam_concentration']
                features.append(ligand_total / (conditions['cs_br_concentration'] + conditions['pb_br2_concentration']))
            elif col == 'temp_normalized':
                features.append((conditions['temperature'] - 80) / (250 - 80))
            elif col == 'solvent_effect':
                solvent_effects = {0: 1.2, 1: 1.0, 2: 0.5, 3: 0.8, 4: 0.9}
                features.append(solvent_effects.get(conditions['solvent_type'], 1.0))
            else:
                features.append(0.0)
        
        features_array = np.array(features).reshape(1, -1)
        
        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)
        
        return features_array.flatten()
    
    def suggest_experiments(self, strategy: str = 'mixed', num_suggestions: int = 5,
                          experimental_data: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate intelligent experiment suggestions"""
        
        if self.active_learner is None:
            logger.error("Active learning not available without trained model")
            return []
        
        try:
            # Load experimental data if not provided
            if experimental_data is None:
                experimental_data = self._load_experimental_data()
            
            # Generate suggestions
            suggestions = self.active_learner.suggest_experiments(
                experimental_data=experimental_data,
                strategy=strategy,
                num_suggestions=num_suggestions
            )
            
            # Convert to serializable format
            suggestions_data = []
            for suggestion in suggestions:
                suggestion_dict = {
                    'id': suggestion.suggestion_id,
                    'conditions': suggestion.synthesis_conditions,
                    'predicted_outcome': suggestion.predicted_outcome,
                    'uncertainty_metrics': suggestion.uncertainty_metrics,
                    'rationale': suggestion.rationale,
                    'priority': suggestion.priority,
                    'acquisition_score': suggestion.acquisition_score,
                    'information_gain': suggestion.estimated_information_gain,
                    'strategy': suggestion.acquisition_type,
                    'timestamp': suggestion.timestamp
                }
                suggestions_data.append(suggestion_dict)
            
            logger.info(f"Generated {len(suggestions_data)} experiment suggestions using {strategy} strategy")
            return suggestions_data
        
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
    
    def _load_experimental_data(self) -> Optional[pd.DataFrame]:
        """Load available experimental data"""
        
        try:
            if self.validator.results_file.exists():
                df = pd.read_csv(self.validator.results_file)
                logger.info(f"Loaded {len(df)} experimental results")
                return df
            else:
                logger.info("No experimental data available")
                return None
        except Exception as e:
            logger.error(f"Error loading experimental data: {e}")
            return None
    
    def optimize_synthesis(self, objective: str = 'cspbbr3_probability', 
                          method: str = 'bayesian') -> Dict[str, Any]:
        """Optimize synthesis conditions for given objective"""
        
        if self.active_learner is None:
            logger.error("Optimization not available without trained model")
            return {}
        
        try:
            experimental_data = self._load_experimental_data()
            
            if method == 'bayesian' and experimental_data is not None and len(experimental_data) >= 3:
                # Bayesian optimization
                self.active_learner.bayesian_optimizer.objective = objective
                self.active_learner.bayesian_optimizer.fit_surrogate_model(experimental_data)
                
                optimal_conditions_list = self.active_learner.bayesian_optimizer.suggest_experiments(
                    num_suggestions=1, acquisition_type='EI'
                )
                
                if optimal_conditions_list:
                    optimal_conditions = {
                        'cs_br_concentration': float(optimal_conditions_list[0][0]),
                        'pb_br2_concentration': float(optimal_conditions_list[0][1]),
                        'temperature': float(optimal_conditions_list[0][2]),
                        'oa_concentration': float(optimal_conditions_list[0][3]),
                        'oam_concentration': float(optimal_conditions_list[0][4]),
                        'reaction_time': float(optimal_conditions_list[0][5]),
                        'solvent_type': int(optimal_conditions_list[0][6])
                    }
                    
                    # Get prediction for optimal conditions
                    prediction = self.predict_with_uncertainty(optimal_conditions)
                    
                    return {
                        'success': True,
                        'method': 'Bayesian Optimization',
                        'objective': objective,
                        'optimal_conditions': optimal_conditions,
                        'prediction': prediction,
                        'optimization_timestamp': datetime.now().isoformat()
                    }
                else:
                    raise ValueError("Bayesian optimization failed")
            
            else:
                # Grid search fallback
                optimal_conditions, score = self._grid_search_optimization(objective)
                prediction = self.predict_with_uncertainty(optimal_conditions)
                
                return {
                    'success': True,
                    'method': 'Grid Search',
                    'objective': objective,
                    'optimal_conditions': optimal_conditions,
                    'prediction': prediction,
                    'score': score,
                    'optimization_timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _grid_search_optimization(self, objective: str) -> Tuple[Dict[str, float], float]:
        """Simple grid search optimization"""
        
        # Define parameter grids
        cs_conc_range = np.linspace(0.8, 1.5, 4)
        pb_conc_range = np.linspace(0.8, 1.5, 4)
        temp_range = np.linspace(160, 220, 3)
        
        best_score = -float('inf')
        best_conditions = None
        
        for cs_conc in cs_conc_range:
            for pb_conc in pb_conc_range:
                for temp in temp_range:
                    conditions = {
                        'cs_br_concentration': cs_conc,
                        'pb_br2_concentration': pb_conc,
                        'temperature': temp,
                        'oa_concentration': 0.4,
                        'oam_concentration': 0.2,
                        'reaction_time': 60,
                        'solvent_type': 0
                    }
                    
                    prediction = self.predict_with_uncertainty(conditions)
                    
                    # Calculate score based on objective
                    if objective == 'cspbbr3_probability':
                        if prediction['predicted_phase'] == 'CsPbBr3_3D':
                            score = prediction['confidence']
                        else:
                            score = 0.1
                    elif objective == 'confidence':
                        score = prediction['confidence']
                    elif objective == 'uncertainty':
                        score = -prediction['uncertainty_score']  # Minimize uncertainty
                    else:
                        score = prediction['confidence']
                    
                    if score > best_score:
                        best_score = score
                        best_conditions = conditions
        
        return best_conditions, best_score
    
    def record_experiment(self, conditions: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Record experimental results in validation system"""
        
        if self.validation_pipeline is None:
            logger.error("Validation system not available")
            return {'success': False, 'error': 'Validation system not available'}
        
        try:
            # Setup experiment
            exp_id = self.validation_pipeline.setup_new_experiment(conditions)
            
            # Record results
            success = self.validation_pipeline.record_experiment_results(exp_id, results)
            
            if success:
                return {
                    'success': True,
                    'experiment_id': exp_id,
                    'message': 'Experiment recorded successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to record experiment'
                }
        
        except Exception as e:
            logger.error(f"Error recording experiment: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get validation system status"""
        
        if self.validation_pipeline is None:
            return {'status': 'not_available'}
        
        try:
            status = self.validation_pipeline.get_pipeline_status()
            
            # Add validation metrics if available
            if self.validator.validation_file.exists():
                report = self.validator.generate_validation_report()
                status['validation_metrics'] = {
                    'total_experiments': report.get('total_experiments', 0),
                    'phase_accuracy': report.get('phase_prediction_accuracy_percent', 0),
                    'mean_confidence': report.get('confidence_analysis', {}).get('mean_confidence', 0)
                }
            else:
                status['validation_metrics'] = {
                    'total_experiments': 0,
                    'phase_accuracy': 0,
                    'mean_confidence': 0
                }
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting validation status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def generate_synthesis_protocol(self, conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed synthesis protocol with predictions"""
        
        # Get prediction
        prediction = self.predict_with_uncertainty(conditions)
        
        # Generate protocol steps
        protocol_steps = [
            {
                'step': 1,
                'title': 'Preparation',
                'description': f'Heat reaction vessel to {conditions["temperature"]}°C under inert atmosphere',
                'duration': '15-20 minutes',
                'critical_parameters': ['temperature', 'atmosphere']
            },
            {
                'step': 2,
                'title': 'Stock Solution Preparation',
                'description': f'Prepare {conditions["cs_br_concentration"]} mol/L CsBr and {conditions["pb_br2_concentration"]} mol/L PbBr₂ solutions',
                'duration': '10-15 minutes',
                'critical_parameters': ['concentration_accuracy', 'solvent_purity']
            },
            {
                'step': 3,
                'title': 'Injection',
                'description': 'Rapidly inject precursor solutions maintaining stoichiometry',
                'duration': '30 seconds',
                'critical_parameters': ['injection_speed', 'mixing']
            },
            {
                'step': 4,
                'title': 'Ligand Addition',
                'description': f'Add {conditions["oa_concentration"]} mol/L oleic acid and {conditions["oam_concentration"]} mol/L oleylamine',
                'duration': '1-2 minutes',
                'critical_parameters': ['ligand_quality', 'addition_order']
            },
            {
                'step': 5,
                'title': 'Synthesis',
                'description': f'Maintain {conditions["temperature"]}°C for {conditions["reaction_time"]} minutes',
                'duration': f'{conditions["reaction_time"]} minutes',
                'critical_parameters': ['temperature_stability', 'timing']
            },
            {
                'step': 6,
                'title': 'Cooling and Purification',
                'description': 'Cool to room temperature and purify via centrifugation',
                'duration': '30-60 minutes',
                'critical_parameters': ['cooling_rate', 'purification_efficiency']
            }
        ]
        
        # Quality indicators
        quality_indicators = [
            {
                'indicator': 'Solution Color',
                'expected': 'Bright green' if prediction['predicted_phase'] == 'CsPbBr3_3D' else 'Yellow-green',
                'critical': True
            },
            {
                'indicator': 'Photoluminescence',
                'expected': f'{prediction["properties"].get("emission_peak", {}).get("mean", 520):.0f} nm emission',
                'critical': True
            },
            {
                'indicator': 'Phase Purity',
                'expected': f'{prediction["confidence"]*100:.1f}% confidence',
                'critical': True
            }
        ]
        
        # Troubleshooting guide
        troubleshooting = [
            {
                'problem': 'Weak or no photoluminescence',
                'possible_causes': ['Low temperature', 'Impure precursors', 'Excess ligands'],
                'solutions': ['Increase temperature by 10-20°C', 'Use fresh precursors', 'Reduce ligand concentrations']
            },
            {
                'problem': 'Wrong emission color',
                'possible_causes': ['Incorrect stoichiometry', 'Wrong phase formation'],
                'solutions': ['Check Cs:Pb ratio', 'Optimize reaction time', 'Adjust temperature']
            }
        ]
        
        return {
            'conditions': conditions,
            'prediction': prediction,
            'protocol_steps': protocol_steps,
            'quality_indicators': quality_indicators,
            'troubleshooting': troubleshooting,
            'estimated_duration': f'{sum([20, 15, 1, 2, conditions["reaction_time"], 45])} minutes',
            'difficulty_level': 'Intermediate',
            'success_probability': prediction['confidence'],
            'protocol_generated': datetime.now().isoformat()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and capabilities"""
        
        return {
            'system_name': 'Enhanced CsPbBr₃ Digital Twin',
            'version': '2.0',
            'capabilities': {
                'uncertainty_quantification': self.uncertainty_model is not None,
                'active_learning': self.active_learner is not None,
                'experimental_validation': self.validation_pipeline is not None,
                'bayesian_optimization': True,
                'synthesis_protocols': True
            },
            'model_info': {
                'uncertainty_model_type': self.model_type,
                'has_baseline': self.baseline_model is not None,
                'feature_count': len(self.feature_columns) if self.feature_columns else 0
            },
            'data_info': {
                'experimental_data_available': self._load_experimental_data() is not None,
                'validation_data_count': len(self._load_experimental_data()) if self._load_experimental_data() is not None else 0
            },
            'system_status': 'operational' if (self.uncertainty_model or self.baseline_model) else 'limited',
            'last_updated': datetime.now().isoformat()
        }

def main():
    """Main function for command-line interface"""
    
    parser = argparse.ArgumentParser(description="Enhanced CsPbBr₃ Digital Twin System")
    parser.add_argument('command', choices=['predict', 'suggest', 'optimize', 'status', 'protocol'],
                       help='Command to execute')
    parser.add_argument('--cs-concentration', type=float, default=1.1,
                       help='Cs-Br concentration (mol/L)')
    parser.add_argument('--pb-concentration', type=float, default=1.1,
                       help='Pb-Br₂ concentration (mol/L)')
    parser.add_argument('--temperature', type=float, default=190,
                       help='Temperature (°C)')
    parser.add_argument('--oa-concentration', type=float, default=0.4,
                       help='Oleic acid concentration (mol/L)')
    parser.add_argument('--oam-concentration', type=float, default=0.2,
                       help='Oleylamine concentration (mol/L)')
    parser.add_argument('--reaction-time', type=float, default=75,
                       help='Reaction time (minutes)')
    parser.add_argument('--solvent-type', type=int, default=0,
                       help='Solvent type (0-4)')
    parser.add_argument('--strategy', type=str, default='mixed',
                       choices=['uncertainty', 'bayesian', 'diversity', 'mixed'],
                       help='Active learning strategy')
    parser.add_argument('--num-suggestions', type=int, default=3,
                       help='Number of experiment suggestions')
    parser.add_argument('--objective', type=str, default='cspbbr3_probability',
                       choices=['cspbbr3_probability', 'confidence', 'uncertainty'],
                       help='Optimization objective')
    
    args = parser.parse_args()
    
    print("🧬 Enhanced CsPbBr₃ Digital Twin System")
    print("=" * 50)
    
    # Initialize system
    twin = EnhancedDigitalTwin()
    
    # Execute command
    if args.command == 'predict':
        conditions = {
            'cs_br_concentration': args.cs_concentration,
            'pb_br2_concentration': args.pb_concentration,
            'temperature': args.temperature,
            'oa_concentration': args.oa_concentration,
            'oam_concentration': args.oam_concentration,
            'reaction_time': args.reaction_time,
            'solvent_type': args.solvent_type
        }
        
        prediction = twin.predict_with_uncertainty(conditions)
        print(f"\n🔮 Prediction Results:")
        print(f"   Phase: {prediction['predicted_phase']}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        if prediction['has_uncertainty']:
            print(f"   Uncertainty: {prediction['uncertainty_score']:.3f}")
        
    elif args.command == 'suggest':
        suggestions = twin.suggest_experiments(
            strategy=args.strategy, 
            num_suggestions=args.num_suggestions
        )
        
        print(f"\n🤖 Experiment Suggestions ({args.strategy} strategy):")
        for i, suggestion in enumerate(suggestions[:3]):
            print(f"   {i+1}. {suggestion['conditions']['temperature']:.0f}°C, "
                  f"Cs:Pb={suggestion['conditions']['cs_br_concentration']:.1f}:{suggestion['conditions']['pb_br2_concentration']:.1f}, "
                  f"Priority: {suggestion['priority']}")
    
    elif args.command == 'optimize':
        optimization = twin.optimize_synthesis(objective=args.objective)
        
        if optimization['success']:
            print(f"\n🎯 Optimization Results ({args.objective}):")
            conditions = optimization['optimal_conditions']
            print(f"   Temperature: {conditions['temperature']:.0f}°C")
            print(f"   Cs-Br: {conditions['cs_br_concentration']:.2f} mol/L")
            print(f"   Pb-Br₂: {conditions['pb_br2_concentration']:.2f} mol/L")
            print(f"   Predicted: {optimization['prediction']['predicted_phase']} "
                  f"(confidence: {optimization['prediction']['confidence']:.3f})")
        else:
            print(f"❌ Optimization failed: {optimization.get('error', 'Unknown error')}")
    
    elif args.command == 'status':
        info = twin.get_system_info()
        print(f"\n📊 System Status:")
        print(f"   Status: {info['system_status']}")
        print(f"   Uncertainty Model: {info['model_info']['uncertainty_model_type'] or 'None'}")
        print(f"   Validation Data: {info['data_info']['validation_data_count']} experiments")
        print(f"   Capabilities: {', '.join([k for k, v in info['capabilities'].items() if v])}")
    
    elif args.command == 'protocol':
        conditions = {
            'cs_br_concentration': args.cs_concentration,
            'pb_br2_concentration': args.pb_concentration,
            'temperature': args.temperature,
            'oa_concentration': args.oa_concentration,
            'oam_concentration': args.oam_concentration,
            'reaction_time': args.reaction_time,
            'solvent_type': args.solvent_type
        }
        
        protocol = twin.generate_synthesis_protocol(conditions)
        print(f"\n📋 Synthesis Protocol:")
        print(f"   Duration: {protocol['estimated_duration']}")
        print(f"   Success Probability: {protocol['success_probability']:.1%}")
        print(f"   Steps: {len(protocol['protocol_steps'])}")

if __name__ == "__main__":
    main()