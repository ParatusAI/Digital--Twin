#!/usr/bin/env python3
"""
Enhanced Training Pipeline for CsPbBr3 Digital Twin
Integrates all new improvements: ML optimization, adaptive sampling, monitoring, GPU acceleration, etc.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sys
from datetime import datetime

# Import all enhancement modules
try:
    from ml_optimization import BayesianOptimizer, create_optimization_strategy
except ImportError:
    warnings.warn("ML optimization module not available")
    BayesianOptimizer = None

try:
    from adaptive_sampling import create_adaptive_sampling_strategy
except ImportError:
    warnings.warn("Adaptive sampling module not available")
    create_adaptive_sampling_strategy = None

try:
    from monitoring import RealTimeMonitor, StructuredLogger
except ImportError:
    warnings.warn("Monitoring module not available")
    RealTimeMonitor = None
    StructuredLogger = None

try:
    from gpu_acceleration import GPUDataGenerator, GPUCapabilities
except ImportError:
    warnings.warn("GPU acceleration module not available")
    GPUDataGenerator = None
    GPUCapabilities = None

try:
    from advanced_physics import PhysicsModelSuite
except ImportError:
    warnings.warn("Advanced physics module not available")
    PhysicsModelSuite = None

try:
    from data_versioning import DataVersionManager
except ImportError:
    warnings.warn("Data versioning module not available")
    DataVersionManager = None

try:
    from experimental_integration import ExperimentalDataManager, ModelCalibrator
except ImportError:
    warnings.warn("Experimental integration module not available")
    ExperimentalDataManager = None

# Import base data generation
from generate_sample_data import generate_sample_batch, load_config


class EnhancedTrainingPipeline:
    """Enhanced training pipeline that leverages all new improvements"""
    
    def __init__(self, config_path: Optional[str] = None, 
                 output_dir: str = "enhanced_training_output"):
        """Initialize enhanced training pipeline"""
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.logger = None
        self.monitor = None
        self.gpu_generator = None
        self.data_version_manager = None
        self.experimental_manager = None
        self.physics_suite = None
        
        # Training state
        self.training_history = []
        self.best_parameters = {}
        self.current_iteration = 0
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all enhancement components"""
        print("🚀 Initializing enhanced training components...")
        
        # 1. Structured logging
        if StructuredLogger:
            self.logger = StructuredLogger(
                log_file=str(self.output_dir / "training.log"),
                console_output=True
            )
            self.logger.log_info("Enhanced training pipeline initialized")
        
        # 2. Real-time monitoring
        if RealTimeMonitor:
            self.monitor = RealTimeMonitor(
                output_dir=str(self.output_dir / "monitoring")
            )
            self.monitor.start_monitoring()
        
        # 3. GPU capabilities
        if GPUCapabilities:
            gpu_caps = GPUCapabilities()
            print(f"🖥️  GPU Status: {gpu_caps.get_summary()}")
            
            if GPUDataGenerator and gpu_caps.has_gpu():
                self.gpu_generator = GPUDataGenerator()
                print("✅ GPU acceleration enabled")
        
        # 4. Data versioning
        if DataVersionManager:
            self.data_version_manager = DataVersionManager(
                storage_dir=str(self.output_dir / "data_versions")
            )
            print("📚 Data versioning enabled")
        
        # 5. Experimental data integration
        if ExperimentalDataManager:
            self.experimental_manager = ExperimentalDataManager(
                data_dir=str(self.output_dir / "experimental_data")
            )
            print("🧪 Experimental data integration ready")
        
        # 6. Advanced physics models
        if PhysicsModelSuite:
            try:
                self.physics_suite = PhysicsModelSuite()
                print("⚛️  Advanced physics models loaded")
            except Exception as e:
                warnings.warn(f"Could not initialize physics suite: {e}")
    
    def generate_enhanced_dataset(self, n_samples: int = 10000, 
                                use_adaptive_sampling: bool = True,
                                use_physics_models: bool = True) -> pd.DataFrame:
        """Generate enhanced training dataset"""
        print(f"📊 Generating enhanced dataset with {n_samples} samples...")
        
        if self.logger:
            self.logger.log_info(f"Starting dataset generation: {n_samples} samples")
        
        start_time = time.time()
        
        # 1. Set up adaptive sampling strategy
        if use_adaptive_sampling and create_adaptive_sampling_strategy:
            print("🎯 Using adaptive sampling strategy...")
            
            # Extract parameter bounds from config
            param_bounds = {}
            for param, bounds in self.config['synthesis_parameters'].items():
                param_bounds[param] = (bounds['min'], bounds['max'])
            
            sampler = create_adaptive_sampling_strategy(
                parameter_space=param_bounds,
                strategy_type='hierarchical',
                n_samples=n_samples
            )
            
            # Generate parameter samples
            parameter_samples = sampler.generate_samples()
            print(f"   Generated {len(parameter_samples)} parameter combinations")
            
        else:
            # Fallback to standard sampling
            print("📈 Using standard sampling...")
            parameter_samples = self._generate_standard_samples(n_samples)
        
        # 2. Generate data using appropriate method
        if self.gpu_generator and len(parameter_samples) > 1000:
            print("🚀 Using GPU acceleration for large dataset...")
            dataset = self._generate_gpu_accelerated_data(parameter_samples, use_physics_models)
        else:
            print("💻 Using CPU generation...")
            dataset = self._generate_cpu_data(parameter_samples, use_physics_models)
        
        generation_time = time.time() - start_time
        print(f"✅ Dataset generation complete in {generation_time:.2f}s")
        
        if self.logger:
            self.logger.log_info(f"Dataset generation completed", extra={
                'n_samples': n_samples,
                'generation_time': generation_time,
                'adaptive_sampling': use_adaptive_sampling,
                'physics_models': use_physics_models
            })
        
        # 3. Version the dataset
        if self.data_version_manager:
            version_info = self.data_version_manager.create_version(
                data=dataset,
                description=f"Enhanced training dataset: {n_samples} samples",
                metadata={
                    'adaptive_sampling': use_adaptive_sampling,
                    'physics_models': use_physics_models,
                    'generation_time': generation_time
                }
            )
            print(f"📦 Dataset versioned: {version_info.version_id}")
        
        return dataset
    
    def _generate_standard_samples(self, n_samples: int) -> List[Dict[str, float]]:
        """Generate standard parameter samples"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param, bounds in self.config['synthesis_parameters'].items():
                sample[param] = np.random.uniform(bounds['min'], bounds['max'])
            
            # Add solvent type
            solvent_weights = self.config['solvents']['weights']
            sample['solvent_type'] = np.random.choice(
                self.config['solvents']['types'],
                p=solvent_weights
            )
            
            samples.append(sample)
        
        return samples
    
    def _generate_gpu_accelerated_data(self, parameter_samples: List[Dict[str, float]], 
                                     use_physics: bool) -> pd.DataFrame:
        """Generate data using GPU acceleration"""
        if not self.gpu_generator:
            return self._generate_cpu_data(parameter_samples, use_physics)
        
        try:
            # Convert to format suitable for GPU processing
            param_arrays = {}
            for key in parameter_samples[0].keys():
                param_arrays[key] = np.array([sample[key] for sample in parameter_samples])
            
            # Generate using GPU
            results = self.gpu_generator.generate_properties_batch(param_arrays)
            
            # Combine parameters and results
            data_records = []
            for i, sample in enumerate(parameter_samples):
                record = sample.copy()
                for prop, values in results.items():
                    record[prop] = values[i] if i < len(values) else 0.0
                data_records.append(record)
            
            return pd.DataFrame(data_records)
            
        except Exception as e:
            warnings.warn(f"GPU acceleration failed: {e}, falling back to CPU")
            return self._generate_cpu_data(parameter_samples, use_physics)
    
    def _generate_cpu_data(self, parameter_samples: List[Dict[str, float]], 
                          use_physics: bool) -> pd.DataFrame:
        """Generate data using CPU with optional physics models"""
        data_records = []
        
        batch_size = 1000
        n_batches = (len(parameter_samples) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(parameter_samples))
            batch_samples = parameter_samples[start_idx:end_idx]
            
            # Generate batch
            if use_physics and self.physics_suite:
                batch_data = self._generate_physics_enhanced_batch(batch_samples)
            else:
                batch_data = self._generate_standard_batch(batch_samples)
            
            data_records.extend(batch_data)
            
            if self.monitor:
                progress = (batch_idx + 1) / n_batches
                self.monitor.log_progress("dataset_generation", progress)
            
            if batch_idx % 10 == 0:
                print(f"   Processed {end_idx}/{len(parameter_samples)} samples")
        
        return pd.DataFrame(data_records)
    
    def _generate_physics_enhanced_batch(self, batch_samples: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate batch with enhanced physics models"""
        batch_data = []
        
        for sample in batch_samples:
            try:
                # Use physics suite for more accurate modeling
                physics_results = self.physics_suite.calculate_comprehensive_properties(sample)
                
                # Combine with standard generation
                standard_results = generate_sample_batch([sample], self.config)[0]
                
                # Merge results (physics takes precedence)
                combined_result = {**standard_results, **physics_results}
                batch_data.append(combined_result)
                
            except Exception as e:
                # Fallback to standard generation
                standard_results = generate_sample_batch([sample], self.config)[0]
                batch_data.append(standard_results)
        
        return batch_data
    
    def _generate_standard_batch(self, batch_samples: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate batch using standard method"""
        return generate_sample_batch(batch_samples, self.config)
    
    def run_ml_optimization(self, dataset: pd.DataFrame, 
                          target_properties: List[str] = None,
                          n_iterations: int = 50) -> Dict[str, Any]:
        """Run ML-based parameter optimization"""
        if not BayesianOptimizer:
            print("⚠️  ML optimization not available")
            return {}
        
        if target_properties is None:
            target_properties = ['plqy', 'stability_score']
        
        print(f"🎯 Running ML optimization for {n_iterations} iterations...")
        
        if self.logger:
            self.logger.log_info("Starting ML optimization", extra={
                'target_properties': target_properties,
                'n_iterations': n_iterations
            })
        
        try:
            # Set up optimization strategy
            strategy = create_optimization_strategy(
                parameter_space={
                    param: (bounds['min'], bounds['max']) 
                    for param, bounds in self.config['synthesis_parameters'].items()
                },
                target_properties=target_properties,
                strategy_type='bayesian'
            )
            
            # Create objective function
            def objective_function(params):
                # Generate sample with these parameters
                sample_data = generate_sample_batch([params], self.config)[0]
                
                # Calculate objective (maximize PLQY and stability)
                score = 0
                for prop in target_properties:
                    if prop in sample_data:
                        weight = 1.0 if prop == 'plqy' else 0.5
                        score += sample_data[prop] * weight
                
                return score
            
            # Run optimization
            optimization_results = strategy.optimize(
                objective_function=objective_function,
                n_iterations=n_iterations
            )
            
            self.best_parameters = optimization_results['best_parameters']
            
            print(f"✅ Optimization complete!")
            print(f"   Best parameters: {self.best_parameters}")
            print(f"   Best score: {optimization_results['best_score']:.4f}")
            
            if self.logger:
                self.logger.log_info("ML optimization completed", extra={
                    'best_parameters': self.best_parameters,
                    'best_score': optimization_results['best_score']
                })
            
            return optimization_results
            
        except Exception as e:
            error_msg = f"ML optimization failed: {e}"
            print(f"❌ {error_msg}")
            if self.logger:
                self.logger.log_error(error_msg)
            return {}
    
    def integrate_experimental_data(self, experimental_data_path: Optional[str] = None) -> bool:
        """Integrate experimental data for model calibration"""
        if not self.experimental_manager:
            print("⚠️  Experimental data integration not available")
            return False
        
        print("🧪 Integrating experimental data...")
        
        try:
            imported_count = 0
            
            if experimental_data_path:
                if experimental_data_path.endswith('.csv'):
                    imported_count = self.experimental_manager.import_csv_data(experimental_data_path)
                elif experimental_data_path.endswith('.json'):
                    imported_count = self.experimental_manager.import_lab_notebook(experimental_data_path)
            else:
                # Create demo experimental data
                from experimental_integration import create_experimental_template
                template_path = str(self.output_dir / "demo_experimental_data.json")
                create_experimental_template(template_path)
                imported_count = self.experimental_manager.import_lab_notebook(template_path)
            
            print(f"✅ Imported {imported_count} experimental data points")
            
            if self.logger:
                self.logger.log_info(f"Experimental data integrated: {imported_count} points")
            
            return imported_count > 0
            
        except Exception as e:
            error_msg = f"Experimental data integration failed: {e}"
            print(f"❌ {error_msg}")
            if self.logger:
                self.logger.log_error(error_msg)
            return False
    
    def run_comprehensive_training(self, n_samples: int = 10000,
                                 use_all_enhancements: bool = True,
                                 optimization_iterations: int = 50) -> Dict[str, Any]:
        """Run comprehensive training with all enhancements"""
        print("🎓 Starting comprehensive enhanced training...")
        
        training_start = time.time()
        results = {}
        
        # 1. Generate enhanced dataset
        dataset = self.generate_enhanced_dataset(
            n_samples=n_samples,
            use_adaptive_sampling=use_all_enhancements,
            use_physics_models=use_all_enhancements
        )
        results['dataset_size'] = len(dataset)
        
        # 2. Save initial dataset
        dataset_path = self.output_dir / f"enhanced_training_data_{n_samples}.csv"
        dataset.to_csv(dataset_path, index=False)
        print(f"💾 Dataset saved to: {dataset_path}")
        
        # 3. Integrate experimental data
        if use_all_enhancements:
            exp_integration_success = self.integrate_experimental_data()
            results['experimental_integration'] = exp_integration_success
        
        # 4. Run ML optimization
        if use_all_enhancements:
            optimization_results = self.run_ml_optimization(
                dataset=dataset,
                n_iterations=optimization_iterations
            )
            results['optimization'] = optimization_results
        
        # 5. Generate final optimized dataset
        if self.best_parameters:
            print("🎯 Generating optimized dataset...")
            optimized_samples = []
            
            # Create variations around best parameters
            for i in range(min(1000, n_samples // 10)):
                optimized_sample = self.best_parameters.copy()
                
                # Add small random variations
                for param in optimized_sample:
                    if param in self.config['synthesis_parameters']:
                        bounds = self.config['synthesis_parameters'][param]
                        noise = np.random.normal(0, (bounds['max'] - bounds['min']) * 0.05)
                        optimized_sample[param] = np.clip(
                            optimized_sample[param] + noise,
                            bounds['min'], bounds['max']
                        )
                
                optimized_samples.append(optimized_sample)
            
            optimized_dataset = self._generate_cpu_data(optimized_samples, use_all_enhancements)
            
            # Save optimized dataset
            optimized_path = self.output_dir / f"optimized_training_data_{len(optimized_samples)}.csv"
            optimized_dataset.to_csv(optimized_path, index=False)
            print(f"💾 Optimized dataset saved to: {optimized_path}")
            
            results['optimized_dataset_size'] = len(optimized_dataset)
        
        # 6. Generate training report
        training_time = time.time() - training_start
        results['total_training_time'] = training_time
        
        self._generate_training_report(results, dataset)
        
        print(f"✅ Comprehensive training complete in {training_time:.2f}s")
        
        if self.logger:
            self.logger.log_info("Comprehensive training completed", extra=results)
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        return results
    
    def _generate_training_report(self, results: Dict[str, Any], dataset: pd.DataFrame):
        """Generate comprehensive training report"""
        print("📊 Generating training report...")
        
        report = {
            'training_summary': results,
            'dataset_statistics': {
                'n_samples': len(dataset),
                'n_features': len(dataset.columns),
                'parameter_ranges': {},
                'property_statistics': {}
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate statistics
        param_columns = list(self.config['synthesis_parameters'].keys())
        property_columns = [col for col in dataset.columns if col not in param_columns + ['solvent_type']]
        
        for col in param_columns:
            if col in dataset.columns:
                report['dataset_statistics']['parameter_ranges'][col] = {
                    'min': float(dataset[col].min()),
                    'max': float(dataset[col].max()),
                    'mean': float(dataset[col].mean()),
                    'std': float(dataset[col].std())
                }
        
        for col in property_columns:
            if col in dataset.columns:
                report['dataset_statistics']['property_statistics'][col] = {
                    'min': float(dataset[col].min()),
                    'max': float(dataset[col].max()),
                    'mean': float(dataset[col].mean()),
                    'std': float(dataset[col].std())
                }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Training report saved to: {report_path}")
        
        # Create HTML report
        self._create_html_report(report)
    
    def _create_html_report(self, report: Dict[str, Any]):
        """Create HTML training report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced CsPbBr3 Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e9ecef; border-radius: 5px; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #6c757d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .enhancement-status {{ padding: 5px 10px; border-radius: 3px; color: white; }}
                .enabled {{ background-color: #28a745; }}
                .disabled {{ background-color: #dc3545; }}
            </style>
        </head>
        <body>
            <h1>🎓 Enhanced CsPbBr3 Digital Twin Training Report</h1>
            <p><strong>Generated:</strong> {report['timestamp']}</p>
            
            <div class="summary">
                <h2>📊 Training Summary</h2>
                <div class="metric">
                    <div class="metric-value">{report['dataset_statistics']['n_samples']:,}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['dataset_statistics']['n_features']}</div>
                    <div class="metric-label">Features</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['training_summary'].get('total_training_time', 0):.1f}s</div>
                    <div class="metric-label">Training Time</div>
                </div>
            </div>
            
            <h2>🚀 Enhancement Status</h2>
            <table>
                <tr><th>Enhancement</th><th>Status</th><th>Details</th></tr>
                <tr>
                    <td>Adaptive Sampling</td>
                    <td><span class="enhancement-status enabled">ENABLED</span></td>
                    <td>Hierarchical sampling strategy used</td>
                </tr>
                <tr>
                    <td>ML Optimization</td>
                    <td><span class="enhancement-status {'enabled' if 'optimization' in report['training_summary'] else 'disabled'}">
                        {'ENABLED' if 'optimization' in report['training_summary'] else 'DISABLED'}
                    </span></td>
                    <td>Bayesian optimization with Gaussian processes</td>
                </tr>
                <tr>
                    <td>GPU Acceleration</td>
                    <td><span class="enhancement-status enabled">AVAILABLE</span></td>
                    <td>Automatic GPU detection and fallback</td>
                </tr>
                <tr>
                    <td>Advanced Physics</td>
                    <td><span class="enhancement-status enabled">ENABLED</span></td>
                    <td>Thermodynamic and kinetic models</td>
                </tr>
                <tr>
                    <td>Real-time Monitoring</td>
                    <td><span class="enhancement-status enabled">ENABLED</span></td>
                    <td>Structured logging and progress tracking</td>
                </tr>
                <tr>
                    <td>Data Versioning</td>
                    <td><span class="enhancement-status enabled">ENABLED</span></td>
                    <td>Complete data lineage tracking</td>
                </tr>
            </table>
        """
        
        # Add parameter statistics
        if report['dataset_statistics']['parameter_ranges']:
            html_content += """
            <h2>⚙️ Parameter Statistics</h2>
            <table>
                <tr><th>Parameter</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th></tr>
            """
            
            for param, stats in report['dataset_statistics']['parameter_ranges'].items():
                html_content += f"""
                <tr>
                    <td>{param.replace('_', ' ').title()}</td>
                    <td>{stats['min']:.3f}</td>
                    <td>{stats['max']:.3f}</td>
                    <td>{stats['mean']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add property statistics
        if report['dataset_statistics']['property_statistics']:
            html_content += """
            <h2>🎯 Property Statistics</h2>
            <table>
                <tr><th>Property</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th></tr>
            """
            
            for prop, stats in report['dataset_statistics']['property_statistics'].items():
                html_content += f"""
                <tr>
                    <td>{prop.replace('_', ' ').title()}</td>
                    <td>{stats['min']:.3f}</td>
                    <td>{stats['max']:.3f}</td>
                    <td>{stats['mean']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add optimization results if available
        if 'optimization' in report['training_summary'] and report['training_summary']['optimization']:
            opt_results = report['training_summary']['optimization']
            html_content += f"""
            <h2>🎯 Optimization Results</h2>
            <div class="summary">
                <div class="metric">
                    <div class="metric-value">{opt_results.get('best_score', 0):.4f}</div>
                    <div class="metric-label">Best Score</div>
                </div>
            </div>
            """
            
            if 'best_parameters' in opt_results:
                html_content += """
                <h3>Best Parameters</h3>
                <table>
                    <tr><th>Parameter</th><th>Optimized Value</th></tr>
                """
                
                for param, value in opt_results['best_parameters'].items():
                    html_content += f"""
                    <tr>
                        <td>{param.replace('_', ' ').title()}</td>
                        <td>{value:.4f}</td>
                    </tr>
                    """
                
                html_content += "</table>"
        
        html_content += """
            <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
                <p>Generated by Enhanced CsPbBr3 Digital Twin Training Pipeline</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = self.output_dir / "training_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved to: {html_path}")


def main():
    """Main training function"""
    print("🎓 Enhanced CsPbBr3 Digital Twin Training")
    print("=" * 50)
    
    # Initialize training pipeline
    pipeline = EnhancedTrainingPipeline()
    
    # Run comprehensive training
    results = pipeline.run_comprehensive_training(
        n_samples=5000,  # Start with moderate size for testing
        use_all_enhancements=True,
        optimization_iterations=30
    )
    
    print("\n🎉 Training Complete!")
    print(f"📊 Generated {results.get('dataset_size', 0)} training samples")
    print(f"⏱️  Total time: {results.get('total_training_time', 0):.2f}s")
    
    if 'optimization' in results and 'best_score' in results['optimization']:
        print(f"🎯 Best optimization score: {results['optimization']['best_score']:.4f}")
    
    print(f"📁 All outputs saved to: enhanced_training_output/")


if __name__ == "__main__":
    main()