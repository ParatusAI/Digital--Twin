#!/usr/bin/env python3
"""
Working Enhanced Training Pipeline for CsPbBr3 Digital Twin
Simplified version that works with available dependencies
"""

import json
import time
import random
import math
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class WorkingEnhancedTrainer:
    """Simplified enhanced training pipeline that actually works"""
    
    def __init__(self, output_dir: str = "working_training_output"):
        """Initialize the working trainer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = self._create_default_config()
        self.training_history = []
        self.best_parameters = {}
        
        print("🚀 Working Enhanced Training Pipeline Initialized")
    
    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        return {
            'synthesis_parameters': {
                'cs_br_concentration': {'min': 0.1, 'max': 3.0},
                'pb_br2_concentration': {'min': 0.1, 'max': 2.0},
                'temperature': {'min': 80.0, 'max': 250.0},
                'oa_concentration': {'min': 0.0, 'max': 1.5},
                'oam_concentration': {'min': 0.0, 'max': 1.5},
                'reaction_time': {'min': 1.0, 'max': 60.0}
            },
            'solvents': {
                'types': [0, 1, 2, 3, 4],
                'weights': [0.3, 0.25, 0.15, 0.2, 0.1]
            },
            'target_properties': ['plqy', 'stability_score', 'bandgap', 'particle_size'],
            'quality_thresholds': {
                'min_plqy': 0.1,
                'min_stability': 0.5,
                'min_bandgap': 1.5,
                'max_bandgap': 3.5
            }
        }
    
    def generate_adaptive_samples(self, n_samples: int, use_adaptive: bool = True) -> List[Dict[str, float]]:
        """Generate parameter samples using adaptive strategy"""
        print(f"📊 Generating {n_samples} samples (adaptive: {use_adaptive})...")
        
        samples = []
        
        if use_adaptive and len(self.training_history) > 10:
            # Use historical data to guide sampling
            print("   Using adaptive sampling based on training history...")
            
            # Analyze best performing regions
            good_samples = [h for h in self.training_history if h.get('score', 0) > 0.8]
            
            if good_samples:
                # Sample around good regions with variations
                for i in range(n_samples):
                    if i < len(good_samples) and random.random() < 0.7:
                        # Sample near a good sample
                        base_sample = random.choice(good_samples)
                        sample = self._create_variation(base_sample)
                    else:
                        # Random sample
                        sample = self._generate_random_sample()
                    
                    samples.append(sample)
            else:
                # Fall back to random sampling
                samples = [self._generate_random_sample() for _ in range(n_samples)]
        else:
            # Pure random sampling
            print("   Using random sampling...")
            samples = [self._generate_random_sample() for _ in range(n_samples)]
        
        return samples
    
    def _create_variation(self, base_sample: Dict[str, float], noise_level: float = 0.1) -> Dict[str, float]:
        """Create a variation of a base sample"""
        sample = {}
        
        for param, bounds in self.config['synthesis_parameters'].items():
            if param in base_sample:
                base_value = base_sample[param]
                range_size = bounds['max'] - bounds['min']
                noise = random.gauss(0, range_size * noise_level)
                new_value = base_value + noise
                new_value = max(bounds['min'], min(bounds['max'], new_value))
                sample[param] = new_value
            else:
                sample[param] = random.uniform(bounds['min'], bounds['max'])
        
        # Add solvent type
        sample['solvent_type'] = random.choices(
            self.config['solvents']['types'],
            weights=self.config['solvents']['weights']
        )[0]
        
        return sample
    
    def _generate_random_sample(self) -> Dict[str, float]:
        """Generate a random parameter sample"""
        sample = {}
        
        for param, bounds in self.config['synthesis_parameters'].items():
            sample[param] = random.uniform(bounds['min'], bounds['max'])
        
        # Add solvent type
        sample['solvent_type'] = random.choices(
            self.config['solvents']['types'],
            weights=self.config['solvents']['weights']
        )[0]
        
        return sample
    
    def simulate_physics_enhanced(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Enhanced physics simulation"""
        # Extract parameters
        cs_conc = parameters.get('cs_br_concentration', 1.0)
        pb_conc = parameters.get('pb_br2_concentration', 1.0)
        temp = parameters.get('temperature', 150.0)
        oa_conc = parameters.get('oa_concentration', 0.1)
        oam_conc = parameters.get('oam_concentration', 0.1)
        time_min = parameters.get('reaction_time', 10.0)
        solvent = parameters.get('solvent_type', 0)
        
        # Advanced physics calculations
        
        # 1. Temperature effects on thermodynamics
        temp_normalized = (temp - 150) / 100
        activation_energy = 50000  # J/mol
        boltzmann_factor = math.exp(-activation_energy / (8.314 * (temp + 273.15)))
        
        # 2. Concentration ratio and stoichiometry effects
        conc_ratio = cs_conc / pb_conc if pb_conc > 0 else 1.0
        stoich_deviation = abs(conc_ratio - 1.0)
        
        # 3. Ligand coordination effects
        total_ligand = oa_conc + oam_conc
        ligand_surface_coverage = min(1.0, total_ligand / 0.5)  # Saturation at 0.5M total
        
        # 4. Solvent effects
        solvent_polarity_effects = [1.0, 0.95, 0.9, 0.85, 0.8][solvent]
        
        # 5. Time-dependent growth kinetics
        growth_rate = 0.1 * boltzmann_factor * (1 + ligand_surface_coverage)
        final_size = 5 + growth_rate * math.sqrt(time_min)
        
        # Calculate properties with enhanced physics
        
        # Bandgap with quantum confinement
        bulk_bandgap = 2.3
        quantum_confinement = 1.8 / (final_size ** 2) if final_size > 0 else 0
        bandgap = bulk_bandgap + quantum_confinement + temp_normalized * 0.1
        bandgap += random.gauss(0, 0.05)
        bandgap = max(1.5, min(3.5, bandgap))
        
        # PLQY with defect considerations
        base_plqy = 0.9 * solvent_polarity_effects
        defect_concentration = stoich_deviation * 0.3 + (1 - ligand_surface_coverage) * 0.2
        plqy = base_plqy * math.exp(-defect_concentration * 3)
        plqy += random.gauss(0, 0.03)
        plqy = max(0.0, min(1.0, plqy))
        
        # Particle size with nucleation-growth competition
        nucleation_rate = cs_conc * pb_conc * boltzmann_factor
        particle_size = final_size * (1 + random.gauss(0, 0.1))
        particle_size = max(1.0, particle_size)
        
        # Emission properties
        emission_peak = 1240 / bandgap + random.gauss(0, 3)
        emission_fwhm = 20 + 200 / particle_size + random.gauss(0, 2)
        emission_fwhm = max(10, emission_fwhm)
        
        # Lifetime with surface effects
        intrinsic_lifetime = 20  # ns
        surface_recombination = 1 / (1 + ligand_surface_coverage * 10)
        lifetime = intrinsic_lifetime * (1 - surface_recombination) + random.gauss(0, 1)
        lifetime = max(1.0, lifetime)
        
        # Stability with ligand protection
        thermal_stability = math.exp(-abs(temp_normalized) * 0.5)
        ligand_protection = ligand_surface_coverage
        stability_score = 0.5 + 0.4 * thermal_stability + 0.3 * ligand_protection
        stability_score += random.gauss(0, 0.05)
        stability_score = max(0.0, min(1.0, stability_score))
        
        # Phase purity with stoichiometry
        phase_purity = 1.0 - stoich_deviation * 0.4 + random.gauss(0, 0.03)
        phase_purity = max(0.5, min(1.0, phase_purity))
        
        return {
            **parameters,
            'bandgap': round(bandgap, 3),
            'plqy': round(plqy, 3),
            'emission_peak': round(emission_peak, 1),
            'particle_size': round(particle_size, 1),
            'emission_fwhm': round(emission_fwhm, 1),
            'lifetime': round(lifetime, 1),
            'stability_score': round(stability_score, 3),
            'phase_purity': round(phase_purity, 3)
        }
    
    def run_ml_optimization(self, n_iterations: int = 50) -> Dict[str, Any]:
        """Run simplified ML optimization"""
        print(f"🎯 Running ML optimization for {n_iterations} iterations...")
        
        best_score = -1
        best_params = {}
        optimization_history = []
        
        for iteration in range(n_iterations):
            # Generate candidate parameters
            if iteration < 10:
                # Initial random exploration
                params = self._generate_random_sample()
            else:
                # Exploitation around best regions
                if best_params:
                    params = self._create_variation(best_params, noise_level=0.2)
                else:
                    params = self._generate_random_sample()
            
            # Evaluate candidate
            results = self.simulate_physics_enhanced(params)
            
            # Calculate objective score (multi-objective)
            plqy_score = results['plqy']
            stability_score = results['stability_score']
            bandgap_score = 1.0 - abs(results['bandgap'] - 2.3) / 1.0  # Target 2.3 eV
            
            total_score = 0.5 * plqy_score + 0.3 * stability_score + 0.2 * bandgap_score
            
            # Update best
            if total_score > best_score:
                best_score = total_score
                best_params = params.copy()
                print(f"   New best at iteration {iteration}: score = {best_score:.4f}")
            
            optimization_history.append({
                'iteration': iteration,
                'parameters': params,
                'results': results,
                'score': total_score
            })
            
            if iteration % 10 == 0:
                print(f"   Iteration {iteration}/{n_iterations}, best score: {best_score:.4f}")
        
        self.best_parameters = best_params
        print(f"✅ Optimization complete! Best score: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_history': optimization_history
        }
    
    def generate_enhanced_dataset(self, n_samples: int, use_all_enhancements: bool = True) -> List[Dict[str, Any]]:
        """Generate enhanced dataset with all improvements"""
        print(f"📊 Generating enhanced dataset with {n_samples} samples...")
        
        start_time = time.time()
        
        # 1. Generate adaptive samples
        parameter_samples = self.generate_adaptive_samples(n_samples, use_adaptive=use_all_enhancements)
        
        # 2. Simulate with enhanced physics
        dataset = []
        for i, params in enumerate(parameter_samples):
            try:
                if use_all_enhancements:
                    results = self.simulate_physics_enhanced(params)
                else:
                    results = self._simulate_basic(params)
                
                dataset.append(results)
                
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{n_samples} samples")
                    
            except Exception as e:
                print(f"   Error processing sample {i}: {e}")
                continue
        
        generation_time = time.time() - start_time
        print(f"✅ Dataset generation complete in {generation_time:.2f}s")
        
        return dataset
    
    def _simulate_basic(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Basic simulation for comparison"""
        # Simple version from our working test
        cs_conc = parameters.get('cs_br_concentration', 1.0)
        pb_conc = parameters.get('pb_br2_concentration', 1.0)
        temp = parameters.get('temperature', 150.0)
        oa_conc = parameters.get('oa_concentration', 0.1)
        oam_conc = parameters.get('oam_concentration', 0.1)
        
        temp_normalized = (temp - 150) / 100
        conc_ratio = cs_conc / pb_conc if pb_conc > 0 else 1.0
        
        bandgap = 2.3 + temp_normalized * 0.2 + random.gauss(0, 0.1)
        bandgap = max(1.5, min(3.5, bandgap))
        
        plqy = 0.8 * math.exp(-abs(conc_ratio - 1.0)) + random.gauss(0, 0.05)
        plqy = max(0.0, min(1.0, plqy))
        
        ligand_effect = (oa_conc + oam_conc) / 2.0
        particle_size = 10 + ligand_effect * 5 + temp_normalized * 3 + random.gauss(0, 1)
        particle_size = max(1, particle_size)
        
        stability_score = 0.9 - abs(temp_normalized) * 0.2 + ligand_effect * 0.1
        stability_score += random.gauss(0, 0.05)
        stability_score = max(0.0, min(1.0, stability_score))
        
        return {
            **parameters,
            'bandgap': round(bandgap, 3),
            'plqy': round(plqy, 3),
            'emission_peak': round(1240 / bandgap, 1),
            'particle_size': round(particle_size, 1),
            'emission_fwhm': round(20 + 100 / particle_size, 1),
            'lifetime': round(10 + random.gauss(0, 2), 1),
            'stability_score': round(stability_score, 3),
            'phase_purity': round(1.0 - abs(conc_ratio - 1.0) * 0.2, 3)
        }
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> str:
        """Save dataset to CSV file"""
        filepath = self.output_dir / filename
        
        if not dataset:
            print("⚠️  No data to save")
            return str(filepath)
        
        # Get headers
        headers = list(dataset[0].keys())
        
        # Write CSV
        with open(filepath, 'w') as f:
            f.write(','.join(headers) + '\n')
            
            for sample in dataset:
                values = [str(sample.get(h, '')) for h in headers]
                f.write(','.join(values) + '\n')
        
        print(f"💾 Dataset saved to: {filepath}")
        return str(filepath)
    
    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the generated dataset"""
        if not dataset:
            return {}
        
        print("📊 Analyzing dataset...")
        
        properties = ['plqy', 'bandgap', 'particle_size', 'stability_score', 'phase_purity']
        analysis = {
            'n_samples': len(dataset),
            'property_stats': {}
        }
        
        for prop in properties:
            values = [sample.get(prop, 0) for sample in dataset if prop in sample]
            
            if values:
                analysis['property_stats'][prop] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
                }
        
        # Find best samples
        best_plqy = max(dataset, key=lambda x: x.get('plqy', 0))
        best_overall = max(dataset, key=lambda x: x.get('plqy', 0) + x.get('stability_score', 0))
        
        analysis['best_samples'] = {
            'highest_plqy': best_plqy,
            'best_overall': best_overall
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], optimization_results: Optional[Dict] = None) -> str:
        """Generate HTML training report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
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
                .good {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; font-weight: bold; }}
                .poor {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🎓 Enhanced CsPbBr3 Digital Twin Training Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            
            <div class="summary">
                <h2>📊 Training Summary</h2>
                <div class="metric">
                    <div class="metric-value">{analysis.get('n_samples', 0):,}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
        """
        
        if optimization_results:
            html_content += f"""
                <div class="metric">
                    <div class="metric-value">{optimization_results.get('best_score', 0):.3f}</div>
                    <div class="metric-label">Optimization Score</div>
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>🎯 Property Statistics</h2>
            <table>
                <tr><th>Property</th><th>Mean</th><th>Min</th><th>Max</th><th>Std Dev</th><th>Quality</th></tr>
        """
        
        # Add property statistics
        for prop, stats in analysis.get('property_stats', {}).items():
            mean_val = stats['mean']
            
            # Determine quality
            if prop == 'plqy':
                quality = 'good' if mean_val > 0.7 else 'warning' if mean_val > 0.4 else 'poor'
            elif prop == 'stability_score':
                quality = 'good' if mean_val > 0.8 else 'warning' if mean_val > 0.6 else 'poor'
            else:
                quality = 'good'
            
            html_content += f"""
                <tr>
                    <td>{prop.replace('_', ' ').title()}</td>
                    <td>{mean_val:.3f}</td>
                    <td>{stats['min']:.3f}</td>
                    <td>{stats['max']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                    <td class="{quality}">{'✓' if quality == 'good' else '⚠' if quality == 'warning' else '✗'}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # Add best samples
        if 'best_samples' in analysis:
            html_content += """
            <h2>🏆 Best Samples</h2>
            <h3>Highest PLQY Sample:</h3>
            <table>
            """
            
            best_plqy = analysis['best_samples']['highest_plqy']
            for key, value in best_plqy.items():
                if isinstance(value, (int, float)):
                    html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"
                else:
                    html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            
            html_content += "</table>"
        
        # Add optimization results
        if optimization_results:
            html_content += """
            <h2>🎯 Optimization Results</h2>
            <h3>Best Parameters:</h3>
            <table>
            """
            
            for param, value in optimization_results['best_parameters'].items():
                if isinstance(value, (int, float)):
                    html_content += f"<tr><td>{param.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"
                else:
                    html_content += f"<tr><td>{param.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            
            html_content += "</table>"
        
        html_content += """
            <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
                <p>Generated by Working Enhanced CsPbBr3 Digital Twin Training Pipeline</p>
            </footer>
        </body>
        </html>
        """
        
        # Save report
        report_path = self.output_dir / "training_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved to: {report_path}")
        return str(report_path)
    
    def run_complete_training(self, n_samples: int = 2000, 
                            run_optimization: bool = True,
                            optimization_iterations: int = 30) -> Dict[str, Any]:
        """Run complete enhanced training pipeline"""
        print("🎓 Starting Complete Enhanced Training Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        results = {}
        
        # 1. Generate base dataset
        print("\n1️⃣ Generating Base Dataset...")
        base_dataset = self.generate_enhanced_dataset(n_samples, use_all_enhancements=True)
        results['base_dataset_size'] = len(base_dataset)
        
        # Save base dataset
        base_path = self.save_dataset(base_dataset, f"base_training_data_{n_samples}.csv")
        results['base_dataset_path'] = base_path
        
        # 2. Run optimization
        if run_optimization:
            print("\n2️⃣ Running ML Optimization...")
            optimization_results = self.run_ml_optimization(optimization_iterations)
            results['optimization'] = optimization_results
            
            # Update training history for adaptive sampling
            self.training_history.extend(optimization_results.get('optimization_history', []))
        
        # 3. Generate optimized dataset
        if self.best_parameters:
            print("\n3️⃣ Generating Optimized Dataset...")
            optimized_samples = min(1000, n_samples // 2)
            optimized_dataset = self.generate_enhanced_dataset(optimized_samples, use_all_enhancements=True)
            
            optimized_path = self.save_dataset(optimized_dataset, f"optimized_training_data_{optimized_samples}.csv")
            results['optimized_dataset_size'] = len(optimized_dataset)
            results['optimized_dataset_path'] = optimized_path
            
            # Combine datasets
            combined_dataset = base_dataset + optimized_dataset
        else:
            combined_dataset = base_dataset
        
        # 4. Analyze results
        print("\n4️⃣ Analyzing Results...")
        analysis = self.analyze_dataset(combined_dataset)
        results['analysis'] = analysis
        
        # 5. Generate report
        print("\n5️⃣ Generating Report...")
        report_path = self.generate_report(analysis, results.get('optimization'))
        results['report_path'] = report_path
        
        # 6. Save final combined dataset
        final_path = self.save_dataset(combined_dataset, "final_enhanced_training_data.csv")
        results['final_dataset_path'] = final_path
        results['final_dataset_size'] = len(combined_dataset)
        
        total_time = time.time() - start_time
        results['total_training_time'] = total_time
        
        print(f"\n🎉 Complete Enhanced Training Finished!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Final dataset: {len(combined_dataset)} samples")
        print(f"📁 All outputs in: {self.output_dir}")
        
        return results


def main():
    """Main function to run the working enhanced training"""
    print("🚀 Working Enhanced CsPbBr3 Digital Twin Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = WorkingEnhancedTrainer()
    
    # Run complete training pipeline
    results = trainer.run_complete_training(
        n_samples=1000,  # Start with reasonable size
        run_optimization=True,
        optimization_iterations=20
    )
    
    print("\n✅ Training Summary:")
    print(f"   Base dataset: {results.get('base_dataset_size', 0)} samples")
    print(f"   Final dataset: {results.get('final_dataset_size', 0)} samples")
    print(f"   Training time: {results.get('total_training_time', 0):.2f}s")
    
    if 'optimization' in results:
        print(f"   Best score: {results['optimization'].get('best_score', 0):.4f}")
    
    print(f"   Report: {results.get('report_path', 'Not generated')}")


if __name__ == "__main__":
    main()