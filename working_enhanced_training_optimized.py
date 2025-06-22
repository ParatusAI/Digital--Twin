#!/usr/bin/env python3
"""
Optimized Working Enhanced Training Pipeline for CsPbBr3 Digital Twin
Performance-optimized version with vectorization, caching, and memory efficiency
"""

import json
import time
import random
import math
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from functools import lru_cache
import pickle

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimized modules
try:
    from generate_sample_data_optimized import (
        create_sample_dataset_parallel_optimized, 
        array_pool, 
        computation_cache
    )
    OPTIMIZED_GENERATION_AVAILABLE = True
except ImportError:
    OPTIMIZED_GENERATION_AVAILABLE = False

class OptimizedComputationCache:
    """High-performance computation cache with LRU eviction"""
    def __init__(self, max_size=2000):
        self._cache = {}
        self._access_order = deque()
        self._max_size = max_size
        
    def get(self, key):
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Remove least recently used
            oldest_key = self._access_order.popleft()
            del self._cache[oldest_key]
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()

class WorkingEnhancedTrainerOptimized:
    """Performance-optimized enhanced training pipeline"""
    
    def __init__(self, output_dir: str = "working_training_output_optimized", max_history_size: int = 1000):
        """Initialize the optimized trainer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = self._create_default_config()
        
        # Use deque with max size to prevent memory growth
        self.training_history = deque(maxlen=max_history_size)
        self.best_parameters = {}
        self.computation_cache = OptimizedComputationCache()
        
        # Pre-compute constants for physics calculations
        self._precompute_physics_constants()
        
        print("🚀 Optimized Working Enhanced Training Pipeline Initialized")
    
    def _precompute_physics_constants(self):
        """Pre-compute expensive constants used in physics calculations"""
        self.physics_constants = {
            'activation_energy': 50000,  # J/mol
            'gas_constant': 8.314,
            'reference_temp': 150.0,
            'temp_scale': 100.0,
            'boltzmann_temps': {},  # Cache for temperature-dependent calculations
            'quantum_factors': {},  # Cache for quantum confinement factors
        }
    
    @lru_cache(maxsize=512)
    def _create_default_config(self) -> Dict:
        """Create default configuration with caching"""
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
    
    def generate_adaptive_samples_vectorized(self, n_samples: int, use_adaptive: bool = True) -> pd.DataFrame:
        """Vectorized adaptive sample generation"""
        print(f"📊 Generating {n_samples} samples (adaptive: {use_adaptive}, vectorized)...")
        
        if use_adaptive and len(self.training_history) > 10:
            print("   Using adaptive sampling based on training history...")
            return self._generate_adaptive_vectorized(n_samples)
        else:
            print("   Using random sampling...")
            return self._generate_random_vectorized(n_samples)
    
    def _generate_adaptive_vectorized(self, n_samples: int) -> pd.DataFrame:
        """Vectorized adaptive sampling around good regions"""
        # Extract good samples efficiently
        good_samples = [h for h in self.training_history if h.get('score', 0) > 0.8]
        
        if not good_samples:
            return self._generate_random_vectorized(n_samples)
        
        # Pre-allocate arrays
        param_names = list(self.config['synthesis_parameters'].keys())
        n_params = len(param_names)
        samples_array = np.empty((n_samples, n_params), dtype=np.float32)
        
        # Vectorized generation around good samples
        n_guided = int(0.7 * n_samples)  # 70% guided, 30% random
        
        # Generate guided samples
        if n_guided > 0:
            # Random selection of good samples to base on
            base_indices = np.random.choice(len(good_samples), n_guided, replace=True)
            
            for i, base_idx in enumerate(base_indices):
                base_sample = good_samples[base_idx]
                samples_array[i] = self._create_variation_vectorized(base_sample, param_names)
        
        # Generate random samples for the rest
        for i in range(n_guided, n_samples):
            samples_array[i] = self._generate_random_sample_vectorized(param_names)
        
        # Create DataFrame
        df = pd.DataFrame(samples_array, columns=param_names)
        
        # Add solvent types efficiently
        df['solvent_type'] = np.random.choice(
            self.config['solvents']['types'],
            size=n_samples,
            p=self.config['solvents']['weights']
        ).astype(np.int8)
        
        return df
    
    def _generate_random_vectorized(self, n_samples: int) -> pd.DataFrame:
        """Vectorized random sample generation"""
        param_names = list(self.config['synthesis_parameters'].keys())
        n_params = len(param_names)
        
        # Pre-allocate array
        samples_array = np.empty((n_samples, n_params), dtype=np.float32)
        
        # Vectorized parameter generation
        for i, param in enumerate(param_names):
            bounds = self.config['synthesis_parameters'][param]
            samples_array[:, i] = np.random.uniform(bounds['min'], bounds['max'], n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(samples_array, columns=param_names)
        
        # Add solvent types
        df['solvent_type'] = np.random.choice(
            self.config['solvents']['types'],
            size=n_samples,
            p=self.config['solvents']['weights']
        ).astype(np.int8)
        
        return df
    
    def _create_variation_vectorized(self, base_sample: Dict[str, float], param_names: List[str], 
                                   noise_level: float = 0.1) -> np.ndarray:
        """Vectorized variation creation"""
        variation = np.empty(len(param_names), dtype=np.float32)
        
        for i, param in enumerate(param_names):
            bounds = self.config['synthesis_parameters'][param]
            if param in base_sample:
                base_value = base_sample[param]
                range_size = bounds['max'] - bounds['min']
                noise = np.random.normal(0, range_size * noise_level)
                new_value = base_value + noise
                variation[i] = np.clip(new_value, bounds['min'], bounds['max'])
            else:
                variation[i] = np.random.uniform(bounds['min'], bounds['max'])
        
        return variation
    
    def _generate_random_sample_vectorized(self, param_names: List[str]) -> np.ndarray:
        """Vectorized random sample generation"""
        sample = np.empty(len(param_names), dtype=np.float32)
        
        for i, param in enumerate(param_names):
            bounds = self.config['synthesis_parameters'][param]
            sample[i] = np.random.uniform(bounds['min'], bounds['max'])
        
        return sample
    
    @lru_cache(maxsize=1000)
    def _get_boltzmann_factor(self, temperature: float) -> float:
        """Cached Boltzmann factor calculation"""
        temp_key = round(temperature, 1)  # Round for cache efficiency
        
        if temp_key not in self.physics_constants['boltzmann_temps']:
            factor = math.exp(-self.physics_constants['activation_energy'] / 
                            (self.physics_constants['gas_constant'] * (temperature + 273.15)))
            self.physics_constants['boltzmann_temps'][temp_key] = factor
        
        return self.physics_constants['boltzmann_temps'][temp_key]
    
    @lru_cache(maxsize=500)
    def _get_quantum_factor(self, size: float) -> float:
        """Cached quantum confinement factor"""
        size_key = round(size, 2)
        
        if size_key not in self.physics_constants['quantum_factors']:
            factor = 1.8 / (size ** 2) if size > 0 else 0
            self.physics_constants['quantum_factors'][size_key] = factor
        
        return self.physics_constants['quantum_factors'][size_key]
    
    def simulate_physics_enhanced_vectorized(self, parameters_df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized physics simulation for batch processing"""
        n_samples = len(parameters_df)
        
        # Pre-allocate result arrays
        results = {
            'bandgap': np.empty(n_samples, dtype=np.float32),
            'plqy': np.empty(n_samples, dtype=np.float32),
            'emission_peak': np.empty(n_samples, dtype=np.float32),
            'particle_size': np.empty(n_samples, dtype=np.float32),
            'emission_fwhm': np.empty(n_samples, dtype=np.float32),
            'lifetime': np.empty(n_samples, dtype=np.float32),
            'stability_score': np.empty(n_samples, dtype=np.float32),
            'phase_purity': np.empty(n_samples, dtype=np.float32)
        }
        
        # Extract parameter arrays for vectorized operations
        cs_conc = parameters_df['cs_br_concentration'].values
        pb_conc = parameters_df['pb_br2_concentration'].values
        temp = parameters_df['temperature'].values
        oa_conc = parameters_df['oa_concentration'].values
        oam_conc = parameters_df['oam_concentration'].values
        time_vals = parameters_df['reaction_time'].values
        solvent = parameters_df['solvent_type'].values
        
        # Vectorized physics calculations
        
        # 1. Temperature effects (vectorized)
        temp_k = temp + 273.15
        boltzmann_factors = np.exp(-self.physics_constants['activation_energy'] / 
                                 (self.physics_constants['gas_constant'] * temp_k))
        
        # 2. Concentration effects (vectorized)
        conc_ratios = cs_conc / (pb_conc + 1e-8)
        stoich_factors = np.exp(-0.5 * (conc_ratios - 1.0) ** 2)
        
        # 3. Ligand effects (vectorized)
        total_ligands = oa_conc + oam_conc
        ligand_factors = np.exp(-total_ligands / 1.5)
        
        # 4. Time-dependent growth (vectorized)
        growth_rates = 0.1 * boltzmann_factors * np.sqrt(time_vals)
        final_sizes = 5 + growth_rates * np.sqrt(time_vals)
        
        # 5. Solvent effects (vectorized)
        solvent_effects = np.array([1.2, 1.0, 0.5, 0.8, 0.9])[solvent]
        
        # 6. Calculate properties (vectorized)
        
        # Particle size
        results['particle_size'][:] = np.clip(final_sizes, 2.0, 30.0)
        
        # Quantum confinement effects (vectorized)
        quantum_factors = 1.8 / (final_sizes ** 2 + 1e-8)
        
        # Bandgap (vectorized)
        base_bandgap = 2.3 + quantum_factors
        temp_shift = -0.0003 * (temp - 150)
        results['bandgap'][:] = np.clip(base_bandgap + temp_shift, 1.5, 3.5)
        
        # PLQY (vectorized)
        base_plqy = 0.8 * stoich_factors * ligand_factors * solvent_effects
        temp_quenching = np.exp(-(temp - 150) / 200)
        size_enhancement = np.where(final_sizes < 10, 1.2, 1.0)
        results['plqy'][:] = np.clip(base_plqy * temp_quenching * size_enhancement, 0, 1)
        
        # Emission peak (vectorized)
        size_shift = 50 * quantum_factors
        results['emission_peak'][:] = 520 - size_shift + np.random.normal(0, 5, n_samples)
        
        # Emission FWHM (vectorized)
        size_broadening = 20 + 100 / (final_sizes + 1e-8)
        disorder_broadening = 10 * (1 - stoich_factors)
        results['emission_fwhm'][:] = size_broadening + disorder_broadening
        
        # Lifetime (vectorized)
        base_lifetime = 15 + 5 * quantum_factors
        quenching_factor = np.exp(-total_ligands / 2.0)
        results['lifetime'][:] = base_lifetime * quenching_factor
        
        # Stability score (vectorized)
        temp_stability = np.exp(-(temp - 150) ** 2 / 5000)
        size_stability = np.where(final_sizes > 5, 1.0, 0.8)
        ligand_stability = np.where(total_ligands > 0.1, 1.0, 0.7)
        results['stability_score'][:] = np.clip(
            temp_stability * size_stability * ligand_stability * stoich_factors,
            0, 1
        )
        
        # Phase purity (vectorized)
        purity_base = stoich_factors * temp_stability
        solvent_purity = np.where(solvent == 0, 1.0, 0.9)  # DMF is best
        results['phase_purity'][:] = np.clip(purity_base * solvent_purity, 0.5, 1.0)
        
        return pd.DataFrame(results)
    
    def run_ml_optimization_vectorized(self, n_iterations: int = 20, batch_size: int = 100) -> Dict[str, Any]:
        """Vectorized ML optimization with batch processing"""
        print(f"🎯 Running ML optimization for {n_iterations} iterations (vectorized)...")
        
        best_score = 0
        best_params = {}
        optimization_history = []
        
        for iteration in range(n_iterations):
            # Generate batch of candidates
            candidates_df = self.generate_adaptive_samples_vectorized(batch_size, use_adaptive=True)
            
            # Simulate physics for entire batch
            properties_df = self.simulate_physics_enhanced_vectorized(candidates_df)
            
            # Vectorized scoring
            scores = self._calculate_scores_vectorized(properties_df)
            
            # Find best in batch
            best_idx = np.argmax(scores)
            batch_best_score = scores[best_idx]
            
            if batch_best_score > best_score:
                best_score = batch_best_score
                best_params = candidates_df.iloc[best_idx].to_dict()
                print(f"   New best at iteration {iteration}: score = {best_score:.4f}")
            
            # Add to history (sample for memory efficiency)
            for i in range(min(10, len(candidates_df))):  # Sample 10 per iteration
                sample_data = candidates_df.iloc[i].to_dict()
                sample_data.update(properties_df.iloc[i].to_dict())
                sample_data['score'] = scores[i]
                self.training_history.append(sample_data)
            
            optimization_history.append({
                'iteration': iteration,
                'best_score': batch_best_score,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })
            
            if iteration % 10 == 0 and iteration > 0:
                print(f"   Iteration {iteration}/{n_iterations}, best score: {best_score:.4f}")
        
        self.best_parameters = best_params
        
        print(f"✅ Optimization complete! Best score: {best_score:.4f}")
        
        return {
            'best_score': best_score,
            'best_parameters': best_params,
            'optimization_history': optimization_history,
            'n_iterations': n_iterations
        }
    
    def _calculate_scores_vectorized(self, properties_df: pd.DataFrame) -> np.ndarray:
        """Vectorized scoring function"""
        # Vectorized multi-objective scoring
        plqy_scores = properties_df['plqy'].values
        stability_scores = properties_df['stability_score'].values
        
        # Bandgap penalty (vectorized)
        bandgaps = properties_df['bandgap'].values
        bandgap_penalty = np.where(
            (bandgaps >= 2.2) & (bandgaps <= 2.5), 1.0,
            np.where((bandgaps >= 2.0) & (bandgaps <= 2.7), 0.8, 0.5)
        )
        
        # Phase purity bonus (vectorized)
        purity_bonus = properties_df['phase_purity'].values
        
        # Combined score (vectorized)
        combined_scores = (0.4 * plqy_scores + 
                          0.3 * stability_scores + 
                          0.2 * bandgap_penalty + 
                          0.1 * purity_bonus)
        
        return combined_scores
    
    def generate_enhanced_dataset_batch(self, n_samples: int, batch_size: int = 1000) -> pd.DataFrame:
        """Generate dataset in batches for memory efficiency"""
        print(f"📊 Generating enhanced dataset with {n_samples} samples (batch size: {batch_size})...")
        
        datasets = []
        start_time = time.time()
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            
            # Generate batch
            params_df = self.generate_adaptive_samples_vectorized(current_batch_size, use_adaptive=True)
            properties_df = self.simulate_physics_enhanced_vectorized(params_df)
            
            # Combine
            batch_df = pd.concat([params_df, properties_df], axis=1)
            datasets.append(batch_df)
            
            if len(datasets) % 5 == 0:
                print(f"   Processed {batch_end}/{n_samples} samples")
        
        # Combine all batches
        final_dataset = pd.concat(datasets, ignore_index=True)
        
        end_time = time.time()
        print(f"✅ Dataset generation complete in {end_time - start_time:.2f}s")
        
        return final_dataset
    
    def run_complete_training_pipeline(self) -> Dict[str, Any]:
        """Run complete optimized training pipeline"""
        print("🎓 Starting Complete Enhanced Training Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Generate base dataset
        print("\n1️⃣ Generating Base Dataset...")
        base_dataset = self.generate_enhanced_dataset_batch(1000, batch_size=200)
        base_file = self.output_dir / "base_training_data_1000.csv"
        base_dataset.to_csv(base_file, index=False, float_format='%.6f')
        print(f"💾 Dataset saved to: {base_file}")
        
        # Step 2: Run ML optimization
        print("\n2️⃣ Running ML Optimization...")
        optimization_results = self.run_ml_optimization_vectorized(n_iterations=20, batch_size=50)
        
        # Step 3: Generate optimized dataset
        print("\n3️⃣ Generating Optimized Dataset...")
        optimized_dataset = self.generate_enhanced_dataset_batch(500, batch_size=100)
        optimized_file = self.output_dir / "optimized_training_data_500.csv"
        optimized_dataset.to_csv(optimized_file, index=False, float_format='%.6f')
        print(f"💾 Dataset saved to: {optimized_file}")
        
        # Step 4: Combine and analyze
        print("\n4️⃣ Analyzing Results...")
        final_dataset = pd.concat([base_dataset, optimized_dataset], ignore_index=True)
        analysis_results = self._analyze_dataset_vectorized(final_dataset)
        
        # Step 5: Save final results
        print("\n5️⃣ Generating Report...")
        final_file = self.output_dir / "final_enhanced_training_data.csv"
        final_dataset.to_csv(final_file, index=False, float_format='%.6f')
        
        report_file = self.output_dir / "training_report.html"
        self._generate_html_report(final_dataset, optimization_results, analysis_results, report_file)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Complete Enhanced Training Finished!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Final dataset: {len(final_dataset)} samples")
        print(f"📁 All outputs in: {self.output_dir}")
        print(f"🚀 Performance: {len(final_dataset) / total_time:.0f} samples/second")
        
        return {
            'base_dataset_size': len(base_dataset),
            'final_dataset_size': len(final_dataset),
            'training_time': total_time,
            'best_score': optimization_results['best_score'],
            'performance_samples_per_sec': len(final_dataset) / total_time,
            'output_directory': str(self.output_dir),
            'report_file': str(report_file)
        }
    
    def _analyze_dataset_vectorized(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Vectorized dataset analysis"""
        print("📊 Analyzing dataset...")
        
        # Vectorized statistics
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        stats = dataset[numeric_cols].describe()
        
        # Quality metrics (vectorized)
        quality_metrics = {
            'high_plqy_samples': np.sum(dataset['plqy'] > 0.7),
            'stable_samples': np.sum(dataset['stability_score'] > 0.8),
            'optimal_bandgap_samples': np.sum((dataset['bandgap'] >= 2.2) & (dataset['bandgap'] <= 2.5)),
            'overall_quality_score': np.mean(self._calculate_scores_vectorized(dataset))
        }
        
        return {
            'statistics': stats.to_dict(),
            'quality_metrics': quality_metrics,
            'sample_count': len(dataset)
        }
    
    def _generate_html_report(self, dataset: pd.DataFrame, optimization_results: Dict, 
                            analysis_results: Dict, output_file: Path):
        """Generate comprehensive HTML report"""
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>🎓 Optimized Enhanced CsPbBr3 Training Report</title>
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
            <h1>🎓 Optimized Enhanced CsPbBr3 Digital Twin Training Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>📊 Training Summary</h2>
                <div class="metric">
                    <div class="metric-value">{len(dataset):,}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
        
                <div class="metric">
                    <div class="metric-value">{optimization_results['best_score']:.3f}</div>
                    <div class="metric-label">Optimization Score</div>
                </div>
            
                <div class="metric">
                    <div class="metric-value">{len(dataset) / optimization_results.get('total_time', 1):.0f}</div>
                    <div class="metric-label">Samples/Second</div>
                </div>
            </div>
            
            <h2>🎯 Property Statistics</h2>
            <table>
                <tr><th>Property</th><th>Mean</th><th>Min</th><th>Max</th><th>Std Dev</th><th>Quality</th></tr>
        '''
        
        # Add property statistics
        for prop in ['plqy', 'bandgap', 'particle_size', 'stability_score', 'phase_purity']:
            if prop in dataset.columns:
                mean_val = dataset[prop].mean()
                min_val = dataset[prop].min()
                max_val = dataset[prop].max()
                std_val = dataset[prop].std()
                
                # Quality assessment
                if prop == 'plqy':
                    quality = "✓" if mean_val > 0.5 else "⚠" if mean_val > 0.3 else "✗"
                    quality_class = "good" if mean_val > 0.5 else "warning" if mean_val > 0.3 else "poor"
                else:
                    quality = "✓"
                    quality_class = "good"
                
                html_content += f'''
                <tr>
                    <td>{prop.replace('_', ' ').title()}</td>
                    <td>{mean_val:.3f}</td>
                    <td>{min_val:.3f}</td>
                    <td>{max_val:.3f}</td>
                    <td>{std_val:.3f}</td>
                    <td class="{quality_class}">{quality}</td>
                </tr>
                '''
        
        html_content += '''
            </table>
            <h2>🏆 Best Samples</h2>
            <h3>Highest PLQY Sample:</h3>
            <table>
        '''
        
        # Find and display best sample
        best_plqy_idx = dataset['plqy'].idxmax()
        best_sample = dataset.iloc[best_plqy_idx]
        
        for param in ['cs_br_concentration', 'pb_br2_concentration', 'temperature', 
                     'oa_concentration', 'oam_concentration', 'reaction_time', 'solvent_type',
                     'bandgap', 'plqy', 'emission_peak', 'particle_size', 'emission_fwhm',
                     'lifetime', 'stability_score', 'phase_purity']:
            if param in best_sample:
                html_content += f'<tr><td>{param.replace("_", " ").title()}</td><td>{best_sample[param]:.3f}</td></tr>'
        
        html_content += '''</table>
            <h2>🎯 Optimization Results</h2>
            <h3>Best Parameters:</h3>
            <table>
        '''
        
        best_params = optimization_results['best_parameters']
        for param, value in best_params.items():
            if param != 'solvent_type':
                html_content += f'<tr><td>{param.replace("_", " ").title()}</td><td>{value:.3f}</td></tr>'
            else:
                html_content += f'<tr><td>{param.replace("_", " ").title()}</td><td>{value:.0f}</td></tr>'
        
        html_content += '''</table>
            <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
                <p>Generated by Optimized Working Enhanced CsPbBr3 Digital Twin Training Pipeline</p>
            </footer>
        </body>
        </html>
        '''
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved to: {output_file}")

def main():
    """Main function for optimized training"""
    print("🚀 Starting Optimized Working Enhanced CsPbBr3 Training")
    print("=" * 50)
    
    # Create optimized trainer
    trainer = WorkingEnhancedTrainerOptimized()
    
    # Run complete pipeline
    results = trainer.run_complete_training_pipeline()
    
    print(f"\n✅ Training Summary:")
    print(f"   Base dataset: {results['base_dataset_size']} samples")
    print(f"   Final dataset: {results['final_dataset_size']} samples")
    print(f"   Training time: {results['training_time']:.2f}s")
    print(f"   Performance: {results['performance_samples_per_sec']:.0f} samples/second")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Report: {results['report_file']}")

if __name__ == "__main__":
    main()