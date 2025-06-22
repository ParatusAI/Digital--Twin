#!/usr/bin/env python3
"""
Optimized Generate Sample Training Data for CsPbBr3 Digital Twin
Performance-optimized version with vectorization, caching, and memory efficiency
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial, lru_cache
from collections import deque
import multiprocessing.shared_memory as shm

# Add project to path
sys.path.insert(0, '.')

# Global memory pool for efficient array allocation
class ArrayMemoryPool:
    def __init__(self):
        self._pools = {}  # (shape, dtype) -> list of arrays
        
    def get_array(self, shape, dtype=np.float32):
        key = (tuple(shape), dtype)
        if key not in self._pools:
            self._pools[key] = []
        
        pool = self._pools[key]
        if pool:
            return pool.pop()
        else:
            return np.empty(shape, dtype=dtype)
    
    def return_array(self, array):
        key = (tuple(array.shape), array.dtype)
        if key not in self._pools:
            self._pools[key] = []
        
        # Clear array data and add to pool
        array.fill(0)
        self._pools[key].append(array)

# Global memory pool instance
array_pool = ArrayMemoryPool()

# Computation cache with size limit
class ComputationCache:
    def __init__(self, max_size=1000):
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

# Global computation cache
computation_cache = ComputationCache()

@lru_cache(maxsize=128)
def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from JSON file with caching"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config' / 'data_generation_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"⚙️  Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        warnings.warn(f"Config file not found at {config_path}, using defaults")
        return get_default_config()
    except json.JSONDecodeError as e:
        warnings.warn(f"Error parsing config file: {e}, using defaults")
        return get_default_config()

@lru_cache(maxsize=1)
def get_default_config() -> Dict:
    """Return default configuration with caching"""
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
            'weights': [0.3, 0.25, 0.15, 0.2, 0.1],
            'effects': [1.2, 1.0, 0.5, 0.8, 0.9]
        },
        'data_generation': {'random_seed': 42}
    }

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    return np.random.RandomState(seed)

def validate_physics_constraints_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized physics constraints validation"""
    print("🧪 Applying physics constraints (vectorized)...")
    
    # Pre-allocate boolean arrays
    n_samples = len(df)
    
    # Vectorized stoichiometric constraints
    cs_pb_ratio = df['cs_br_concentration'].values / (df['pb_br2_concentration'].values + 1e-8)
    stoichiometry_valid = (cs_pb_ratio >= 0.5) & (cs_pb_ratio <= 3.0)
    
    # Vectorized temperature-dependent solubility
    temp_factor = (df['temperature'].values - 25) / 100
    max_cs_solubility = 0.1 + 0.01 * temp_factor
    max_pb_solubility = 0.05 + 0.008 * temp_factor
    
    concentration_valid = (
        (df['cs_br_concentration'].values <= max_cs_solubility * 10) &
        (df['pb_br2_concentration'].values <= max_pb_solubility * 10)
    )
    
    # Vectorized ligand constraints
    total_ligand = df['oa_concentration'].values + df['oam_concentration'].values
    ligand_valid = (
        (df['oa_concentration'].values <= 2.0) & 
        (df['oam_concentration'].values <= 2.0) &
        (total_ligand <= 2.5)
    )
    
    # Efficient constraint violation calculation
    constraint_violations = (~stoichiometry_valid).astype(int) + (~concentration_valid).astype(int) + (~ligand_valid).astype(int)
    
    # Add results to dataframe
    df['stoichiometry_valid'] = stoichiometry_valid
    df['concentration_valid'] = concentration_valid
    df['ligand_valid'] = ligand_valid
    df['physics_failure_prob'] = np.clip(constraint_violations * 0.3, 0, 0.9)
    
    return df

def generate_synthesis_parameters_optimized(n_samples: int, config: Optional[Dict] = None) -> pd.DataFrame:
    """Optimized parameter generation with pre-allocation and vectorization"""
    if config is None:
        config = load_config()
    
    # Pre-allocate all arrays with optimal dtype
    data = {}
    param_names = list(config['synthesis_parameters'].keys())
    
    # Create arrays all at once to improve memory locality
    for param in param_names:
        data[param] = array_pool.get_array((n_samples,), dtype=np.float32)
    
    # Extract parameter ranges correctly  
    param_ranges = {}
    for param, bounds in config['synthesis_parameters'].items():
        param_ranges[param] = (bounds['min'], bounds['max'])
    
    # Vectorized parameter generation
    for param, (min_val, max_val) in param_ranges.items():
        arr = data[param]
        
        if param in ['cs_br_concentration', 'pb_br2_concentration']:
            # Log-normal distribution for concentrations
            log_mean = np.log((min_val + max_val) / 2)
            arr[:] = np.random.lognormal(log_mean, 0.5, size=n_samples)
            np.clip(arr, min_val, max_val, out=arr)
        elif param == 'temperature':
            # Slight bias toward moderate temperatures
            center = (min_val + max_val) / 2
            spread = (max_val - min_val) / 4
            arr[:] = np.random.normal(center, spread, size=n_samples)
            np.clip(arr, min_val, max_val, out=arr)
        elif param in ['oa_concentration', 'oam_concentration']:
            # Exponential distribution for ligand concentrations
            scale = (max_val - min_val) / 3
            arr[:] = np.random.exponential(scale, size=n_samples)
            arr += min_val
            np.clip(arr, min_val, max_val, out=arr)
        else:
            # Uniform distribution for other parameters
            arr[:] = np.random.uniform(min_val, max_val, size=n_samples)
    
    # Generate solvent types efficiently
    solvent_config = config['solvents']
    solvent_types = np.random.choice(
        solvent_config['types'], 
        size=n_samples, 
        p=solvent_config['weights']
    )
    data['solvent_type'] = solvent_types.astype(np.int8)  # Use smaller int type
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Return arrays to pool for reuse
    for param in param_names:
        # Don't return arrays that are now owned by DataFrame
        pass
    
    return df

def determine_phase_outcomes_optimized(params_df: pd.DataFrame) -> np.ndarray:
    """Optimized phase outcome determination with vectorization"""
    n_samples = len(params_df)
    
    # Cache key for phase probability calculation
    cache_key = f"phase_probs_{n_samples}_{hash(tuple(params_df.values.flatten()[:100]))}"  # Use subset for hash
    cached_result = computation_cache.get(cache_key)
    
    if cached_result is not None:
        return cached_result
    
    # Vectorized parameter extraction
    temp = params_df['temperature'].values
    cs_conc = params_df['cs_br_concentration'].values
    pb_conc = params_df['pb_br2_concentration'].values
    oa_conc = params_df['oa_concentration'].values
    oam_conc = params_df['oam_concentration'].values
    time_vals = params_df['reaction_time'].values
    solvent = params_df['solvent_type'].values
    
    # Vectorized temperature factor calculation
    temp_factor = 1.0 / (1.0 + np.exp(-(temp - 150) / 30))
    
    # Vectorized stoichiometry calculation
    cs_pb_ratio = cs_conc / (pb_conc + 1e-8)
    stoich_factor = np.exp(-0.5 * (cs_pb_ratio - 1.0) ** 2)
    
    # Vectorized ligand effects
    total_ligand = oa_conc + oam_conc
    ligand_factor = np.exp(-total_ligand / 2.0)
    
    # Vectorized time effects
    time_factor = 1.0 - np.exp(-time_vals / 20.0)
    
    # Vectorized solvent effects
    solvent_effects = np.array([1.2, 1.0, 0.5, 0.8, 0.9])
    solvent_factor = solvent_effects[solvent]
    
    # Vectorized probability calculations
    base_3d = 0.4 * temp_factor * stoich_factor * time_factor * solvent_factor
    base_0d = 0.3 * (1 - temp_factor) * ligand_factor
    base_2d = 0.2 * temp_factor * (1 - stoich_factor)
    base_mixed = 0.1 * np.ones(n_samples)
    
    # Normalize probabilities efficiently
    total_prob = base_3d + base_0d + base_2d + base_mixed
    prob_3d = base_3d / total_prob
    prob_0d = base_0d / total_prob
    prob_2d = base_2d / total_prob
    prob_mixed = base_mixed / total_prob
    prob_failed = np.maximum(0, params_df.get('physics_failure_prob', np.zeros(n_samples)).values)
    
    # Efficient multinomial sampling
    phase_labels = np.empty(n_samples, dtype=np.int8)
    phase_probs = np.column_stack([prob_3d, prob_0d, prob_2d, prob_mixed, prob_failed])
    
    # Vectorized sampling using cumulative probabilities
    cumsum_probs = np.cumsum(phase_probs, axis=1)
    random_vals = np.random.random(n_samples)
    
    for i in range(n_samples):
        phase_labels[i] = np.searchsorted(cumsum_probs[i], random_vals[i])
    
    # Cache result
    computation_cache.put(cache_key, phase_labels)
    
    return phase_labels

def calculate_properties_vectorized(params_df: pd.DataFrame, phase_labels: np.ndarray) -> pd.DataFrame:
    """Vectorized property calculation for better performance"""
    print("🔬 Calculating optical and electronic properties (vectorized)...")
    
    n_samples = len(params_df)
    
    # Pre-allocate output arrays
    properties = {
        'bandgap': array_pool.get_array((n_samples,), dtype=np.float32),
        'plqy': array_pool.get_array((n_samples,), dtype=np.float32),
        'emission_peak': array_pool.get_array((n_samples,), dtype=np.float32),
        'particle_size': array_pool.get_array((n_samples,), dtype=np.float32),
        'emission_fwhm': array_pool.get_array((n_samples,), dtype=np.float32),
        'lifetime': array_pool.get_array((n_samples,), dtype=np.float32),
        'stability_score': array_pool.get_array((n_samples,), dtype=np.float32),
        'phase_purity': array_pool.get_array((n_samples,), dtype=np.float32)
    }
    
    # Vectorized base calculations
    temp = params_df['temperature'].values
    cs_conc = params_df['cs_br_concentration'].values
    pb_conc = params_df['pb_br2_concentration'].values
    time_vals = params_df['reaction_time'].values
    
    # Phase-specific vectorized calculations
    for phase_idx in range(5):  # 0: 3D, 1: 0D, 2: 2D, 3: mixed, 4: failed
        mask = phase_labels == phase_idx
        if not np.any(mask):
            continue
            
        n_phase = np.sum(mask)
        
        if phase_idx == 0:  # 3D CsPbBr3
            properties['bandgap'][mask] = 2.3 + 0.1 * np.random.normal(0, 0.05, n_phase)
            properties['plqy'][mask] = np.clip(
                0.8 * np.exp(-0.01 * (temp[mask] - 150)) + np.random.normal(0, 0.1, n_phase),
                0, 1
            )
            properties['emission_peak'][mask] = 520 + np.random.normal(0, 10, n_phase)
            properties['particle_size'][mask] = 10 + 5 * np.log1p(time_vals[mask]) + np.random.normal(0, 2, n_phase)
            
        elif phase_idx == 1:  # 0D quantum dots
            size_factor = 15 / (properties['particle_size'][mask] + 1e-8)
            properties['bandgap'][mask] = 2.3 + size_factor * 0.5 + np.random.normal(0, 0.1, n_phase)
            properties['plqy'][mask] = np.clip(0.9 + np.random.normal(0, 0.05, n_phase), 0, 1)
            properties['emission_peak'][mask] = 520 - 50 * size_factor + np.random.normal(0, 5, n_phase)
            properties['particle_size'][mask] = 2 + 8 * np.random.beta(0.5, 2, n_phase)
            
        elif phase_idx == 2:  # 2D nanoplatelets
            properties['bandgap'][mask] = 2.4 + np.random.normal(0, 0.08, n_phase)
            properties['plqy'][mask] = np.clip(0.6 + np.random.normal(0, 0.15, n_phase), 0, 1)
            properties['emission_peak'][mask] = 510 + np.random.normal(0, 15, n_phase)
            properties['particle_size'][mask] = 5 + 3 * np.random.gamma(2, 1, n_phase)
            
        elif phase_idx == 3:  # Mixed phases
            properties['bandgap'][mask] = 2.0 + 0.4 * np.random.random(n_phase)
            properties['plqy'][mask] = np.clip(0.3 * np.random.random(n_phase), 0, 1)
            properties['emission_peak'][mask] = 450 + 150 * np.random.random(n_phase)
            properties['particle_size'][mask] = 3 + 20 * np.random.random(n_phase)
            
        else:  # Failed synthesis
            properties['bandgap'][mask] = 0
            properties['plqy'][mask] = 0
            properties['emission_peak'][mask] = 0
            properties['particle_size'][mask] = 0
    
    # Vectorized secondary property calculations
    # FWHM based on particle size and phase
    size_broadening = 20 + 100 / (properties['particle_size'] + 1e-8)
    phase_broadening = np.where(phase_labels == 1, 5, np.where(phase_labels == 2, 15, 10))
    properties['emission_fwhm'][:] = size_broadening + phase_broadening + np.random.normal(0, 3, n_samples)
    
    # Lifetime calculations
    base_lifetime = np.where(phase_labels == 0, 15, np.where(phase_labels == 1, 25, 8))
    properties['lifetime'][:] = np.maximum(0.1, base_lifetime + np.random.normal(0, 3, n_samples))
    
    # Stability score
    temp_stability = np.exp(-(temp - 150) ** 2 / 5000)
    phase_stability = np.where(phase_labels == 0, 1.0, np.where(phase_labels == 1, 0.8, 0.6))
    properties['stability_score'][:] = np.clip(
        temp_stability * phase_stability + np.random.normal(0, 0.05, n_samples),
        0, 1
    )
    
    # Phase purity
    properties['phase_purity'][:] = np.where(
        phase_labels == 4, 0.5,  # Failed synthesis
        np.clip(0.8 + np.random.normal(0, 0.15, n_samples), 0.5, 1.0)
    )
    
    # Create DataFrame
    result_df = pd.DataFrame(properties)
    
    # Return arrays to memory pool
    for key, arr in properties.items():
        array_pool.return_array(arr)
    
    return result_df

def generate_batch_optimized(args):
    """Optimized batch generation function (moved to module level for pickling)"""
    batch_size, config = args
    
    # Set different random seed for each process
    np.random.seed(np.random.randint(0, 10000))
    
    # Generate parameters
    params_df = generate_synthesis_parameters_optimized(batch_size, config)
    
    # Apply physics constraints
    params_df = validate_physics_constraints_vectorized(params_df)
    
    # Determine phases
    phase_labels = determine_phase_outcomes_optimized(params_df)
    
    # Calculate properties
    properties_df = calculate_properties_vectorized(params_df, phase_labels)
    
    # Combine results
    result_df = pd.concat([params_df, properties_df], axis=1)
    result_df['phase_label'] = phase_labels
    
    return result_df

def create_sample_dataset_parallel_optimized(n_samples: int, n_processes: Optional[int] = None, 
                                           config: Optional[Dict] = None) -> pd.DataFrame:
    """Optimized parallel dataset creation with efficient batch processing"""
    if config is None:
        config = load_config()
    
    if n_processes is None:
        n_processes = min(cpu_count(), 8)  # Limit to avoid overhead
    
    # Larger batch sizes to reduce overhead
    min_batch_size = 5000
    batch_size = max(min_batch_size, n_samples // n_processes)
    
    print(f"🚀 Generating {n_samples} samples with {n_processes} processes (batch size: {batch_size})")
    
    # Create batch arguments
    batch_args = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_args.append((end - start, config))
    
    # Parallel processing
    if n_processes > 1 and len(batch_args) > 1:
        with Pool(processes=n_processes) as pool:
            batch_results = pool.map(generate_batch_optimized, batch_args)
    else:
        batch_results = [generate_batch_optimized(args) for args in batch_args]
    
    # Combine all batches efficiently
    final_df = pd.concat(batch_results, ignore_index=True)
    
    print(f"✅ Generated {len(final_df)} samples successfully")
    return final_df

def save_dataset_optimized(df: pd.DataFrame, filename: str):
    """Optimized dataset saving with compression"""
    print(f"💾 Saving dataset to {filename}...")
    
    # Use optimized CSV writing
    if filename.endswith('.csv'):
        df.to_csv(filename, index=False, float_format='%.6f')
    elif filename.endswith('.parquet'):
        # Parquet is much faster for large datasets
        df.to_parquet(filename, index=False, compression='snappy')
    else:
        df.to_csv(filename, index=False, float_format='%.6f')

def main():
    """Main function for optimized data generation"""
    print("🚀 Starting Optimized CsPbBr3 Sample Data Generation")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    seed = config.get('data_generation', {}).get('random_seed', 42)
    set_random_seed(seed)
    
    # Generate samples with optimizations
    n_samples = 10000  # Larger default for testing optimization
    
    import time
    start_time = time.time()
    
    dataset = create_sample_dataset_parallel_optimized(n_samples, config=config)
    
    end_time = time.time()
    print(f"⏱️  Generation completed in {end_time - start_time:.2f} seconds")
    print(f"📊 Performance: {n_samples / (end_time - start_time):.0f} samples/second")
    
    # Save dataset
    output_file = "optimized_training_data.csv"
    save_dataset_optimized(dataset, output_file)
    
    # Display summary statistics
    print("\n📊 Dataset Summary:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Successful syntheses: {np.sum(dataset['phase_label'] != 4)}")
    print(f"   Average PLQY: {dataset['plqy'].mean():.3f}")
    print(f"   Average stability: {dataset['stability_score'].mean():.3f}")
    print("\n✅ Optimized data generation complete!")

if __name__ == "__main__":
    main()