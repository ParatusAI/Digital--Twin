#!/usr/bin/env python3
"""
Generate Sample Training Data for CsPbBr3 Digital Twin
Creates realistic synthesis data based on literature and physics models
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project to path
sys.path.insert(0, '.')

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from JSON file"""
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

def get_default_config() -> Dict:
    """Return default configuration if config file is not available"""
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
    """Set random seed for reproducibility across numpy operations"""
    np.random.seed(seed)
    # Also set the global random state for consistent results
    global_rng = np.random.RandomState(seed)
    return global_rng

def validate_physics_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physics-based constraints and validation to generated data"""
    print("🧪 Applying physics constraints...")
    
    # Stoichiometric constraints for perovskite formation
    cs_pb_ratio = df['cs_br_concentration'] / (df['pb_br2_concentration'] + 1e-8)
    
    # Flag unrealistic stoichiometry (too far from ideal ratios)
    df['stoichiometry_valid'] = (
        (cs_pb_ratio >= 0.5) & (cs_pb_ratio <= 3.0)
    )
    
    # Temperature-dependent solubility limits
    max_cs_solubility = 0.1 + 0.01 * (df['temperature'] - 25) / 100  # Rough approximation
    max_pb_solubility = 0.05 + 0.008 * (df['temperature'] - 25) / 100
    
    df['concentration_valid'] = (
        (df['cs_br_concentration'] <= max_cs_solubility * 10) &  # Allow some supersaturation
        (df['pb_br2_concentration'] <= max_pb_solubility * 10)
    )
    
    # Ligand concentration limits (based on typical synthesis protocols)
    df['ligand_valid'] = (
        (df['oa_concentration'] <= 2.0) & 
        (df['oam_concentration'] <= 2.0) &
        ((df['oa_concentration'] + df['oam_concentration']) <= 2.5)  # Total ligand limit
    )
    
    # Mark samples that violate multiple constraints as likely failed
    constraint_violations = (
        ~df['stoichiometry_valid'] + 
        ~df['concentration_valid'] + 
        ~df['ligand_valid']
    )
    
    # Increase failure probability for samples with constraint violations
    df['physics_failure_prob'] = np.clip(constraint_violations * 0.3, 0, 0.9)
    
    return df

def generate_synthesis_parameters(n_samples: int = 1000, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Generate realistic synthesis parameters based on literature ranges
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthesis parameters
    """
    print(f"📊 Generating {n_samples} synthesis parameter sets...")
    
    # Load configuration
    if config is None:
        config = load_config()
    
    # Extract parameter ranges from config
    param_config = config['synthesis_parameters']
    param_ranges = {
        param: (param_config[param]['min'], param_config[param]['max'])
        for param in param_config.keys()
    }
    
    # Extract solvent configuration
    solvent_config = config['solvents']
    solvent_types = solvent_config['types']
    solvent_weights = solvent_config['weights']
    
    data = {}
    
    # Generate continuous parameters with optimized dtypes
    for param, (min_val, max_val) in param_ranges.items():
        if param in ['cs_br_concentration', 'pb_br2_concentration']:
            # Log-normal distribution for concentrations with dtype specification
            log_mean = np.log((min_val + max_val) / 2)
            data[param] = np.clip(
                np.random.lognormal(log_mean, 0.5, size=n_samples),
                min_val, max_val
            ).astype(np.float32)
        elif param == 'temperature':
            # Normal distribution for temperature with dtype specification
            data[param] = np.clip(
                np.random.normal(150.0, 30.0, size=n_samples),
                min_val, max_val
            ).astype(np.float32)
        elif param in ['oa_concentration', 'oam_concentration']:
            # Many experiments with no ligands, some with ligands - vectorized
            prob_no_ligand = 0.3
            has_ligand_mask = np.random.random(n_samples) >= prob_no_ligand
            ligand_values = np.random.uniform(0.01, max_val, n_samples)
            data[param] = np.where(has_ligand_mask, ligand_values, 0.0).astype(np.float32)
        else:
            # Uniform distribution for other parameters with dtype specification
            data[param] = np.random.uniform(min_val, max_val, n_samples).astype(np.float32)
    
    # Generate solvent types with appropriate dtype
    data['solvent_type'] = np.random.choice(
        solvent_types, size=n_samples, p=solvent_weights
    ).astype(np.int8)
    
    return pd.DataFrame(data)

def calculate_physics_features(params_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate physics-based features for the synthesis parameters
    
    Args:
        params_df: DataFrame with synthesis parameters
        
    Returns:
        DataFrame with added physics features
    """
    print("⚗️ Calculating physics features...")
    
    # Simple physics calculations (approximations for speed)
    df = params_df.copy()
    
    # Supersaturation calculation (simplified) - optimized with constants
    R = 8.314  # Gas constant
    activation_energy = 5000  # Activation energy
    kelvin_offset = 273.15
    
    temp_kelvin = df['temperature'] + kelvin_offset
    df['supersaturation'] = (
        df['cs_br_concentration'] * df['pb_br2_concentration'] * 
        np.exp(-activation_energy / (R * temp_kelvin))
    ).astype(np.float32)
    
    # Nucleation rate approximation - using numpy power for efficiency
    nucleation_const = 1e10
    temp_scale = 100.0
    df['nucleation_rate'] = (
        nucleation_const * np.square(df['supersaturation']) * 
        np.exp(-df['temperature'] / temp_scale)
    ).astype(np.float32)
    
    # Growth rate approximation - vectorized calculation
    ligand_total = df['oa_concentration'] + df['oam_concentration']
    ligand_effect = 1.0 + ligand_total * 10.0
    df['growth_rate'] = np.divide(
        df['supersaturation'] * df['temperature'],
        ligand_effect
    ).astype(np.float32)
    
    # Solvent effects using numpy fancy indexing (faster than dict.map())
    # Array indices correspond to solvent types: [DMSO, DMF, Water, Toluene, Octadecene]
    solvent_effect_array = np.array([1.2, 1.0, 0.5, 0.8, 0.9], dtype=np.float32)
    df['solvent_effect'] = solvent_effect_array[df['solvent_type'].astype(int)]
    
    return df

def determine_phase_outcomes(params_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine phase outcomes based on synthesis conditions
    
    Args:
        params_df: DataFrame with synthesis parameters
        
    Returns:
        DataFrame with phase labels
    """
    print("🔬 Determining phase outcomes...")
    
    df = params_df.copy()
    
    # Calculate phase probabilities based on conditions
    # Safe division to prevent numerical instability
    # Use machine epsilon for float32 to avoid division by zero
    epsilon = np.finfo(np.float32).eps
    cs_pb_ratio = np.divide(
        df['cs_br_concentration'], 
        df['pb_br2_concentration'] + epsilon,
        out=np.ones_like(df['cs_br_concentration']),  # Default to 1.0 if division fails
        where=(df['pb_br2_concentration'] + epsilon) != 0
    )
    temp_normalized = (df['temperature'] - 80) / (250 - 80)
    
    # Phase determination logic (simplified but physically motivated)
    # 0: CsPbBr3_3D, 1: Cs4PbBr6_0D, 2: CsPb2Br5_2D, 3: Mixed, 4: Failed
    
    prob_3d = (
        0.6 * (1 - np.abs(cs_pb_ratio - 1.0)) *  # Stoichiometric favors 3D
        (0.5 + 0.5 * temp_normalized) *           # Higher temp favors 3D
        df['solvent_effect']                      # Solvent effects
    )
    
    prob_0d = (
        0.4 * np.maximum(0, cs_pb_ratio - 1.5) *  # Excess Cs favors 0D
        (1 - temp_normalized) *                    # Lower temp favors 0D
        df['solvent_effect']
    )
    
    prob_2d = (
        0.3 * np.maximum(0, 1.0 / cs_pb_ratio - 1.0) *  # Excess Pb favors 2D
        temp_normalized *                                 # Higher temp favors 2D
        df['solvent_effect']
    )
    
    # Mixed phases when conditions are borderline
    prob_mixed = 0.2 * (1 - prob_3d - prob_0d - prob_2d)
    
    # Enhanced failure probability based on physics constraints
    base_failure_prob = np.where(
        (df['supersaturation'] < 0.1) | 
        (df['temperature'] < 90) | 
        (df['temperature'] > 240),
        0.8, 0.05
    )
    
    # Add physics-based failure probability if available
    if 'physics_failure_prob' in df.columns:
        prob_failed = np.maximum(base_failure_prob, df['physics_failure_prob'])
    else:
        prob_failed = base_failure_prob
    
    # Normalize probabilities with safe division
    total_prob = prob_3d + prob_0d + prob_2d + prob_mixed + prob_failed
    # Prevent division by zero
    total_prob = np.maximum(total_prob, epsilon)
    
    prob_3d = np.divide(prob_3d, total_prob)
    prob_0d = np.divide(prob_0d, total_prob)
    prob_2d = np.divide(prob_2d, total_prob)
    prob_mixed = np.divide(prob_mixed, total_prob)
    prob_failed = np.divide(prob_failed, total_prob)
    
    # Sample phase labels - vectorized approach
    # This replaces the slow Python loop with efficient numpy operations
    # cumsum creates cumulative probability distribution for each sample
    phase_probs = np.column_stack([prob_3d, prob_0d, prob_2d, prob_mixed, prob_failed])
    # Ensure probabilities are non-negative and normalized
    phase_probs = np.abs(phase_probs)
    phase_probs = phase_probs / phase_probs.sum(axis=1, keepdims=True)
    
    # Vectorized sampling using cumulative probabilities
    cumsum_probs = np.cumsum(phase_probs, axis=1)
    # Generate random values and broadcast for comparison
    random_vals = np.random.random(len(phase_probs))[:, np.newaxis]
    # Find first index where random < cumulative prob (vectorized choice)
    phase_labels = np.argmax(random_vals < cumsum_probs, axis=1)
    
    df['phase_label'] = phase_labels
    
    # Store probabilities for uncertainty analysis
    for i, phase_name in enumerate(['prob_3d', 'prob_0d', 'prob_2d', 'prob_mixed', 'prob_failed']):
        df[phase_name] = phase_probs[:, i]
    
    return df

def generate_material_properties(params_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate material properties based on phase and synthesis conditions
    
    Args:
        params_df: DataFrame with synthesis parameters and phase labels
        
    Returns:
        DataFrame with material properties
    """
    print("💎 Generating material properties...")
    
    df = params_df.copy()
    
    # Base properties for each phase (literature values)
    phase_properties = {
        0: {'bandgap': 2.3, 'plqy': 0.85, 'emission_peak': 520, 'size': 12},  # 3D
        1: {'bandgap': 3.9, 'plqy': 0.15, 'emission_peak': 410, 'size': 5},   # 0D
        2: {'bandgap': 2.9, 'plqy': 0.45, 'emission_peak': 460, 'size': 8},   # 2D
        3: {'bandgap': 2.6, 'plqy': 0.35, 'emission_peak': 480, 'size': 10},  # Mixed
        4: {'bandgap': 0.0, 'plqy': 0.0, 'emission_peak': 0, 'size': 0}       # Failed
    }
    
    # Initialize properties
    properties = ['bandgap', 'plqy', 'emission_peak', 'particle_size']
    for prop in properties:
        df[prop] = 0.0
    
    # Generate properties based on phase
    for phase in range(5):
        mask = df['phase_label'] == phase
        if not mask.any():
            continue
            
        base_props = phase_properties[phase]
        n_samples = mask.sum()
        
        # Bandgap with temperature dependence and noise
        temp_coeff = -0.0003  # eV/K
        temp_ref = 150.0  # Reference temperature
        bandgap_noise_std = 0.1
        min_bandgap = 0.5
        
        temp_effect = temp_coeff * (df.loc[mask, 'temperature'] - temp_ref)
        bandgap_values = (
            base_props['bandgap'] + temp_effect + 
            np.random.normal(0, bandgap_noise_std, n_samples)
        )
        df.loc[mask, 'bandgap'] = np.maximum(bandgap_values, min_bandgap).astype(np.float32)
        
        # PLQY with ligand effects
        ligand_effect = (df.loc[mask, 'oa_concentration'] + df.loc[mask, 'oam_concentration']) * 0.1
        df.loc[mask, 'plqy'] = np.clip(
            base_props['plqy'] + ligand_effect + np.random.normal(0, 0.1, n_samples),
            0.0, 1.0
        )
        
        # Emission peak (correlated with bandgap) - safe division
        planck_constant_c = 1240  # eV⋅nm (hc for energy-wavelength conversion)
        emission_noise_std = 10  # nm
        
        # Safe division to prevent issues with very small bandgaps
        emission_base = np.divide(
            planck_constant_c,
            df.loc[mask, 'bandgap'],
            out=np.full(n_samples, 500.0),  # Default 500nm if division fails
            where=df.loc[mask, 'bandgap'] > 0.1
        )
        
        df.loc[mask, 'emission_peak'] = (
            emission_base + np.random.normal(0, emission_noise_std, n_samples)
        ).astype(np.float32)
        
        # Particle size with time and temperature dependence
        time_scale = 2.0
        temp_scale = 0.05
        size_noise_std = 2.0
        min_size = 1.0
        
        # Safe log calculation
        time_effect = np.log(np.maximum(df.loc[mask, 'reaction_time'], 0.1)) * time_scale
        temp_effect = (df.loc[mask, 'temperature'] - temp_ref) * temp_scale
        
        size_values = (
            base_props['size'] + time_effect + temp_effect + 
            np.random.normal(0, size_noise_std, n_samples)
        )
        df.loc[mask, 'particle_size'] = np.maximum(size_values, min_size).astype(np.float32)
    
    # Additional properties with uncertainty quantification
    df['emission_fwhm'] = 20 + np.random.normal(0, 5, len(df))
    df['emission_fwhm'] = np.maximum(df['emission_fwhm'], 10)
    df['emission_fwhm_uncertainty'] = np.random.uniform(0.05, 0.15, len(df)) * df['emission_fwhm']
    
    df['size_distribution_width'] = 0.2 + np.random.exponential(0.1, len(df))
    df['size_distribution_width'] = np.minimum(df['size_distribution_width'], 0.8)
    
    # Add uncertainty estimates for key properties
    df['bandgap_uncertainty'] = np.random.uniform(0.02, 0.08, len(df))  # ±2-8% uncertainty
    df['plqy_uncertainty'] = np.random.uniform(0.05, 0.15, len(df))     # ±5-15% uncertainty  
    df['particle_size_uncertainty'] = np.random.uniform(0.10, 0.25, len(df))  # ±10-25% uncertainty
    
    # Synthesis confidence score (0-1) based on parameter reliability
    synthesis_confidence = np.ones(len(df))
    
    # Lower confidence for extreme conditions
    temp_normalized = (df['temperature'] - 150) / 100
    synthesis_confidence *= np.exp(-0.5 * temp_normalized**2)  # Gaussian confidence around 150°C
    
    # Lower confidence for non-stoichiometric ratios
    if 'cs_br_concentration' in df.columns and 'pb_br2_concentration' in df.columns:
        cs_pb_ratio = df['cs_br_concentration'] / (df['pb_br2_concentration'] + 1e-8)
        stoich_deviation = np.abs(cs_pb_ratio - 1.0)
        synthesis_confidence *= np.exp(-stoich_deviation)
    
    df['synthesis_confidence'] = np.clip(synthesis_confidence, 0.1, 1.0)
    
    df['lifetime'] = np.where(
        df['phase_label'] == 0,
        np.random.normal(15, 5, len(df)),    # 3D perovskites
        np.random.normal(5, 2, len(df))      # Other phases
    )
    df['lifetime'] = np.maximum(df['lifetime'], 0.5)
    
    df['stability_score'] = np.clip(
        0.7 + 0.2 * df['plqy'] + np.random.normal(0, 0.1, len(df)),
        0.0, 1.0
    )
    
    df['phase_purity'] = np.where(
        df['phase_label'] == 3,  # Mixed phases
        np.random.uniform(0.3, 0.7, len(df)),
        np.random.uniform(0.8, 1.0, len(df))
    )
    
    return df

def validate_generated_data(df: pd.DataFrame) -> Dict:
    """Validate generated data against known literature ranges"""
    print("📊 Validating generated data...")
    
    validation_results = {
        'n_samples': len(df),
        'physics_violations': 0,
        'property_ranges': {},
        'warnings': []
    }
    
    # Literature ranges for validation
    expected_ranges = {
        'bandgap': (1.5, 4.5),  # eV
        'plqy': (0.0, 1.0),     # Unitless
        'emission_peak': (350, 700),  # nm
        'particle_size': (1, 100),    # nm
        'emission_fwhm': (10, 80),    # nm
    }
    
    for prop, (min_val, max_val) in expected_ranges.items():
        if prop in df.columns:
            actual_min = df[prop].min()
            actual_max = df[prop].max()
            
            validation_results['property_ranges'][prop] = {
                'expected': (min_val, max_val),
                'actual': (float(actual_min), float(actual_max)),
                'in_range': (actual_min >= min_val * 0.8) and (actual_max <= max_val * 1.2)
            }
            
            if not validation_results['property_ranges'][prop]['in_range']:
                validation_results['warnings'].append(
                    f"{prop} values outside expected range: {actual_min:.2f}-{actual_max:.2f} vs {min_val}-{max_val}"
                )
    
    # Check for physics violations
    if 'stoichiometry_valid' in df.columns:
        violations = (~df['stoichiometry_valid']).sum()
        validation_results['physics_violations'] += violations
        if violations > len(df) * 0.1:  # More than 10% violations
            validation_results['warnings'].append(
                f"High stoichiometry violations: {violations}/{len(df)} samples"
            )
    
    # Additional quality checks
    
    # Check for NaN or infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_counts = df[numeric_cols].isna().sum()
    inf_counts = np.isinf(df[numeric_cols]).sum()
    
    if nan_counts.sum() > 0:
        validation_results['warnings'].append(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
    
    if inf_counts.sum() > 0:
        validation_results['warnings'].append(f"Found infinite values: {inf_counts[inf_counts > 0].to_dict()}")
    
    # Check for duplicate samples (unlikely but possible)
    param_cols = ['cs_br_concentration', 'pb_br2_concentration', 'temperature', 'solvent_type']
    if all(col in df.columns for col in param_cols):
        duplicates = df[param_cols].duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"Found {duplicates} duplicate parameter combinations")
    
    # Physical consistency checks
    if 'bandgap' in df.columns and 'emission_peak' in df.columns:
        # Check bandgap-emission correlation (E = hc/λ)
        expected_wavelength = 1240 / df['bandgap']  # eV to nm
        wavelength_diff = np.abs(df['emission_peak'] - expected_wavelength)
        large_deviations = (wavelength_diff > 100).sum()  # >100nm deviation
        
        if large_deviations > len(df) * 0.05:  # More than 5% with large deviations
            validation_results['warnings'].append(
                f"Large bandgap-emission inconsistencies: {large_deviations} samples with >100nm deviation"
            )
    
    return validation_results

def filter_low_quality_samples(df: pd.DataFrame, quality_threshold: float = 0.8) -> pd.DataFrame:
    """Remove samples that fail multiple quality checks"""
    print(f"🔍 Filtering low-quality samples (threshold: {quality_threshold})...")
    
    initial_count = len(df)
    
    # Create quality score based on multiple factors
    quality_score = np.ones(len(df))
    
    # Penalize physics violations
    if 'stoichiometry_valid' in df.columns:
        quality_score *= df['stoichiometry_valid'].astype(float)
    if 'concentration_valid' in df.columns:
        quality_score *= df['concentration_valid'].astype(float)
    if 'ligand_valid' in df.columns:
        quality_score *= df['ligand_valid'].astype(float)
    
    # Penalize extreme values
    for col in ['supersaturation', 'nucleation_rate', 'growth_rate']:
        if col in df.columns:
            # Flag samples in extreme percentiles
            lower_thresh = df[col].quantile(0.01)
            upper_thresh = df[col].quantile(0.99)
            extreme_mask = (df[col] < lower_thresh) | (df[col] > upper_thresh)
            quality_score *= (~extreme_mask).astype(float)
    
    # Keep only high-quality samples
    high_quality_mask = quality_score >= quality_threshold
    filtered_df = df[high_quality_mask].copy()
    
    removed_count = initial_count - len(filtered_df)
    print(f"   Removed {removed_count} low-quality samples ({removed_count/initial_count*100:.1f}%)")
    
    return filtered_df

def generate_batch(batch_args: Tuple[int, int, Dict]) -> pd.DataFrame:
    """Generate a batch of samples for parallel processing"""
    batch_size, random_offset, config = batch_args
    
    # Set unique random seed for this batch
    base_seed = config.get('data_generation', {}).get('random_seed', 42)
    np.random.seed(base_seed + random_offset)
    
    # Generate batch
    params_df = generate_synthesis_parameters(batch_size, config)
    validated_params = validate_physics_constraints(params_df, config)
    physics_df = calculate_physics_features(validated_params, config)
    phase_df = determine_phase_outcomes(physics_df, config)
    complete_df = generate_material_properties(phase_df, config)
    
    return complete_df

def create_sample_dataset(
    n_samples: int = 1000, 
    output_dir: str = "data", 
    config_path: Optional[str] = None,
    use_parallel: bool = True,
    n_processes: Optional[int] = None
) -> str:
    """
    Create complete sample dataset
    
    Args:
        n_samples: Number of samples to generate
        output_dir: Output directory for data files
        
    Returns:
        Path to generated CSV file
    """
    print("🎯 Creating complete sample dataset...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Determine if parallel processing should be used
    parallel_threshold = 10000  # Use parallel processing for datasets > 10k samples
    use_parallel = use_parallel and n_samples > parallel_threshold
    
    if use_parallel:
        print(f"🚀 Using parallel processing for {n_samples} samples...")
        
        # Determine number of processes
        if n_processes is None:
            n_processes = min(cpu_count(), 8)  # Limit to 8 processes max
        
        # Calculate batch sizes
        batch_size = max(1000, n_samples // n_processes)  # Minimum 1000 samples per batch
        n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
        
        print(f"   Using {n_processes} processes with {n_batches} batches of ~{batch_size} samples each")
        
        # Prepare batch arguments
        batch_args = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            batch_args.append((current_batch_size, i * 1000, config))  # Offset random seeds
        
        # Generate batches in parallel
        with Pool(processes=n_processes) as pool:
            batch_results = pool.map(generate_batch, batch_args)
        
        # Combine results
        complete_df = pd.concat(batch_results, ignore_index=True)
        print(f"   ✅ Parallel generation complete: {len(complete_df)} samples")
        
    else:
        print(f"🔄 Using sequential processing for {n_samples} samples...")
        # Sequential processing for smaller datasets
        params_df = generate_synthesis_parameters(n_samples, config)
        validated_params = validate_physics_constraints(params_df, config)
        physics_df = calculate_physics_features(validated_params, config)
        phase_df = determine_phase_outcomes(physics_df, config)
        complete_df = generate_material_properties(phase_df, config)
    
    # Optional quality filtering based on config
    quality_config = config.get('quality_control', {})
    if quality_config.get('enable_quality_filter', True):
        quality_threshold = quality_config.get('quality_threshold', 0.8)
        complete_df = filter_low_quality_samples(complete_df, quality_threshold)
    
    # Validate the final dataset
    validation_results = validate_generated_data(complete_df)
    
    # Add some metadata
    complete_df['sample_id'] = range(len(complete_df))
    complete_df['dataset_version'] = '1.0'
    
    # Save to CSV
    csv_path = Path(output_dir) / 'synthesis_training_data.csv'
    complete_df.to_csv(csv_path, index=False)
    
    # Save data info
    info = {
        'n_samples': len(complete_df),
        'columns': list(complete_df.columns),
        'phase_distribution': complete_df['phase_label'].value_counts().to_dict(),
        'parameter_ranges': {
            col: {'min': float(complete_df[col].min()), 'max': float(complete_df[col].max())}
            for col in ['cs_br_concentration', 'pb_br2_concentration', 'temperature']
        }
    }
    
    info_path = Path(output_dir) / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Dataset saved to: {csv_path}")
    print(f"📊 Dataset info saved to: {info_path}")
    
    # Print summary
    print("\n📈 Dataset Summary:")
    print(f"   Total samples: {len(complete_df)}")
    print(f"   Features: {len(complete_df.columns)}")
    print("   Phase distribution:")
    phase_names = {0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 3: 'Mixed', 4: 'Failed'}
    for phase, count in complete_df['phase_label'].value_counts().sort_index().items():
        print(f"     {phase_names[phase]}: {count} ({count/len(complete_df)*100:.1f}%)")
    
    return str(csv_path)

def main():
    """Main function to generate sample data"""
    print("🧪 CsPbBr3 Digital Twin - Sample Data Generator")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Generate datasets of different sizes
    datasets = [
        (500, "Training dataset (small)"),
        (2000, "Training dataset (medium)"),
        (5000, "Training dataset (large)")
    ]
    
    generated_files = []
    
    for n_samples, description in datasets:
        print(f"\n🎲 Generating {description}...")
        output_dir = f"data/samples_{n_samples}"
        # Use parallel processing for larger datasets
        use_parallel = n_samples >= 2000
        csv_path = create_sample_dataset(n_samples, output_dir, use_parallel=use_parallel)
        generated_files.append(csv_path)
    
    print("\n" + "=" * 50)
    print("🎉 Sample data generation complete!")
    print("\n📁 Generated files:")
    for file_path in generated_files:
        print(f"   {file_path}")
    
    print("\n📝 Next steps:")
    print("1. Examine the data: pandas.read_csv('data/samples_2000/synthesis_training_data.csv')")
    print("2. Train the model: python train_pytorch_models.py --data-dir data/samples_2000")
    print("3. Make predictions with the trained model")

if __name__ == "__main__":
    main()