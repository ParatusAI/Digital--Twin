#!/usr/bin/env python3
"""
Simple Training Data Generator for CsPbBr₃ Digital Twin
Creates reliable synthetic data for model training
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse

def generate_synthesis_parameters(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthesis parameters"""
    np.random.seed(seed)
    
    data = {
        # Main concentrations (mol/L)
        'cs_br_concentration': np.random.uniform(0.5, 3.0, n_samples),
        'pb_br2_concentration': np.random.uniform(0.3, 2.0, n_samples),
        
        # Temperature (°C)
        'temperature': np.random.uniform(80, 250, n_samples),
        
        # Ligand concentrations (mol/L)
        'oa_concentration': np.random.uniform(0.0, 1.0, n_samples),
        'oam_concentration': np.random.uniform(0.0, 1.0, n_samples),
        
        # Reaction time (minutes)
        'reaction_time': np.random.uniform(5, 120, n_samples),
        
        # Solvent type (0: DMSO, 1: DMF, 2: water, 3: toluene, 4: octadecene)
        'solvent_type': np.random.randint(0, 5, n_samples)
    }
    
    return pd.DataFrame(data)

def calculate_simple_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate simplified physics-based features"""
    result_df = df.copy()
    
    # Stoichiometric ratio
    result_df['cs_pb_ratio'] = df['cs_br_concentration'] / df['pb_br2_concentration']
    
    # Simple supersaturation approximation
    result_df['supersaturation'] = np.log(
        (df['cs_br_concentration'] * df['pb_br2_concentration']) / 
        (0.1 + df['temperature'] / 1000)  # Temperature-dependent solubility
    )
    
    # Ligand effects
    ligand_total = df['oa_concentration'] + df['oam_concentration']
    result_df['ligand_ratio'] = ligand_total / (df['cs_br_concentration'] + df['pb_br2_concentration'])
    
    # Normalized temperature
    result_df['temp_normalized'] = (df['temperature'] - 80) / (250 - 80)
    
    # Solvent effects (simplified)
    solvent_effects = {0: 1.2, 1: 1.0, 2: 0.5, 3: 0.8, 4: 0.9}
    result_df['solvent_effect'] = df['solvent_type'].map(solvent_effects)
    
    return result_df

def determine_phase_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Determine phase outcomes using simplified rules"""
    result_df = df.copy()
    
    # Phase probabilities based on physics
    cs_pb_ratio = df['cs_pb_ratio']
    temp_norm = df['temp_normalized']
    solvent_eff = df['solvent_effect']
    ligand_ratio = df['ligand_ratio']
    
    # Calculate probabilities for each phase
    # 0: CsPbBr3_3D, 1: Cs4PbBr6_0D, 2: CsPb2Br5_2D, 3: Mixed, 4: Failed
    
    # 3D CsPbBr3 favored by stoichiometric ratios and moderate temps
    prob_3d = np.clip(
        0.7 * np.exp(-np.abs(cs_pb_ratio - 1.0)) * 
        (0.3 + 0.7 * temp_norm) * 
        solvent_eff * 
        (1.0 - ligand_ratio * 0.5), 
        0.01, 0.98
    )
    
    # 0D Cs4PbBr6 favored by excess Cs and lower temps
    prob_0d = np.clip(
        0.5 * np.maximum(0, cs_pb_ratio - 1.5) * 
        (1.0 - temp_norm) * 
        solvent_eff * 0.8,
        0.01, 0.6
    )
    
    # 2D CsPb2Br5 favored by excess Pb and high ligand concentration
    prob_2d = np.clip(
        0.4 * np.maximum(0, 1.0 - cs_pb_ratio) * 
        temp_norm * 
        ligand_ratio * 2.0,
        0.01, 0.5
    )
    
    # Mixed phases
    prob_mixed = np.clip(0.15 + 0.1 * np.random.random(len(df)), 0.01, 0.3)
    
    # Failed synthesis (low probability)
    prob_failed = np.clip(0.05 + 0.05 * np.random.random(len(df)), 0.01, 0.15)
    
    # Normalize probabilities
    total_prob = prob_3d + prob_0d + prob_2d + prob_mixed + prob_failed
    prob_3d /= total_prob
    prob_0d /= total_prob
    prob_2d /= total_prob
    prob_mixed /= total_prob
    prob_failed /= total_prob
    
    # Sample phases
    phase_labels = []
    for i in range(len(df)):
        probs = [prob_3d.iloc[i], prob_0d.iloc[i], prob_2d.iloc[i], 
                prob_mixed.iloc[i], prob_failed.iloc[i]]
        phase = np.random.choice(5, p=probs)
        phase_labels.append(phase)
    
    result_df['phase_label'] = phase_labels
    
    return result_df

def calculate_material_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate material properties based on phase and conditions"""
    result_df = df.copy()
    
    # Base properties by phase
    phase_properties = {
        0: {'bandgap_base': 2.3, 'plqy_base': 0.8, 'size_base': 15.0},  # CsPbBr3_3D
        1: {'bandgap_base': 2.8, 'plqy_base': 0.6, 'size_base': 5.0},   # Cs4PbBr6_0D
        2: {'bandgap_base': 2.1, 'plqy_base': 0.4, 'size_base': 25.0},  # CsPb2Br5_2D
        3: {'bandgap_base': 2.4, 'plqy_base': 0.5, 'size_base': 12.0},  # Mixed
        4: {'bandgap_base': 1.8, 'plqy_base': 0.1, 'size_base': 50.0}   # Failed
    }
    
    properties = {'bandgap': [], 'plqy': [], 'particle_size': [], 'emission_peak': []}
    
    for _, row in df.iterrows():
        phase = int(row['phase_label'])
        base_props = phase_properties[phase]
        
        # Add variations based on synthesis conditions
        temp_factor = 1.0 + (row['temp_normalized'] - 0.5) * 0.2
        ligand_factor = 1.0 - row['ligand_ratio'] * 0.3
        
        # Bandgap (eV)
        bandgap = base_props['bandgap_base'] * temp_factor
        bandgap += np.random.normal(0, 0.1)
        properties['bandgap'].append(np.clip(bandgap, 1.5, 3.5))
        
        # PLQY (0-1)
        plqy = base_props['plqy_base'] * ligand_factor
        plqy += np.random.normal(0, 0.05)
        properties['plqy'].append(np.clip(plqy, 0.0, 1.0))
        
        # Particle size (nm)
        size = base_props['size_base'] * temp_factor * ligand_factor
        size += np.random.normal(0, 2.0)
        properties['particle_size'].append(np.clip(size, 2.0, 100.0))
        
        # Emission peak (nm) - approximate from bandgap
        emission = 1240 / bandgap  # eV to nm conversion
        emission += np.random.normal(0, 5)
        properties['emission_peak'].append(np.clip(emission, 400, 700))
    
    for prop, values in properties.items():
        result_df[prop] = values
    
    return result_df

def create_training_dataset(n_samples: int = 1000, output_file: str = "training_data.csv", seed: int = 42):
    """Create complete training dataset"""
    print(f"🧪 Generating {n_samples} training samples...")
    
    # Generate synthesis parameters
    print("📊 Generating synthesis parameters...")
    df = generate_synthesis_parameters(n_samples, seed)
    
    # Calculate physics features
    print("⚗️ Calculating physics features...")
    df = calculate_simple_physics_features(df)
    
    # Determine phase outcomes
    print("🔬 Determining phase outcomes...")
    df = determine_phase_outcomes(df)
    
    # Calculate material properties
    print("💎 Calculating material properties...")
    df = calculate_material_properties(df)
    
    # Save to CSV
    print(f"💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n✅ Training data created successfully!")
    print(f"   Samples: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    print("   Phase distribution:")
    phase_names = {0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 3: 'Mixed', 4: 'Failed'}
    for phase, count in df['phase_label'].value_counts().sort_index().items():
        print(f"     {phase_names[phase]}: {count} ({count/len(df)*100:.1f}%)")
    
    return output_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate training data for CsPbBr₃ Digital Twin")
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='training_data.csv', help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_training_dataset(args.samples, args.output, args.seed)

if __name__ == "__main__":
    main()