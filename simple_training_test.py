#!/usr/bin/env python3
"""
Simple training test without complex dependencies
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_basic_config():
    """Create a basic configuration for testing"""
    config = {
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
        'quality_control': {
            'min_plqy': 0.1,
            'max_plqy': 1.0,
            'min_bandgap': 1.5,
            'max_bandgap': 3.5
        }
    }
    
    # Save config
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / 'data_generation_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def generate_simple_sample(parameters, config):
    """Generate a simple sample without complex dependencies"""
    import random
    import math
    
    # Extract parameters
    cs_conc = parameters.get('cs_br_concentration', 1.0)
    pb_conc = parameters.get('pb_br2_concentration', 1.0)
    temp = parameters.get('temperature', 150.0)
    oa_conc = parameters.get('oa_concentration', 0.1)
    oam_conc = parameters.get('oam_concentration', 0.1)
    time_min = parameters.get('reaction_time', 10.0)
    solvent = parameters.get('solvent_type', 0)
    
    # Simple physics-based calculations
    # Temperature effect on bandgap
    temp_normalized = (temp - 150) / 100  # Normalize around 150°C
    bandgap_base = 2.3
    bandgap = bandgap_base + temp_normalized * 0.2 + random.gauss(0, 0.1)
    bandgap = max(1.5, min(3.5, bandgap))
    
    # Concentration ratio effect on PLQY
    conc_ratio = cs_conc / pb_conc if pb_conc > 0 else 1.0
    plqy_base = 0.8
    plqy = plqy_base * math.exp(-abs(conc_ratio - 1.0)) + random.gauss(0, 0.05)
    plqy = max(0.0, min(1.0, plqy))
    
    # Ligand concentration effects
    ligand_effect = (oa_conc + oam_conc) / 2.0
    particle_size = 10 + ligand_effect * 5 + temp_normalized * 3 + random.gauss(0, 1)
    particle_size = max(1, particle_size)
    
    # Emission peak based on bandgap
    emission_peak = 1240 / bandgap  # Simple approximation (nm)
    emission_peak += random.gauss(0, 5)
    
    # FWHM based on particle size (quantum confinement)
    emission_fwhm = 20 + 100 / particle_size + random.gauss(0, 3)
    emission_fwhm = max(10, emission_fwhm)
    
    # Stability based on temperature and ligands
    stability_score = 0.9 - abs(temp_normalized) * 0.2 + ligand_effect * 0.1
    stability_score += random.gauss(0, 0.05)
    stability_score = max(0.0, min(1.0, stability_score))
    
    # Phase purity based on stoichiometry
    phase_purity = 1.0 - abs(conc_ratio - 1.0) * 0.2 + random.gauss(0, 0.05)
    phase_purity = max(0.5, min(1.0, phase_purity))
    
    return {
        **parameters,
        'bandgap': round(bandgap, 3),
        'plqy': round(plqy, 3),
        'emission_peak': round(emission_peak, 1),
        'particle_size': round(particle_size, 1),
        'emission_fwhm': round(emission_fwhm, 1),
        'lifetime': round(10 + random.gauss(0, 2), 1),
        'stability_score': round(stability_score, 3),
        'phase_purity': round(phase_purity, 3)
    }

def run_simple_training(n_samples=1000):
    """Run simple training without complex dependencies"""
    print("🚀 Starting simple training test...")
    
    # Create config
    config = create_basic_config()
    print("✅ Configuration created")
    
    # Generate training data
    print(f"📊 Generating {n_samples} training samples...")
    
    training_data = []
    for i in range(n_samples):
        # Generate random parameters
        params = {}
        for param, bounds in config['synthesis_parameters'].items():
            import random
            params[param] = random.uniform(bounds['min'], bounds['max'])
        
        # Add solvent type
        import random
        params['solvent_type'] = random.choices(
            config['solvents']['types'],
            weights=config['solvents']['weights']
        )[0]
        
        # Generate sample
        sample = generate_simple_sample(params, config)
        training_data.append(sample)
        
        if (i + 1) % 100 == 0:
            print(f"   Generated {i + 1}/{n_samples} samples")
    
    print("✅ Data generation complete")
    
    # Save training data
    output_file = "simple_training_data.csv"
    
    # Create CSV manually to avoid pandas dependency
    if training_data:
        headers = list(training_data[0].keys())
        
        with open(output_file, 'w') as f:
            # Write headers
            f.write(','.join(headers) + '\n')
            
            # Write data
            for sample in training_data:
                values = [str(sample[h]) for h in headers]
                f.write(','.join(values) + '\n')
        
        print(f"💾 Training data saved to: {output_file}")
    
    # Calculate basic statistics
    if training_data:
        print("\n📊 Basic Statistics:")
        
        # Calculate averages for key properties
        avg_plqy = sum(sample['plqy'] for sample in training_data) / len(training_data)
        avg_bandgap = sum(sample['bandgap'] for sample in training_data) / len(training_data)
        avg_particle_size = sum(sample['particle_size'] for sample in training_data) / len(training_data)
        avg_stability = sum(sample['stability_score'] for sample in training_data) / len(training_data)
        
        print(f"   Average PLQY: {avg_plqy:.3f}")
        print(f"   Average Bandgap: {avg_bandgap:.3f} eV")
        print(f"   Average Particle Size: {avg_particle_size:.1f} nm")
        print(f"   Average Stability: {avg_stability:.3f}")
        
        # Find best sample
        best_sample = max(training_data, key=lambda x: x['plqy'] + x['stability_score'])
        print(f"\n🎯 Best Sample (PLQY + Stability):")
        for key, value in best_sample.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    
    print("\n✅ Simple training test complete!")
    return training_data

if __name__ == "__main__":
    try:
        training_data = run_simple_training(500)  # Start with 500 samples
        print("🎉 Success! Simple training completed without errors.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()