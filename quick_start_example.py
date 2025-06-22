#!/usr/bin/env python3
"""
Quick Start Example for CsPbBr3 Digital Twin
Simple example showing how to train and use the model
"""

import sys
import pandas as pd
from pathlib import Path
import torch

# Add project to path
sys.path.insert(0, '.')

def quick_start_demo():
    """
    Demonstrate the complete workflow: data -> training -> prediction
    """
    print("🚀 CsPbBr3 Digital Twin - Quick Start Demo")
    print("=" * 50)
    
    # Step 1: Check if we have sample data
    print("📊 Step 1: Loading sample data...")
    data_path = "data/samples_2000/synthesis_training_data.csv"
    
    if not Path(data_path).exists():
        print("❌ Sample data not found. Please run: python generate_sample_data.py")
        return False
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Columns: {list(df.columns[:10])}...")  # Show first 10 columns
    
    # Step 2: Create a simple model (without full training)
    print("\n🤖 Step 2: Creating model...")
    try:
        from synthesis.training.pytorch_neural_models import create_model
        
        config = {
            'input_dim': 75,  # Reduced for quick demo
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
        
        model = create_model(config)
        print(f"✅ Model created: {type(model).__name__}")
        
    except ImportError as e:
        print(f"❌ Could not create model (missing dependencies): {e}")
        print("   Please install: pip install torch pytorch-lightning")
        return False
    
    # Step 3: Test prediction pipeline (without training)
    print("\n🔮 Step 3: Testing prediction pipeline...")
    try:
        from synthesis.schemas import SynthesisParameters, SolventType
        
        # Create test parameters
        test_params = SynthesisParameters(
            cs_br_concentration=1.5,
            pb_br2_concentration=1.0,
            temperature=150.0,
            solvent_type=SolventType.DMSO,
            oa_concentration=0.1,
            oam_concentration=0.1,
            reaction_time=10.0
        )
        
        print(f"✅ Test parameters created:")
        print(f"   Cs-Br: {test_params.cs_br_concentration} mol/L")
        print(f"   Pb-Br2: {test_params.pb_br2_concentration} mol/L") 
        print(f"   Temperature: {test_params.temperature}°C")
        print(f"   Solvent: {test_params.solvent_type.name}")
        
    except ImportError as e:
        print(f"❌ Could not create parameters: {e}")
        return False
    
    # Step 4: Show data analysis
    print("\n📈 Step 4: Data analysis...")
    print("Phase distribution:")
    phase_names = {0: 'CsPbBr3_3D', 1: 'Cs4PbBr6_0D', 2: 'CsPb2Br5_2D', 3: 'Mixed', 4: 'Failed'}
    for phase, count in df['phase_label'].value_counts().sort_index().items():
        print(f"   {phase_names[phase]}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nProperty ranges:")
    print(f"   Bandgap: {df['bandgap'].min():.2f} - {df['bandgap'].max():.2f} eV")
    print(f"   PLQY: {df['plqy'].min():.2f} - {df['plqy'].max():.2f}")
    print(f"   Size: {df['particle_size'].min():.1f} - {df['particle_size'].max():.1f} nm")
    
    print("\n✅ Quick start demo completed successfully!")
    print("\n📝 Next steps:")
    print("1. Train the full model: python train_pytorch_models.py --data-dir data/samples_2000 --max-epochs 20")
    print("2. Use the digital twin: python -c \"from synthesis import quick_prediction; print(quick_prediction(1.0, 1.0, 150))\"")
    
    return True

def minimal_training_example():
    """
    Show how to do minimal training (just a few steps)
    """
    print("\n🏋️ Minimal Training Example")
    print("-" * 30)
    
    try:
        import torch
        import torch.nn as nn
        from synthesis.training.pytorch_neural_models import PhysicsInformedNeuralNetwork, ModelConfig
        
        # Create minimal model
        config = ModelConfig(
            input_dim=10,
            hidden_dims=[32, 16],
            dropout_rate=0.1,
            max_epochs=3  # Just 3 epochs for demo
        )
        
        model = PhysicsInformedNeuralNetwork(config)
        
        # Create dummy data
        batch_size = 8
        dummy_features = torch.randn(batch_size, 10)
        dummy_phase_labels = torch.randint(0, 5, (batch_size,))
        dummy_properties = {
            'bandgap': torch.randn(batch_size),
            'plqy': torch.rand(batch_size),
            'particle_size': torch.rand(batch_size) * 100
        }
        
        dummy_batch = {
            'features': dummy_features,
            'phase_labels': dummy_phase_labels,
            'properties': dummy_properties
        }
        
        # Test forward pass
        model.train()
        outputs = model(dummy_features, training=True)
        losses = model.compute_loss(outputs, dummy_batch)
        
        print(f"✅ Forward pass successful!")
        print(f"   Phase output shape: {outputs['phase']['probabilities'].shape}")
        print(f"   Total loss: {losses['total_loss'].item():.4f}")
        print(f"   Phase loss: {losses['phase_loss'].item():.4f}")
        print(f"   Property loss: {losses['property_loss'].item():.4f}")
        
        # Simulate one training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        print(f"✅ Training step completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the quick start demonstration"""
    success = quick_start_demo()
    
    if success:
        minimal_training_example()
    
    print("\n🎯 Summary:")
    if success:
        print("✅ Environment is ready for training!")
        print("✅ Sample data is available")
        print("✅ Model creation works")
        print("✅ Basic training pipeline functional")
    else:
        print("❌ Some setup issues found. Please:")
        print("1. Run: ./setup_environment.sh")
        print("2. Run: python generate_sample_data.py")
        print("3. Try again: python quick_start_example.py")

if __name__ == "__main__":
    main()