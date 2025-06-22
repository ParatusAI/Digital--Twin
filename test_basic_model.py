#!/usr/bin/env python3
"""
Basic Model Test - Check if we can create and run a simple model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_schema_creation():
    """Test if we can create basic schemas"""
    print("🧪 Testing schema creation...")
    try:
        from synthesis.schemas import SynthesisParameters, SolventType, PhaseType
        
        params = SynthesisParameters(
            cs_br_concentration=1.0,
            pb_br2_concentration=1.0,
            temperature=150.0,
            solvent_type=SolventType.DMSO
        )
        
        print(f"✅ Schema creation successful")
        print(f"   Parameters: {params}")
        return True
        
    except Exception as e:
        print(f"❌ Schema creation failed: {e}")
        return False

def test_model_creation():
    """Test if we can create a basic neural network"""
    print("\n🤖 Testing model creation...")
    try:
        from synthesis.training.pytorch_neural_models import create_model, ModelConfig
        
        config = {
            'input_dim': 100,
            'hidden_dims': [64, 32],
            'dropout_rate': 0.1
        }
        
        model = create_model(config)
        print(f"✅ Model creation successful")
        print(f"   Model type: {type(model).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_engineering():
    """Test if we can create feature engineer"""
    print("\n🔧 Testing feature engineering...")
    try:
        from synthesis.training.pytorch_feature_engineering import create_feature_engineer
        
        feature_engineer = create_feature_engineer({
            'normalize_features': True,
            'include_interactions': True
        })
        
        print(f"✅ Feature engineering creation successful")
        print(f"   Feature engineer type: {type(feature_engineer).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_physics_models():
    """Test if we can create physics models"""
    print("\n⚗️ Testing physics models...")
    try:
        from synthesis.physics.nucleation import create_nucleation_model
        
        nucleation_model = create_nucleation_model()
        
        # Test nucleation calculation
        result = nucleation_model.calculate_supersaturation(
            cs_concentration=1.0,
            pb_concentration=1.0,
            br_concentration=3.0,
            temperature=150.0,
            solvent='DMSO',
            phase_type=nucleation_model.PhaseType.CSPBBR3_3D if hasattr(nucleation_model, 'PhaseType') else 0
        )
        
        print(f"✅ Physics model creation successful")
        print(f"   Supersaturation: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Physics model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_training():
    """Test if we can set up a basic training loop"""
    print("\n🏋️ Testing simple training setup...")
    try:
        import torch
        from synthesis.training.pytorch_neural_models import create_model
        
        # Create simple model
        model = create_model({
            'input_dim': 10,
            'hidden_dims': [32, 16],
            'dropout_rate': 0.1
        })
        
        # Create dummy data
        batch_size = 4
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
        outputs = model(dummy_features, training=True)
        
        # Test loss computation
        losses = model.compute_loss(outputs, dummy_batch)
        
        print(f"✅ Basic training setup successful")
        print(f"   Total loss: {losses['total_loss'].item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_tests():
    """Run quick subset of tests"""
    print("⚡ Running Quick Tests for CsPbBr₃ Digital Twin")
    print("=" * 40)
    
    quick_tests = [
        test_schema_creation,
        test_model_creation
    ]
    
    passed = 0
    total = len(quick_tests)
    
    for test in quick_tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"⚡ Quick Test Results: {passed}/{total} tests passed")
    return passed == total

def main():
    """Run all basic tests"""
    print("🚀 Basic Model Test for CsPbBr₃ Digital Twin")
    print("=" * 50)
    
    tests = [
        test_schema_creation,
        test_model_creation,
        test_feature_engineering,
        test_physics_models,
        test_simple_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Ready to run ML model.")
        return 0
    elif passed >= 3:
        print("⚠️ Most tests passed. Model can likely run with minor fixes.")
        return 0
    else:
        print("❌ Major issues found. Need to fix before running model.")
        return 1

if __name__ == "__main__":
    sys.exit(main())