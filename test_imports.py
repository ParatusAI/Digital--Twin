#!/usr/bin/env python3
"""
Test script to check all imports and identify missing dependencies
"""

import sys
import warnings

print("🔍 Testing imports for enhanced training pipeline...")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Test basic scientific packages
try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    from sklearn.preprocessing import StandardScaler
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

# Test our custom modules
print("\n🧪 Testing custom module imports...")

try:
    from generate_sample_data import generate_sample_batch, load_config
    print("✅ generate_sample_data imported successfully")
except ImportError as e:
    print(f"❌ generate_sample_data import failed: {e}")

try:
    from ml_optimization import BayesianOptimizer, create_optimization_strategy
    print("✅ ml_optimization imported successfully")
except ImportError as e:
    print(f"❌ ml_optimization import failed: {e}")

try:
    from adaptive_sampling import create_adaptive_sampling_strategy
    print("✅ adaptive_sampling imported successfully")
except ImportError as e:
    print(f"❌ adaptive_sampling import failed: {e}")

try:
    from monitoring import RealTimeMonitor, StructuredLogger
    print("✅ monitoring imported successfully")
except ImportError as e:
    print(f"❌ monitoring import failed: {e}")

try:
    from gpu_acceleration import GPUDataGenerator, GPUCapabilities
    print("✅ gpu_acceleration imported successfully")
except ImportError as e:
    print(f"❌ gpu_acceleration import failed: {e}")

try:
    from advanced_physics import PhysicsModelSuite
    print("✅ advanced_physics imported successfully")
except ImportError as e:
    print(f"❌ advanced_physics import failed: {e}")

try:
    from data_versioning import DataVersionManager
    print("✅ data_versioning imported successfully")
except ImportError as e:
    print(f"❌ data_versioning import failed: {e}")

try:
    from experimental_integration import ExperimentalDataManager, ModelCalibrator
    print("✅ experimental_integration imported successfully")
except ImportError as e:
    print(f"❌ experimental_integration import failed: {e}")

print("\n✅ Import testing complete!")