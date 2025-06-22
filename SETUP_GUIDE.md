# CsPbBr₃ Digital Twin - Complete Setup Guide

## 🎯 Overview

This guide will help you set up and run the CsPbBr₃ Digital Twin machine learning system from scratch. The system predicts perovskite synthesis outcomes using physics-informed neural networks.

## ✅ What's Ready

All components are complete and ready to run:

- ✅ **Physics-informed neural networks** with uncertainty quantification
- ✅ **Classical physics models** (nucleation, growth, phase selection, ligand effects)
- ✅ **Advanced feature engineering** with 100+ physics-based features
- ✅ **Complete training pipeline** with PyTorch Lightning
- ✅ **Sample data generation** with realistic synthesis parameters
- ✅ **Production-ready API** and experiment management

## 🚀 Step-by-Step Setup

### Step 1: Environment Setup

```bash
# Navigate to the project directory
cd cspbbr3_digital_twin

# Make setup script executable
chmod +x setup_environment.sh

# Run the setup script (installs all dependencies)
./setup_environment.sh

# Activate the virtual environment
source venv/bin/activate
```

**What this does:**
- Creates a Python virtual environment
- Installs PyTorch, PyTorch Lightning, and all dependencies
- Verifies the installation

### Step 2: Generate Training Data

```bash
# Create sample datasets (works without dependencies)
python3 create_sample_csv.py

# OR generate more sophisticated data (requires numpy/pandas)
python generate_sample_data.py
```

**Sample data includes:**
- 1000 realistic synthesis parameter sets
- Phase outcomes (3D, 0D, 2D, mixed, failed)
- Material properties (bandgap, PLQY, size, etc.)
- Physics-based relationships

### Step 3: Test the Setup

```bash
# Run basic tests
python quick_start_example.py

# Test model components
python test_basic_model.py
```

**This verifies:**
- All imports work correctly
- Model creation is functional
- Basic training pipeline works

### Step 4: Train the Model

```bash
# Basic training (CPU, small dataset)
python train_pytorch_models.py \
    --data-file data/sample_data_1000.csv \
    --max-epochs 20 \
    --batch-size 16

# Advanced training (GPU, larger dataset)
python train_pytorch_models.py \
    --data-dir data/samples_5000 \
    --max-epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --n-folds 5 \
    --use-wandb \
    --experiment-name "my_experiment"
```

**Training features:**
- Cross-validation support
- Experiment tracking with TensorBoard/WandB
- Model checkpointing and early stopping
- Physics constraint validation

### Step 5: Make Predictions

```python
# Quick prediction
from synthesis import quick_prediction

result = quick_prediction(
    cs_concentration=1.5,
    pb_concentration=1.0,
    temperature=150.0,
    solvent="DMSO"
)

print(f"Predicted phase: {result.primary_phase.name}")
print(f"Confidence: {result.confidence:.3f}")
print(f"PLQY: {result.properties.plqy:.3f}")
```

## 📊 Data Format

### Input Parameters
- `cs_br_concentration` (mol/L) - Cesium bromide concentration
- `pb_br2_concentration` (mol/L) - Lead bromide concentration
- `temperature` (°C) - Synthesis temperature
- `solvent_type` (0-4) - Solvent type (DMSO, DMF, Water, Toluene, Octadecene)
- `oa_concentration` (mol/L) - Oleic acid concentration
- `oam_concentration` (mol/L) - Oleylamine concentration
- `reaction_time` (min) - Reaction time

### Target Outputs
- `phase_label` (0-4) - Phase outcome (3D, 0D, 2D, Mixed, Failed)
- Material properties: bandgap, PLQY, emission peak, particle size, etc.

## 🛠️ Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install torch pytorch-lightning numpy pandas pydantic
```

**2. CUDA/GPU Issues**
```bash
# Use CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. Memory Issues**
```bash
# Reduce batch size
python train_pytorch_models.py --batch-size 8 --max-epochs 10
```

**4. Data Issues**
```bash
# Regenerate sample data
python create_sample_csv.py
```

### Environment Verification

```python
# Test basic functionality
python -c "
import torch
import synthesis
print('✅ PyTorch version:', torch.__version__)
print('✅ Synthesis package loaded')
"
```

## 🎯 Performance Expectations

### Training Time
- **Small dataset (1K samples)**: ~5-10 minutes on CPU
- **Medium dataset (5K samples)**: ~20-30 minutes on CPU, ~5 minutes on GPU
- **Large dataset (10K+ samples)**: ~1-2 hours on CPU, ~15-30 minutes on GPU

### Model Performance
- **Phase Classification**: >90% accuracy expected
- **Property Prediction**: <5% MAE for key properties
- **Physics Consistency**: >95% constraint satisfaction

## 🔧 Development Tips

### Adding Your Own Data
1. Format your data as CSV with required columns
2. Ensure phase labels are integers (0-4)
3. Include material property measurements
4. Run validation: `python -c "from synthesis.utils import validate_synthesis_parameters"`

### Customizing the Model
1. Edit `synthesis/training/pytorch_neural_models.py` for architecture changes
2. Modify `synthesis/training/pytorch_feature_engineering.py` for new features
3. Update `synthesis/physics/` modules for new physics models

### Experiment Management
- Use `--use-wandb` for advanced experiment tracking
- Model checkpoints saved in `experiments/` directory
- TensorBoard logs for training visualization

## 📚 Next Steps

1. **Train on your data**: Replace sample data with experimental results
2. **Optimize hyperparameters**: Use grid search or Bayesian optimization
3. **Deploy the model**: Use the digital twin API for real-time predictions
4. **Extend physics**: Add new physics models for different perovskites

## 🎉 Success Criteria

You'll know the setup is successful when:
- ✅ All dependencies install without errors
- ✅ Sample data generates correctly
- ✅ Training completes without crashes
- ✅ Predictions return reasonable results
- ✅ Physics constraints are satisfied

---

**Need help?** Check the main README.md or open an issue on GitHub.