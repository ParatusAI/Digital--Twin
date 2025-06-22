#!/bin/bash
# Environment Setup Script for CsPbBr3 Digital Twin
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up CsPbBr3 Digital Twin Environment"
echo "=" * 50

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo "🏗️ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "📁 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Lightning
echo "⚡ Installing PyTorch Lightning..."
pip install pytorch-lightning torchmetrics

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Install additional dependencies
echo "🛠️ Installing additional dependencies..."
pip install pydantic pyyaml tqdm h5py

# Install development tools (optional)
echo "🔍 Installing development tools..."
pip install pytest black flake8

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import pydantic
print('🎉 All core dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Lightning version: {pl.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Generate sample data: python generate_sample_data.py"
echo "3. Train the model: python train_pytorch_models.py"
echo ""
echo "🚀 You're ready to run the CsPbBr3 Digital Twin!"