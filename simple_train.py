#!/usr/bin/env python3
"""
Simple Training Script for CsPbBr₃ Digital Twin
Simplified version that focuses on getting a working model quickly
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sklearn_split
import os
import json
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SynthesisDataset(Dataset):
    """Simple dataset for synthesis data"""
    
    def __init__(self, features, phase_labels, properties):
        self.features = torch.FloatTensor(features)
        self.phase_labels = torch.LongTensor(phase_labels)
        self.properties = {
            key: torch.FloatTensor(values) for key, values in properties.items()
        }
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'phase_labels': self.phase_labels[idx],
            'properties': {key: values[idx] for key, values in self.properties.items()}
        }

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for synthesis prediction"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], n_phases=5, n_properties=4):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Phase classification head
        self.phase_classifier = nn.Linear(prev_dim, n_phases)
        
        # Property regression heads
        self.property_regressors = nn.ModuleDict({
            'bandgap': nn.Linear(prev_dim, 1),
            'plqy': nn.Linear(prev_dim, 1),
            'particle_size': nn.Linear(prev_dim, 1),
            'emission_peak': nn.Linear(prev_dim, 1)
        })
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        phase_logits = self.phase_classifier(features)
        
        properties = {}
        for prop_name, regressor in self.property_regressors.items():
            properties[prop_name] = regressor(features).squeeze()
        
        return {
            'phase_logits': phase_logits,
            'properties': properties
        }

def load_and_prepare_data(data_file):
    """Load and prepare training data"""
    logger.info(f"Loading data from {data_file}")
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Feature columns (exclude target variables)
    feature_cols = [col for col in df.columns if col not in 
                   ['phase_label', 'bandgap', 'plqy', 'particle_size', 'emission_peak']]
    
    # Prepare features
    X = df[feature_cols].values
    
    # Prepare targets
    y_phase = df['phase_label'].values
    y_properties = {
        'bandgap': df['bandgap'].values,
        'plqy': df['plqy'].values,
        'particle_size': df['particle_size'].values,
        'emission_peak': df['emission_peak'].values
    }
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Features shape: {X_scaled.shape}")
    logger.info(f"Phase distribution: {np.bincount(y_phase)}")
    
    return X_scaled, y_phase, y_properties, scaler, feature_cols

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train the neural network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Loss functions
    phase_criterion = nn.CrossEntropyLoss()
    property_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            features = batch['features'].to(device)
            phase_labels = batch['phase_labels'].to(device)
            properties = {k: v.to(device) for k, v in batch['properties'].items()}
            
            # Forward pass
            outputs = model(features)
            
            # Calculate losses
            phase_loss = phase_criterion(outputs['phase_logits'], phase_labels)
            
            property_loss = 0
            for prop_name, pred in outputs['properties'].items():
                target = properties[prop_name]
                property_loss += property_criterion(pred, target)
            
            total_loss = phase_loss + 0.5 * property_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_samples += features.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        correct_phases = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                phase_labels = batch['phase_labels'].to(device)
                properties = {k: v.to(device) for k, v in batch['properties'].items()}
                
                outputs = model(features)
                
                # Calculate losses
                phase_loss = phase_criterion(outputs['phase_logits'], phase_labels)
                
                property_loss = 0
                for prop_name, pred in outputs['properties'].items():
                    target = properties[prop_name]
                    property_loss += property_criterion(pred, target)
                
                total_loss = phase_loss + 0.5 * property_loss
                val_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs['phase_logits'].data, 1)
                correct_phases += (predicted == phase_labels).sum().item()
                val_samples += features.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_phases / val_samples
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_history

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Simple training for CsPbBr₃ Digital Twin")
    parser.add_argument('--data-file', type=str, required=True, help='Training data CSV file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='simple_training_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("🚀 Starting CsPbBr₃ Digital Twin Training")
    logger.info("=" * 50)
    
    # Load and prepare data
    X, y_phase, y_properties, scaler, feature_cols = load_and_prepare_data(args.data_file)
    
    # Split data
    X_train, X_val, y_phase_train, y_phase_val = sklearn_split(
        X, y_phase, test_size=0.2, random_state=42, stratify=y_phase
    )
    
    # Split properties using the same indices as X split
    _, _, *property_splits = sklearn_split(
        X, y_phase, *y_properties.values(), 
        test_size=0.2, random_state=42, stratify=y_phase
    )
    
    # Reconstruct property dictionaries
    prop_names = list(y_properties.keys())
    y_props_train = {prop_names[i]: property_splits[i*2] for i in range(len(prop_names))}
    y_props_val = {prop_names[i]: property_splits[i*2+1] for i in range(len(prop_names))}
    
    # Create datasets and dataloaders
    train_dataset = SynthesisDataset(X_train, y_phase_train, y_props_train)
    val_dataset = SynthesisDataset(X_val, y_phase_val, y_props_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = SimpleNeuralNetwork(input_dim=X.shape[1])
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    history = train_model(model, train_loader, val_loader, args.epochs, args.lr)
    
    # Save results
    results = {
        'training_history': history,
        'feature_columns': feature_cols,
        'model_config': {
            'input_dim': X.shape[1],
            'hidden_dims': [128, 64],
            'n_phases': 5,
            'n_properties': 4
        }
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save scaler
    import pickle
    with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"Model saved to: best_model.pth")
    logger.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()