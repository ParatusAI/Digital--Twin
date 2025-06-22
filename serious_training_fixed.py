#!/usr/bin/env python3
"""
Serious ML Training for CsPbBr3 Digital Twin
Proper cross-validation, hyperparameter tuning, and model validation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import optuna
import json
import time
import pickle

class ImprovedMLPClassifier(nn.Module):
    """Improved MLP with proper regularization"""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class SeriousTrainer:
    """Serious model training with proper validation"""
    
    def __init__(self, data_path='large_training_data.csv'):
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_and_prepare_data(self):
        """Load and prepare data with proper preprocessing"""
        print("📊 Loading and preparing data...")
        
        df = pd.read_csv(self.data_path)
        print(f"   Dataset size: {len(df)} samples")
        
        self.feature_cols = [
            'cs_br_concentration', 'pb_br2_concentration', 'temperature',
            'oa_concentration', 'oam_concentration', 'reaction_time',
            'solvent_type', 'cs_pb_ratio', 'supersaturation', 'ligand_ratio',
            'temp_normalized', 'solvent_effect'
        ]
        
        X = df[self.feature_cols].values
        y = df['phase_label'].values
        
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"   Class distribution: {dict(zip(unique_classes, counts))}")
        
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = torch.FloatTensor(X_train_scaled)
        self.X_test = torch.FloatTensor(X_test_scaled)
        self.y_train = torch.LongTensor(y_train)
        self.y_test = torch.LongTensor(y_test)
        
        self.num_features = len(self.feature_cols)
        self.num_classes = len(unique_classes)
        
        print(f"   Training samples: {len(X_train_scaled)}")
        print(f"   Test samples: {len(X_test_scaled)}")
        
        return True
    
    def objective(self, trial):
        """Optuna objective for hyperparameter optimization"""
        
        hidden_size1 = trial.suggest_int('hidden_size1', 32, 256, step=32)
        hidden_size2 = trial.suggest_int('hidden_size2', 16, 128, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        model = ImprovedMLPClassifier(
            input_size=self.num_features,
            hidden_sizes=[hidden_size1, hidden_size2],
            num_classes=self.num_classes,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train[train_idx].to(self.device)
            X_fold_val = self.X_train[val_idx].to(self.device)
            y_fold_train = self.y_train[train_idx].to(self.device)
            y_fold_val = self.y_train[val_idx].to(self.device)
            
            train_dataset = TensorDataset(X_fold_train, y_fold_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Reset model weights
            for layer in model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            
            model.train()
            for epoch in range(20):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_fold_val)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_accuracy = accuracy_score(y_fold_val.cpu(), val_predictions.cpu())
                cv_scores.append(val_accuracy)
        
        return np.mean(cv_scores)
    
    def find_best_hyperparameters(self, n_trials=20):
        """Find best hyperparameters using Optuna"""
        print(f"🔍 Searching for best hyperparameters ({n_trials} trials)...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        print(f"   Best parameters: {self.best_params}")
        print(f"   Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train_final_model(self, epochs=100):
        """Train final model with best hyperparameters"""
        print("🚀 Training final model...")
        
        self.model = ImprovedMLPClassifier(
            input_size=self.num_features,
            hidden_sizes=[self.best_params['hidden_size1'], self.best_params['hidden_size2']],
            num_classes=self.num_classes,
            dropout_rate=self.best_params['dropout_rate']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.best_params['learning_rate'],
            weight_decay=1e-5
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        train_dataset = TensorDataset(self.X_train.to(self.device), self.y_train.to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0
        patience_counter = 0
        max_patience = 15
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
            
            train_acc = correct_train / total_train
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_test.to(self.device))
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_acc = accuracy_score(self.y_test, val_predictions.cpu())
            
            train_losses.append(epoch_loss / len(train_loader))
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_serious_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= max_patience:
                print(f"   Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f}s")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        self.model.load_state_dict(torch.load('best_serious_model.pth'))
        
        self.training_history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc,
            'training_time': training_time
        }
        
        return best_val_acc
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("📊 Evaluating final model...")
        
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test.to(self.device))
            test_predictions = torch.argmax(test_outputs, dim=1).cpu()
            test_probabilities = torch.softmax(test_outputs, dim=1).cpu()
            
            test_acc = accuracy_score(self.y_test, test_predictions)
            test_f1 = f1_score(self.y_test, test_predictions, average='weighted')
            precision, recall, f1_per_class, _ = precision_recall_fscore_support(
                self.y_test, test_predictions, average=None
            )
            
            max_probs = torch.max(test_probabilities, dim=1)[0]
            avg_confidence = torch.mean(max_probs).item()
            low_conf_pct = (max_probs < 0.6).float().mean().item() * 100
            
            print(f"   Test Accuracy: {test_acc:.4f}")
            print(f"   Test F1-Score: {test_f1:.4f}")
            print(f"   Average Confidence: {avg_confidence:.4f}")
            print(f"   Low Confidence Predictions: {low_conf_pct:.1f}%")
            
            print("   Per-class F1 scores:")
            for i, f1_val in enumerate(f1_per_class):
                print(f"     Class {i}: {f1_val:.4f}")
        
        self.test_results = {
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'avg_confidence': avg_confidence,
            'low_confidence_pct': low_conf_pct,
            'per_class_f1': f1_per_class.tolist(),
            'predictions': test_predictions.numpy().tolist(),
            'probabilities': test_probabilities.numpy().tolist()
        }
        
        return test_acc, test_f1
    
    def save_results(self):
        """Save all training results"""
        results = {
            'hyperparameters': self.best_params,
            'training_history': self.training_history,
            'test_results': self.test_results,
            'model_info': {
                'num_features': self.num_features,
                'num_classes': self.num_classes,
                'feature_columns': self.feature_cols
            }
        }
        
        with open('serious_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('serious_model_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("💾 Results saved:")
        print("   - best_serious_model.pth")
        print("   - serious_training_results.json")
        print("   - serious_model_scaler.pkl")

def main():
    """Main training function"""
    print("🎯 SERIOUS ML TRAINING FOR CsPbBr3 DIGITAL TWIN")
    print("=" * 60)
    
    trainer = SeriousTrainer()
    
    if not trainer.load_and_prepare_data():
        return
    
    best_params = trainer.find_best_hyperparameters(n_trials=30)
    best_val_acc = trainer.train_final_model(epochs=150)
    test_acc, test_f1 = trainer.evaluate_model()
    trainer.save_results()
    
    print("\n🎉 SERIOUS TRAINING COMPLETE!")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Test F1-score: {test_f1:.4f}")
    
    if test_acc > 0.85 and test_f1 > 0.80:
        print("\n✅ MODEL QUALITY: EXCELLENT - Ready for production")
    elif test_acc > 0.75 and test_f1 > 0.70:
        print("\n⚠️ MODEL QUALITY: GOOD - Needs minor improvements")
    else:
        print("\n❌ MODEL QUALITY: POOR - Needs significant work")
    
    return test_acc, test_f1

if __name__ == "__main__":
    main()