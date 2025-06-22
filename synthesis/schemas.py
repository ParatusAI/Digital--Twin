#!/usr/bin/env python3
"""
Unified Schema Definitions for CsPbBr₃ Digital Twin
Centralized data structures, enums, and type definitions
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass
from datetime import datetime
import torch
import numpy as np


class PhaseType(Enum):
    """Standardized phase type enumeration"""
    CSPBBR3_3D = 0           # 3D Perovskite phase
    CS4PBBR6_0D = 1          # 0D Zero-dimensional phase  
    CSPB2BR5_2D = 2          # 2D Quasi-layered phase
    MIXED_PHASES = 3         # Multiple phase mixture
    FAILED_SYNTHESIS = 4     # Synthesis failure


class SolventType(Enum):
    """Standardized solvent type enumeration"""
    DMSO = 0         # Dimethyl sulfoxide
    DMF = 1          # N,N-Dimethylformamide  
    WATER = 2        # Water
    TOLUENE = 3      # Toluene
    OCTADECENE = 4   # 1-Octadecene
    
    @classmethod
    def from_string(cls, solvent_str: str) -> 'SolventType':
        """Convert string to SolventType enum"""
        mapping = {
            'DMSO': cls.DMSO,
            'DMF': cls.DMF,
            'water': cls.WATER,
            'toluene': cls.TOLUENE,
            'octadecene': cls.OCTADECENE
        }
        return mapping.get(solvent_str, cls.DMSO)
    
    def to_string(self) -> str:
        """Convert enum to string"""
        mapping = {
            self.DMSO: 'DMSO',
            self.DMF: 'DMF',
            self.WATER: 'water',
            self.TOLUENE: 'toluene',
            self.OCTADECENE: 'octadecene'
        }
        return mapping[self]


@dataclass
class SynthesisParameters:
    """Synthesis parameters for CsPbBr₃ formation"""
    cs_br_concentration: float      # mol/L
    pb_br2_concentration: float     # mol/L
    temperature: float              # °C
    solvent_type: Union[SolventType, int, str]  # Flexible input
    oa_concentration: float = 0.0   # mol/L
    oam_concentration: float = 0.0  # mol/L
    reaction_time: float = 10.0     # minutes
    
    def __post_init__(self):
        """Normalize solvent_type to SolventType enum"""
        if isinstance(self.solvent_type, str):
            self.solvent_type = SolventType.from_string(self.solvent_type)
        elif isinstance(self.solvent_type, int):
            self.solvent_type = SolventType(self.solvent_type)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for tensor creation"""
        return {
            'cs_br_concentration': self.cs_br_concentration,
            'pb_br2_concentration': self.pb_br2_concentration,
            'temperature': self.temperature,
            'solvent_type': float(self.solvent_type.value),
            'oa_concentration': self.oa_concentration,
            'oam_concentration': self.oam_concentration,
            'reaction_time': self.reaction_time
        }
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        values = [
            self.cs_br_concentration,
            self.pb_br2_concentration,
            self.temperature,
            float(self.solvent_type.value),
            self.oa_concentration,
            self.oam_concentration,
            self.reaction_time
        ]
        return torch.tensor(values, dtype=torch.float32)


@dataclass
class MaterialProperties:
    """Material properties for synthesis outcome"""
    bandgap: float                    # eV
    plqy: float                      # 0-1
    emission_peak: float             # nm
    emission_fwhm: float             # nm
    particle_size: float             # nm
    size_distribution_width: float   # relative std dev
    lifetime: float                  # ns
    stability_score: float           # 0-1
    phase_purity: float = 1.0        # 0-1
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'bandgap': self.bandgap,
            'plqy': self.plqy,
            'emission_peak': self.emission_peak,
            'emission_fwhm': self.emission_fwhm,
            'particle_size': self.particle_size,
            'size_distribution_width': self.size_distribution_width,
            'lifetime': self.lifetime,
            'stability_score': self.stability_score,
            'phase_purity': self.phase_purity
        }


@dataclass
class PhysicsFeatures:
    """Physics-calculated features from nucleation and growth models"""
    # Nucleation features
    supersaturation: float
    nucleation_rate_3d: float
    nucleation_rate_0d: float
    nucleation_rate_2d: float
    critical_radius_3d: float
    critical_radius_0d: float
    critical_radius_2d: float
    
    # Growth features
    growth_rate: float
    diffusion_length: float
    ligand_coverage: float
    
    # Thermodynamic features
    gibbs_energy_3d: float
    gibbs_energy_0d: float
    gibbs_energy_2d: float
    
    # Additional physics features
    thermal_energy: float
    ionic_strength: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor"""
        values = [
            self.supersaturation,
            self.nucleation_rate_3d,
            self.nucleation_rate_0d, 
            self.nucleation_rate_2d,
            self.critical_radius_3d,
            self.critical_radius_0d,
            self.critical_radius_2d,
            self.growth_rate,
            self.diffusion_length,
            self.ligand_coverage,
            self.gibbs_energy_3d,
            self.gibbs_energy_0d,
            self.gibbs_energy_2d,
            self.thermal_energy,
            self.ionic_strength
        ]
        return torch.tensor(values, dtype=torch.float32)


class PredictionResult(BaseModel):
    """Complete synthesis prediction result"""
    # Primary predictions
    primary_phase: PhaseType = Field(description="Most likely phase")
    phase_probabilities: Dict[PhaseType, float] = Field(description="Phase probability distribution")
    properties: MaterialProperties = Field(description="Predicted material properties")
    
    # Uncertainty information
    phase_uncertainty: float = Field(description="Phase prediction uncertainty")
    property_uncertainties: Dict[str, float] = Field(description="Property prediction uncertainties")
    
    # Physics analysis
    physics_features: PhysicsFeatures = Field(description="Physics-based features")
    physics_consistency: float = Field(description="Physics constraint satisfaction")
    
    # Metadata
    confidence: float = Field(ge=0, le=1, description="Overall prediction confidence")
    prediction_id: str = Field(description="Unique prediction identifier")
    timestamp: str = Field(description="Prediction timestamp")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    model_version: str = Field(description="Model version used")
    
    class Config:
        # Allow enum types in JSON serialization
        use_enum_values = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'primary_phase': self.primary_phase.value,
            'phase_probabilities': {phase.value: prob for phase, prob in self.phase_probabilities.items()},
            'properties': self.properties.to_dict(),
            'phase_uncertainty': self.phase_uncertainty,
            'property_uncertainties': self.property_uncertainties,
            'physics_features': self.physics_features.__dict__,
            'physics_consistency': self.physics_consistency,
            'confidence': self.confidence,
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp,
            'processing_time_ms': self.processing_time_ms,
            'model_version': self.model_version
        }


@dataclass
class TrainingConfig:
    """Configuration for neural network training"""
    # Model architecture
    input_dim: int = 100
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    activation: str = "relu"
    
    # Physics constraints
    physics_weight: float = 0.2
    uncertainty_weight: float = 0.1
    constraint_weight: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 20
    
    # Multi-task weights
    phase_weight: float = 1.0
    property_weight: float = 1.0
    
    # Cross-validation
    n_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    # Phase classification metrics
    phase_accuracy: float
    phase_f1_score: float
    phase_precision: float
    phase_recall: float
    
    # Property regression metrics
    property_mae: Dict[str, float]
    property_rmse: Dict[str, float]
    property_r2: Dict[str, float]
    
    # Uncertainty metrics
    calibration_error: float
    uncertainty_quality: float
    
    # Physics consistency
    physics_violation_rate: float
    constraint_satisfaction: float
    
    # Overall metrics
    total_loss: float
    validation_loss: float
    training_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'phase_accuracy': self.phase_accuracy,
            'phase_f1_score': self.phase_f1_score,
            'phase_precision': self.phase_precision,
            'phase_recall': self.phase_recall,
            'property_mae': self.property_mae,
            'property_rmse': self.property_rmse,
            'property_r2': self.property_r2,
            'calibration_error': self.calibration_error,
            'uncertainty_quality': self.uncertainty_quality,
            'physics_violation_rate': self.physics_violation_rate,
            'constraint_satisfaction': self.constraint_satisfaction,
            'total_loss': self.total_loss,
            'validation_loss': self.validation_loss,
            'training_time': self.training_time
        }


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment"""
    experiment_id: str
    description: str
    training_config: TrainingConfig
    data_config: Dict[str, Any]
    physics_config: Dict[str, Any]
    output_dir: str
    
    # Metadata
    created_by: str = "CsPbBr3_Digital_Twin"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


# Type aliases for convenience
TensorDict = Dict[str, torch.Tensor]
ParameterDict = Dict[str, float]
PredictionDict = Dict[str, Any]

# Constants
PHASE_NAMES = {
    PhaseType.CSPBBR3_3D: "CsPbBr₃ (3D)",
    PhaseType.CS4PBBR6_0D: "Cs₄PbBr₆ (0D)",
    PhaseType.CSPB2BR5_2D: "CsPb₂Br₅ (2D)",
    PhaseType.MIXED_PHASES: "Mixed Phases",
    PhaseType.FAILED_SYNTHESIS: "Failed Synthesis"
}

SOLVENT_NAMES = {
    SolventType.DMSO: "DMSO",
    SolventType.DMF: "DMF", 
    SolventType.WATER: "Water",
    SolventType.TOLUENE: "Toluene",
    SolventType.OCTADECENE: "1-Octadecene"
}

# Validation ranges
PARAMETER_RANGES = {
    'cs_br_concentration': (0.1, 5.0),
    'pb_br2_concentration': (0.1, 3.0),
    'temperature': (60.0, 300.0),
    'oa_concentration': (0.0, 2.0),
    'oam_concentration': (0.0, 2.0),
    'reaction_time': (0.5, 120.0)
}

PROPERTY_RANGES = {
    'bandgap': (0.5, 5.0),
    'plqy': (0.0, 1.0),
    'emission_peak': (300.0, 800.0),
    'particle_size': (1.0, 1000.0),
    'lifetime': (0.1, 100.0),
    'stability_score': (0.0, 1.0)
}