#!/usr/bin/env python3
"""
Machine Learning-Based Parameter Optimization for CsPbBr3 Synthesis
Implements Bayesian optimization and Gaussian Process models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML optimization features will be limited.")

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some optimization features will be limited.")


@dataclass
class OptimizationTarget:
    """Define optimization target for synthesis conditions"""
    property_name: str
    target_value: float
    weight: float = 1.0
    minimize: bool = False  # If True, minimize the property; if False, maximize
    tolerance: float = 0.1  # Acceptable deviation from target


class BayesianOptimizer:
    """Bayesian optimizer for synthesis parameter optimization"""
    
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 targets: List[OptimizationTarget],
                 kernel_type: str = 'matern',
                 acquisition_function: str = 'ei',  # Expected Improvement
                 random_seed: int = 42):
        """
        Initialize Bayesian optimizer
        
        Args:
            parameter_bounds: Dict mapping parameter names to (min, max) bounds
            targets: List of optimization targets
            kernel_type: GP kernel type ('rbf', 'matern', 'combined')
            acquisition_function: Acquisition function ('ei', 'pi', 'ucb')
            random_seed: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Bayesian optimization")
        
        self.parameter_bounds = parameter_bounds
        self.targets = targets
        self.kernel_type = kernel_type
        self.acquisition_function = acquisition_function
        self.random_seed = random_seed
        
        # Initialize components
        self._setup_kernel()
        self._setup_scaler()
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []
        
        # Track optimization history
        self.optimization_history = []
        
    def _setup_kernel(self):
        """Setup Gaussian Process kernel"""
        n_dims = len(self.parameter_bounds)
        
        if self.kernel_type == 'rbf':
            self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        elif self.kernel_type == 'matern':
            self.kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        elif self.kernel_type == 'combined':
            self.kernel = (
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _setup_scaler(self):
        """Setup parameter scaling for GP"""
        self.scaler = StandardScaler()
        
        # Create dummy data for fitting scaler
        param_names = list(self.parameter_bounds.keys())
        bounds_array = np.array([self.parameter_bounds[name] for name in param_names])
        dummy_data = np.array([bounds_array[:, 0], bounds_array[:, 1]])
        self.scaler.fit(dummy_data)
        
        self.param_names = param_names
    
    def _parameters_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array"""
        return np.array([params[name] for name in self.param_names])
    
    def _array_to_parameters(self, array: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dict"""
        return {name: float(value) for name, value in zip(self.param_names, array)}
    
    def _calculate_objective(self, properties: Dict[str, float]) -> float:
        """Calculate objective function value from material properties"""
        objective = 0.0
        
        for target in self.targets:
            if target.property_name in properties:
                prop_value = properties[target.property_name]
                
                if target.minimize:
                    # For minimization: lower is better
                    score = -prop_value * target.weight
                else:
                    # For maximization or target matching
                    if target.target_value is not None:
                        # Target matching: penalize deviation from target
                        deviation = abs(prop_value - target.target_value)
                        if deviation <= target.tolerance:
                            score = target.weight  # Perfect score within tolerance
                        else:
                            score = target.weight * np.exp(-(deviation - target.tolerance))
                    else:
                        # Simple maximization
                        score = prop_value * target.weight
                
                objective += score
        
        return objective
    
    def add_observation(self, parameters: Dict[str, float], properties: Dict[str, float]):
        """Add new observation to the optimizer"""
        param_array = self._parameters_to_array(parameters)
        objective_value = self._calculate_objective(properties)
        
        self.X_observed.append(param_array)
        self.y_observed.append(objective_value)
        
        # Record in history
        self.optimization_history.append({
            'parameters': parameters.copy(),
            'properties': properties.copy(),
            'objective': objective_value,
            'iteration': len(self.optimization_history)
        })
        
        # Retrain GP model
        self._update_model()
    
    def _update_model(self):
        """Update Gaussian Process model with new observations"""
        if len(self.X_observed) < 2:
            return
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Fit GP model
        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_seed
        )
        
        self.gp_model.fit(X_scaled, y)
    
    def _acquisition_function_value(self, x: np.ndarray) -> float:
        """Calculate acquisition function value"""
        if self.gp_model is None or len(self.X_observed) < 2:
            return np.random.random()  # Random exploration if no model
        
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        mu, sigma = self.gp_model.predict(x_scaled, return_std=True)
        
        mu, sigma = mu[0], sigma[0]
        
        if sigma < 1e-6:
            return 0.0
        
        if self.acquisition_function == 'ei':
            # Expected Improvement
            best_y = max(self.y_observed) if self.y_observed else 0
            z = (mu - best_y) / sigma
            ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei
        
        elif self.acquisition_function == 'pi':
            # Probability of Improvement
            best_y = max(self.y_observed) if self.y_observed else 0
            z = (mu - best_y) / sigma
            return norm.cdf(z)
        
        elif self.acquisition_function == 'ucb':
            # Upper Confidence Bound
            kappa = 2.0  # Exploration parameter
            return mu + kappa * sigma
        
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def suggest_next_parameters(self, n_candidates: int = 1000) -> Dict[str, float]:
        """Suggest next parameters to evaluate"""
        if not SCIPY_AVAILABLE:
            # Fallback to random sampling
            return self._random_parameter_suggestion()
        
        best_acquisition = -np.inf
        best_params = None
        
        # Random search over parameter space
        for _ in range(n_candidates):
            # Generate random parameters within bounds
            random_params = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                random_params[param_name] = np.random.uniform(min_val, max_val)
            
            param_array = self._parameters_to_array(random_params)
            acquisition_value = self._acquisition_function_value(param_array)
            
            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_params = random_params
        
        return best_params or self._random_parameter_suggestion()
    
    def _random_parameter_suggestion(self) -> Dict[str, float]:
        """Fallback random parameter suggestion"""
        random_params = {}
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            random_params[param_name] = np.random.uniform(min_val, max_val)
        return random_params
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress"""
        if not self.optimization_history:
            return {"status": "No observations yet"}
        
        objectives = [obs['objective'] for obs in self.optimization_history]
        best_idx = np.argmax(objectives)
        best_observation = self.optimization_history[best_idx]
        
        return {
            "total_iterations": len(self.optimization_history),
            "best_objective": best_observation['objective'],
            "best_parameters": best_observation['parameters'],
            "best_properties": best_observation['properties'],
            "improvement_over_time": objectives,
            "convergence_rate": self._calculate_convergence_rate(),
            "model_performance": self._evaluate_model_performance()
        }
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate of optimization"""
        if len(self.optimization_history) < 3:
            return 0.0
        
        objectives = [obs['objective'] for obs in self.optimization_history]
        recent_improvement = np.mean(objectives[-3:]) - np.mean(objectives[:3])
        return float(recent_improvement)
    
    def _evaluate_model_performance(self) -> Dict[str, float]:
        """Evaluate GP model performance using cross-validation"""
        if self.gp_model is None or len(self.X_observed) < 5:
            return {"status": "Insufficient data for evaluation"}
        
        try:
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            X_scaled = self.scaler.transform(X)
            
            cv_scores = cross_val_score(self.gp_model, X_scaled, y, cv=min(5, len(X)), scoring='r2')
            
            return {
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "model_score": float(self.gp_model.score(X_scaled, y))
            }
        except Exception as e:
            return {"error": str(e)}


class ActiveLearningOptimizer:
    """Active learning optimizer for efficient parameter space exploration"""
    
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 uncertainty_threshold: float = 0.1,
                 exploitation_ratio: float = 0.7):
        """
        Initialize active learning optimizer
        
        Args:
            parameter_bounds: Parameter bounds for optimization
            uncertainty_threshold: Threshold for uncertainty-based sampling
            exploitation_ratio: Ratio of exploitation vs exploration
        """
        self.parameter_bounds = parameter_bounds
        self.uncertainty_threshold = uncertainty_threshold
        self.exploitation_ratio = exploitation_ratio
        
        self.observations = []
        self.uncertainty_model = None
        
    def add_observation(self, parameters: Dict[str, float], 
                       properties: Dict[str, float], 
                       uncertainty: Dict[str, float]):
        """Add observation with uncertainty information"""
        self.observations.append({
            'parameters': parameters,
            'properties': properties,
            'uncertainty': uncertainty,
            'total_uncertainty': sum(uncertainty.values())
        })
    
    def suggest_next_parameters(self, strategy: str = 'hybrid') -> Dict[str, float]:
        """
        Suggest next parameters using active learning strategy
        
        Args:
            strategy: 'exploit', 'explore', or 'hybrid'
        """
        if strategy == 'exploit':
            return self._exploitation_strategy()
        elif strategy == 'explore':
            return self._exploration_strategy()
        else:  # hybrid
            if np.random.random() < self.exploitation_ratio:
                return self._exploitation_strategy()
            else:
                return self._exploration_strategy()
    
    def _exploitation_strategy(self) -> Dict[str, float]:
        """Focus on regions with good known performance"""
        if not self.observations:
            return self._random_parameters()
        
        # Find best performing regions
        best_obs = max(self.observations, 
                      key=lambda x: x['properties'].get('plqy', 0))
        
        # Add noise around best parameters
        best_params = best_obs['parameters']
        noisy_params = {}
        
        for param_name, value in best_params.items():
            min_val, max_val = self.parameter_bounds[param_name]
            noise_scale = (max_val - min_val) * 0.1  # 10% noise
            noisy_value = value + np.random.normal(0, noise_scale)
            noisy_params[param_name] = np.clip(noisy_value, min_val, max_val)
        
        return noisy_params
    
    def _exploration_strategy(self) -> Dict[str, float]:
        """Focus on unexplored or uncertain regions"""
        if len(self.observations) < 5:
            return self._random_parameters()
        
        # Use uncertainty to guide exploration
        high_uncertainty_obs = [
            obs for obs in self.observations 
            if obs['total_uncertainty'] > self.uncertainty_threshold
        ]
        
        if high_uncertainty_obs:
            # Sample near uncertain regions
            uncertain_obs = np.random.choice(high_uncertainty_obs)
            base_params = uncertain_obs['parameters']
            
            # Add larger noise for exploration
            exploration_params = {}
            for param_name, value in base_params.items():
                min_val, max_val = self.parameter_bounds[param_name]
                noise_scale = (max_val - min_val) * 0.2  # 20% noise for exploration
                noisy_value = value + np.random.normal(0, noise_scale)
                exploration_params[param_name] = np.clip(noisy_value, min_val, max_val)
            
            return exploration_params
        
        return self._random_parameters()
    
    def _random_parameters(self) -> Dict[str, float]:
        """Generate random parameters within bounds"""
        random_params = {}
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            random_params[param_name] = np.random.uniform(min_val, max_val)
        return random_params


def optimize_synthesis_conditions(
    targets: List[OptimizationTarget],
    initial_samples: int = 10,
    optimization_iterations: int = 20,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Run complete optimization workflow for synthesis conditions
    
    Args:
        targets: List of optimization targets
        initial_samples: Number of initial random samples
        optimization_iterations: Number of optimization iterations
        parameter_bounds: Custom parameter bounds (optional)
    
    Returns:
        Optimization results and best parameters
    """
    if parameter_bounds is None:
        # Default parameter bounds
        parameter_bounds = {
            'cs_br_concentration': (0.1, 3.0),
            'pb_br2_concentration': (0.1, 2.0),
            'temperature': (80.0, 250.0),
            'oa_concentration': (0.0, 1.5),
            'oam_concentration': (0.0, 1.5),
            'reaction_time': (1.0, 60.0)
        }
    
    # Initialize optimizer
    optimizer = BayesianOptimizer(parameter_bounds, targets)
    
    # Import here to avoid circular imports
    from generate_sample_data import (
        generate_synthesis_parameters, validate_physics_constraints,
        calculate_physics_features, determine_phase_outcomes,
        generate_material_properties, load_config
    )
    
    config = load_config()
    
    print(f"🎯 Starting optimization with {len(targets)} targets...")
    print(f"   Initial samples: {initial_samples}")
    print(f"   Optimization iterations: {optimization_iterations}")
    
    # Initial random sampling
    print("📊 Generating initial samples...")
    for i in range(initial_samples):
        # Generate random parameters
        params = optimizer._random_parameter_suggestion()
        
        # Simulate synthesis with these parameters
        # Create single-row DataFrame for simulation
        param_df = pd.DataFrame([params])
        
        # Run simulation pipeline
        validated_df = validate_physics_constraints(param_df, config)
        physics_df = calculate_physics_features(validated_df, config)
        phase_df = determine_phase_outcomes(physics_df, config)
        material_df = generate_material_properties(phase_df, config)
        
        # Extract properties
        properties = material_df.iloc[0].to_dict()
        
        # Add to optimizer
        optimizer.add_observation(params, properties)
        
        if (i + 1) % 5 == 0:
            print(f"   Completed {i + 1}/{initial_samples} initial samples")
    
    # Optimization loop
    print("🚀 Starting Bayesian optimization...")
    for iteration in range(optimization_iterations):
        # Suggest next parameters
        suggested_params = optimizer.suggest_next_parameters()
        
        # Simulate synthesis
        param_df = pd.DataFrame([suggested_params])
        validated_df = validate_physics_constraints(param_df, config)
        physics_df = calculate_physics_features(validated_df, config)
        phase_df = determine_phase_outcomes(physics_df, config)
        material_df = generate_material_properties(phase_df, config)
        
        properties = material_df.iloc[0].to_dict()
        
        # Add observation
        optimizer.add_observation(suggested_params, properties)
        
        if (iteration + 1) % 5 == 0:
            current_best = max(optimizer.y_observed)
            print(f"   Iteration {iteration + 1}/{optimization_iterations}, Best objective: {current_best:.3f}")
    
    # Get final results
    results = optimizer.get_optimization_summary()
    
    print("✅ Optimization complete!")
    print(f"   Best objective value: {results['best_objective']:.3f}")
    print(f"   Best parameters: {results['best_parameters']}")
    
    return results


if __name__ == "__main__":
    # Example optimization run
    targets = [
        OptimizationTarget("plqy", target_value=0.9, weight=1.0),
        OptimizationTarget("bandgap", target_value=2.3, weight=0.5, tolerance=0.1),
        OptimizationTarget("stability_score", target_value=None, weight=0.3)  # Maximize
    ]
    
    results = optimize_synthesis_conditions(
        targets=targets,
        initial_samples=15,
        optimization_iterations=25
    )
    
    # Save results
    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to optimization_results.json")