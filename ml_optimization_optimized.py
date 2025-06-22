#!/usr/bin/env python3
"""
Optimized Machine Learning-Based Parameter Optimization for CsPbBr3 Synthesis
Performance-optimized version with caching, vectorization, and incremental learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
import pickle
import time
from functools import lru_cache
from collections import deque
import copy

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML optimization features will be limited.")

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some optimization features will be limited.")

# High-performance caching system
class OptimizationCache:
    """High-performance cache for optimization computations"""
    def __init__(self, max_size=5000):
        self._cache = {}
        self._access_order = deque()
        self._max_size = max_size
        self._hit_count = 0
        self._miss_count = 0
        
    def get(self, key):
        if key in self._cache:
            self._hit_count += 1
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        self._miss_count += 1
        return None
    
    def put(self, key, value):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Remove least recently used
            oldest_key = self._access_order.popleft()
            del self._cache[oldest_key]
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def get_stats(self):
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        return {
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'hits': self._hit_count,
            'misses': self._miss_count
        }

# Global optimization cache
optimization_cache = OptimizationCache()

@dataclass
class OptimizationTarget:
    """Define optimization target for synthesis conditions"""
    property_name: str
    target_value: float
    weight: float = 1.0
    minimize: bool = False
    tolerance: float = 0.1

class BayesianOptimizerOptimized:
    """Performance-optimized Bayesian optimizer with caching and incremental learning"""
    
    def __init__(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 targets: List[OptimizationTarget],
                 kernel_type: str = 'matern',
                 acquisition_function: str = 'ei',
                 random_seed: int = 42,
                 max_history_size: int = 2000):
        """
        Initialize optimized Bayesian optimizer
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Bayesian optimization")
        
        self.parameter_bounds = parameter_bounds
        self.targets = targets
        self.kernel_type = kernel_type
        self.acquisition_function = acquisition_function
        self.random_seed = random_seed
        
        # Performance optimizations
        self.max_history_size = max_history_size
        self.model_cache = {}
        self.last_fit_size = 0
        self.scaling_cache = {}
        self.acquisition_cache = {}
        
        # Setup components
        self._setup_kernel()
        self._setup_scaler()
        self.gp_model = None
        
        # Use deque with max size for memory efficiency
        self.X_observed = deque(maxlen=max_history_size)
        self.y_observed = deque(maxlen=max_history_size)
        
        # Track optimization history
        self.optimization_history = deque(maxlen=1000)
        
        # Pre-compute parameter ranges for vectorization
        self.param_names = list(parameter_bounds.keys())
        self.bounds_array = np.array([parameter_bounds[param] for param in self.param_names])
        self.param_ranges = self.bounds_array[:, 1] - self.bounds_array[:, 0]
        
        np.random.seed(random_seed)
        
        print(f"🚀 Optimized Bayesian Optimizer initialized with {len(self.param_names)} parameters")
    
    @lru_cache(maxsize=32)
    def _setup_kernel(self):
        """Cached kernel setup"""
        n_dims = len(self.parameter_bounds)
        
        if self.kernel_type == 'rbf':
            self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        elif self.kernel_type == 'matern':
            self.kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        elif self.kernel_type == 'combined':
            self.kernel = (
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
                WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _setup_scaler(self):
        """Setup optimized parameter scaling"""
        # Use MinMaxScaler for better numerical stability
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_fitted = False
    
    def _parameters_to_array(self, parameters: Dict[str, float]) -> np.ndarray:
        """Vectorized parameter conversion"""
        return np.array([parameters.get(param, 0.0) for param in self.param_names], dtype=np.float32)
    
    def _array_to_parameters(self, array: np.ndarray) -> Dict[str, float]:
        """Vectorized array to parameters conversion"""
        return {param: float(array[i]) for i, param in enumerate(self.param_names)}
    
    def _calculate_objective_vectorized(self, properties_list: List[Dict[str, float]]) -> np.ndarray:
        """Vectorized objective calculation for multiple samples"""
        n_samples = len(properties_list)
        objectives = np.zeros(n_samples, dtype=np.float32)
        
        for i, properties in enumerate(properties_list):
            objectives[i] = self._calculate_single_objective(properties)
        
        return objectives
    
    @lru_cache(maxsize=10000)
    def _calculate_single_objective(self, properties_tuple: tuple) -> float:
        """Cached single objective calculation"""
        # Convert tuple back to dict for calculation
        properties = dict(zip(['plqy', 'stability_score', 'bandgap', 'particle_size', 'phase_purity'], properties_tuple))
        
        total_score = 0.0
        total_weight = 0.0
        
        for target in self.targets:
            if target.property_name in properties:
                value = properties[target.property_name]
                
                if target.minimize:
                    # For minimization targets
                    score = max(0, 1 - abs(value - target.target_value) / target.tolerance)
                else:
                    # For maximization targets
                    if target.property_name == 'plqy':
                        score = value  # Direct PLQY score
                    elif target.property_name == 'stability_score':
                        score = value  # Direct stability score
                    elif target.property_name == 'bandgap':
                        # Penalty for bandgap deviation from target
                        if 2.2 <= value <= 2.5:
                            score = 1.0
                        elif 2.0 <= value <= 2.7:
                            score = 0.8
                        else:
                            score = 0.5
                    else:
                        score = min(1.0, value / target.target_value) if target.target_value > 0 else 0
                
                total_score += score * target.weight
                total_weight += target.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def observe_samples_batch(self, parameters_list: List[Dict[str, float]], 
                            properties_list: List[Dict[str, float]]):
        """Optimized batch observation processing"""
        if len(parameters_list) != len(properties_list):
            raise ValueError("Parameters and properties lists must have same length")
        
        # Vectorized parameter conversion
        X_new = np.array([self._parameters_to_array(params) for params in parameters_list], dtype=np.float32)
        
        # Vectorized objective calculation
        properties_tuples = [
            tuple(props.get(key, 0.0) for key in ['plqy', 'stability_score', 'bandgap', 'particle_size', 'phase_purity'])
            for props in properties_list
        ]
        y_new = np.array([self._calculate_single_objective(props) for props in properties_tuples], dtype=np.float32)
        
        # Batch append to observations
        self.X_observed.extend(X_new)
        self.y_observed.extend(y_new)
        
        # Update model if enough new data
        if len(self.X_observed) - self.last_fit_size >= 10 or len(self.X_observed) < 20:
            self._update_model_incremental()
        
        # Add to history (sample for memory efficiency)
        for i in range(min(5, len(parameters_list))):  # Sample 5 per batch
            self.optimization_history.append({
                'parameters': parameters_list[i],
                'properties': properties_list[i],
                'objective': y_new[i],
                'timestamp': time.time()
            })
    
    def _update_model_incremental(self):
        """Incremental model update with caching"""
        current_size = len(self.X_observed)
        
        # Skip update if minimal new data and model exists
        if current_size - self.last_fit_size < 5 and current_size > 20 and self.gp_model is not None:
            return
        
        # Check cache for existing model
        cache_key = f"gp_model_{current_size}_{hash(tuple(map(tuple, list(self.X_observed)[-10:])))}"
        cached_model = optimization_cache.get(cache_key)
        
        if cached_model is not None:
            self.gp_model = cached_model
            self.last_fit_size = current_size
            return
        
        if current_size < 2:
            return
        
        try:
            # Convert deque to numpy arrays efficiently
            X = np.array(list(self.X_observed), dtype=np.float32)
            y = np.array(list(self.y_observed), dtype=np.float32)
            
            # Incremental scaling update
            if not self.scaler_fitted:
                X_scaled = self.scaler.fit_transform(X)
                self.scaler_fitted = True
            else:
                # Use partial fit if available, otherwise refit on recent data
                if hasattr(self.scaler, 'partial_fit'):
                    recent_data = X[-min(50, len(X)):]
                    self.scaler.partial_fit(recent_data)
                X_scaled = self.scaler.transform(X)
            
            # Initialize or update GP model
            if self.gp_model is None:
                self.gp_model = GaussianProcessRegressor(
                    kernel=self.kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=2,  # Reduced for speed
                    random_state=self.random_seed
                )
            
            # Fit model
            self.gp_model.fit(X_scaled, y)
            
            # Cache the model (deep copy to avoid modification)
            optimization_cache.put(cache_key, copy.deepcopy(self.gp_model))
            self.last_fit_size = current_size
            
        except Exception as e:
            warnings.warn(f"Failed to update GP model: {e}")
    
    def _acquisition_function_value_vectorized(self, X_candidates: np.ndarray) -> np.ndarray:
        """Vectorized acquisition function evaluation"""
        if self.gp_model is None:
            return np.random.random(len(X_candidates)).astype(np.float32)
        
        try:
            # Batch prediction (much faster than individual predictions)
            X_scaled = self.scaler.transform(X_candidates)
            mu, sigma = self.gp_model.predict(X_scaled, return_std=True)
            
            # Vectorized acquisition calculation
            if self.acquisition_function == 'ei':
                return self._expected_improvement_vectorized(mu, sigma)
            elif self.acquisition_function == 'pi':
                return self._probability_improvement_vectorized(mu, sigma)
            elif self.acquisition_function == 'ucb':
                return self._upper_confidence_bound_vectorized(mu, sigma)
            else:
                return mu + 2 * sigma  # Simple UCB as fallback
                
        except Exception as e:
            warnings.warn(f"Acquisition function evaluation failed: {e}")
            return np.random.random(len(X_candidates)).astype(np.float32)
    
    def _expected_improvement_vectorized(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Vectorized Expected Improvement calculation"""
        if len(self.y_observed) == 0:
            return mu
        
        best_y = max(self.y_observed)
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        z = (mu - best_y) / sigma
        
        # Vectorized normal CDF and PDF
        ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei.astype(np.float32)
    
    def _probability_improvement_vectorized(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Vectorized Probability of Improvement calculation"""
        if len(self.y_observed) == 0:
            return np.ones_like(mu)
        
        best_y = max(self.y_observed)
        sigma = np.maximum(sigma, 1e-9)
        z = (mu - best_y) / sigma
        return norm.cdf(z).astype(np.float32)
    
    def _upper_confidence_bound_vectorized(self, mu: np.ndarray, sigma: np.ndarray, 
                                         kappa: float = 2.0) -> np.ndarray:
        """Vectorized Upper Confidence Bound calculation"""
        return (mu + kappa * sigma).astype(np.float32)
    
    def suggest_next_parameters_vectorized(self, n_candidates: int = 2000) -> Dict[str, float]:
        """Vectorized parameter suggestion with efficient candidate evaluation"""
        # Check cache first
        cache_key = f"suggest_{n_candidates}_{len(self.X_observed)}_{hash(tuple(self.y_observed)[-10:]) if self.y_observed else 0}"
        cached_result = optimization_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Generate all candidates at once (vectorized)
        candidates = np.random.uniform(
            self.bounds_array[:, 0],
            self.bounds_array[:, 1],
            (n_candidates, len(self.param_names))
        ).astype(np.float32)
        
        # Add some candidates near best observed points
        if len(self.X_observed) > 0:
            best_indices = np.argsort(list(self.y_observed))[-min(5, len(self.y_observed)):]
            X_array = np.array(list(self.X_observed))
            
            for i, best_idx in enumerate(best_indices):
                if i * 100 < n_candidates:
                    # Add variations around best points
                    best_point = X_array[best_idx]
                    noise = np.random.normal(0, 0.1 * self.param_ranges, (min(100, n_candidates - i * 100), len(self.param_names)))
                    variations = best_point + noise
                    
                    # Clip to bounds
                    variations = np.clip(variations, self.bounds_array[:, 0], self.bounds_array[:, 1])
                    
                    start_idx = i * 100
                    end_idx = min(start_idx + 100, n_candidates)
                    candidates[start_idx:end_idx] = variations[:end_idx - start_idx]
        
        # Vectorized evaluation
        acquisition_values = self._acquisition_function_value_vectorized(candidates)
        
        # Find best candidate
        best_idx = np.argmax(acquisition_values)
        best_params = self._array_to_parameters(candidates[best_idx])
        
        # Cache result
        optimization_cache.put(cache_key, best_params)
        
        return best_params
    
    def optimize_batch(self, n_iterations: int = 20, batch_size: int = 5, 
                      evaluation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Optimized batch optimization with parallel evaluation"""
        print(f"🎯 Starting batch optimization: {n_iterations} iterations, batch size {batch_size}")
        
        optimization_start = time.time()
        best_objective = float('-inf')
        best_parameters = {}
        
        iteration_times = []
        
        for iteration in range(n_iterations):
            iter_start = time.time()
            
            # Generate batch of candidates
            candidates = []
            for _ in range(batch_size):
                candidate = self.suggest_next_parameters_vectorized()
                candidates.append(candidate)
            
            # Evaluate candidates (batch if function supports it)
            if evaluation_function:
                if hasattr(evaluation_function, 'evaluate_batch'):
                    # Batch evaluation
                    properties_list = evaluation_function.evaluate_batch(candidates)
                else:
                    # Sequential evaluation
                    properties_list = [evaluation_function(candidate) for candidate in candidates]
                
                # Observe all results
                self.observe_samples_batch(candidates, properties_list)
                
                # Track best result
                objectives = [self._calculate_single_objective(
                    tuple(props.get(key, 0.0) for key in ['plqy', 'stability_score', 'bandgap', 'particle_size', 'phase_purity'])
                ) for props in properties_list]
                
                batch_best_idx = np.argmax(objectives)
                batch_best_objective = objectives[batch_best_idx]
                
                if batch_best_objective > best_objective:
                    best_objective = batch_best_objective
                    best_parameters = candidates[batch_best_idx].copy()
                    print(f"   Iteration {iteration}: New best objective = {best_objective:.4f}")
            
            iter_time = time.time() - iter_start
            iteration_times.append(iter_time)
            
            if iteration % 5 == 0 and iteration > 0:
                avg_time = np.mean(iteration_times[-5:])
                print(f"   Iteration {iteration}/{n_iterations}, avg time: {avg_time:.2f}s, best: {best_objective:.4f}")
        
        total_time = time.time() - optimization_start
        
        # Generate summary
        cache_stats = optimization_cache.get_stats()
        
        results = {
            'best_objective': best_objective,
            'best_parameters': best_parameters,
            'n_iterations': n_iterations,
            'n_observations': len(self.X_observed),
            'optimization_time': total_time,
            'avg_iteration_time': np.mean(iteration_times),
            'cache_hit_rate': cache_stats['hit_rate'],
            'optimization_history': list(self.optimization_history)[-50:]  # Last 50 entries
        }
        
        print(f"✅ Optimization complete! Best objective: {best_objective:.4f}")
        print(f"   Total time: {total_time:.2f}s, Cache hit rate: {cache_stats['hit_rate']:.2%}")
        
        return results
    
    def get_model_predictions_vectorized(self, parameters_list: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized model predictions for multiple parameter sets"""
        if self.gp_model is None or not parameters_list:
            n_params = len(parameters_list)
            return np.zeros(n_params), np.ones(n_params)
        
        # Convert to array
        X = np.array([self._parameters_to_array(params) for params in parameters_list])
        
        try:
            X_scaled = self.scaler.transform(X)
            mu, sigma = self.gp_model.predict(X_scaled, return_std=True)
            return mu.astype(np.float32), sigma.astype(np.float32)
        except Exception as e:
            warnings.warn(f"Model prediction failed: {e}")
            n_params = len(parameters_list)
            return np.zeros(n_params), np.ones(n_params)
    
    def save_optimizer_state(self, filepath: str):
        """Save optimizer state for resuming optimization"""
        state = {
            'parameter_bounds': self.parameter_bounds,
            'targets': self.targets,
            'X_observed': list(self.X_observed),
            'y_observed': list(self.y_observed),
            'optimization_history': list(self.optimization_history),
            'best_parameters': getattr(self, 'best_parameters', {}),
            'scaler_fitted': self.scaler_fitted
        }
        
        if self.scaler_fitted:
            state['scaler_state'] = {
                'scale_': self.scaler.scale_,
                'min_': self.scaler.min_,
                'data_min_': self.scaler.data_min_,
                'data_max_': self.scaler.data_max_,
                'data_range_': self.scaler.data_range_
            }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Optimizer state saved to {filepath}")
    
    def load_optimizer_state(self, filepath: str):
        """Load optimizer state to resume optimization"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.X_observed = deque(state['X_observed'], maxlen=self.max_history_size)
            self.y_observed = deque(state['y_observed'], maxlen=self.max_history_size)
            self.optimization_history = deque(state['optimization_history'], maxlen=1000)
            
            if 'scaler_state' in state and state['scaler_fitted']:
                scaler_state = state['scaler_state']
                self.scaler.scale_ = scaler_state['scale_']
                self.scaler.min_ = scaler_state['min_']
                self.scaler.data_min_ = scaler_state['data_min_']
                self.scaler.data_max_ = scaler_state['data_max_']
                self.scaler.data_range_ = scaler_state['data_range_']
                self.scaler_fitted = True
            
            # Update model with loaded data
            self._update_model_incremental()
            
            print(f"📂 Optimizer state loaded from {filepath}")
            print(f"   Restored {len(self.X_observed)} observations")
            
        except Exception as e:
            warnings.warn(f"Failed to load optimizer state: {e}")

def create_default_targets() -> List[OptimizationTarget]:
    """Create default optimization targets for CsPbBr3 synthesis"""
    return [
        OptimizationTarget('plqy', target_value=0.8, weight=0.4, minimize=False),
        OptimizationTarget('stability_score', target_value=0.9, weight=0.3, minimize=False),
        OptimizationTarget('bandgap', target_value=2.35, weight=0.2, minimize=False, tolerance=0.3),
        OptimizationTarget('phase_purity', target_value=0.95, weight=0.1, minimize=False)
    ]

def run_optimization_example():
    """Example of running optimized ML optimization"""
    print("🚀 Running Optimized ML Optimization Example")
    
    # Define parameter bounds
    parameter_bounds = {
        'cs_br_concentration': (0.1, 3.0),
        'pb_br2_concentration': (0.1, 2.0),
        'temperature': (80.0, 250.0),
        'oa_concentration': (0.0, 1.5),
        'oam_concentration': (0.0, 1.5),
        'reaction_time': (1.0, 60.0)
    }
    
    # Create targets
    targets = create_default_targets()
    
    # Initialize optimizer
    optimizer = BayesianOptimizerOptimized(
        parameter_bounds=parameter_bounds,
        targets=targets,
        kernel_type='matern',
        acquisition_function='ei'
    )
    
    # Mock evaluation function
    def mock_evaluation(parameters):
        """Mock evaluation function for testing"""
        # Simulate physics-based evaluation
        cs_conc = parameters['cs_br_concentration']
        pb_conc = parameters['pb_br2_concentration']
        temp = parameters['temperature']
        
        # Mock properties based on parameters
        plqy = min(1.0, 0.8 * np.exp(-(temp - 150)**2 / 5000) * np.exp(-(cs_conc/pb_conc - 1.0)**2))
        stability = min(1.0, 0.9 * np.exp(-(temp - 150)**2 / 10000))
        bandgap = 2.3 + 0.1 * np.random.normal()
        phase_purity = min(1.0, 0.95 * np.exp(-(cs_conc/pb_conc - 1.0)**2 / 2))
        
        return {
            'plqy': plqy,
            'stability_score': stability,
            'bandgap': bandgap,
            'particle_size': 5 + 10 * np.random.random(),
            'phase_purity': phase_purity
        }
    
    # Run optimization
    results = optimizer.optimize_batch(
        n_iterations=10,
        batch_size=3,
        evaluation_function=mock_evaluation
    )
    
    print(f"🎯 Optimization Results:")
    print(f"   Best objective: {results['best_objective']:.4f}")
    print(f"   Best parameters: {results['best_parameters']}")
    print(f"   Total time: {results['optimization_time']:.2f}s")
    print(f"   Cache hit rate: {results['cache_hit_rate']:.2%}")

if __name__ == "__main__":
    run_optimization_example()