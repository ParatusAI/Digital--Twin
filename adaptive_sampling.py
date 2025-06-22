#!/usr/bin/env python3
"""
Advanced Adaptive Sampling Strategies for CsPbBr3 Synthesis
Implements Latin Hypercube, importance sampling, and hierarchical strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from pathlib import Path

try:
    from scipy.stats import qmc, norm, uniform
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced sampling features will be limited.")

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Clustering-based sampling will be limited.")


@dataclass
class SamplingRegion:
    """Define a region of parameter space for targeted sampling"""
    bounds: Dict[str, Tuple[float, float]]
    weight: float = 1.0
    priority: int = 1  # Higher numbers = higher priority
    success_rate: float = 0.5  # Historical success rate in this region
    uncertainty: float = 0.1  # Uncertainty about this region
    description: str = ""


@dataclass 
class SamplingStrategy:
    """Configuration for sampling strategy"""
    method: str  # 'latin_hypercube', 'importance', 'hierarchical', 'adaptive'
    n_samples: int
    regions: List[SamplingRegion] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42


class BaseSampler(ABC):
    """Abstract base class for sampling strategies"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 random_seed: int = 42):
        self.parameter_bounds = parameter_bounds
        self.random_seed = random_seed
        self.param_names = list(parameter_bounds.keys())
        np.random.seed(random_seed)
    
    @abstractmethod
    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate n samples using the specific strategy"""
        pass
    
    def _validate_samples(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Validate that samples are within bounds"""
        for param_name in self.param_names:
            min_val, max_val = self.parameter_bounds[param_name]
            samples[param_name] = np.clip(samples[param_name], min_val, max_val)
        return samples


class LatinHypercubeSampler(BaseSampler):
    """Latin Hypercube Sampling for better space coverage"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 random_seed: int = 42, 
                 criterion: str = 'maximin'):
        """
        Initialize Latin Hypercube Sampler
        
        Args:
            parameter_bounds: Parameter bounds
            random_seed: Random seed
            criterion: LHS criterion ('center', 'maximin', 'correlation')
        """
        super().__init__(parameter_bounds, random_seed)
        self.criterion = criterion
        
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available. Falling back to random sampling.")
    
    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate Latin Hypercube samples"""
        if not SCIPY_AVAILABLE:
            return self._fallback_random_sampling(n_samples)
        
        n_dims = len(self.param_names)
        
        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=n_dims, seed=self.random_seed)
        
        if self.criterion == 'maximin':
            # Optimize for maximin distance
            lhs_samples = sampler.random(n_samples)
            
            # Try to improve with optimization (if enough samples)
            if n_samples >= 20 and n_samples <= 1000:
                try:
                    for _ in range(10):  # Simple optimization iterations
                        candidate = sampler.random(n_samples)
                        if self._evaluate_maximin_criterion(candidate) > self._evaluate_maximin_criterion(lhs_samples):
                            lhs_samples = candidate
                except:
                    pass  # Use original if optimization fails
        else:
            lhs_samples = sampler.random(n_samples)
        
        # Scale to parameter bounds
        samples_dict = {}
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.parameter_bounds[param_name]
            scaled_values = lhs_samples[:, i] * (max_val - min_val) + min_val
            samples_dict[param_name] = scaled_values
        
        df = pd.DataFrame(samples_dict)
        return self._validate_samples(df)
    
    def _evaluate_maximin_criterion(self, samples: np.ndarray) -> float:
        """Evaluate maximin criterion for sample quality"""
        try:
            distances = pdist(samples)
            return np.min(distances) if len(distances) > 0 else 0.0
        except:
            return 0.0
    
    def _fallback_random_sampling(self, n_samples: int) -> pd.DataFrame:
        """Fallback to random sampling if SciPy unavailable"""
        samples_dict = {}
        for param_name in self.param_names:
            min_val, max_val = self.parameter_bounds[param_name]
            samples_dict[param_name] = np.random.uniform(min_val, max_val, n_samples)
        
        return pd.DataFrame(samples_dict)


class ImportanceSampler(BaseSampler):
    """Importance sampling for focusing on interesting regions"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 importance_function: Optional[Callable] = None,
                 regions: Optional[List[SamplingRegion]] = None,
                 random_seed: int = 42):
        """
        Initialize importance sampler
        
        Args:
            parameter_bounds: Parameter bounds
            importance_function: Function to evaluate importance of regions
            regions: Pre-defined important regions
            random_seed: Random seed
        """
        super().__init__(parameter_bounds, random_seed)
        self.importance_function = importance_function
        self.regions = regions or []
        
    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate importance-weighted samples"""
        if not self.regions:
            # No regions defined, fall back to weighted random sampling
            return self._weighted_random_sampling(n_samples)
        
        # Allocate samples to regions based on importance
        region_samples = self._allocate_samples_to_regions(n_samples)
        
        all_samples = []
        for region, n_region_samples in zip(self.regions, region_samples):
            if n_region_samples > 0:
                region_df = self._sample_from_region(region, n_region_samples)
                all_samples.append(region_df)
        
        if all_samples:
            combined_df = pd.concat(all_samples, ignore_index=True)
            return self._validate_samples(combined_df)
        else:
            return self._weighted_random_sampling(n_samples)
    
    def _allocate_samples_to_regions(self, n_samples: int) -> List[int]:
        """Allocate samples to regions based on weights and priorities"""
        if not self.regions:
            return []
        
        # Calculate weights considering both weight and priority
        total_weight = sum(region.weight * region.priority for region in self.regions)
        
        allocations = []
        remaining_samples = n_samples
        
        for i, region in enumerate(self.regions):
            if i == len(self.regions) - 1:  # Last region gets remaining samples
                allocations.append(remaining_samples)
            else:
                region_weight = (region.weight * region.priority) / total_weight
                n_region_samples = int(n_samples * region_weight)
                allocations.append(n_region_samples)
                remaining_samples -= n_region_samples
        
        return allocations
    
    def _sample_from_region(self, region: SamplingRegion, n_samples: int) -> pd.DataFrame:
        """Sample from a specific region"""
        samples_dict = {}
        
        for param_name in self.param_names:
            if param_name in region.bounds:
                min_val, max_val = region.bounds[param_name]
            else:
                min_val, max_val = self.parameter_bounds[param_name]
            
            # Add some randomness based on region uncertainty
            uncertainty_factor = region.uncertainty
            range_width = max_val - min_val
            
            # Generate samples with bias toward successful regions
            if region.success_rate > 0.7:  # High success regions
                # Sample more densely in center of range
                center = (min_val + max_val) / 2
                std = range_width * 0.2  # 20% of range as std
                samples = np.random.normal(center, std, n_samples)
                samples = np.clip(samples, min_val, max_val)
            else:
                # Uniform sampling for uncertain regions
                samples = np.random.uniform(min_val, max_val, n_samples)
            
            samples_dict[param_name] = samples
        
        return pd.DataFrame(samples_dict)
    
    def _weighted_random_sampling(self, n_samples: int) -> pd.DataFrame:
        """Fallback weighted random sampling"""
        samples_dict = {}
        
        for param_name in self.param_names:
            min_val, max_val = self.parameter_bounds[param_name]
            
            # Apply simple importance weighting based on parameter
            if self.importance_function:
                # Custom importance function
                samples = self._sample_with_importance(param_name, min_val, max_val, n_samples)
            else:
                # Default: focus on middle ranges for most parameters
                if 'temperature' in param_name:
                    # Favor typical synthesis temperatures
                    samples = np.random.normal(150, 30, n_samples)
                    samples = np.clip(samples, min_val, max_val)
                elif 'concentration' in param_name:
                    # Log-normal for concentrations
                    log_mean = np.log((min_val + max_val) / 2)
                    samples = np.random.lognormal(log_mean, 0.5, n_samples)
                    samples = np.clip(samples, min_val, max_val)
                else:
                    samples = np.random.uniform(min_val, max_val, n_samples)
            
            samples_dict[param_name] = samples
        
        return pd.DataFrame(samples_dict)
    
    def _sample_with_importance(self, param_name: str, min_val: float, 
                               max_val: float, n_samples: int) -> np.ndarray:
        """Sample using custom importance function"""
        samples = []
        for _ in range(n_samples):
            # Simple rejection sampling
            for _ in range(100):  # Max attempts
                candidate = np.random.uniform(min_val, max_val)
                importance = self.importance_function({param_name: candidate})
                if np.random.random() < importance:
                    samples.append(candidate)
                    break
            else:
                samples.append(np.random.uniform(min_val, max_val))  # Fallback
        
        return np.array(samples)


class HierarchicalSampler(BaseSampler):
    """Hierarchical sampling with multi-level parameter exploration"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 hierarchy_levels: int = 3,
                 refinement_factor: float = 0.5,
                 random_seed: int = 42):
        """
        Initialize hierarchical sampler
        
        Args:
            parameter_bounds: Parameter bounds
            hierarchy_levels: Number of refinement levels
            refinement_factor: Factor by which to refine each level
            random_seed: Random seed
        """
        super().__init__(parameter_bounds, random_seed)
        self.hierarchy_levels = hierarchy_levels
        self.refinement_factor = refinement_factor
        
    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate hierarchical samples"""
        all_samples = []
        
        # Level 0: Coarse global sampling
        level_0_samples = max(1, n_samples // (2 ** self.hierarchy_levels))
        coarse_samples = self._generate_coarse_samples(level_0_samples)
        all_samples.append(coarse_samples)
        
        # Subsequent levels: Refine around promising regions
        current_samples = coarse_samples
        remaining_samples = n_samples - level_0_samples
        
        for level in range(1, self.hierarchy_levels):
            if remaining_samples <= 0:
                break
            
            # Determine samples for this level
            level_samples = max(1, remaining_samples // (self.hierarchy_levels - level))
            remaining_samples -= level_samples
            
            # Find promising regions from current samples
            promising_regions = self._identify_promising_regions(current_samples)
            
            # Generate refined samples
            refined_samples = self._generate_refined_samples(
                promising_regions, level_samples, level
            )
            
            all_samples.append(refined_samples)
            current_samples = pd.concat([current_samples, refined_samples], ignore_index=True)
        
        # Combine all samples
        combined_df = pd.concat(all_samples, ignore_index=True)
        return self._validate_samples(combined_df)
    
    def _generate_coarse_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate coarse initial samples"""
        # Use Latin Hypercube for good space coverage
        lhs_sampler = LatinHypercubeSampler(self.parameter_bounds, self.random_seed)
        return lhs_sampler.generate_samples(n_samples)
    
    def _identify_promising_regions(self, samples: pd.DataFrame) -> List[Dict[str, Tuple[float, float]]]:
        """Identify promising regions from existing samples"""
        if len(samples) < 3:
            return [self.parameter_bounds]  # Fall back to full space
        
        # Simple clustering-based approach
        if SKLEARN_AVAILABLE and len(samples) >= 5:
            try:
                # Standardize samples
                scaler = StandardScaler()
                scaled_samples = scaler.fit_transform(samples[self.param_names])
                
                # Cluster samples
                n_clusters = min(3, len(samples) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
                clusters = kmeans.fit_predict(scaled_samples)
                
                # Create regions around cluster centers
                regions = []
                for cluster_id in range(n_clusters):
                    cluster_mask = clusters == cluster_id
                    cluster_samples = samples[cluster_mask]
                    
                    if len(cluster_samples) == 0:
                        continue
                    
                    # Define region as expanded bounding box of cluster
                    region_bounds = {}
                    for param_name in self.param_names:
                        param_values = cluster_samples[param_name]
                        param_min = float(param_values.min())
                        param_max = float(param_values.max())
                        
                        # Expand region
                        param_range = param_max - param_min
                        expansion = max(param_range * 0.5, 
                                      (self.parameter_bounds[param_name][1] - 
                                       self.parameter_bounds[param_name][0]) * 0.1)
                        
                        region_min = max(self.parameter_bounds[param_name][0], 
                                       param_min - expansion)
                        region_max = min(self.parameter_bounds[param_name][1], 
                                       param_max + expansion)
                        
                        region_bounds[param_name] = (region_min, region_max)
                    
                    regions.append(region_bounds)
                
                return regions if regions else [self.parameter_bounds]
                
            except Exception:
                pass  # Fall back to simple approach
        
        # Simple fallback: divide space based on parameter ranges
        n_regions = min(3, len(samples))
        regions = []
        
        for i in range(n_regions):
            region_bounds = {}
            for param_name in self.param_names:
                min_val, max_val = self.parameter_bounds[param_name]
                range_width = max_val - min_val
                
                # Create overlapping regions
                region_start = min_val + (i / n_regions) * range_width * 0.7
                region_end = min_val + ((i + 1) / n_regions) * range_width * 1.3
                
                region_bounds[param_name] = (
                    max(min_val, region_start),
                    min(max_val, region_end)
                )
            
            regions.append(region_bounds)
        
        return regions
    
    def _generate_refined_samples(self, regions: List[Dict[str, Tuple[float, float]]], 
                                 n_samples: int, level: int) -> pd.DataFrame:
        """Generate refined samples in promising regions"""
        if not regions:
            return pd.DataFrame()
        
        # Distribute samples among regions
        samples_per_region = max(1, n_samples // len(regions))
        remaining_samples = n_samples - (samples_per_region * len(regions))
        
        all_samples = []
        
        for i, region_bounds in enumerate(regions):
            # Add extra samples to first regions if there are remainders
            current_samples = samples_per_region
            if i < remaining_samples:
                current_samples += 1
            
            if current_samples <= 0:
                continue
            
            # Generate samples in this region
            region_df = self._sample_from_bounds(region_bounds, current_samples, level)
            all_samples.append(region_df)
        
        if all_samples:
            return pd.concat(all_samples, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _sample_from_bounds(self, bounds: Dict[str, Tuple[float, float]], 
                           n_samples: int, level: int) -> pd.DataFrame:
        """Sample from specific bounds with level-dependent strategy"""
        samples_dict = {}
        
        # Adjust sampling strategy based on level
        concentration_factor = self.refinement_factor ** level
        
        for param_name in self.param_names:
            if param_name in bounds:
                min_val, max_val = bounds[param_name]
            else:
                min_val, max_val = self.parameter_bounds[param_name]
            
            if level == 0:
                # Uniform sampling for initial level
                samples = np.random.uniform(min_val, max_val, n_samples)
            else:
                # More concentrated sampling for higher levels
                center = (min_val + max_val) / 2
                std = (max_val - min_val) * concentration_factor * 0.25
                samples = np.random.normal(center, std, n_samples)
                samples = np.clip(samples, min_val, max_val)
            
            samples_dict[param_name] = samples
        
        return pd.DataFrame(samples_dict)


class AdaptiveSampler(BaseSampler):
    """Adaptive sampler that combines multiple strategies"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 adaptation_frequency: int = 50,
                 strategy_weights: Optional[Dict[str, float]] = None,
                 random_seed: int = 42):
        """
        Initialize adaptive sampler
        
        Args:
            parameter_bounds: Parameter bounds
            adaptation_frequency: How often to adapt strategy
            strategy_weights: Weights for different strategies
            random_seed: Random seed
        """
        super().__init__(parameter_bounds, random_seed)
        self.adaptation_frequency = adaptation_frequency
        self.strategy_weights = strategy_weights or {
            'latin_hypercube': 0.3,
            'importance': 0.3,
            'hierarchical': 0.2,
            'random': 0.2
        }
        
        # Initialize sub-samplers
        self.lhs_sampler = LatinHypercubeSampler(parameter_bounds, random_seed)
        self.importance_sampler = ImportanceSampler(parameter_bounds, random_seed=random_seed)
        self.hierarchical_sampler = HierarchicalSampler(parameter_bounds, random_seed=random_seed)
        
        # Track performance
        self.strategy_performance = {strategy: [] for strategy in self.strategy_weights.keys()}
        self.sample_count = 0
        
    def generate_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate samples using adaptive strategy selection"""
        all_samples = []
        remaining_samples = n_samples
        
        while remaining_samples > 0:
            # Determine batch size
            batch_size = min(self.adaptation_frequency, remaining_samples)
            
            # Select strategy
            strategy = self._select_strategy()
            
            # Generate samples with selected strategy
            if strategy == 'latin_hypercube':
                batch_samples = self.lhs_sampler.generate_samples(batch_size)
            elif strategy == 'importance':
                batch_samples = self.importance_sampler.generate_samples(batch_size)
            elif strategy == 'hierarchical':
                batch_samples = self.hierarchical_sampler.generate_samples(batch_size)
            else:  # random
                batch_samples = self._random_sampling(batch_size)
            
            all_samples.append(batch_samples)
            remaining_samples -= batch_size
            self.sample_count += batch_size
            
            # Update strategy performance (simplified)
            self._update_strategy_performance(strategy, batch_samples)
        
        # Combine all samples
        combined_df = pd.concat(all_samples, ignore_index=True)
        return self._validate_samples(combined_df)
    
    def _select_strategy(self) -> str:
        """Select sampling strategy based on performance and weights"""
        if self.sample_count < self.adaptation_frequency:
            # Initial phase: use weighted random selection
            strategies = list(self.strategy_weights.keys())
            weights = list(self.strategy_weights.values())
            return np.random.choice(strategies, p=weights)
        
        # Adaptive phase: adjust weights based on performance
        adjusted_weights = self._calculate_adjusted_weights()
        
        strategies = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        
        return np.random.choice(strategies, p=weights)
    
    def _calculate_adjusted_weights(self) -> Dict[str, float]:
        """Calculate adjusted weights based on strategy performance"""
        adjusted_weights = {}
        
        for strategy, base_weight in self.strategy_weights.items():
            performance_scores = self.strategy_performance.get(strategy, [])
            
            if performance_scores:
                # Simple performance metric: average "quality" of samples
                avg_performance = np.mean(performance_scores)
                # Adjust weight based on performance
                adjusted_weight = base_weight * (1 + avg_performance)
            else:
                adjusted_weight = base_weight
            
            adjusted_weights[strategy] = adjusted_weight
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        else:
            adjusted_weights = self.strategy_weights
        
        return adjusted_weights
    
    def _update_strategy_performance(self, strategy: str, samples: pd.DataFrame):
        """Update performance tracking for strategy"""
        # Simple quality metric: diversity of samples
        try:
            if len(samples) > 1:
                # Calculate sample diversity as quality metric
                diversity = self._calculate_sample_diversity(samples)
                self.strategy_performance[strategy].append(diversity)
                
                # Keep only recent performance data
                max_history = 10
                if len(self.strategy_performance[strategy]) > max_history:
                    self.strategy_performance[strategy] = \
                        self.strategy_performance[strategy][-max_history:]
        except:
            pass  # Ignore errors in performance tracking
    
    def _calculate_sample_diversity(self, samples: pd.DataFrame) -> float:
        """Calculate diversity score for samples"""
        try:
            # Simple diversity metric: average pairwise distance
            if len(samples) < 2:
                return 0.0
            
            # Normalize samples to [0,1] range
            normalized_samples = []
            for param_name in self.param_names:
                min_val, max_val = self.parameter_bounds[param_name]
                if max_val > min_val:
                    normalized = (samples[param_name] - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros(len(samples))
                normalized_samples.append(normalized)
            
            normalized_array = np.column_stack(normalized_samples)
            
            # Calculate average pairwise distance
            distances = pdist(normalized_array)
            return float(np.mean(distances)) if len(distances) > 0 else 0.0
            
        except:
            return 0.0
    
    def _random_sampling(self, n_samples: int) -> pd.DataFrame:
        """Simple random sampling fallback"""
        samples_dict = {}
        for param_name in self.param_names:
            min_val, max_val = self.parameter_bounds[param_name]
            samples_dict[param_name] = np.random.uniform(min_val, max_val, n_samples)
        
        return pd.DataFrame(samples_dict)


def create_adaptive_sampling_strategy(
    parameter_bounds: Dict[str, Tuple[float, float]],
    target_regions: Optional[List[SamplingRegion]] = None,
    strategy_config: Optional[SamplingStrategy] = None
) -> BaseSampler:
    """
    Create an adaptive sampling strategy based on configuration
    
    Args:
        parameter_bounds: Parameter bounds for sampling
        target_regions: Specific regions to focus sampling on
        strategy_config: Configuration for sampling strategy
    
    Returns:
        Configured sampler instance
    """
    if strategy_config is None:
        strategy_config = SamplingStrategy(
            method='adaptive',
            n_samples=1000
        )
    
    if strategy_config.method == 'latin_hypercube':
        return LatinHypercubeSampler(
            parameter_bounds, 
            random_seed=strategy_config.random_seed
        )
    
    elif strategy_config.method == 'importance':
        return ImportanceSampler(
            parameter_bounds,
            regions=target_regions,
            random_seed=strategy_config.random_seed
        )
    
    elif strategy_config.method == 'hierarchical':
        return HierarchicalSampler(
            parameter_bounds,
            random_seed=strategy_config.random_seed
        )
    
    elif strategy_config.method == 'adaptive':
        return AdaptiveSampler(
            parameter_bounds,
            random_seed=strategy_config.random_seed
        )
    
    else:
        raise ValueError(f"Unknown sampling method: {strategy_config.method}")


if __name__ == "__main__":
    # Example usage
    parameter_bounds = {
        'cs_br_concentration': (0.1, 3.0),
        'pb_br2_concentration': (0.1, 2.0),
        'temperature': (80.0, 250.0),
        'oa_concentration': (0.0, 1.5),
        'oam_concentration': (0.0, 1.5),
        'reaction_time': (1.0, 60.0)
    }
    
    # Define regions of interest
    high_plqy_region = SamplingRegion(
        bounds={
            'temperature': (140.0, 160.0),
            'cs_br_concentration': (0.8, 1.2),
            'pb_br2_concentration': (0.8, 1.2)
        },
        weight=2.0,
        priority=3,
        success_rate=0.8,
        description="High PLQY region based on literature"
    )
    
    # Test different sampling strategies
    strategies = ['latin_hypercube', 'importance', 'hierarchical', 'adaptive']
    
    for method in strategies:
        print(f"\n🎯 Testing {method} sampling...")
        
        config = SamplingStrategy(
            method=method,
            n_samples=100,
            regions=[high_plqy_region]
        )
        
        sampler = create_adaptive_sampling_strategy(
            parameter_bounds, 
            target_regions=[high_plqy_region],
            strategy_config=config
        )
        
        samples = sampler.generate_samples(100)
        
        print(f"   Generated {len(samples)} samples")
        print(f"   Temperature range: {samples['temperature'].min():.1f} - {samples['temperature'].max():.1f}")
        print(f"   Cs/Pb ratio range: {(samples['cs_br_concentration']/samples['pb_br2_concentration']).min():.2f} - {(samples['cs_br_concentration']/samples['pb_br2_concentration']).max():.2f}")
        
        # Save samples
        output_file = f"samples_{method}.csv"
        samples.to_csv(output_file, index=False)
        print(f"   Saved to {output_file}")
    
    print("\n✅ Adaptive sampling demonstration complete!")