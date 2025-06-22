#!/usr/bin/env python3
"""
Interactive Parameter Exploration for CsPbBr3 Synthesis
Provides Jupyter widgets and real-time visualization tools
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import warnings
from pathlib import Path
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import Slider, Button, CheckButtons
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive plotting features will be limited.")

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import IPython
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False
    warnings.warn("Jupyter widgets not available. Interactive features will be limited.")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Dimensionality reduction features will be limited.")


class ParameterExplorer:
    """Interactive parameter space explorer"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 simulation_function: Optional[Callable] = None,
                 cached_data: Optional[pd.DataFrame] = None):
        """
        Initialize parameter explorer
        
        Args:
            parameter_bounds: Dictionary of parameter bounds
            simulation_function: Function to simulate synthesis with given parameters
            cached_data: Pre-computed simulation data for faster exploration
        """
        self.parameter_bounds = parameter_bounds
        self.simulation_function = simulation_function
        self.cached_data = cached_data
        
        # Current parameter values
        self.current_parameters = {
            param: (bounds[0] + bounds[1]) / 2 
            for param, bounds in parameter_bounds.items()
        }
        
        # Exploration history
        self.exploration_history = []
        
        # Interactive widgets
        self.widgets = {}
        self.output_widget = None
        
    def create_parameter_sliders(self) -> Dict[str, Any]:
        """Create interactive parameter sliders"""
        if not JUPYTER_AVAILABLE:
            print("Jupyter widgets not available. Cannot create interactive sliders.")
            return {}
        
        sliders = {}
        
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            # Create slider with appropriate step size
            step = (max_val - min_val) / 100
            current_val = self.current_parameters[param_name]
            
            slider = widgets.FloatSlider(
                value=current_val,
                min=min_val,
                max=max_val,
                step=step,
                description=param_name.replace('_', ' ').title(),
                readout_format='.3f',
                style={'description_width': '150px'},
                layout={'width': '400px'}
            )
            
            # Add callback to update parameters
            slider.observe(self._on_parameter_change, names='value')
            sliders[param_name] = slider
        
        self.widgets = sliders
        return sliders
    
    def _on_parameter_change(self, change):
        """Handle parameter slider changes"""
        # Find which parameter changed
        for param_name, slider in self.widgets.items():
            if slider == change['owner']:
                self.current_parameters[param_name] = change['new']
                break
        
        # Update visualization
        self._update_visualization()
    
    def _update_visualization(self):
        """Update visualization based on current parameters"""
        if self.output_widget is None:
            return
        
        with self.output_widget:
            clear_output(wait=True)
            
            # Run simulation with current parameters
            if self.simulation_function:
                try:
                    results = self.simulation_function(self.current_parameters)
                    self._display_simulation_results(results)
                except Exception as e:
                    print(f"Simulation error: {e}")
            
            # Add to exploration history
            self.exploration_history.append({
                'parameters': self.current_parameters.copy(),
                'timestamp': pd.Timestamp.now()
            })
    
    def _display_simulation_results(self, results: Dict[str, Any]):
        """Display simulation results"""
        print("🧪 Synthesis Prediction Results:")
        print(f"Parameters: {self.current_parameters}")
        print(f"Results: {results}")
        
        # Create simple visualization if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            self._create_results_plot(results)
    
    def _create_results_plot(self, results: Dict[str, Any]):
        """Create results visualization plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot key properties
        properties = ['bandgap', 'plqy', 'particle_size']
        values = [results.get(prop, 0) for prop in properties]
        colors = ['blue', 'green', 'red']
        
        for i, (prop, val, color) in enumerate(zip(properties, values, colors)):
            axes[i].bar([prop], [val], color=color, alpha=0.7)
            axes[i].set_title(f'{prop.replace("_", " ").title()}: {val:.3f}')
            axes[i].set_ylim(0, max(val * 1.2, 1))
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create complete interactive dashboard"""
        if not JUPYTER_AVAILABLE:
            print("Jupyter widgets not available. Cannot create interactive dashboard.")
            return
        
        # Create parameter sliders
        sliders = self.create_parameter_sliders()
        
        # Create output widget for results
        self.output_widget = widgets.Output(layout={'border': '1px solid black'})
        
        # Create control buttons
        reset_button = widgets.Button(
            description='Reset Parameters',
            button_style='warning',
            layout={'width': '150px'}
        )
        reset_button.on_click(self._reset_parameters)
        
        optimize_button = widgets.Button(
            description='Optimize',
            button_style='success',
            layout={'width': '150px'}
        )
        optimize_button.on_click(self._run_optimization)
        
        # Layout
        slider_box = widgets.VBox(list(sliders.values()))
        button_box = widgets.HBox([reset_button, optimize_button])
        controls = widgets.VBox([slider_box, button_box])
        
        dashboard = widgets.HBox([controls, self.output_widget])
        
        # Initial update
        self._update_visualization()
        
        return dashboard
    
    def _reset_parameters(self, button):
        """Reset parameters to default values"""
        for param_name, slider in self.widgets.items():
            min_val, max_val = self.parameter_bounds[param_name]
            default_val = (min_val + max_val) / 2
            slider.value = default_val
    
    def _run_optimization(self, button):
        """Run parameter optimization"""
        with self.output_widget:
            clear_output(wait=True)
            print("🎯 Running optimization...")
            
            # Simple optimization example
            if self.simulation_function:
                try:
                    best_params, best_score = self._simple_optimization()
                    
                    print(f"Optimization complete!")
                    print(f"Best parameters: {best_params}")
                    print(f"Best score: {best_score:.3f}")
                    
                    # Update sliders to best parameters
                    for param_name, value in best_params.items():
                        if param_name in self.widgets:
                            self.widgets[param_name].value = value
                    
                except Exception as e:
                    print(f"Optimization error: {e}")
    
    def _simple_optimization(self, n_iterations: int = 50) -> Tuple[Dict[str, float], float]:
        """Simple random search optimization"""
        best_score = -np.inf
        best_params = self.current_parameters.copy()
        
        for i in range(n_iterations):
            # Generate random parameters
            random_params = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                random_params[param_name] = np.random.uniform(min_val, max_val)
            
            # Evaluate
            results = self.simulation_function(random_params)
            score = results.get('plqy', 0) + results.get('stability_score', 0) * 0.5
            
            if score > best_score:
                best_score = score
                best_params = random_params.copy()
            
            if i % 10 == 0:
                print(f"  Iteration {i}/{n_iterations}, Best score: {best_score:.3f}")
        
        return best_params, best_score


class ParameterSpaceVisualizer:
    """Advanced parameter space visualization tools"""
    
    def __init__(self, data: pd.DataFrame, parameter_columns: List[str],
                 property_columns: List[str]):
        """
        Initialize parameter space visualizer
        
        Args:
            data: DataFrame with parameter and property data
            parameter_columns: List of parameter column names
            property_columns: List of property column names
        """
        self.data = data
        self.parameter_columns = parameter_columns
        self.property_columns = property_columns
        
    def create_correlation_heatmap(self, save_path: Optional[str] = None) -> None:
        """Create correlation heatmap between parameters and properties"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create heatmap.")
            return
        
        # Combine parameter and property columns
        columns_to_analyze = self.parameter_columns + self.property_columns
        
        # Calculate correlation matrix
        correlation_data = self.data[columns_to_analyze].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap
        colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', 
                 '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        sns.heatmap(correlation_data, 
                   annot=True, 
                   cmap=custom_cmap, 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Parameter-Property Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_parameter_space_plot(self, x_param: str, y_param: str, 
                                  color_property: str, 
                                  save_path: Optional[str] = None) -> None:
        """Create 2D parameter space plot colored by property"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create plot.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(self.data[x_param], 
                           self.data[y_param], 
                           c=self.data[color_property], 
                           cmap='viridis', 
                           alpha=0.7, 
                           s=50)
        
        # Customize plot
        ax.set_xlabel(x_param.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_param.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Parameter Space: {x_param} vs {y_param}\\nColored by {color_property}', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_property.replace('_', ' ').title(), fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_3d_plot(self, x_param: str, y_param: str, z_param: str,
                                  color_property: str, save_html: Optional[str] = None):
        """Create interactive 3D plot with plotly"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive plot.")
            return
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=self.data[x_param],
            y=self.data[y_param],
            z=self.data[z_param],
            mode='markers',
            marker=dict(
                size=5,
                color=self.data[color_property],
                colorscale='Viridis',
                colorbar=dict(title=color_property.replace('_', ' ').title()),
                opacity=0.8
            ),
            text=[f'{x_param}: {x:.3f}<br>{y_param}: {y:.3f}<br>{z_param}: {z:.3f}<br>{color_property}: {c:.3f}'
                  for x, y, z, c in zip(self.data[x_param], self.data[y_param], 
                                       self.data[z_param], self.data[color_property])],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title=f'3D Parameter Space: {x_param} vs {y_param} vs {z_param}',
            scene=dict(
                xaxis_title=x_param.replace('_', ' ').title(),
                yaxis_title=y_param.replace('_', ' ').title(),
                zaxis_title=z_param.replace('_', ' ').title()
            ),
            width=800,
            height=600
        )
        
        if save_html:
            fig.write_html(save_html)
        
        fig.show()
    
    def create_parallel_coordinates_plot(self, properties_to_show: Optional[List[str]] = None,
                                       save_html: Optional[str] = None):
        """Create parallel coordinates plot for multi-dimensional visualization"""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create parallel coordinates plot.")
            return
        
        if properties_to_show is None:
            properties_to_show = self.parameter_columns + self.property_columns[:3]
        
        # Prepare data for parallel coordinates
        plot_data = self.data[properties_to_show].copy()
        
        # Create parallel coordinates plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=self.data[self.property_columns[0]], 
                     colorscale='Viridis',
                     showscale=True,
                     colorbar=dict(title=self.property_columns[0].replace('_', ' ').title())),
            dimensions=[
                dict(label=col.replace('_', ' ').title(), 
                     values=plot_data[col])
                for col in properties_to_show
            ]
        ))
        
        fig.update_layout(
            title='Multi-Dimensional Parameter Space Exploration',
            width=1000,
            height=600
        )
        
        if save_html:
            fig.write_html(save_html)
        
        fig.show()
    
    def create_dimensionality_reduction_plot(self, method: str = 'pca', 
                                           color_property: Optional[str] = None,
                                           save_path: Optional[str] = None):
        """Create dimensionality reduction visualization"""
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            print("scikit-learn and/or matplotlib not available. Cannot create dimensionality reduction plot.")
            return
        
        # Prepare data
        X = self.data[self.parameter_columns].copy()
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            title_method = 'PCA'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            title_method = 't-SNE'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_reduced = reducer.fit_transform(X_scaled)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_property and color_property in self.data.columns:
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                               c=self.data[color_property], 
                               cmap='viridis', alpha=0.7, s=50)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_property.replace('_', ' ').title(), fontsize=12)
        else:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, s=50)
        
        ax.set_xlabel(f'{title_method} Component 1', fontsize=12)
        ax.set_ylabel(f'{title_method} Component 2', fontsize=12)
        ax.set_title(f'{title_method} Visualization of Parameter Space', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add explained variance for PCA
        if method.lower() == 'pca':
            explained_var = reducer.explained_variance_ratio_
            ax.text(0.02, 0.98, f'Explained Variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class SensitivityAnalyzer:
    """Sensitivity analysis tools for parameter exploration"""
    
    def __init__(self, simulation_function: Callable, 
                 parameter_bounds: Dict[str, Tuple[float, float]]):
        """Initialize sensitivity analyzer"""
        self.simulation_function = simulation_function
        self.parameter_bounds = parameter_bounds
        
    def local_sensitivity_analysis(self, base_parameters: Dict[str, float],
                                 property_name: str,
                                 perturbation_percent: float = 0.1,
                                 n_samples: int = 10) -> Dict[str, List[float]]:
        """Perform local sensitivity analysis around base parameters"""
        sensitivities = {}
        
        for param_name in self.parameter_bounds.keys():
            param_sensitivities = []
            base_value = base_parameters[param_name]
            min_val, max_val = self.parameter_bounds[param_name]
            
            # Calculate perturbation range
            perturbation = (max_val - min_val) * perturbation_percent
            
            for i in range(n_samples):
                # Create perturbed parameters
                perturbed_params = base_parameters.copy()
                
                # Apply perturbation
                perturbation_value = (i - n_samples//2) * perturbation / (n_samples//2)
                new_value = base_value + perturbation_value
                new_value = np.clip(new_value, min_val, max_val)
                perturbed_params[param_name] = new_value
                
                # Run simulation
                try:
                    results = self.simulation_function(perturbed_params)
                    property_value = results.get(property_name, 0)
                    param_sensitivities.append(property_value)
                except:
                    param_sensitivities.append(0)
            
            sensitivities[param_name] = param_sensitivities
        
        return sensitivities
    
    def global_sensitivity_analysis(self, property_name: str, 
                                  n_samples: int = 1000) -> Dict[str, float]:
        """Perform global sensitivity analysis using variance-based methods"""
        # Generate random samples
        samples = []
        property_values = []
        
        for _ in range(n_samples):
            # Generate random parameters
            random_params = {}
            for param_name, (min_val, max_val) in self.parameter_bounds.items():
                random_params[param_name] = np.random.uniform(min_val, max_val)
            
            samples.append(random_params)
            
            # Run simulation
            try:
                results = self.simulation_function(random_params)
                property_values.append(results.get(property_name, 0))
            except:
                property_values.append(0)
        
        # Convert to DataFrame for analysis
        param_df = pd.DataFrame(samples)
        property_values = np.array(property_values)
        
        # Calculate sensitivity indices (simplified Sobol-like approach)
        total_variance = np.var(property_values)
        sensitivities = {}
        
        for param_name in self.parameter_bounds.keys():
            # Calculate conditional variance
            param_values = param_df[param_name]
            
            # Bin parameter values and calculate variance within bins
            n_bins = 10
            bins = np.linspace(param_values.min(), param_values.max(), n_bins + 1)
            bin_indices = np.digitize(param_values, bins)
            
            conditional_variances = []
            for bin_idx in range(1, n_bins + 1):
                mask = bin_indices == bin_idx
                if mask.sum() > 1:
                    conditional_var = np.var(property_values[mask])
                    conditional_variances.append(conditional_var)
            
            if conditional_variances:
                avg_conditional_var = np.mean(conditional_variances)
                sensitivity = 1 - (avg_conditional_var / total_variance) if total_variance > 0 else 0
            else:
                sensitivity = 0
            
            sensitivities[param_name] = max(0, min(1, sensitivity))
        
        return sensitivities
    
    def create_sensitivity_plot(self, sensitivity_data: Dict[str, float],
                              title: str = "Parameter Sensitivity Analysis",
                              save_path: Optional[str] = None):
        """Create sensitivity analysis visualization"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot create sensitivity plot.")
            return
        
        # Sort parameters by sensitivity
        sorted_items = sorted(sensitivity_data.items(), key=lambda x: x[1], reverse=True)
        param_names = [item[0].replace('_', ' ').title() for item in sorted_items]
        sensitivities = [item[1] for item in sorted_items]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sensitivities)))
        bars = ax.barh(param_names, sensitivities, color=colors)
        
        # Customize plot
        ax.set_xlabel('Sensitivity Index', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, sensitivity in zip(bars, sensitivities):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{sensitivity:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_exploration_report(data: pd.DataFrame, 
                            parameter_columns: List[str],
                            property_columns: List[str],
                            output_dir: str = "exploration_report") -> str:
    """Create comprehensive exploration report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = ParameterSpaceVisualizer(data, parameter_columns, property_columns)
    
    print("📊 Generating exploration report...")
    
    # 1. Correlation heatmap
    print("   Creating correlation heatmap...")
    visualizer.create_correlation_heatmap(save_path=output_path / "correlation_heatmap.png")
    
    # 2. Parameter space plots for key combinations
    key_params = parameter_columns[:3]  # Use first 3 parameters
    key_property = property_columns[0]  # Use first property
    
    print("   Creating parameter space plots...")
    if len(key_params) >= 2:
        visualizer.create_parameter_space_plot(
            key_params[0], key_params[1], key_property,
            save_path=output_path / f"param_space_{key_params[0]}_{key_params[1]}.png"
        )
    
    # 3. Dimensionality reduction
    print("   Creating dimensionality reduction plots...")
    visualizer.create_dimensionality_reduction_plot(
        method='pca', color_property=key_property,
        save_path=output_path / "pca_analysis.png"
    )
    
    # 4. Generate summary statistics
    print("   Generating summary statistics...")
    summary_stats = {
        'dataset_info': {
            'n_samples': len(data),
            'n_parameters': len(parameter_columns),
            'n_properties': len(property_columns)
        },
        'parameter_ranges': {
            param: {
                'min': float(data[param].min()),
                'max': float(data[param].max()),
                'mean': float(data[param].mean()),
                'std': float(data[param].std())
            }
            for param in parameter_columns
        },
        'property_ranges': {
            prop: {
                'min': float(data[prop].min()),
                'max': float(data[prop].max()),
                'mean': float(data[prop].mean()),
                'std': float(data[prop].std())
            }
            for prop in property_columns
        }
    }
    
    # Save summary
    with open(output_path / "summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # 5. Create HTML report
    html_report = _generate_html_report(summary_stats, output_path)
    with open(output_path / "exploration_report.html", 'w') as f:
        f.write(html_report)
    
    print(f"✅ Exploration report saved to: {output_path}")
    return str(output_path)


def _generate_html_report(summary_stats: Dict[str, Any], 
                         output_path: Path) -> str:
    """Generate HTML exploration report"""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CsPbBr3 Parameter Exploration Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                     background-color: #e9ecef; border-radius: 3px; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>🧪 CsPbBr3 Digital Twin - Parameter Exploration Report</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>📊 Dataset Summary</h2>
            <div class="metric">
                <strong>Samples:</strong> {summary_stats['dataset_info']['n_samples']}
            </div>
            <div class="metric">
                <strong>Parameters:</strong> {summary_stats['dataset_info']['n_parameters']}
            </div>
            <div class="metric">
                <strong>Properties:</strong> {summary_stats['dataset_info']['n_properties']}
            </div>
        </div>
        
        <h2>🔗 Parameter Correlations</h2>
        <img src="correlation_heatmap.png" alt="Correlation Heatmap">
        
        <h2>🗺️ Parameter Space Visualization</h2>
        <img src="pca_analysis.png" alt="PCA Analysis">
        
        <h2>📈 Parameter Statistics</h2>
        <table>
            <tr><th>Parameter</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th></tr>
    """
    
    # Add parameter statistics rows
    for param, stats in summary_stats['parameter_ranges'].items():
        html_template += f"""
            <tr>
                <td>{param.replace('_', ' ').title()}</td>
                <td>{stats['min']:.3f}</td>
                <td>{stats['max']:.3f}</td>
                <td>{stats['mean']:.3f}</td>
                <td>{stats['std']:.3f}</td>
            </tr>
        """
    
    html_template += """
        </table>
        
        <h2>🎯 Property Statistics</h2>
        <table>
            <tr><th>Property</th><th>Min</th><th>Max</th><th>Mean</th><th>Std Dev</th></tr>
    """
    
    # Add property statistics rows
    for prop, stats in summary_stats['property_ranges'].items():
        html_template += f"""
            <tr>
                <td>{prop.replace('_', ' ').title()}</td>
                <td>{stats['min']:.3f}</td>
                <td>{stats['max']:.3f}</td>
                <td>{stats['mean']:.3f}</td>
                <td>{stats['std']:.3f}</td>
            </tr>
        """
    
    html_template += """
        </table>
        
        <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
            <p>Generated by CsPbBr3 Digital Twin Interactive Exploration Tools</p>
        </footer>
    </body>
    </html>
    """
    
    return html_template


if __name__ == "__main__":
    # Demo of interactive exploration tools
    print("🎮 Interactive Parameter Exploration Demo")
    
    # Define parameter bounds
    parameter_bounds = {
        'cs_br_concentration': (0.1, 3.0),
        'pb_br2_concentration': (0.1, 2.0),
        'temperature': (80.0, 250.0),
        'oa_concentration': (0.0, 1.5),
        'oam_concentration': (0.0, 1.5),
        'reaction_time': (1.0, 60.0)
    }
    
    # Create mock simulation function
    def mock_simulation(params):
        """Mock simulation for demonstration"""
        # Simple mock that creates reasonable correlations
        temp_effect = (params['temperature'] - 150) / 100
        conc_effect = params['cs_br_concentration'] / params['pb_br2_concentration']
        
        return {
            'bandgap': 2.3 + temp_effect * 0.2 + np.random.normal(0, 0.1),
            'plqy': 0.8 * np.exp(-abs(conc_effect - 1.0)) + np.random.normal(0, 0.05),
            'particle_size': 10 + temp_effect * 5 + np.random.normal(0, 1),
            'stability_score': 0.7 + temp_effect * 0.1 + np.random.normal(0, 0.05)
        }
    
    # Generate demo data
    print("📊 Generating demo data...")
    demo_data = []
    for i in range(500):
        # Random parameters
        params = {
            param: np.random.uniform(bounds[0], bounds[1])
            for param, bounds in parameter_bounds.items()
        }
        
        # Simulate
        results = mock_simulation(params)
        
        # Combine
        demo_data.append({**params, **results})
    
    df = pd.DataFrame(demo_data)
    
    # Create visualizations
    parameter_columns = list(parameter_bounds.keys())
    property_columns = ['bandgap', 'plqy', 'particle_size', 'stability_score']
    
    print("🎨 Creating visualizations...")
    
    # Create exploration report
    report_path = create_exploration_report(df, parameter_columns, property_columns)
    
    # Test sensitivity analysis
    print("🔍 Running sensitivity analysis...")
    analyzer = SensitivityAnalyzer(mock_simulation, parameter_bounds)
    
    base_params = {param: (bounds[0] + bounds[1]) / 2 
                  for param, bounds in parameter_bounds.items()}
    
    sensitivities = analyzer.global_sensitivity_analysis('plqy', n_samples=200)
    analyzer.create_sensitivity_plot(sensitivities, 
                                   title="PLQY Sensitivity Analysis",
                                   save_path="plqy_sensitivity.png")
    
    print("✅ Interactive exploration demo complete!")
    print(f"📁 Report available at: {report_path}/exploration_report.html")
    print(f"🎯 Sensitivity plot saved as: plqy_sensitivity.png")