#!/usr/bin/env python3
"""
Experimental Data Integration for CsPbBr3 Digital Twin
Handles integration of real experimental data with simulation results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings
from datetime import datetime, timezone
import re

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML calibration features will be limited.")

try:
    from scipy import stats
    from scipy.optimize import minimize, differential_evolution
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical and optimization features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization features will be limited.")


@dataclass
class ExperimentalDataPoint:
    """Container for experimental data point"""
    experiment_id: str
    timestamp: str
    parameters: Dict[str, float]
    measurements: Dict[str, float]
    measurement_errors: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0  # 0-1 quality assessment
    notes: str = ""


@dataclass
class CalibrationResult:
    """Results from model calibration"""
    calibration_method: str
    calibrated_parameters: Dict[str, float]
    calibration_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    residual_analysis: Dict[str, Any]
    calibration_data: pd.DataFrame
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExperimentalDataManager:
    """Manager for experimental data import, validation, and preprocessing"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize experimental data manager"""
        self.data_dir = Path(data_dir) if data_dir else Path("experimental_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.experimental_data: List[ExperimentalDataPoint] = []
        self.data_schemas = self._load_data_schemas()
        
    def _load_data_schemas(self) -> Dict[str, Any]:
        """Load data schemas for validation"""
        default_schemas = {
            "synthesis_parameters": {
                "required": ["cs_br_concentration", "pb_br2_concentration", "temperature"],
                "optional": ["oa_concentration", "oam_concentration", "reaction_time", "solvent_type"],
                "ranges": {
                    "cs_br_concentration": (0.01, 10.0),
                    "pb_br2_concentration": (0.01, 10.0),
                    "temperature": (20.0, 300.0),
                    "oa_concentration": (0.0, 5.0),
                    "oam_concentration": (0.0, 5.0),
                    "reaction_time": (0.1, 1440.0)  # minutes
                }
            },
            "measurements": {
                "required": [],
                "optional": ["bandgap", "plqy", "emission_peak", "particle_size", 
                           "emission_fwhm", "lifetime", "stability_score", "phase_purity"],
                "ranges": {
                    "bandgap": (1.0, 5.0),
                    "plqy": (0.0, 1.0),
                    "emission_peak": (300.0, 800.0),
                    "particle_size": (1.0, 1000.0),
                    "emission_fwhm": (5.0, 200.0),
                    "lifetime": (0.1, 1000.0),
                    "stability_score": (0.0, 1.0),
                    "phase_purity": (0.0, 1.0)
                }
            }
        }
        
        # Try to load custom schemas
        schema_path = self.data_dir / "schemas.json"
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    custom_schemas = json.load(f)
                    default_schemas.update(custom_schemas)
            except Exception as e:
                warnings.warn(f"Error loading custom schemas: {e}")
        
        return default_schemas
    
    def import_csv_data(self, csv_path: str, 
                       parameter_columns: Optional[List[str]] = None,
                       measurement_columns: Optional[List[str]] = None,
                       error_columns: Optional[List[str]] = None,
                       metadata_columns: Optional[List[str]] = None) -> int:
        """Import experimental data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Auto-detect columns if not specified
        if parameter_columns is None:
            parameter_columns = self._auto_detect_columns(df, "parameters")
        
        if measurement_columns is None:
            measurement_columns = self._auto_detect_columns(df, "measurements")
        
        imported_count = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract parameters
                parameters = {}
                for col in parameter_columns:
                    if col in df.columns and pd.notna(row[col]):
                        parameters[col] = float(row[col])
                
                # Extract measurements
                measurements = {}
                for col in measurement_columns:
                    if col in df.columns and pd.notna(row[col]):
                        measurements[col] = float(row[col])
                
                # Extract errors
                measurement_errors = {}
                if error_columns:
                    for col in error_columns:
                        if col in df.columns and pd.notna(row[col]):
                            error_key = col.replace('_error', '').replace('_uncertainty', '')
                            measurement_errors[error_key] = float(row[col])
                else:
                    # Default 5% error if not specified
                    for prop, value in measurements.items():
                        measurement_errors[prop] = abs(value) * 0.05
                
                # Extract metadata
                metadata = {}
                if metadata_columns:
                    for col in metadata_columns:
                        if col in df.columns and pd.notna(row[col]):
                            metadata[col] = row[col]
                
                # Create data point
                experiment_id = f"CSV_{Path(csv_path).stem}_{idx}"
                data_point = ExperimentalDataPoint(
                    experiment_id=experiment_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    parameters=parameters,
                    measurements=measurements,
                    measurement_errors=measurement_errors,
                    metadata=metadata
                )
                
                # Validate and add
                if self.validate_data_point(data_point):
                    self.experimental_data.append(data_point)
                    imported_count += 1
                
            except Exception as e:
                warnings.warn(f"Error processing row {idx}: {e}")
        
        return imported_count
    
    def _auto_detect_columns(self, df: pd.DataFrame, category: str) -> List[str]:
        """Auto-detect parameter or measurement columns"""
        schema = self.data_schemas.get(category, {})
        required = schema.get("required", [])
        optional = schema.get("optional", [])
        
        detected_columns = []
        
        # Check for exact matches first
        for col in required + optional:
            if col in df.columns:
                detected_columns.append(col)
        
        # Check for partial matches
        for df_col in df.columns:
            df_col_lower = df_col.lower()
            for target_col in required + optional:
                if target_col in df_col_lower or df_col_lower in target_col:
                    if df_col not in detected_columns:
                        detected_columns.append(df_col)
        
        return detected_columns
    
    def validate_data_point(self, data_point: ExperimentalDataPoint) -> bool:
        """Validate experimental data point"""
        # Check parameter ranges
        param_schema = self.data_schemas.get("synthesis_parameters", {})
        param_ranges = param_schema.get("ranges", {})
        
        for param, value in data_point.parameters.items():
            if param in param_ranges:
                min_val, max_val = param_ranges[param]
                if not (min_val <= value <= max_val):
                    warnings.warn(f"Parameter {param} value {value} outside valid range [{min_val}, {max_val}]")
                    return False
        
        # Check measurement ranges
        meas_schema = self.data_schemas.get("measurements", {})
        meas_ranges = meas_schema.get("ranges", {})
        
        for measurement, value in data_point.measurements.items():
            if measurement in meas_ranges:
                min_val, max_val = meas_ranges[measurement]
                if not (min_val <= value <= max_val):
                    warnings.warn(f"Measurement {measurement} value {value} outside valid range [{min_val}, {max_val}]")
                    return False
        
        return True
    
    def import_lab_notebook(self, notebook_path: str) -> int:
        """Import data from structured lab notebook format"""
        # This would parse structured lab notebook files
        # For now, implement a simple JSON format
        try:
            with open(notebook_path, 'r') as f:
                notebook_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error reading lab notebook: {e}")
        
        imported_count = 0
        
        if isinstance(notebook_data, list):
            experiments = notebook_data
        else:
            experiments = notebook_data.get('experiments', [])
        
        for exp_data in experiments:
            try:
                data_point = ExperimentalDataPoint(
                    experiment_id=exp_data.get('id', f"notebook_{imported_count}"),
                    timestamp=exp_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    parameters=exp_data.get('parameters', {}),
                    measurements=exp_data.get('measurements', {}),
                    measurement_errors=exp_data.get('errors', {}),
                    metadata=exp_data.get('metadata', {}),
                    quality_score=exp_data.get('quality_score', 1.0),
                    notes=exp_data.get('notes', '')
                )
                
                if self.validate_data_point(data_point):
                    self.experimental_data.append(data_point)
                    imported_count += 1
                    
            except Exception as e:
                warnings.warn(f"Error processing experiment {exp_data.get('id', 'unknown')}: {e}")
        
        return imported_count
    
    def get_combined_dataset(self, min_quality_score: float = 0.0) -> pd.DataFrame:
        """Get combined experimental dataset"""
        if not self.experimental_data:
            return pd.DataFrame()
        
        records = []
        
        for data_point in self.experimental_data:
            if data_point.quality_score >= min_quality_score:
                record = {
                    'experiment_id': data_point.experiment_id,
                    'timestamp': data_point.timestamp,
                    'quality_score': data_point.quality_score,
                    **data_point.parameters,
                    **data_point.measurements,
                    **{f"{k}_error": v for k, v in data_point.measurement_errors.items()}
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def export_data(self, output_path: str, format: str = "csv"):
        """Export experimental data"""
        df = self.get_combined_dataset()
        
        if format.lower() == "csv":
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            data_for_export = [
                {
                    'experiment_id': dp.experiment_id,
                    'timestamp': dp.timestamp,
                    'parameters': dp.parameters,
                    'measurements': dp.measurements,
                    'measurement_errors': dp.measurement_errors,
                    'metadata': dp.metadata,
                    'quality_score': dp.quality_score,
                    'notes': dp.notes
                }
                for dp in self.experimental_data
            ]
            
            with open(output_path, 'w') as f:
                json.dump(data_for_export, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class ModelCalibrator:
    """Calibrate simulation models using experimental data"""
    
    def __init__(self, simulation_function: Callable, 
                 experimental_data: pd.DataFrame):
        """
        Initialize model calibrator
        
        Args:
            simulation_function: Function that runs simulation with given parameters
            experimental_data: DataFrame with experimental data
        """
        self.simulation_function = simulation_function
        self.experimental_data = experimental_data
        self.calibration_results = {}
        
    def calibrate_bias_correction(self, target_properties: List[str],
                                parameter_columns: List[str]) -> CalibrationResult:
        """Calibrate using bias correction approach"""
        print("🎯 Performing bias correction calibration...")
        
        # Run simulations for experimental conditions
        sim_results = []
        exp_results = []
        
        for idx, row in self.experimental_data.iterrows():
            # Extract parameters
            params = {col: row[col] for col in parameter_columns if col in row}
            
            try:
                # Run simulation
                sim_result = self.simulation_function(params)
                
                # Extract experimental measurements
                exp_measurements = {prop: row[prop] for prop in target_properties 
                                  if prop in row and pd.notna(row[prop])}
                
                if exp_measurements and sim_result:
                    sim_results.append(sim_result)
                    exp_results.append(exp_measurements)
                    
            except Exception as e:
                warnings.warn(f"Error in simulation for row {idx}: {e}")
        
        if not sim_results:
            raise ValueError("No valid simulation results obtained")
        
        # Calculate bias corrections
        bias_corrections = {}
        calibration_metrics = {}
        
        for prop in target_properties:
            sim_values = [result.get(prop, np.nan) for result in sim_results]
            exp_values = [result.get(prop, np.nan) for result in exp_results]
            
            # Filter out NaN values
            valid_pairs = [(s, e) for s, e in zip(sim_values, exp_values) 
                          if not (np.isnan(s) or np.isnan(e))]
            
            if len(valid_pairs) > 2:
                sim_vals, exp_vals = zip(*valid_pairs)
                sim_vals = np.array(sim_vals)
                exp_vals = np.array(exp_vals)
                
                # Simple bias correction: additive offset
                bias = np.mean(exp_vals - sim_vals)
                bias_corrections[prop] = bias
                
                # Calculate metrics
                corrected_sim = sim_vals + bias
                calibration_metrics[f"{prop}_bias"] = bias
                calibration_metrics[f"{prop}_rmse_before"] = np.sqrt(mean_squared_error(exp_vals, sim_vals))
                calibration_metrics[f"{prop}_rmse_after"] = np.sqrt(mean_squared_error(exp_vals, corrected_sim))
                calibration_metrics[f"{prop}_r2_before"] = r2_score(exp_vals, sim_vals)
                calibration_metrics[f"{prop}_r2_after"] = r2_score(exp_vals, corrected_sim)
        
        # Create calibration result
        result = CalibrationResult(
            calibration_method="bias_correction",
            calibrated_parameters=bias_corrections,
            calibration_metrics=calibration_metrics,
            validation_metrics={},
            residual_analysis={},
            calibration_data=self.experimental_data.copy()
        )
        
        self.calibration_results["bias_correction"] = result
        return result
    
    def calibrate_parameter_adjustment(self, target_properties: List[str],
                                     parameter_columns: List[str],
                                     adjustable_params: Optional[List[str]] = None) -> CalibrationResult:
        """Calibrate by adjusting model parameters"""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for parameter optimization")
        
        print("⚙️ Performing parameter adjustment calibration...")
        
        if adjustable_params is None:
            adjustable_params = ['temperature_scale', 'concentration_scale', 'time_scale']
        
        # Define objective function
        def objective_function(adjustment_factors):
            total_error = 0
            n_comparisons = 0
            
            for idx, row in self.experimental_data.iterrows():
                # Extract parameters and apply adjustments
                params = {col: row[col] for col in parameter_columns if col in row}
                
                # Apply adjustment factors (this would be model-specific)
                adjusted_params = params.copy()
                if 'temperature_scale' in adjustable_params and 'temperature' in params:
                    adjusted_params['temperature'] = params['temperature'] * adjustment_factors[0]
                
                try:
                    # Run simulation with adjusted parameters
                    sim_result = self.simulation_function(adjusted_params)
                    
                    # Calculate error for each target property
                    for prop in target_properties:
                        if prop in row and prop in sim_result and pd.notna(row[prop]):
                            exp_val = row[prop]
                            sim_val = sim_result[prop]
                            error = ((sim_val - exp_val) / exp_val) ** 2  # Relative error
                            total_error += error
                            n_comparisons += 1
                            
                except Exception:
                    continue
            
            return total_error / max(n_comparisons, 1)
        
        # Initial guess and bounds
        initial_guess = [1.0] * len(adjustable_params)
        bounds = [(0.5, 2.0)] * len(adjustable_params)
        
        # Optimize
        try:
            result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
            optimal_factors = result.x
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            optimal_factors = initial_guess
        
        # Calculate calibration metrics
        calibration_metrics = {}
        calibrated_parameters = {}
        
        for i, param in enumerate(adjustable_params):
            calibrated_parameters[param] = optimal_factors[i]
            calibration_metrics[f"{param}_factor"] = optimal_factors[i]
        
        # Validation metrics
        validation_metrics = {"final_objective": objective_function(optimal_factors)}
        
        result = CalibrationResult(
            calibration_method="parameter_adjustment",
            calibrated_parameters=calibrated_parameters,
            calibration_metrics=calibration_metrics,
            validation_metrics=validation_metrics,
            residual_analysis={},
            calibration_data=self.experimental_data.copy()
        )
        
        self.calibration_results["parameter_adjustment"] = result
        return result
    
    def calibrate_machine_learning(self, target_properties: List[str],
                                 parameter_columns: List[str],
                                 method: str = "gaussian_process") -> CalibrationResult:
        """Calibrate using machine learning surrogate models"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ML calibration")
        
        print(f"🤖 Performing {method} calibration...")
        
        # Prepare training data
        X_train = []
        y_train = {prop: [] for prop in target_properties}
        
        for idx, row in self.experimental_data.iterrows():
            params = [row[col] for col in parameter_columns if col in row and pd.notna(row[col])]
            
            if len(params) == len(parameter_columns):
                X_train.append(params)
                
                for prop in target_properties:
                    if prop in row and pd.notna(row[prop]):
                        y_train[prop].append(row[prop])
                    else:
                        y_train[prop].append(np.nan)
        
        X_train = np.array(X_train)
        
        # Train models for each property
        trained_models = {}
        calibration_metrics = {}
        
        for prop in target_properties:
            y_prop = np.array(y_train[prop])
            
            # Filter out NaN values
            valid_mask = ~np.isnan(y_prop)
            if valid_mask.sum() < 3:
                continue
            
            X_valid = X_train[valid_mask]
            y_valid = y_prop[valid_mask]
            
            # Split data
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42
            )
            
            # Choose model
            if method == "gaussian_process":
                model = GaussianProcessRegressor(
                    kernel=RBF() + Matern(),
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=42
                )
            elif method == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Train model
            model.fit(X_train_split, y_train_split)
            
            # Evaluate
            y_pred_train = model.predict(X_train_split)
            y_pred_test = model.predict(X_test_split)
            
            # Store metrics
            calibration_metrics[f"{prop}_train_r2"] = r2_score(y_train_split, y_pred_train)
            calibration_metrics[f"{prop}_test_r2"] = r2_score(y_test_split, y_pred_test)
            calibration_metrics[f"{prop}_train_rmse"] = np.sqrt(mean_squared_error(y_train_split, y_pred_train))
            calibration_metrics[f"{prop}_test_rmse"] = np.sqrt(mean_squared_error(y_test_split, y_pred_test))
            
            trained_models[prop] = model
        
        result = CalibrationResult(
            calibration_method=f"ml_{method}",
            calibrated_parameters={"trained_models": trained_models},
            calibration_metrics=calibration_metrics,
            validation_metrics={},
            residual_analysis={},
            calibration_data=self.experimental_data.copy()
        )
        
        self.calibration_results[f"ml_{method}"] = result
        return result
    
    def generate_calibration_report(self, output_path: str = "calibration_report.html"):
        """Generate comprehensive calibration report"""
        if not self.calibration_results:
            print("No calibration results available")
            return
        
        html_content = self._create_calibration_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Calibration report saved to: {output_path}")
    
    def _create_calibration_html_report(self) -> str:
        """Create HTML calibration report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Calibration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .method-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ margin: 5px 0; }}
                .good {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .poor {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🎯 Model Calibration Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Dataset: {len(self.experimental_data)} experimental data points</p>
        """
        
        # Add results for each calibration method
        for method, result in self.calibration_results.items():
            html += f"""
            <div class="method-section">
                <h2>Method: {result.calibration_method}</h2>
                <h3>Calibration Metrics:</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Quality</th></tr>
            """
            
            for metric, value in result.calibration_metrics.items():
                if 'r2' in metric.lower():
                    quality_class = 'good' if value > 0.8 else 'warning' if value > 0.6 else 'poor'
                elif 'rmse' in metric.lower():
                    quality_class = 'good' if value < 0.1 else 'warning' if value < 0.2 else 'poor'
                else:
                    quality_class = ''
                
                html += f"<tr><td>{metric}</td><td>{value:.4f}</td><td class='{quality_class}'>{'✓' if quality_class == 'good' else '⚠' if quality_class == 'warning' else '✗' if quality_class == 'poor' else '-'}</td></tr>"
            
            html += """
                </table>
            </div>
            """
        
        html += """
            </body>
        </html>
        """
        
        return html


class DataFusion:
    """Fuse experimental and simulation data for improved predictions"""
    
    def __init__(self):
        """Initialize data fusion system"""
        self.fusion_weights = {}
        self.uncertainty_models = {}
        
    def weight_based_fusion(self, simulation_results: Dict[str, float],
                          experimental_data: Dict[str, float],
                          simulation_uncertainty: Dict[str, float],
                          experimental_uncertainty: Dict[str, float]) -> Dict[str, float]:
        """Perform uncertainty-weighted data fusion"""
        fused_results = {}
        
        for property_name in simulation_results.keys():
            if property_name in experimental_data:
                sim_val = simulation_results[property_name]
                exp_val = experimental_data[property_name]
                
                sim_unc = simulation_uncertainty.get(property_name, 0.1)
                exp_unc = experimental_uncertainty.get(property_name, 0.05)
                
                # Inverse variance weighting
                sim_weight = 1.0 / (sim_unc ** 2) if sim_unc > 0 else 1.0
                exp_weight = 1.0 / (exp_unc ** 2) if exp_unc > 0 else 1.0
                
                total_weight = sim_weight + exp_weight
                
                fused_value = (sim_val * sim_weight + exp_val * exp_weight) / total_weight
                fused_uncertainty = 1.0 / np.sqrt(total_weight)
                
                fused_results[property_name] = fused_value
                fused_results[f"{property_name}_uncertainty"] = fused_uncertainty
                fused_results[f"{property_name}_sim_weight"] = sim_weight / total_weight
                fused_results[f"{property_name}_exp_weight"] = exp_weight / total_weight
            else:
                # Use simulation result if no experimental data
                fused_results[property_name] = simulation_results[property_name]
                fused_results[f"{property_name}_uncertainty"] = simulation_uncertainty.get(property_name, 0.1)
        
        return fused_results
    
    def adaptive_fusion(self, simulation_history: List[Dict[str, float]],
                       experimental_history: List[Dict[str, float]],
                       validation_data: Dict[str, float]) -> Dict[str, float]:
        """Adaptive fusion that learns optimal weights"""
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not available. Using simple averaging.")
            return self._simple_average_fusion(simulation_history[-1], experimental_history[-1])
        
        # This would implement a more sophisticated adaptive fusion algorithm
        # For now, return a placeholder
        return self._simple_average_fusion(simulation_history[-1], experimental_history[-1])
    
    def _simple_average_fusion(self, sim_results: Dict[str, float], 
                              exp_results: Dict[str, float]) -> Dict[str, float]:
        """Simple average fusion as fallback"""
        fused = {}
        
        for prop in sim_results.keys():
            if prop in exp_results:
                fused[prop] = (sim_results[prop] + exp_results[prop]) / 2
            else:
                fused[prop] = sim_results[prop]
        
        return fused


def create_experimental_template(output_path: str = "experimental_template.json"):
    """Create template for experimental data entry"""
    template = {
        "experiments": [
            {
                "id": "EXP_001",
                "timestamp": "2024-01-01T10:00:00Z",
                "parameters": {
                    "cs_br_concentration": 1.0,
                    "pb_br2_concentration": 1.0,
                    "temperature": 150.0,
                    "oa_concentration": 0.1,
                    "oam_concentration": 0.1,
                    "reaction_time": 30.0,
                    "solvent_type": 0
                },
                "measurements": {
                    "bandgap": 2.3,
                    "plqy": 0.85,
                    "emission_peak": 520.0,
                    "particle_size": 12.0,
                    "emission_fwhm": 25.0,
                    "lifetime": 15.0,
                    "stability_score": 0.8,
                    "phase_purity": 0.95
                },
                "errors": {
                    "bandgap": 0.05,
                    "plqy": 0.05,
                    "emission_peak": 2.0,
                    "particle_size": 1.0,
                    "emission_fwhm": 2.0,
                    "lifetime": 1.0,
                    "stability_score": 0.05,
                    "phase_purity": 0.02
                },
                "metadata": {
                    "researcher": "Dr. Smith",
                    "lab": "Materials Lab A",
                    "equipment": "XRD-123, PL-456",
                    "notes": "Standard synthesis protocol"
                },
                "quality_score": 1.0,
                "notes": "High quality sample with good reproducibility"
            }
        ],
        "schema_info": {
            "version": "1.0",
            "description": "Template for CsPbBr3 experimental data",
            "required_parameters": ["cs_br_concentration", "pb_br2_concentration", "temperature"],
            "optional_parameters": ["oa_concentration", "oam_concentration", "reaction_time", "solvent_type"],
            "measurements": ["bandgap", "plqy", "emission_peak", "particle_size", "emission_fwhm", "lifetime", "stability_score", "phase_purity"]
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"📝 Experimental template created: {output_path}")


if __name__ == "__main__":
    # Demonstration of experimental data integration
    print("🧪 Experimental Data Integration Demo")
    
    # Create experimental data manager
    data_manager = ExperimentalDataManager()
    
    # Create template
    create_experimental_template("demo_template.json")
    
    # Import demo data
    print("📥 Importing demo experimental data...")
    imported_count = data_manager.import_lab_notebook("demo_template.json")
    print(f"   Imported {imported_count} experiments")
    
    # Create demo simulation function
    def demo_simulation(params):
        """Demo simulation function"""
        temp_effect = (params.get('temperature', 150) - 150) / 100
        conc_effect = params.get('cs_br_concentration', 1) / params.get('pb_br2_concentration', 1)
        
        return {
            'bandgap': 2.3 + temp_effect * 0.1 + np.random.normal(0, 0.05),
            'plqy': 0.8 * np.exp(-abs(conc_effect - 1.0)) + np.random.normal(0, 0.02),
            'particle_size': 10 + temp_effect * 3 + np.random.normal(0, 0.5),
            'stability_score': 0.7 + temp_effect * 0.05 + np.random.normal(0, 0.02)
        }
    
    # Get combined dataset
    experimental_df = data_manager.get_combined_dataset()
    print(f"📊 Combined dataset: {len(experimental_df)} data points")
    
    if len(experimental_df) > 0:
        # Calibrate model
        print("⚙️ Calibrating model...")
        parameter_columns = ['cs_br_concentration', 'pb_br2_concentration', 'temperature']
        target_properties = ['bandgap', 'plqy', 'particle_size', 'stability_score']
        
        calibrator = ModelCalibrator(demo_simulation, experimental_df)
        
        # Try bias correction
        try:
            bias_result = calibrator.calibrate_bias_correction(target_properties, parameter_columns)
            print(f"   Bias correction completed: {len(bias_result.calibrated_parameters)} corrections")
        except Exception as e:
            print(f"   Bias correction failed: {e}")
        
        # Generate report
        calibrator.generate_calibration_report("demo_calibration_report.html")
    
    # Demo data fusion
    print("🔗 Demonstrating data fusion...")
    fusion = DataFusion()
    
    sim_results = {'bandgap': 2.25, 'plqy': 0.82}
    exp_results = {'bandgap': 2.30, 'plqy': 0.85}
    sim_uncertainty = {'bandgap': 0.05, 'plqy': 0.03}
    exp_uncertainty = {'bandgap': 0.02, 'plqy': 0.05}
    
    fused_results = fusion.weight_based_fusion(sim_results, exp_results, sim_uncertainty, exp_uncertainty)
    
    print(f"   Simulation: {sim_results}")
    print(f"   Experimental: {exp_results}")
    print(f"   Fused: {fused_results}")
    
    # Export data
    print("💾 Exporting processed data...")
    data_manager.export_data("processed_experimental_data.csv", "csv")
    data_manager.export_data("processed_experimental_data.json", "json")
    
    print("✅ Experimental integration demo complete!")
    print("📁 Generated files:")
    print("   - demo_template.json")
    print("   - demo_calibration_report.html") 
    print("   - processed_experimental_data.csv")
    print("   - processed_experimental_data.json")