#!/usr/bin/env python3
"""
Advanced Physics Models for CsPbBr3 Synthesis
Implements detailed thermodynamic, kinetic, and crystallographic models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import warnings
from abc import ABC, abstractmethod
import math

try:
    from scipy import constants
    from scipy.integrate import odeint, solve_ivp
    from scipy.optimize import fsolve, root_scalar
    from scipy.special import erf
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced physics calculations will be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Physical constants
R_GAS = 8.314  # J/(mol·K)
k_B = constants.Boltzmann  # J/K
N_A = constants.Avogadro  # 1/mol
h_PLANCK = constants.Planck  # J·s


@dataclass
class ThermodynamicParameters:
    """Container for thermodynamic parameters"""
    formation_enthalpy: float  # J/mol
    formation_entropy: float   # J/(mol·K)
    heat_capacity: float      # J/(mol·K)
    molar_volume: float       # m³/mol
    surface_energy: float     # J/m²
    solubility_product: float # mol²/L²
    activity_coefficients: Dict[str, float] = field(default_factory=dict)


@dataclass
class KineticParameters:
    """Container for kinetic parameters"""
    nucleation_rate_constant: float    # 1/(s·m³)
    growth_rate_constant: float        # m/s
    nucleation_activation_energy: float # J/mol
    growth_activation_energy: float     # J/mol
    diffusion_coefficient: float        # m²/s
    interfacial_energy: float          # J/m²


@dataclass
class CrystalStructure:
    """Container for crystal structure information"""
    lattice_parameters: Dict[str, float]  # a, b, c in Å
    space_group: str
    density: float  # g/cm³
    coordination_numbers: Dict[str, int]
    bond_lengths: Dict[str, float]  # Å
    bond_angles: Dict[str, float]   # degrees


class AdvancedPhysicsModel(ABC):
    """Abstract base class for advanced physics models"""
    
    @abstractmethod
    def calculate(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate model outputs for given conditions"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters"""
        pass


class ThermodynamicModel(AdvancedPhysicsModel):
    """Advanced thermodynamic model for phase stability and solubility"""
    
    def __init__(self):
        """Initialize thermodynamic model with literature parameters"""
        # Thermodynamic data for CsPbBr3 phases (literature values and estimates)
        self.phase_data = {
            'CsPbBr3_3D': ThermodynamicParameters(
                formation_enthalpy=-800000,  # J/mol (estimated)
                formation_entropy=-150,      # J/(mol·K)
                heat_capacity=120,           # J/(mol·K)
                molar_volume=0.000162,       # m³/mol
                surface_energy=0.1,          # J/m²
                solubility_product=1e-12     # mol²/L²
            ),
            'Cs4PbBr6_0D': ThermodynamicParameters(
                formation_enthalpy=-850000,  # J/mol
                formation_entropy=-180,      # J/(mol·K)
                heat_capacity=140,           # J/(mol·K)
                molar_volume=0.000180,       # m³/mol
                surface_energy=0.08,         # J/m²
                solubility_product=1e-14     # mol²/L²
            ),
            'CsPb2Br5_2D': ThermodynamicParameters(
                formation_enthalpy=-780000,  # J/mol
                formation_entropy=-140,      # J/(mol·K)
                heat_capacity=110,           # J/(mol·K)
                molar_volume=0.000155,       # m³/mol
                surface_energy=0.12,         # J/m²
                solubility_product=5e-13     # mol²/L²
            )
        }
        
        # Solvent parameters
        self.solvent_data = {
            'DMSO': {'dielectric': 47.2, 'viscosity': 1.996e-3, 'density': 1100},
            'DMF': {'dielectric': 36.7, 'viscosity': 0.92e-3, 'density': 944},
            'Water': {'dielectric': 80.1, 'viscosity': 1.0e-3, 'density': 1000},
            'Toluene': {'dielectric': 2.38, 'viscosity': 0.59e-3, 'density': 867},
            'Octadecene': {'dielectric': 2.1, 'viscosity': 3.5e-3, 'density': 789}
        }
    
    def calculate_gibbs_energy(self, phase: str, temperature: float, 
                              pressure: float = 101325) -> float:
        """Calculate Gibbs free energy for a phase"""
        if phase not in self.phase_data:
            raise ValueError(f"Unknown phase: {phase}")
        
        params = self.phase_data[phase]
        
        # G = H - TS + ∫Cp dT - T∫(Cp/T) dT (simplified)
        T_ref = 298.15  # K
        
        # Enthalpy contribution
        H = params.formation_enthalpy + params.heat_capacity * (temperature - T_ref)
        
        # Entropy contribution (simplified)
        S = params.formation_entropy + params.heat_capacity * np.log(temperature / T_ref)
        
        # Gibbs energy
        G = H - temperature * S
        
        return G
    
    def calculate_phase_stability(self, temperature: float, 
                                cs_activity: float, pb_activity: float,
                                br_activity: float) -> Dict[str, float]:
        """Calculate relative phase stability based on chemical potentials"""
        phase_stabilities = {}
        
        for phase_name in self.phase_data.keys():
            G_formation = self.calculate_gibbs_energy(phase_name, temperature)
            
            # Chemical potential contribution (simplified)
            if '3D' in phase_name:
                # CsPbBr3: μ_Cs + μ_Pb + 3μ_Br
                mu_contribution = (np.log(cs_activity) + np.log(pb_activity) + 
                                 3 * np.log(br_activity)) * R_GAS * temperature
            elif '0D' in phase_name:
                # Cs4PbBr6: 4μ_Cs + μ_Pb + 6μ_Br (simplified)
                mu_contribution = (4 * np.log(cs_activity) + np.log(pb_activity) + 
                                 6 * np.log(br_activity)) * R_GAS * temperature
            elif '2D' in phase_name:
                # CsPb2Br5: μ_Cs + 2μ_Pb + 5μ_Br
                mu_contribution = (np.log(cs_activity) + 2 * np.log(pb_activity) + 
                                 5 * np.log(br_activity)) * R_GAS * temperature
            else:
                mu_contribution = 0
            
            total_energy = G_formation + mu_contribution
            phase_stabilities[phase_name] = total_energy
        
        return phase_stabilities
    
    def calculate_solubility(self, phase: str, temperature: float, 
                           solvent: str = 'DMSO') -> Dict[str, float]:
        """Calculate solubility of phase in given solvent"""
        if phase not in self.phase_data:
            raise ValueError(f"Unknown phase: {phase}")
        
        if solvent not in self.solvent_data:
            raise ValueError(f"Unknown solvent: {solvent}")
        
        params = self.phase_data[phase]
        solvent_props = self.solvent_data[solvent]
        
        # Van't Hoff equation for temperature dependence
        # ln(S) = -ΔH_sol/(RT) + ΔS_sol/R
        
        # Estimate dissolution enthalpy (rough approximation)
        delta_H_sol = -params.formation_enthalpy * 0.1  # Rough estimate
        delta_S_sol = 50  # J/(mol·K) typical dissolution entropy
        
        # Base solubility at reference temperature
        T_ref = 298.15
        ln_S_ref = delta_S_sol / R_GAS
        
        # Temperature correction
        ln_S = ln_S_ref - delta_H_sol / (R_GAS * temperature)
        solubility_mol_L = np.exp(ln_S) * params.solubility_product
        
        # Solvent effect (dielectric constant correction)
        dielectric_factor = solvent_props['dielectric'] / 47.2  # Normalized to DMSO
        solubility_mol_L *= dielectric_factor
        
        return {
            'solubility_mol_L': solubility_mol_L,
            'solubility_g_L': solubility_mol_L * self._get_molar_mass(phase),
            'dissolution_enthalpy': delta_H_sol,
            'dissolution_entropy': delta_S_sol
        }
    
    def _get_molar_mass(self, phase: str) -> float:
        """Get molar mass of phase (g/mol)"""
        molar_masses = {
            'CsPbBr3_3D': 572.8,   # g/mol
            'Cs4PbBr6_0D': 1289.1, # g/mol  
            'CsPb2Br5_2D': 1053.4  # g/mol
        }
        return molar_masses.get(phase, 500.0)
    
    def calculate(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate thermodynamic properties"""
        temperature = conditions.get('temperature', 150) + 273.15  # Convert to K
        cs_conc = conditions.get('cs_br_concentration', 1.0)
        pb_conc = conditions.get('pb_br2_concentration', 1.0)
        
        # Estimate activities (simplified)
        cs_activity = cs_conc * 0.8  # Activity coefficient approximation
        pb_activity = pb_conc * 0.7
        br_activity = (cs_conc + 2 * pb_conc) * 0.9  # From dissociation
        
        # Calculate phase stabilities
        stabilities = self.calculate_phase_stability(temperature, cs_activity, 
                                                   pb_activity, br_activity)
        
        # Find most stable phase
        most_stable_phase = min(stabilities.keys(), key=lambda k: stabilities[k])
        
        # Calculate solubilities
        solubilities = {}
        for phase in self.phase_data.keys():
            sol_data = self.calculate_solubility(phase, temperature)
            solubilities[f"{phase}_solubility"] = sol_data['solubility_mol_L']
        
        results = {
            'most_stable_phase': most_stable_phase,
            'phase_stability_3D': stabilities.get('CsPbBr3_3D', 0),
            'phase_stability_0D': stabilities.get('Cs4PbBr6_0D', 0),
            'phase_stability_2D': stabilities.get('CsPb2Br5_2D', 0),
            **solubilities,
            'temperature_K': temperature
        }
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get thermodynamic parameters"""
        return {
            'phase_data': self.phase_data,
            'solvent_data': self.solvent_data
        }


class NucleationGrowthModel(AdvancedPhysicsModel):
    """Advanced nucleation and growth kinetics model"""
    
    def __init__(self):
        """Initialize nucleation-growth model"""
        self.kinetic_params = {
            'CsPbBr3_3D': KineticParameters(
                nucleation_rate_constant=1e15,     # 1/(s·m³)
                growth_rate_constant=1e-8,         # m/s
                nucleation_activation_energy=80000, # J/mol
                growth_activation_energy=40000,     # J/mol
                diffusion_coefficient=1e-10,        # m²/s at 298K
                interfacial_energy=0.1              # J/m²
            ),
            'Cs4PbBr6_0D': KineticParameters(
                nucleation_rate_constant=5e14,
                growth_rate_constant=5e-9,
                nucleation_activation_energy=60000,
                growth_activation_energy=30000,
                diffusion_coefficient=1.5e-10,
                interfacial_energy=0.08
            ),
            'CsPb2Br5_2D': KineticParameters(
                nucleation_rate_constant=2e14,
                growth_rate_constant=2e-8,
                nucleation_activation_energy=70000,
                growth_activation_energy=35000,
                diffusion_coefficient=8e-11,
                interfacial_energy=0.12
            )
        }
    
    def calculate_nucleation_rate(self, phase: str, supersaturation: float, 
                                temperature: float) -> float:
        """Calculate nucleation rate using classical nucleation theory"""
        if phase not in self.kinetic_params:
            return 0.0
        
        params = self.kinetic_params[phase]
        T_K = temperature + 273.15
        
        if supersaturation <= 1.0:
            return 0.0
        
        # Classical nucleation theory
        # J = A * exp(-ΔG*/kT)
        
        # Pre-exponential factor
        A = params.nucleation_rate_constant * np.exp(-params.nucleation_activation_energy / (R_GAS * T_K))
        
        # Critical nucleus free energy barrier
        # ΔG* = 16πσ³Vm²/(3(kT ln S)²) where S is supersaturation
        
        # Estimate molecular volume (very rough)
        V_mol = 1e-28  # m³/molecule (order of magnitude estimate)
        sigma = params.interfacial_energy
        
        ln_S = np.log(supersaturation)
        
        if ln_S > 0:
            delta_G_star = (16 * np.pi * sigma**3 * V_mol**2) / (3 * (k_B * T_K * ln_S)**2)
            
            # Nucleation rate
            J = A * np.exp(-delta_G_star / (k_B * T_K))
        else:
            J = 0.0
        
        return max(0.0, J)
    
    def calculate_growth_rate(self, phase: str, supersaturation: float,
                            temperature: float, particle_size: float = 1e-8) -> float:
        """Calculate crystal growth rate"""
        if phase not in self.kinetic_params:
            return 0.0
        
        params = self.kinetic_params[phase]
        T_K = temperature + 273.15
        
        if supersaturation <= 1.0:
            return 0.0
        
        # Growth rate: v = k * (S - 1) * exp(-Ea/RT)
        # with size-dependent corrections
        
        # Base growth rate
        k_growth = params.growth_rate_constant * np.exp(-params.growth_activation_energy / (R_GAS * T_K))
        
        # Supersaturation driving force
        driving_force = supersaturation - 1.0
        
        # Gibbs-Thomson effect (size dependence)
        # S_r = S_∞ * exp(2σVm/(rkT))
        V_mol = 1e-28  # m³/molecule
        radius = particle_size / 2
        
        if radius > 0:
            gibbs_thomson = np.exp(2 * params.interfacial_energy * V_mol / (radius * k_B * T_K))
            effective_supersaturation = supersaturation / gibbs_thomson
            driving_force = max(0.0, effective_supersaturation - 1.0)
        
        # Growth rate
        growth_rate = k_growth * driving_force
        
        return max(0.0, growth_rate)
    
    def simulate_particle_size_evolution(self, conditions: Dict[str, float],
                                       time_points: np.ndarray,
                                       initial_size: float = 1e-9) -> Dict[str, np.ndarray]:
        """Simulate particle size evolution over time"""
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available. Cannot perform time evolution simulation.")
            return {'time': time_points, 'size': np.full_like(time_points, initial_size)}
        
        temperature = conditions.get('temperature', 150)
        supersaturation = conditions.get('supersaturation', 2.0)
        phase = conditions.get('phase', 'CsPbBr3_3D')
        
        def size_derivative(t, y):
            """Derivative for particle size evolution"""
            size = y[0]
            growth_rate = self.calculate_growth_rate(phase, supersaturation, temperature, size)
            return [growth_rate]
        
        # Solve ODE
        sol = solve_ivp(size_derivative, [time_points[0], time_points[-1]], 
                       [initial_size], t_eval=time_points, method='RK45')
        
        return {
            'time': sol.t,
            'size': sol.y[0],
            'growth_rate': [self.calculate_growth_rate(phase, supersaturation, temperature, s) 
                           for s in sol.y[0]]
        }
    
    def calculate_final_particle_size(self, conditions: Dict[str, float]) -> float:
        """Calculate steady-state particle size"""
        temperature = conditions.get('temperature', 150)
        supersaturation = conditions.get('supersaturation', 2.0)
        reaction_time = conditions.get('reaction_time', 30) * 60  # Convert to seconds
        phase = conditions.get('phase', 'CsPbBr3_3D')
        
        # Simplified analytical solution for constant growth rate
        avg_growth_rate = self.calculate_growth_rate(phase, supersaturation, temperature)
        
        # Estimate final size considering nucleation and growth competition
        nucleation_rate = self.calculate_nucleation_rate(phase, supersaturation, temperature)
        
        if nucleation_rate > 0:
            # More nucleation → smaller final particles
            nucleation_factor = 1.0 / (1.0 + nucleation_rate * 1e-15)  # Empirical scaling
        else:
            nucleation_factor = 1.0
        
        final_size = avg_growth_rate * reaction_time * nucleation_factor
        
        # Add initial nucleus size
        critical_radius = 1e-9  # 1 nm
        final_size += critical_radius
        
        return max(critical_radius, final_size)
    
    def calculate(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate nucleation and growth kinetics"""
        temperature = conditions.get('temperature', 150)
        supersaturation = conditions.get('supersaturation', 2.0)
        
        results = {}
        
        # Calculate for each phase
        for phase in self.kinetic_params.keys():
            nucleation_rate = self.calculate_nucleation_rate(phase, supersaturation, temperature)
            growth_rate = self.calculate_growth_rate(phase, supersaturation, temperature)
            
            # Use simplified phase name for results
            phase_short = phase.split('_')[1]  # Extract '3D', '0D', '2D'
            
            results[f'nucleation_rate_{phase_short}'] = nucleation_rate
            results[f'growth_rate_{phase_short}'] = growth_rate
        
        # Calculate predicted final particle size
        phase = conditions.get('phase', 'CsPbBr3_3D')
        final_size = self.calculate_final_particle_size(conditions)
        results['predicted_particle_size'] = final_size * 1e9  # Convert to nm
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get kinetic parameters"""
        return {'kinetic_params': self.kinetic_params}


class CrystalStructureModel(AdvancedPhysicsModel):
    """Crystal structure and lattice dynamics model"""
    
    def __init__(self):
        """Initialize crystal structure model"""
        self.structures = {
            'CsPbBr3_3D': CrystalStructure(
                lattice_parameters={'a': 5.874, 'b': 5.874, 'c': 5.874},  # Å, cubic
                space_group='Pm-3m',
                density=4.84,  # g/cm³
                coordination_numbers={'Cs': 12, 'Pb': 6, 'Br': 2},
                bond_lengths={'Pb-Br': 2.937, 'Cs-Br': 3.71},  # Å
                bond_angles={'Br-Pb-Br': 90.0, 'Pb-Br-Pb': 180.0}  # degrees
            ),
            'Cs4PbBr6_0D': CrystalStructure(
                lattice_parameters={'a': 13.84, 'b': 13.84, 'c': 13.84},  # Å, cubic
                space_group='Fm-3m',
                density=4.56,  # g/cm³
                coordination_numbers={'Cs': 8, 'Pb': 6, 'Br': 1},
                bond_lengths={'Pb-Br': 2.94, 'Cs-Br': 3.85},  # Å
                bond_angles={'Br-Pb-Br': 90.0}  # degrees
            ),
            'CsPb2Br5_2D': CrystalStructure(
                lattice_parameters={'a': 8.31, 'b': 8.31, 'c': 14.87},  # Å, tetragonal
                space_group='I4/mcm',
                density=5.12,  # g/cm³
                coordination_numbers={'Cs': 9, 'Pb': 6, 'Br': 3},
                bond_lengths={'Pb-Br': 2.92, 'Cs-Br': 3.65},  # Å
                bond_angles={'Br-Pb-Br': 90.0, 'Pb-Br-Pb': 145.0}  # degrees
            )
        }
    
    def calculate_lattice_strain(self, phase: str, temperature: float, 
                               pressure: float = 1.0) -> Dict[str, float]:
        """Calculate lattice strain due to temperature and pressure"""
        if phase not in self.structures:
            return {}
        
        structure = self.structures[phase]
        
        # Thermal expansion (linear approximation)
        # α ≈ 3-5 × 10^-5 K^-1 for perovskites
        thermal_expansion_coeff = 4e-5  # K^-1
        T_ref = 298.15  # K
        T_K = temperature + 273.15
        
        thermal_strain = thermal_expansion_coeff * (T_K - T_ref)
        
        # Pressure effect (bulk modulus approximation)
        # K ≈ 20-30 GPa for halide perovskites
        bulk_modulus = 25e9  # Pa
        pressure_Pa = pressure * 101325  # Convert atm to Pa
        
        volumetric_strain = -pressure_Pa / bulk_modulus
        linear_strain = volumetric_strain / 3
        
        total_strain = thermal_strain + linear_strain
        
        # Update lattice parameters
        strained_lattice = {}
        for param, value in structure.lattice_parameters.items():
            strained_lattice[param] = value * (1 + total_strain)
        
        return {
            'thermal_strain': thermal_strain,
            'pressure_strain': linear_strain,
            'total_strain': total_strain,
            'strained_lattice_a': strained_lattice.get('a', 0),
            'strained_lattice_b': strained_lattice.get('b', 0),
            'strained_lattice_c': strained_lattice.get('c', 0)
        }
    
    def calculate_bandgap_from_structure(self, phase: str, temperature: float) -> float:
        """Calculate bandgap from crystal structure using empirical relationships"""
        if phase not in self.structures:
            return 0.0
        
        structure = self.structures[phase]
        
        # Empirical relationship: Eg ∝ 1/d^n where d is bond length
        pb_br_bond = structure.bond_lengths.get('Pb-Br', 2.9)
        
        # Base bandgaps (experimental values)
        base_bandgaps = {
            'CsPbBr3_3D': 2.3,
            'Cs4PbBr6_0D': 3.95,
            'CsPb2Br5_2D': 2.9
        }
        
        base_eg = base_bandgaps.get(phase, 2.3)
        
        # Temperature dependence (Varshni equation approximation)
        # Eg(T) = Eg(0) - αT²/(T + β)
        alpha = 3e-4  # eV/K
        beta = 300    # K
        T_K = temperature + 273.15
        
        bandgap = base_eg - alpha * T_K**2 / (T_K + beta)
        
        # Bond length correction
        ref_bond_length = 2.937  # Å (CsPbBr3 reference)
        bond_correction = (ref_bond_length / pb_br_bond)**2
        
        return bandgap * bond_correction
    
    def calculate_phonon_properties(self, phase: str, temperature: float) -> Dict[str, float]:
        """Calculate phonon-related properties"""
        if phase not in self.structures:
            return {}
        
        structure = self.structures[phase]
        T_K = temperature + 273.15
        
        # Estimate Debye temperature (rough approximation)
        # θ_D ≈ (h/k_B) * (6π²n)^(1/3) * v_s
        # where n is number density and v_s is sound velocity
        
        # Rough estimates
        if '3D' in phase:
            debye_temp = 180  # K
            sound_velocity = 2000  # m/s
        elif '0D' in phase:
            debye_temp = 120  # K
            sound_velocity = 1500  # m/s
        else:  # 2D
            debye_temp = 150  # K
            sound_velocity = 1800  # m/s
        
        # Heat capacity (Debye model)
        x = debye_temp / T_K
        if x < 50:  # Avoid overflow
            debye_factor = (x**4 * np.exp(x)) / (np.exp(x) - 1)**2
            heat_capacity = 3 * R_GAS * debye_factor  # J/(mol·K)
        else:
            heat_capacity = 3 * R_GAS * np.exp(-x)
        
        # Thermal conductivity (rough estimate)
        # κ ≈ (1/3) * C * v * l where C is heat capacity, v is velocity, l is mean free path
        mean_free_path = 1e-9  # m (rough estimate)
        thermal_conductivity = (1/3) * heat_capacity * sound_velocity * mean_free_path
        
        return {
            'debye_temperature': debye_temp,
            'heat_capacity': heat_capacity,
            'thermal_conductivity': thermal_conductivity,
            'sound_velocity': sound_velocity
        }
    
    def calculate(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate structure-related properties"""
        temperature = conditions.get('temperature', 150)
        phase = conditions.get('phase', 'CsPbBr3_3D')
        pressure = conditions.get('pressure', 1.0)
        
        # Calculate lattice strain
        strain_results = self.calculate_lattice_strain(phase, temperature, pressure)
        
        # Calculate structure-derived bandgap
        structure_bandgap = self.calculate_bandgap_from_structure(phase, temperature)
        
        # Calculate phonon properties
        phonon_props = self.calculate_phonon_properties(phase, temperature)
        
        results = {
            'structure_bandgap': structure_bandgap,
            **strain_results,
            **phonon_props
        }
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get crystal structure parameters"""
        return {'crystal_structures': self.structures}


class IntegratedPhysicsModel:
    """Integration of all advanced physics models"""
    
    def __init__(self):
        """Initialize integrated physics model"""
        self.thermodynamic_model = ThermodynamicModel()
        self.nucleation_model = NucleationGrowthModel()
        self.structure_model = CrystalStructureModel()
        
    def calculate_comprehensive_properties(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive physics-based properties"""
        # Get results from each model
        thermo_results = self.thermodynamic_model.calculate(conditions)
        kinetic_results = self.nucleation_model.calculate(conditions)
        structure_results = self.structure_model.calculate(conditions)
        
        # Determine most likely phase from thermodynamics
        phase_energies = {
            'CsPbBr3_3D': thermo_results.get('phase_stability_3D', 0),
            'Cs4PbBr6_0D': thermo_results.get('phase_stability_0D', 0),
            'CsPb2Br5_2D': thermo_results.get('phase_stability_2D', 0)
        }
        
        most_stable_phase = min(phase_energies.keys(), key=lambda k: phase_energies[k])
        
        # Update conditions with determined phase
        enhanced_conditions = conditions.copy()
        enhanced_conditions['phase'] = most_stable_phase
        
        # Recalculate structure properties for specific phase
        structure_results = self.structure_model.calculate(enhanced_conditions)
        
        # Enhanced particle size prediction combining kinetics and thermodynamics
        kinetic_size = kinetic_results.get('predicted_particle_size', 10)
        
        # Thermodynamic size limit (rough estimate)
        temperature = conditions.get('temperature', 150)
        supersaturation = conditions.get('supersaturation', 2.0)
        
        # Critical nucleus size from thermodynamics
        if supersaturation > 1:
            critical_size = 2 / (np.log(supersaturation) + 1e-6)  # Simplified
        else:
            critical_size = 10
        
        # Combine kinetic and thermodynamic predictions
        predicted_size = max(critical_size, kinetic_size * 0.8)  # Kinetics dominates but thermodynamics sets minimum
        
        # Enhanced bandgap prediction
        structure_bandgap = structure_results.get('structure_bandgap', 2.3)
        
        # Size-dependent bandgap correction (quantum confinement)
        if predicted_size < 10:  # nm
            confinement_correction = 0.1 * (10 / predicted_size - 1)
            enhanced_bandgap = structure_bandgap + confinement_correction
        else:
            enhanced_bandgap = structure_bandgap
        
        # Compile comprehensive results
        comprehensive_results = {
            'predicted_phase': most_stable_phase,
            'enhanced_particle_size': predicted_size,
            'enhanced_bandgap': enhanced_bandgap,
            'phase_probability_3D': np.exp(-phase_energies['CsPbBr3_3D'] / (R_GAS * (temperature + 273.15))),
            'phase_probability_0D': np.exp(-phase_energies['Cs4PbBr6_0D'] / (R_GAS * (temperature + 273.15))),
            'phase_probability_2D': np.exp(-phase_energies['CsPb2Br5_2D'] / (R_GAS * (temperature + 273.15))),
            **thermo_results,
            **kinetic_results,
            **structure_results
        }
        
        # Normalize phase probabilities
        total_prob = (comprehensive_results['phase_probability_3D'] + 
                     comprehensive_results['phase_probability_0D'] + 
                     comprehensive_results['phase_probability_2D'])
        
        if total_prob > 0:
            comprehensive_results['phase_probability_3D'] /= total_prob
            comprehensive_results['phase_probability_0D'] /= total_prob
            comprehensive_results['phase_probability_2D'] /= total_prob
        
        return comprehensive_results
    
    def create_physics_report(self, conditions: Dict[str, float], 
                            output_path: str = "physics_report.html") -> None:
        """Create comprehensive physics analysis report"""
        results = self.calculate_comprehensive_properties(conditions)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Physics Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .parameter {{ margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🧪 Advanced Physics Analysis Report</h1>
            <p>Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>📋 Input Conditions</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
        """
        
        # Add input conditions
        for param, value in conditions.items():
            unit = self._get_parameter_unit(param)
            html_content += f"<tr><td>{param.replace('_', ' ').title()}</td><td>{value:.3f}</td><td>{unit}</td></tr>"
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 Phase Prediction</h2>
                <p><strong>Most Likely Phase:</strong> {results.get('predicted_phase', 'Unknown')}</p>
                <table>
                    <tr><th>Phase</th><th>Probability</th><th>Stability Energy</th></tr>
                    <tr><td>CsPbBr₃ (3D)</td><td>{results.get('phase_probability_3D', 0):.3f}</td><td>{results.get('phase_stability_3D', 0):.0f} J/mol</td></tr>
                    <tr><td>Cs₄PbBr₆ (0D)</td><td>{results.get('phase_probability_0D', 0):.3f}</td><td>{results.get('phase_stability_0D', 0):.0f} J/mol</td></tr>
                    <tr><td>CsPb₂Br₅ (2D)</td><td>{results.get('phase_probability_2D', 0):.3f}</td><td>{results.get('phase_stability_2D', 0):.0f} J/mol</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>📏 Structure Predictions</h2>
                <div class="parameter"><strong>Particle Size:</strong> {results.get('enhanced_particle_size', 0):.1f} nm</div>
                <div class="parameter"><strong>Bandgap:</strong> {results.get('enhanced_bandgap', 0):.3f} eV</div>
                <div class="parameter"><strong>Lattice Strain:</strong> {results.get('total_strain', 0)*100:.4f} %</div>
                <div class="parameter"><strong>Heat Capacity:</strong> {results.get('heat_capacity', 0):.1f} J/(mol·K)</div>
            </div>
            
            <div class="section">
                <h2>⚗️ Kinetic Analysis</h2>
                <div class="parameter"><strong>Nucleation Rate (3D):</strong> {results.get('nucleation_rate_3D', 0):.2e} nuclei/(s·m³)</div>
                <div class="parameter"><strong>Growth Rate (3D):</strong> {results.get('growth_rate_3D', 0):.2e} m/s</div>
                <div class="parameter"><strong>Thermal Conductivity:</strong> {results.get('thermal_conductivity', 0):.2e} W/(m·K)</div>
            </div>
            
            <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
                <p>Generated by CsPbBr₃ Digital Twin Advanced Physics Models</p>
            </footer>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"📊 Physics report saved to: {output_path}")
    
    def _get_parameter_unit(self, param: str) -> str:
        """Get unit for parameter"""
        units = {
            'temperature': '°C',
            'cs_br_concentration': 'mol/L',
            'pb_br2_concentration': 'mol/L',
            'oa_concentration': 'mol/L',
            'oam_concentration': 'mol/L',
            'reaction_time': 'min',
            'pressure': 'atm',
            'supersaturation': '-'
        }
        return units.get(param, '-')


if __name__ == "__main__":
    # Demonstration of advanced physics models
    print("⚛️ Advanced Physics Models Demo")
    
    # Test conditions
    conditions = {
        'temperature': 150,
        'cs_br_concentration': 1.0,
        'pb_br2_concentration': 1.0,
        'oa_concentration': 0.1,
        'oam_concentration': 0.1,
        'reaction_time': 30,
        'supersaturation': 2.5,
        'pressure': 1.0
    }
    
    print(f"📋 Test conditions: {conditions}")
    
    # Initialize integrated model
    physics_model = IntegratedPhysicsModel()
    
    # Calculate comprehensive properties
    print("🔬 Calculating comprehensive physics properties...")
    results = physics_model.calculate_comprehensive_properties(conditions)
    
    print("\n📊 Key Results:")
    print(f"   Predicted Phase: {results.get('predicted_phase', 'Unknown')}")
    print(f"   Particle Size: {results.get('enhanced_particle_size', 0):.1f} nm")
    print(f"   Bandgap: {results.get('enhanced_bandgap', 0):.3f} eV")
    print(f"   Phase Probabilities:")
    print(f"     3D: {results.get('phase_probability_3D', 0):.3f}")
    print(f"     0D: {results.get('phase_probability_0D', 0):.3f}")
    print(f"     2D: {results.get('phase_probability_2D', 0):.3f}")
    
    # Test individual models
    print("\n🧪 Testing individual models...")
    
    thermo_model = ThermodynamicModel()
    thermo_results = thermo_model.calculate(conditions)
    print(f"   Thermodynamics: Most stable = {thermo_results.get('most_stable_phase', 'Unknown')}")
    
    nucleation_model = NucleationGrowthModel()
    kinetic_results = nucleation_model.calculate(conditions)
    print(f"   Kinetics: Predicted size = {kinetic_results.get('predicted_particle_size', 0):.1f} nm")
    
    structure_model = CrystalStructureModel()
    structure_results = structure_model.calculate(conditions)
    print(f"   Structure: Bandgap = {structure_results.get('structure_bandgap', 0):.3f} eV")
    
    # Generate comprehensive report
    print("\n📄 Generating physics report...")
    physics_model.create_physics_report(conditions, "advanced_physics_report.html")
    
    print("✅ Advanced physics demo complete!")
    print("📁 Report saved as: advanced_physics_report.html")