#!/usr/bin/env python3
"""
Data Collection Templates for CsPbBr₃ Experimental Validation
Pre-built forms and templates for easy data entry
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import argparse

def create_experiment_template() -> Dict[str, Any]:
    """Create template for experimental conditions"""
    
    template = {
        "experiment_id": f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "date_conducted": datetime.now().strftime("%Y-%m-%d"),
        "researcher": "",
        
        # Synthesis conditions
        "cs_br_concentration": 0.0,  # mol/L
        "pb_br2_concentration": 0.0,  # mol/L
        "temperature": 0.0,  # °C
        "oa_concentration": 0.0,  # mol/L (oleic acid)
        "oam_concentration": 0.0,  # mol/L (oleylamine)
        "reaction_time": 0.0,  # minutes
        "solvent_type": 0,  # 0=DMSO, 1=DMF, 2=Toluene, 3=Octadecene, 4=Mixed
        
        # Additional parameters
        "atmosphere": "inert",  # inert, air, vacuum
        "stirring_rate": 0,  # rpm
        "heating_rate": 0.0,  # °C/min
        "cooling_rate": 0.0,  # °C/min
        
        # Equipment details
        "reaction_vessel": "",  # flask type, volume
        "heating_method": "",  # oil bath, heating mantle, microwave
        "synthesis_scale": 0.0,  # mL total volume
        
        "notes": ""
    }
    
    return template

def create_results_template() -> Dict[str, Any]:
    """Create template for experimental results"""
    
    template = {
        "experiment_id": "",  # Must match conditions
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        
        # Visual observations
        "solution_color": "",  # green, yellow, colorless, etc.
        "precipitate_observed": False,
        "solution_clarity": "",  # clear, turbid, opaque
        
        # Phase characterization (XRD)
        "dominant_phase": "",  # CsPbBr3_3D, Cs4PbBr6_0D, CsPb2Br5_2D, Mixed, Failed
        "phase_purity": 0.0,  # 0-1 from XRD analysis
        "secondary_phases": [],  # list of other phases detected
        "crystal_structure": "",  # cubic, orthorhombic, etc.
        
        # Optical properties
        "bandgap": None,  # eV from UV-vis spectroscopy
        "absorption_onset": None,  # nm
        "plqy": None,  # photoluminescence quantum yield (0-1)
        "emission_peak": None,  # nm from PL spectroscopy
        "emission_fwhm": None,  # nm full width at half maximum
        "stokes_shift": None,  # nm
        
        # Morphology (TEM/SEM)
        "particle_size": None,  # nm average size
        "particle_size_distribution": None,  # nm standard deviation
        "morphology": "",  # cubic, spherical, platelet, etc.
        "size_uniformity": "",  # uniform, polydisperse, bimodal
        
        # Synthesis metrics
        "synthesis_success": False,
        "yield_percentage": None,  # % of theoretical yield
        "reaction_completion": "",  # complete, partial, failed
        
        # Characterization methods used
        "characterization_methods": [],  # ['XRD', 'UV-vis', 'PL', 'TEM', 'SEM', 'DLS', 'NMR']
        
        # Additional measurements
        "additional_properties": {},  # dictionary for other measurements
        
        # File references
        "data_files": {
            "xrd_file": "",
            "uv_vis_file": "",
            "pl_file": "",
            "tem_images": [],
            "other_files": []
        },
        
        "notes": ""
    }
    
    return template

def create_csv_templates(output_dir: str = "experimental_data"):
    """Create CSV templates for easy data entry"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Experimental conditions template
    conditions_template = pd.DataFrame([create_experiment_template()])
    conditions_file = output_path / "experiment_conditions_template.csv"
    conditions_template.to_csv(conditions_file, index=False)
    
    # Results template
    results_template = create_results_template()
    # Flatten for CSV
    results_flat = {}
    for key, value in results_template.items():
        if isinstance(value, list):
            results_flat[key] = ""  # Will be comma-separated strings
        elif isinstance(value, dict):
            results_flat[key] = ""  # Will be JSON strings
        else:
            results_flat[key] = value
    
    results_df = pd.DataFrame([results_flat])
    results_file = output_path / "experiment_results_template.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"📝 CSV templates created:")
    print(f"   Conditions: {conditions_file}")
    print(f"   Results: {results_file}")
    
    return conditions_file, results_file

def create_json_templates(output_dir: str = "experimental_data"):
    """Create JSON templates for structured data entry"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Experimental conditions
    conditions_template = create_experiment_template()
    conditions_file = output_path / "experiment_conditions_template.json"
    with open(conditions_file, 'w') as f:
        json.dump(conditions_template, f, indent=2)
    
    # Results template
    results_template = create_results_template()
    results_file = output_path / "experiment_results_template.json"
    with open(results_file, 'w') as f:
        json.dump(results_template, f, indent=2)
    
    print(f"📋 JSON templates created:")
    print(f"   Conditions: {conditions_file}")
    print(f"   Results: {results_file}")
    
    return conditions_file, results_file

def create_lab_notebook_template(output_dir: str = "experimental_data"):
    """Create printable lab notebook template"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    template_content = """
# CsPbBr₃ Synthesis Lab Notebook Template

## Experiment Information
- **Experiment ID**: ___________________
- **Date**: ___________________
- **Researcher**: ___________________
- **Objective**: ___________________

## Pre-Synthesis Checklist
- [ ] All chemicals weighed and ready
- [ ] Equipment cleaned and dried
- [ ] Inert atmosphere established
- [ ] Safety equipment ready
- [ ] Model prediction recorded

## Synthesis Conditions

### Reactant Concentrations
- **CsBr concentration**: _________ mol/L
- **PbBr₂ concentration**: _________ mol/L
- **Cs:Pb molar ratio**: _________

### Ligands
- **Oleic acid concentration**: _________ mol/L
- **Oleylamine concentration**: _________ mol/L

### Reaction Parameters
- **Temperature**: _________ °C
- **Reaction time**: _________ minutes
- **Solvent**: [ ] DMSO [ ] DMF [ ] Toluene [ ] Octadecene [ ] Mixed
- **Atmosphere**: [ ] Nitrogen [ ] Argon [ ] Air
- **Stirring rate**: _________ rpm

### Equipment
- **Reaction vessel**: _________
- **Heating method**: _________
- **Total volume**: _________ mL

## Model Prediction
- **Predicted phase**: _________
- **Confidence**: _________
- **Predicted bandgap**: _________ eV
- **Predicted PLQY**: _________

## Synthesis Observations

### Time: _______ | Temperature: _______
Observations: _________________________________

### Time: _______ | Temperature: _______
Observations: _________________________________

### Time: _______ | Temperature: _______
Observations: _________________________________

### Final Observations
- **Solution color**: _________
- **Precipitate**: [ ] Yes [ ] No
- **Clarity**: [ ] Clear [ ] Turbid [ ] Opaque
- **Apparent success**: [ ] Yes [ ] No [ ] Partial

## Post-Synthesis
- **Yield**: _________ mg (_________ % theoretical)
- **Storage conditions**: _________
- **Next steps**: _________

## Characterization Plan
- [ ] XRD analysis
- [ ] UV-vis spectroscopy  
- [ ] Photoluminescence
- [ ] TEM imaging
- [ ] SEM imaging
- [ ] DLS size analysis
- [ ] Other: _________

## Results Summary (Fill after characterization)

### Phase Analysis (XRD)
- **Dominant phase**: _________
- **Phase purity**: _________ %
- **Secondary phases**: _________

### Optical Properties
- **Bandgap**: _________ eV
- **Emission peak**: _________ nm
- **PLQY**: _________
- **Stokes shift**: _________ nm

### Morphology
- **Particle size**: _________ nm
- **Size distribution**: _________ nm (std dev)
- **Morphology**: _________

## Model Validation
- **Phase prediction correct**: [ ] Yes [ ] No
- **Property predictions**:
  - Bandgap error: _________ %
  - PLQY error: _________ %
  - Size error: _________ %

## Notes and Observations
_________________________________________
_________________________________________
_________________________________________

## Files and Data
- **XRD file**: _________
- **UV-vis file**: _________
- **PL file**: _________
- **TEM images**: _________
- **Data folder**: _________

---
**Experiment completed by**: _________________ **Date**: _________
"""
    
    notebook_file = output_path / "lab_notebook_template.md"
    with open(notebook_file, 'w') as f:
        f.write(template_content)
    
    print(f"📓 Lab notebook template created: {notebook_file}")
    return notebook_file

def create_characterization_checklists(output_dir: str = "experimental_data"):
    """Create characterization method checklists"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    checklists = {
        "XRD_checklist": {
            "sample_preparation": [
                "Sample dried completely",
                "Sample ground to fine powder", 
                "Sufficient sample amount (>10 mg)",
                "Sample mounted properly on holder"
            ],
            "measurement_parameters": [
                "2θ range: 10-80°",
                "Step size: 0.02°",
                "Count time: 1-2 s per step",
                "Generator settings recorded"
            ],
            "analysis_steps": [
                "Background subtraction",
                "Peak identification using database",
                "Phase quantification (Rietveld if needed)",
                "Lattice parameter calculation"
            ]
        },
        
        "UV_vis_checklist": {
            "sample_preparation": [
                "Sample dispersed in appropriate solvent",
                "Concentration optimized for measurement",
                "Sample degassed if needed",
                "Reference/blank prepared"
            ],
            "measurement_parameters": [
                "Wavelength range: 300-800 nm",
                "Scan speed appropriate",
                "Cuvette path length recorded",
                "Temperature controlled"
            ],
            "analysis_steps": [
                "Baseline correction",
                "Absorption onset determination",
                "Tauc plot for bandgap",
                "Extinction coefficient calculation"
            ]
        },
        
        "PL_checklist": {
            "sample_preparation": [
                "Sample concentration optimized",
                "Excitation wavelength determined",
                "Sample holder cleaned",
                "Reference standard measured"
            ],
            "measurement_parameters": [
                "Excitation wavelength recorded",
                "Emission range appropriate",
                "Slit widths optimized",
                "Integration time sufficient"
            ],
            "analysis_steps": [
                "Spectrum corrected for detector response",
                "Peak position and FWHM determined",
                "PLQY calculated with reference",
                "Stokes shift calculated"
            ]
        },
        
        "TEM_checklist": {
            "sample_preparation": [
                "Sample diluted appropriately",
                "Grid type recorded (carbon, holey, etc.)",
                "Sample drop-cast or dip-coated",
                "Grid dried completely"
            ],
            "imaging_parameters": [
                "Accelerating voltage recorded",
                "Magnification series taken",
                "Multiple areas imaged",
                "High resolution images if needed"
            ],
            "analysis_steps": [
                "Size distribution measured (>100 particles)",
                "Morphology described",
                "Crystal structure analysis (SAED)",
                "Statistical analysis completed"
            ]
        }
    }
    
    checklist_file = output_path / "characterization_checklists.json"
    with open(checklist_file, 'w') as f:
        json.dump(checklists, f, indent=2)
    
    print(f"✅ Characterization checklists created: {checklist_file}")
    return checklist_file

def main():
    """Generate all experimental templates"""
    
    parser = argparse.ArgumentParser(description="Generate experimental data templates")
    parser.add_argument("--output-dir", type=str, default="experimental_data",
                       help="Output directory for templates")
    parser.add_argument("--format", type=str, choices=["csv", "json", "all"], default="all",
                       help="Template format to generate")
    
    args = parser.parse_args()
    
    print("🧪 Creating CsPbBr₃ Experimental Data Templates")
    print("=" * 50)
    
    if args.format in ["csv", "all"]:
        create_csv_templates(args.output_dir)
    
    if args.format in ["json", "all"]:
        create_json_templates(args.output_dir)
    
    create_lab_notebook_template(args.output_dir)
    create_characterization_checklists(args.output_dir)
    
    print(f"\n📂 All templates created in: {args.output_dir}")
    print("\n💡 Usage instructions:")
    print("1. Copy template files before filling them out")
    print("2. Use lab notebook template for experiment planning")
    print("3. Follow characterization checklists for quality data")
    print("4. Import filled templates using experimental_validation.py")

if __name__ == "__main__":
    main()