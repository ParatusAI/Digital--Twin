#!/usr/bin/env python3
"""
CsPbBr₃ Digital Twin - Main Application Entry Point
Command-line interface for training, prediction, and data generation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from synthesis.utils import setup_logging
from synthesis import CsPbBr3DigitalTwin, quick_prediction


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="CsPbBr₃ Digital Twin - Perovskite Synthesis Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make synthesis predictions')
    predict_parser.add_argument('--cs-concentration', type=float, required=True,
                               help='Cs-Br concentration (mol/L)')
    predict_parser.add_argument('--pb-concentration', type=float, required=True,
                               help='Pb-Br2 concentration (mol/L)')
    predict_parser.add_argument('--temperature', type=float, required=True,
                               help='Temperature (°C)')
    predict_parser.add_argument('--solvent', type=str, default='DMSO',
                               choices=['DMSO', 'DMF', 'water', 'toluene', 'octadecene'],
                               help='Solvent type')
    predict_parser.add_argument('--model-path', type=str, default=None,
                               help='Path to pretrained model')
    predict_parser.add_argument('--output', type=str, default=None,
                               help='Output file for predictions')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train neural networks')
    train_parser.add_argument('--config', type=str, default=None,
                             help='Training configuration file')
    train_parser.add_argument('--data-file', type=str, required=True,
                             help='Training data CSV file')
    train_parser.add_argument('--output-dir', type=str, default='experiments',
                             help='Output directory for models')
    train_parser.add_argument('--epochs', type=int, default=100,
                             help='Number of training epochs')
    
    # Generate data command  
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    generate_parser.add_argument('--samples', type=int, default=1000,
                                help='Number of samples to generate')
    generate_parser.add_argument('--output', type=str, default='data/synthetic_data.csv',
                                help='Output CSV file')
    generate_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducibility')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate synthesis parameters')
    validate_parser.add_argument('--input', type=str, required=True,
                                help='Input parameter file (JSON)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run basic model tests')
    test_parser.add_argument('--quick', action='store_true',
                            help='Run quick tests only')
    
    # Common arguments
    for subparser in [predict_parser, train_parser, generate_parser, validate_parser, test_parser]:
        subparser.add_argument('--log-level', type=str, default='INFO',
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              help='Logging level')
        subparser.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output')
    
    return parser


def handle_predict(args) -> None:
    """Handle prediction command"""
    try:
        print(f"Making prediction for:")
        print(f"  Cs concentration: {args.cs_concentration} mol/L")
        print(f"  Pb concentration: {args.pb_concentration} mol/L") 
        print(f"  Temperature: {args.temperature}°C")
        print(f"  Solvent: {args.solvent}")
        
        # Make prediction
        result = quick_prediction(
            cs_concentration=args.cs_concentration,
            pb_concentration=args.pb_concentration,
            temperature=args.temperature,
            solvent=args.solvent,
            model_path=args.model_path
        )
        
        print("\nPrediction Results:")
        print("=" * 50)
        
        if 'phase' in result:
            phase_data = result['phase']
            print(f"Predicted Phase: {phase_data.get('predicted_class', 'Unknown')}")
            if 'probabilities' in phase_data:
                print(f"Confidence: {max(phase_data['probabilities']):.3f}")
        
        if 'properties' in result:
            print("\nPredicted Properties:")
            for prop_name, prop_data in result['properties'].items():
                if 'mean' in prop_data:
                    mean_val = prop_data['mean']
                    std_val = prop_data.get('std', 0)
                    print(f"  {prop_name}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Save output if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def handle_train(args) -> None:
    """Handle training command"""
    try:
        print(f"Training model with data: {args.data_file}")
        
        # Import training module
        from train_pytorch_models import main as train_main
        
        # Set up arguments for training script
        train_args = [
            '--data-file', args.data_file,
            '--output-dir', args.output_dir,
            '--max-epochs', str(args.epochs)
        ]
        
        if args.config:
            train_args.extend(['--config', args.config])
        
        # Set sys.argv for the training script
        old_argv = sys.argv
        sys.argv = ['train_pytorch_models.py'] + train_args
        
        try:
            train_main()
        finally:
            sys.argv = old_argv
            
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


def handle_generate(args) -> None:
    """Handle data generation command"""
    try:
        print(f"Generating {args.samples} synthetic samples...")
        
        # Import generation module
        from generate_sample_data import main as generate_main
        
        # Set up arguments
        gen_args = [
            '--num-samples', str(args.samples),
            '--output-file', args.output,
            '--seed', str(args.seed)
        ]
        
        # Set sys.argv for the generation script
        old_argv = sys.argv
        sys.argv = ['generate_sample_data.py'] + gen_args
        
        try:
            generate_main()
            print(f"Data generated successfully: {args.output}")
        finally:
            sys.argv = old_argv
            
    except Exception as e:
        print(f"Error generating data: {e}")
        sys.exit(1)


def handle_validate(args) -> None:
    """Handle validation command"""
    try:
        import json
        from synthesis.utils import validate_synthesis_parameters
        
        print(f"Validating parameters from: {args.input}")
        
        with open(args.input, 'r') as f:
            params = json.load(f)
        
        is_valid, errors = validate_synthesis_parameters(params)
        
        if is_valid:
            print("✅ All parameters are valid!")
        else:
            print("❌ Validation errors found:")
            for error in errors:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"Error validating parameters: {e}")
        sys.exit(1)


def handle_test(args) -> None:
    """Handle test command"""
    try:
        if args.quick:
            print("Running quick tests...")
            from test_basic_model import run_quick_tests
            run_quick_tests()
        else:
            print("Running full test suite...")
            from test_basic_model import main as test_main
            from test_integration import main as integration_main
            test_main()
            integration_main()
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


def main():
    """Main application entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = args.log_level if hasattr(args, 'log_level') else 'INFO'
    setup_logging(log_level)
    
    if args.verbose if hasattr(args, 'verbose') else False:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print("🧬 CsPbBr₃ Digital Twin v2.0.0")
    print("Advanced Perovskite Synthesis Prediction")
    print("=" * 50)
    
    # Handle commands
    if args.command == 'predict':
        handle_predict(args)
    elif args.command == 'train':
        handle_train(args)
    elif args.command == 'generate':
        handle_generate(args)
    elif args.command == 'validate':
        handle_validate(args)
    elif args.command == 'test':
        handle_test(args)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py predict --cs-concentration 1.5 --pb-concentration 1.0 --temperature 150")
        print("  python main.py generate --samples 5000 --output data/training_data.csv")
        print("  python main.py train --data-file data/training_data.csv --epochs 50")
        print("  python main.py test --quick")


if __name__ == "__main__":
    main()