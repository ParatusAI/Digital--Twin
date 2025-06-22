#!/bin/bash
# CsPbBr₃ Digital Twin - Easy Startup Script
# This script sets up the environment and runs the digital twin application

echo "🧬 CsPbBr₃ Digital Twin - Startup Script"
echo "========================================"

# Set up Python path to include all required packages
export PYTHONPATH="/home/aroyston/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the project directory."
    exit 1
fi

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Available commands:"
    echo "  help                    - Show detailed help"
    echo "  test                    - Run quick tests"
    echo "  validate [file.json]    - Validate synthesis parameters"
    echo "  predict [options]       - Make synthesis predictions"
    echo "  generate [options]      - Generate synthetic data"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 validate test_params.json"
    echo "  $0 predict --cs-concentration 1.5 --pb-concentration 1.0 --temperature 150"
    echo "  $0 generate --samples 100 --output data.csv"
    echo ""
    echo "For full help: $0 help"
    exit 0
fi

# Handle special commands
case "$1" in
    "help")
        python main.py --help
        echo ""
        echo "=== Examples ==="
        echo "Quick test:"
        echo "  $0 test"
        echo ""
        echo "Parameter validation:"
        echo "  $0 validate test_params.json"
        echo ""
        echo "Make a prediction:"
        echo "  $0 predict --cs-concentration 1.5 --pb-concentration 1.0 --temperature 150 --solvent DMSO"
        echo ""
        echo "Generate training data:"
        echo "  $0 generate --samples 1000 --output training_data.csv --seed 42"
        ;;
    "test")
        echo "🧪 Running Digital Twin Tests..."
        python main.py test --quick
        ;;
    "validate")
        if [ -z "$2" ]; then
            echo "❌ Error: Please provide a JSON file to validate"
            echo "Usage: $0 validate [file.json]"
            exit 1
        fi
        echo "🔍 Validating parameters from: $2"
        python main.py validate --input "$2"
        ;;
    *)
        # Pass all arguments to main.py
        echo "🚀 Running: python main.py $@"
        echo ""
        python main.py "$@"
        ;;
esac