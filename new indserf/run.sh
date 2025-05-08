#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Function to check if command was successful
check_status() {
    if [ $? -eq 0 ]; then
        print_message "✔ Success: $1" "${GREEN}"
    else
        print_message "✘ Error: $1" "${RED}"
        exit 1
    fi
}

# Help message
show_help() {
    echo "Unsupervised Trading Pattern Analysis - Command Runner"
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  setup        - Install dependencies and setup environment"
    echo "  test         - Run all tests"
    echo "  train        - Train the model with default configuration"
    echo "  generate     - Generate synthetic data for testing"
    echo "  analyze      - Run pattern analysis on existing data"
    echo "  clean        - Clean temporary files and cached data"
    echo "  example      - Run usage examples"
    echo "  doc          - Generate documentation"
    echo "  gpu-check    - Check GPU availability"
    echo "  help         - Show this help message"
}

# Setup environment
setup_env() {
    print_message "Setting up environment..." "${YELLOW}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        check_status "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    check_status "Activated virtual environment"
    
    # Install dependencies
    pip install -r requirements.txt
    check_status "Installed dependencies"
    
    # Install package in development mode
    pip install -e .
    check_status "Installed package"
    
    print_message "Environment setup complete!" "${GREEN}"
}

# Run tests
run_tests() {
    print_message "Running tests..." "${YELLOW}"
    python -m pytest tests/
    check_status "Tests completed"
}

# Train model
train_model() {
    print_message "Training model..." "${YELLOW}"
    python main.py --config config.json
    check_status "Training completed"
}

# Generate synthetic data
generate_data() {
    print_message "Generating synthetic data..." "${YELLOW}"
    python utils/synthetic_data.py --output_dir data --num_symbols 10 --timeframe M15 --num_days 30
    check_status "Data generation completed"
}

# Run analysis
run_analysis() {
    print_message "Running pattern analysis..." "${YELLOW}"
    python scripts/pattern_analyzer.py --data_dir data --results_dir results
    check_status "Analysis completed"
}

# Clean temporary files
clean_files() {
    print_message "Cleaning temporary files..." "${YELLOW}"
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete
    
    # Remove temporary directories
    rm -rf build/ dist/ *.egg-info/
    
    check_status "Cleanup completed"
}

# Run examples
run_examples() {
    print_message "Running usage examples..." "${YELLOW}"
    python examples/usage_example.py
    check_status "Examples completed"
}

# Generate documentation
generate_docs() {
    print_message "Generating documentation..." "${YELLOW}"
    
    # Check if Sphinx is installed
    if ! command -v sphinx-build &> /dev/null; then
        pip install sphinx sphinx-rtd-theme
    fi
    
    # Generate documentation
    cd docs
    make html
    cd ..
    
    check_status "Documentation generation completed"
}

# Check GPU availability
check_gpu() {
    print_message "Checking GPU availability..." "${YELLOW}"
    python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('No GPU available. Using CPU only.')
"
    check_status "GPU check completed"
}

# Main script logic
case "$1" in
    "setup")
        setup_env
        ;;
    "test")
        run_tests
        ;;
    "train")
        train_model
        ;;
    "generate")
        generate_data
        ;;
    "analyze")
        run_analysis
        ;;
    "clean")
        clean_files
        ;;
    "example")
        run_examples
        ;;
    "doc")
        generate_docs
        ;;
    "gpu-check")
        check_gpu
        ;;
    "help"|"")
        show_help
        ;;
    *)
        print_message "Unknown command: $1" "${RED}"
        show_help
        exit 1
        ;;
esac

exit 0
