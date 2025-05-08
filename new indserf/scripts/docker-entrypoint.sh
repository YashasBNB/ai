#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
log() {
    echo -e "${2}$(date '+%Y-%m-%d %H:%M:%S') - ${1}${NC}"
}

# Check if running in GPU mode
check_gpu() {
    python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('No GPU available. Using CPU only.')
"
}

# Initialize directories
init_dirs() {
    dirs=("$DATA_DIR" "$MODEL_DIR" "$RESULTS_DIR" "$LOG_DIR")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created directory: $dir" "$GREEN"
        fi
    done
}

# Validate environment variables
validate_env() {
    required_vars=("DATA_DIR" "MODEL_DIR" "RESULTS_DIR" "LOG_DIR" "CONFIG_PATH")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log "Error: Required environment variable $var is not set" "$RED"
            exit 1
        fi
    done
}

# Check data directory
check_data() {
    if [ -z "$(ls -A $DATA_DIR)" ]; then
        log "Warning: Data directory is empty. You may need to mount data volume." "$YELLOW"
    else
        log "Found data files in $DATA_DIR" "$GREEN"
    fi
}

# Check model files
check_models() {
    if [ -z "$(ls -A $MODEL_DIR)" ]; then
        log "Note: No pre-trained models found. Will train new models." "$YELLOW"
    else
        log "Found model files in $MODEL_DIR" "$GREEN"
    fi
}

# Main entrypoint logic
main() {
    log "Starting Unsupervised Trading Pattern Analysis Container" "$GREEN"
    
    # Validate environment
    validate_env
    
    # Initialize directories
    init_dirs
    
    # Check GPU availability
    log "Checking GPU status..." "$YELLOW"
    check_gpu
    
    # Check data and models
    check_data
    check_models
    
    # Execute the main command
    if [ "$1" = "train" ]; then
        log "Starting training process..." "$YELLOW"
        exec python main.py --mode train
        
    elif [ "$1" = "analyze" ]; then
        log "Starting pattern analysis..." "$YELLOW"
        exec python main.py --mode analyze
        
    elif [ "$1" = "serve" ]; then
        log "Starting API server..." "$YELLOW"
        exec python main.py --mode serve
        
    elif [ "$1" = "shell" ]; then
        log "Starting interactive shell..." "$YELLOW"
        exec /bin/bash
        
    else
        # Default: run whatever command was passed
        log "Executing command: $@" "$YELLOW"
        exec "$@"
    fi
}

# Handle SIGTERM
trap 'log "Received SIGTERM - shutting down..." "$YELLOW"; exit 0' SIGTERM

# Start main logic
main "$@"
