.PHONY: all setup clean test lint check docs build docker run help

# Default target
all: help

# Environment variables
PYTHON := python3
PIP := pip3
PROJECT := indserf-unsupervised
VERSION := 1.0.0
DOCKER_IMAGE := indserf/unsupervised
DATA_DIR := data
MODELS_DIR := models
RESULTS_DIR := results

# Colors for output
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Help target
help:
	@echo "$(BOLD)Unsupervised Trading Pattern Analysis - Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Available targets:$(RESET)"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  clean      - Remove build artifacts and temporary files"
	@echo "  test       - Run all tests"
	@echo "  lint       - Run code linting (black, flake8)"
	@echo "  check      - Run type checking (mypy)"
	@echo "  docs       - Generate documentation"
	@echo "  build      - Build Python package"
	@echo "  docker     - Build Docker image"
	@echo "  run        - Run the application"
	@echo ""
	@echo "$(BOLD)Environment:$(RESET)"
	@echo "  PYTHON     = $(PYTHON)"
	@echo "  PROJECT    = $(PROJECT)"
	@echo "  VERSION    = $(VERSION)"

# Setup development environment
setup:
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	@test -d venv || $(PYTHON) -m venv venv
	@. venv/bin/activate && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r requirements.txt && \
		$(PIP) install -e ".[dev]"
	@mkdir -p $(DATA_DIR) $(MODELS_DIR) $(RESULTS_DIR)
	@echo "$(GREEN)Setup complete!$(RESET)"

# Clean build artifacts and temporary files
clean:
	@echo "$(YELLOW)Cleaning project...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type d -name "*.egg" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@find . -type d -name "build" -exec rm -rf {} +
	@find . -type d -name "dist" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "$(GREEN)Clean complete!$(RESET)"

# Run tests
test:
	@echo "$(YELLOW)Running tests...$(RESET)"
	@. venv/bin/activate && \
		pytest tests/ --cov=./ --cov-report=term-missing
	@echo "$(GREEN)Tests complete!$(RESET)"

# Run linting
lint:
	@echo "$(YELLOW)Running linting...$(RESET)"
	@. venv/bin/activate && \
		black . && \
		flake8 .
	@echo "$(GREEN)Linting complete!$(RESET)"

# Run type checking
check:
	@echo "$(YELLOW)Running type checking...$(RESET)"
	@. venv/bin/activate && \
		mypy .
	@echo "$(GREEN)Type checking complete!$(RESET)"

# Generate documentation
docs:
	@echo "$(YELLOW)Generating documentation...$(RESET)"
	@. venv/bin/activate && \
		cd docs && \
		make html
	@echo "$(GREEN)Documentation generated in docs/_build/html/$(RESET)"

# Build Python package
build:
	@echo "$(YELLOW)Building package...$(RESET)"
	@. venv/bin/activate && \
		$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)Build complete!$(RESET)"

# Build Docker image
docker:
	@echo "$(YELLOW)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE):$(VERSION) .
	docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_IMAGE):latest
	@echo "$(GREEN)Docker build complete!$(RESET)"

# Run the application
run:
	@echo "$(YELLOW)Running application...$(RESET)"
	@. venv/bin/activate && \
		$(PYTHON) main.py

# Generate synthetic data
generate-data:
	@echo "$(YELLOW)Generating synthetic data...$(RESET)"
	@. venv/bin/activate && \
		$(PYTHON) utils/synthetic_data.py \
			--output_dir $(DATA_DIR) \
			--num_symbols 10 \
			--timeframe M15 \
			--num_days 30
	@echo "$(GREEN)Data generation complete!$(RESET)"

# Run analysis
analyze:
	@echo "$(YELLOW)Running pattern analysis...$(RESET)"
	@. venv/bin/activate && \
		$(PYTHON) scripts/pattern_analyzer.py \
			--data_dir $(DATA_DIR) \
			--results_dir $(RESULTS_DIR)
	@echo "$(GREEN)Analysis complete!$(RESET)"

# Run Docker container
docker-run:
	@echo "$(YELLOW)Running Docker container...$(RESET)"
	docker run -it --rm \
		-v $(PWD)/$(DATA_DIR):/home/appuser/data \
		-v $(PWD)/$(MODELS_DIR):/home/appuser/models \
		-v $(PWD)/$(RESULTS_DIR):/home/appuser/results \
		$(DOCKER_IMAGE):latest

# Install pre-commit hooks
install-hooks:
	@echo "$(YELLOW)Installing pre-commit hooks...$(RESET)"
	@. venv/bin/activate && \
		pre-commit install
	@echo "$(GREEN)Hooks installed!$(RESET)"

# Run examples
examples:
	@echo "$(YELLOW)Running examples...$(RESET)"
	@. venv/bin/activate && \
		$(PYTHON) examples/usage_example.py
	@echo "$(GREEN)Examples complete!$(RESET)"

# Create a new release
release:
	@echo "$(YELLOW)Creating release $(VERSION)...$(RESET)"
	@git tag -a v$(VERSION) -m "Release version $(VERSION)"
	@git push origin v$(VERSION)
	@echo "$(GREEN)Release $(VERSION) created!$(RESET)"
