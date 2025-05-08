from dataclasses import dataclass
from typing import List, Dict, Optional
import torch

@dataclass
class DataConfig:
    data_dir: str = "data"
    timeframe: str = "M15"  # M1, M5, M15, M30, H1, H4, D1
    window_size: int = 10
    min_samples: int = 1000
    train_test_split: float = 0.8
    validation_size: float = 0.1

@dataclass
class FeatureConfig:
    # Candlestick features
    use_candle_patterns: bool = True
    use_volume: bool = True
    normalize_method: str = "standard"  # standard, minmax, robust
    
    # Technical indicators
    use_momentum: bool = True  # RSI, MACD, etc.
    use_volatility: bool = True  # ATR, Bollinger Bands
    use_trend: bool = True  # Moving averages, trend indicators
    
    # Advanced features
    use_wavelet: bool = False  # Wavelet transformation
    use_fourier: bool = False  # Fourier transformation
    correlation_window: int = 20

@dataclass
class ModelConfig:
    # VAE Architecture
    input_dim: Optional[int] = None  # Set dynamically
    latent_dim: int = 8
    hidden_dims: List[int] = None
    dropout_rate: float = 0.1
    activation: str = "leaky_relu"  # relu, leaky_relu, elu, gelu
    
    # Training
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10
    
    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ClusterConfig:
    # DBSCAN parameters
    eps: float = 0.5
    min_samples: int = 5
    
    # Anomaly detection
    anomaly_threshold: float = 3.0  # Standard deviations
    contamination: float = 0.1
    
    # Pattern analysis
    min_pattern_size: int = 20
    max_patterns: int = 50

@dataclass
class VisualizationConfig:
    # General
    style: str = "seaborn"
    context: str = "paper"
    figure_dpi: int = 300
    
    # Colors
    color_palette: str = "tab20"
    anomaly_color: str = "red"
    normal_color: str = "blue"
    
    # Plotting
    plot_dimensions: tuple = (12, 8)
    font_scale: float = 1.2
    
    # Interactive visualizations
    use_plotly: bool = True
    plot_3d: bool = True

@dataclass
class Config:
    # Main configuration sections
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    cluster: ClusterConfig = ClusterConfig()
    viz: VisualizationConfig = VisualizationConfig()
    
    # Project structure
    project_name: str = "unsupervised_trading"
    model_dir: str = "models"
    results_dir: str = "results"
    log_dir: str = "logs"
    
    # Logging
    log_level: str = "INFO"
    save_memory: bool = True
    
    # Runtime
    random_seed: int = 42
    num_workers: int = 4
    
    def __post_init__(self):
        import os
        
        # Create necessary directories
        for directory in [self.model_dir, self.results_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
            return obj
        
        config_dict = convert_to_dict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        def dict_to_dataclass(cls, d):
            if hasattr(cls, '__dataclass_fields__'):
                fields = {}
                for key, value in d.items():
                    field_type = cls.__dataclass_fields__[key].type
                    if hasattr(field_type, '__dataclass_fields__'):
                        fields[key] = dict_to_dataclass(field_type, value)
                    else:
                        fields[key] = value
                return cls(**fields)
            return d
            
        return dict_to_dataclass(cls, config_dict)

def get_default_config() -> Config:
    """Get default configuration"""
    return Config()

if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    
    # Save configuration
    config.save("config.json")
    
    # Load configuration
    loaded_config = Config.load("config.json")
    
    # Access configuration
    print(f"Data timeframe: {loaded_config.data.timeframe}")
    print(f"Model latent dimension: {loaded_config.model.latent_dim}")
    print(f"Using device: {loaded_config.model.device}")
