import os
import logging
import logging.config
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict

# Default logging configuration
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "json_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "logs/app.json",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": True
        },
        "indserf": {  # Application logger
            "handlers": ["console", "file", "error_file", "json_file"],
            "level": "DEBUG",
            "propagate": False
        },
        "indserf.model": {  # Model-specific logger
            "handlers": ["console", "file", "json_file"],
            "level": "INFO",
            "propagate": False
        },
        "indserf.data": {  # Data processing logger
            "handlers": ["console", "file", "json_file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

class LogManager:
    """Manage logging configuration and setup"""
    
    def __init__(self, config: Optional[Dict] = None, log_dir: str = "logs"):
        self.config = config or DEFAULT_CONFIG
        self.log_dir = Path(log_dir)
        self._setup_log_directory()
        
    def _setup_log_directory(self):
        """Create log directory if it doesn't exist"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Update log file paths in config
        for handler in self.config["handlers"].values():
            if "filename" in handler:
                handler["filename"] = str(self.log_dir / Path(handler["filename"]).name)
                
    def setup_logging(self):
        """Configure logging based on configuration"""
        try:
            logging.config.dictConfig(self.config)
            logging.info("Logging setup completed successfully")
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fall back to basic configuration
            logging.basicConfig(level=logging.INFO)
            
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        return logging.getLogger(name)
    
    def update_log_level(self, level: str):
        """Update log level for all handlers"""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
            
        for handler in logging.root.handlers:
            handler.setLevel(numeric_level)
            
    def add_file_handler(self, 
                        logger_name: str,
                        filename: str,
                        level: str = "INFO",
                        formatter: str = "standard"):
        """Add a new file handler to a logger"""
        logger = logging.getLogger(logger_name)
        
        handler = logging.FileHandler(self.log_dir / filename)
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(logging.Formatter(
            self.config["formatters"][formatter]["format"]
        ))
        
        logger.addHandler(handler)
        
    def rotate_logs(self):
        """Rotate all log files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for handler_config in self.config["handlers"].values():
            if "filename" in handler_config:
                log_file = Path(handler_config["filename"])
                if log_file.exists():
                    backup = log_file.parent / f"{log_file.stem}_{timestamp}{log_file.suffix}"
                    log_file.rename(backup)
                    
    def cleanup_old_logs(self, days: int = 30):
        """Remove log files older than specified days"""
        current_time = datetime.now().timestamp()
        
        for log_file in self.log_dir.glob("*.log"):
            file_time = os.path.getctime(log_file)
            if (current_time - file_time) > (days * 24 * 3600):
                try:
                    os.remove(log_file)
                    print(f"Removed old log file: {log_file}")
                except Exception as e:
                    print(f"Error removing {log_file}: {e}")

def setup_logging(log_dir: str = "logs", config_file: Optional[str] = None):
    """Setup logging with optional custom configuration"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG
        
    log_manager = LogManager(config, log_dir)
    log_manager.setup_logging()
    return log_manager

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Logging Configuration")
    parser.add_argument('--log_dir', type=str, default="logs",
                       help="Directory for log files")
    parser.add_argument('--config', type=str,
                       help="Path to custom logging configuration file")
    parser.add_argument('--cleanup', action='store_true',
                       help="Clean up old log files")
    parser.add_argument('--days', type=int, default=30,
                       help="Days threshold for cleanup")
    
    args = parser.parse_args()
    
    if args.cleanup:
        log_manager = LogManager(log_dir=args.log_dir)
        log_manager.cleanup_old_logs(args.days)
    else:
        log_manager = setup_logging(args.log_dir, args.config)
