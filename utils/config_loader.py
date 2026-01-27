"""
Configuration loader utility for TubeViT.

This module provides utilities to load configuration from YAML files
and merge them with command-line arguments.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    return config


def merge_config_with_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration dictionary with command-line arguments.
    Command-line arguments take precedence over config values.
    
    Args:
        config: Configuration dictionary from YAML file
        args: Command-line arguments dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = config.copy()
    
    # Override with non-None command-line arguments
    for key, value in args.items():
        if value is not None:
            merged[key] = value
    
    return merged


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a value from configuration dictionary with optional default.
    
    Args:
        config: Configuration dictionary
        key: Key to look up (supports nested keys with dot notation, e.g., "dataset.root")
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value if value is not None else default
