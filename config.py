import os
import yaml
import argparse

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    def __repr__(self):
        return str(self.__dict__)


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config():
    parser = argparse.ArgumentParser(description="YAML config parser.")
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='Path to the YAML configuration file.')
    
    args = parser.parse_args()
    config_dict = load_config(args.config_path)
    return Config(config_dict)
