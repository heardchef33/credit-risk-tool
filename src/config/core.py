from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict
from strictyaml import YAML, load


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config" / "config.yml"
DATASET_DIR = ROOT / "data"
TRAINED_MODEL_DIR = ROOT / "model" / "saved_models"

class AppConfig(BaseModel): 

    package_name: str
    raw_data_file: str
    X_train_file: str  
    X_test_file: str 
    y_train_file: str
    y_test_file: str
    pipeline_save_file: str

class ModelConfig(BaseModel): 

    target: str
    random_state: int
    test_size: float 
    features: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    categorical_features_to_encode_one_hot: List[str]
    categorical_features_to_encode_ordinally: List[str]
    target_mappings: Dict[str, int]
    class_ratio: float
    scale_pos_weight: float
    n_estimators: int
    booster: str
    learning_rate: float 
    max_depth: int 
    min_child_weight: int 
    objective: str


class Config(BaseModel): 

    app_config: AppConfig
    model_settings: ModelConfig

def find_config_file(): 
    if CONFIG_FILE_PATH.is_file(): 
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Path = None) -> YAML: 
    if not cfg_path: 
        cfg_path = find_config_file()
    
    if cfg_path: 
        with open(cfg_path, "r") as conf_file: 
            parsed_config = load(conf_file.read()) # the load function requires a Loader parameter (https://github.com/YAML/pyyaml/wiki/PyYAML-YAML.load(input)-Deprecation)
            return parsed_config
    raise OSError(f"Did not find the config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config: 
    if parsed_config is None: 
        parsed_config = fetch_config_from_yaml()

    print(parsed_config.data)

    _config = Config(
        app_config=AppConfig(**parsed_config.data), 
        model_settings=ModelConfig(**parsed_config.data) # is model settings the standard name? 
    )

    return _config 

config = create_and_validate_config()


