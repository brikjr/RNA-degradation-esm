from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class ModelConfig:
    name: str
    esm_model: str
    hidden_dim: int
    num_layers: int
    dropout: float
    embedding_dim: int
    use_prototypes: bool = False
    prototype_dim: Optional[int] = None

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    gradient_clip: float
    scheduler_patience: int
    scheduler_factor: float

@dataclass
class DataConfig:
    train_path: str
    test_path: str
    processed_dir: str
    embeddings_dir: str
    sn_threshold: float

@dataclass
class LoggingConfig:
    tensorboard_dir: str
    save_dir: str
    log_interval: int

@dataclass
class FewShotConfig:
    n_support: int
    n_query: int
    episodes: int
    adaptation_steps: int
    meta_lr: float
    update_step_size: float

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    logging: LoggingConfig
    few_shot: Optional[FewShotConfig] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from yaml file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            logging=LoggingConfig(**config_dict['logging']),
            few_shot=FewShotConfig(**config_dict['few_shot']) 
                if 'few_shot' in config_dict else None
        )
