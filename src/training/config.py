from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    gradient_clip: float
    scheduler_patience: int
    scheduler_factor: float

    def __post_init__(self):
        # Convert string values to appropriate types
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.weight_decay = float(self.weight_decay)
        self.gradient_clip = float(self.gradient_clip)
        self.scheduler_patience = int(self.scheduler_patience)
        self.scheduler_factor = float(self.scheduler_factor)

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

    def __post_init__(self):
        # Convert numeric values to appropriate types
        self.hidden_dim = int(self.hidden_dim)
        self.num_layers = int(self.num_layers)
        self.dropout = float(self.dropout)
        self.embedding_dim = int(self.embedding_dim)
        if self.prototype_dim is not None:
            self.prototype_dim = int(self.prototype_dim)

@dataclass
class DataConfig:
    train_path: str
    test_path: str
    processed_dir: str
    embeddings_dir: str
    sn_threshold: float

    def __post_init__(self):
        self.sn_threshold = float(self.sn_threshold)

@dataclass
class LoggingConfig:
    tensorboard_dir: str
    save_dir: str
    log_interval: int

    def __post_init__(self):
        self.log_interval = int(self.log_interval)

@dataclass
class FewShotConfig:
    n_support: int
    n_query: int
    episodes: int
    adaptation_steps: int
    meta_lr: float
    update_step_size: float

    def __post_init__(self):
        self.n_support = int(self.n_support)
        self.n_query = int(self.n_query)
        self.episodes = int(self.episodes)
        self.adaptation_steps = int(self.adaptation_steps)
        self.meta_lr = float(self.meta_lr)
        self.update_step_size = float(self.update_step_size)

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
