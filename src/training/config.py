from dataclasses import dataclass
from typing import Optional, Dict
import yaml
from copy import deepcopy

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
   distance_metric: str = "euclidean"

   @classmethod
   def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
       # Default values
       defaults = {
           'name': "ESMRNAPredictor",
           'esm_model': "esm2_t33_650M_UR50D", 
           'hidden_dim': 256,
           'num_layers': 3,
           'dropout': 0.2,
           'embedding_dim': 1280,
           'use_prototypes': False,
           'prototype_dim': None,
           'distance_metric': "euclidean"
       }
       
       # Update defaults with provided config
       config = deepcopy(defaults)
       config.update(config_dict)
       
       return cls(**config)

   def __post_init__(self):
       self.hidden_dim = int(self.hidden_dim)
       self.num_layers = int(self.num_layers)
       self.dropout = float(self.dropout)
       self.embedding_dim = int(self.embedding_dim)
       if self.prototype_dim is not None:
           self.prototype_dim = int(self.prototype_dim)
           
       valid_metrics = ["euclidean", "cosine"]
       if self.distance_metric not in valid_metrics:
           raise ValueError(f"distance_metric must be one of {valid_metrics}")

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
       """Load config from yaml file with inheritance"""
       with open(path) as f:
           config_dict = yaml.safe_load(f)
           
       # Handle inheritance
       if 'defaults' in config_dict:
           for default in config_dict['defaults']:
               if isinstance(default, str):
                   default_path = f"configs/{default}.yaml"
               else:
                   for key, value in default.items():
                       default_path = f"configs/{value}.yaml"
               
               with open(default_path) as f:
                   default_config = yaml.safe_load(f)
                   # Update with values from default config
                   for key, value in default_config.items():
                       if key not in config_dict:
                           config_dict[key] = value
                       elif isinstance(value, dict):
                           config_dict[key].update(value)
                           
       # Create configs
       return cls(
           model=ModelConfig.from_dict(config_dict['model']),
           training=TrainingConfig(**config_dict['training']),
           data=DataConfig(**config_dict['data']),
           logging=LoggingConfig(**config_dict['logging']),
           few_shot=FewShotConfig(**config_dict['few_shot']) 
               if 'few_shot' in config_dict else None
       )
