"""
Configuration classes for 3-digit addition transformer.
"""

import os
import torch


class TaskConfig:
    """Configuration for the addition task."""
    
    K_DIGITS = 3
    TOKENIZATION = "character"
    
    # Vocabulary: digits 0-9, operators, special tokens
    DIGITS = [str(i) for i in range(10)]
    OPERATORS = ["+", "="]
    SPECIAL_TOKENS = ["PAD", "BOS", "EOS"]
    
    @property
    def vocab(self):
        """Get the full vocabulary list."""
        return self.SPECIAL_TOKENS + self.DIGITS + self.OPERATORS
    
    def get_vocab_size(self):
        """Get the vocabulary size."""
        return len(self.vocab)
    
    def get_token_to_idx(self):
        """Get token to index mapping."""
        return {token: idx for idx, token in enumerate(self.vocab)}
    
    def get_idx_to_token(self):
        """Get index to token mapping."""
        return {idx: token for idx, token in enumerate(self.vocab)}


class DataConfig:
    """Configuration for data generation and loading."""
    
    TRAIN_SIZE = 50000
    VAL_SIZE = 5000
    TEST_SIZE = 10000
    
    SAMPLING_STRATEGY = "uniform"
    SEED = 42
    
    # Data paths
    DATA_DIR = "data"
    TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
    VAL_FILE = os.path.join(DATA_DIR, "val.json")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")
    
    @classmethod
    def ensure_data_dir(cls):
        """Ensure the data directory exists."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)


class ModelConfig:
    """Configuration for the transformer model architecture."""
    
    N_LAYERS = 6
    D_MODEL = 256
    N_HEADS = 8
    D_FF = 1024
    DROPOUT = 0.1
    MAX_SEQ_LEN = 32
    
    def __init__(self, task_config: TaskConfig):
        """Initialize model config with task config to get vocab size."""
        self.VOCAB_SIZE = task_config.get_vocab_size()


class TrainConfig:
    """Configuration for training hyperparameters."""
    
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 50
    PATIENCE = 5  # Early stopping patience
    SEED = 42
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint directory
    SAVE_DIR = "checkpoints"
    
    @classmethod
    def ensure_save_dir(cls):
        """Ensure the checkpoint directory exists."""
        os.makedirs(cls.SAVE_DIR, exist_ok=True)


class EvalConfig:
    """Configuration for evaluation settings."""
    
    EVAL_STRATEGIES = ["uniform", "carry", "no_carry", "edge_cases"]
    TEST_SIZE_PER_STRATEGY = 1000
    DECODING = "greedy"
    METRICS = ["accuracy", "exact_match", "digit_accuracy"]
    RESULTS_FILE = "results.json"
    
    # Results directory
    RESULTS_DIR = "report"
    
    @classmethod
    def ensure_results_dir(cls):
        """Ensure the results directory exists."""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)


def print_config():
    """Print all configuration settings."""
    task_config = TaskConfig()
    data_config = DataConfig()
    model_config = ModelConfig(task_config)
    train_config = TrainConfig()
    eval_config = EvalConfig()
    
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    
    print("\n[TASK CONFIG]")
    print(f"  K_DIGITS: {task_config.K_DIGITS}")
    print(f"  TOKENIZATION: {task_config.TOKENIZATION}")
    print(f"  VOCAB_SIZE: {task_config.get_vocab_size()}")
    print(f"  Vocabulary: {task_config.vocab}")
    
    print("\n[DATA CONFIG]")
    print(f"  TRAIN_SIZE: {data_config.TRAIN_SIZE:,}")
    print(f"  VAL_SIZE: {data_config.VAL_SIZE:,}")
    print(f"  TEST_SIZE: {data_config.TEST_SIZE:,}")
    print(f"  SAMPLING_STRATEGY: {data_config.SAMPLING_STRATEGY}")
    print(f"  SEED: {data_config.SEED}")
    print(f"  DATA_DIR: {data_config.DATA_DIR}")
    
    print("\n[MODEL CONFIG]")
    print(f"  N_LAYERS: {model_config.N_LAYERS}")
    print(f"  D_MODEL: {model_config.D_MODEL}")
    print(f"  N_HEADS: {model_config.N_HEADS}")
    print(f"  D_FF: {model_config.D_FF}")
    print(f"  DROPOUT: {model_config.DROPOUT}")
    print(f"  MAX_SEQ_LEN: {model_config.MAX_SEQ_LEN}")
    print(f"  VOCAB_SIZE: {model_config.VOCAB_SIZE}")
    
    print("\n[TRAIN CONFIG]")
    print(f"  BATCH_SIZE: {train_config.BATCH_SIZE}")
    print(f"  LEARNING_RATE: {train_config.LEARNING_RATE}")
    print(f"  WEIGHT_DECAY: {train_config.WEIGHT_DECAY}")
    print(f"  MAX_EPOCHS: {train_config.MAX_EPOCHS}")
    print(f"  PATIENCE: {train_config.PATIENCE}")
    print(f"  DEVICE: {train_config.DEVICE}")
    print(f"  SAVE_DIR: {train_config.SAVE_DIR}")
    print(f"  SEED: {train_config.SEED}")
    
    print("\n[EVAL CONFIG]")
    print(f"  EVAL_STRATEGIES: {eval_config.EVAL_STRATEGIES}")
    print(f"  TEST_SIZE_PER_STRATEGY: {eval_config.TEST_SIZE_PER_STRATEGY}")
    print(f"  DECODING: {eval_config.DECODING}")
    print(f"  METRICS: {eval_config.METRICS}")
    print(f"  RESULTS_FILE: {eval_config.RESULTS_FILE}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test the code
    print("Testing...")
    print_config()
    print("Test complete.")
