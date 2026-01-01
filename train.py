"""
Training script for the Addition Transformer.

This module handles the complete training pipeline including:
- Dataset loading and preprocessing
- Model initialization
- Training loop with validation
- Checkpointing and early stopping
- Learning rate scheduling
"""

import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import TaskConfig, DataConfig, ModelConfig, TrainConfig
from model import AdditionTransformer, Tokenizer
from data_generator import AdditionDataGenerator, generate_all_datasets



# Dataset Class

class AdditionDataset(Dataset):
    """
    PyTorch Dataset for addition examples.
    
    Loads JSON data and tokenizes input/output strings for training.
    Each example is formatted as:
        Input:  BOS + "123+456=" 
        Target: "123+456=" + "579" + EOS (shifted by one for next-token prediction)
    """
    
    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 32):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSON file containing examples
            tokenizer: Tokenizer instance for encoding strings
            max_seq_len: Maximum sequence length (for padding)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load data from JSON file
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single example.
        
        For decoder-only transformers with next-token prediction:
        - Full sequence: BOS + input + output + EOS
        - Input to model: sequence[:-1] (everything except last token)
        - Target: sequence[1:] (everything except first token)
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with input_ids, target_ids, and attention_mask tensors
        """
        example = self.data[idx]
        
        # Get input and output strings
        input_str = example['input']   # e.g., "123+456="
        output_str = example['output']  # e.g., "579"
        
        # Create full sequence: BOS + input + output + EOS
        # Tokenize the combined string
        full_text = input_str + output_str
        token_ids = self.tokenizer.encode(full_text)
        
        # Add BOS at start and EOS at end
        full_sequence = (
            [self.tokenizer.bos_token_id] + 
            token_ids + 
            [self.tokenizer.eos_token_id]
        )
        
        # For next-token prediction:
        # Input: full_sequence[:-1] (predict from BOS...output[-1])
        # Target: full_sequence[1:] (predict input[0]...EOS)
        input_ids = full_sequence[:-1]
        target_ids = full_sequence[1:]
        
        # Pad sequences to max_seq_len
        seq_len = len(input_ids)
        padding_len = self.max_seq_len - seq_len
        
        if padding_len > 0:
            # Pad with PAD token
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            target_ids = target_ids + [self.tokenizer.pad_token_id] * padding_len
        elif padding_len < 0:
            # Truncate if too long (shouldn't happen for k=3)
            input_ids = input_ids[:self.max_seq_len]
            target_ids = target_ids[:self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * seq_len + [0] * max(0, padding_len)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }



# DataLoader Creation


def create_dataloaders(
    tokenizer: Tokenizer,
    max_seq_len: int = 32
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = AdditionDataset(
        data_path=DataConfig.TRAIN_FILE,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    
    val_dataset = AdditionDataset(
        data_path=DataConfig.VAL_FILE,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 for compatibility
        pin_memory=True if TrainConfig.DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if TrainConfig.DEVICE == "cuda" else False
    )
    
    return train_loader, val_loader


# Training Functions


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The transformer model
        dataloader: Training DataLoader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar for training
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in pbar:
        # Move tensors to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, loss = model(input_ids, target_ids)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    tokenizer: Tokenizer
) -> tuple[float, float]:
    """
    Evaluate the model on validation data.
    
    Args:
        model: The transformer model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to evaluate on
        tokenizer: Tokenizer for decoding predictions
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        for batch in pbar:
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits, loss = model(input_ids, target_ids)
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy (exact match on non-padded positions)
            predictions = logits.argmax(dim=-1)
            
            # Compare predictions to targets for each sequence
            for i in range(predictions.size(0)):
                # Get the actual sequence length (excluding padding)
                seq_len = attention_mask[i].sum().item()
                
                # Compare predictions to targets (excluding padding)
                pred_seq = predictions[i, :seq_len]
                target_seq = target_ids[i, :seq_len]
                
                # Check if entire sequence matches (exact match)
                if torch.equal(pred_seq, target_seq):
                    correct += 1
                total += 1
    
    avg_loss = total_loss / num_batches
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def compute_answer_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    tokenizer: Tokenizer,
    num_samples: int = 500
) -> float:
    """
    Compute accuracy specifically on the answer portion using generation.
    
    This is a more meaningful metric - does the model produce the correct answer?
    
    Args:
        model: The transformer model
        dataloader: DataLoader to sample from
        device: Device to use
        tokenizer: Tokenizer instance
        num_samples: Number of samples to evaluate
        
    Returns:
        Answer accuracy (0.0 to 1.0)
    """
    model.eval()
    correct = 0
    total = 0
    
    # Get underlying dataset
    dataset = dataloader.dataset
    
    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    with torch.no_grad():
        for idx in indices:
            example = dataset.data[idx]
            input_str = example['input']  # e.g., "123+456="
            expected_output = example['output']  # e.g., "579"
            
            # Prepare input: BOS + input_str
            input_ids = [tokenizer.bos_token_id] + tokenizer.encode(input_str)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            # Generate output
            max_new_tokens = len(expected_output) + 2  # +2 for safety margin
            generated = model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                greedy=True
            )
            
            # Decode generated sequence
            generated_tokens = generated[0].tolist()
            
            # Extract the answer portion (after the = sign)
            try:
                # Find where the input ends in the generated sequence
                input_len = len(input_ids)
                answer_tokens = generated_tokens[input_len:]
                
                # Remove EOS if present
                if tokenizer.eos_token_id in answer_tokens:
                    eos_idx = answer_tokens.index(tokenizer.eos_token_id)
                    answer_tokens = answer_tokens[:eos_idx]
                
                # Decode answer
                predicted_answer = tokenizer.decode(answer_tokens)
                
                # Check if correct
                if predicted_answer == expected_output:
                    correct += 1
            except Exception:
                pass  # Decoding error, count as incorrect
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy



# Main Training Function


def train():
    """
    Main training function.
    
    Handles the complete training pipeline:
    1. Setup (seeds, configs, device)
    2. Data loading/generation
    3. Model initialization
    4. Training loop with validation
    5. Checkpointing and early stopping
    """
    print("=" * 60)
    print("ADDITION TRANSFORMER TRAINING")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    
    # Set random seeds for reproducibility
    random.seed(TrainConfig.SEED)
    torch.manual_seed(TrainConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TrainConfig.SEED)
    
    # Initialize configurations
    task_config = TaskConfig()
    model_config = ModelConfig(task_config)
    
    # Device setup
    device = TrainConfig.DEVICE
    print(f"\n[Setup]")
    print(f"  Device: {device}")
    print(f"  Random seed: {TrainConfig.SEED}")
    
    # Create tokenizer
    tokenizer = Tokenizer(task_config)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    

    # Data Loading/Generation

    
    print(f"\n[Data]")
    
    # Check if datasets exist, otherwise generate them
    if not os.path.exists(DataConfig.TRAIN_FILE) or not os.path.exists(DataConfig.VAL_FILE):
        print("  Datasets not found. Generating...")
        generate_all_datasets()
    else:
        print(f"  Found existing datasets in {DataConfig.DATA_DIR}/")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        max_seq_len=model_config.MAX_SEQ_LEN
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    

    # Model Initialization

    
    print(f"\n[Model]")
    
    # Create model
    model = AdditionTransformer(model_config, task_config)
    model = model.to(device)
    
    # Print parameter count
    num_params = model.count_parameters()
    print(f"  Parameters: {num_params:,}")
    print(f"  Architecture: {model_config.N_LAYERS} layers, d_model={model_config.D_MODEL}")
    
    # -------------------------------------------------------------------------
    # Training Setup
    # -------------------------------------------------------------------------
    
    print(f"\n[Training Setup]")
    
    # Optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY
    )
    print(f"  Optimizer: AdamW (lr={TrainConfig.LEARNING_RATE}, wd={TrainConfig.WEIGHT_DECAY})")
    
    # Learning rate scheduler (Cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TrainConfig.MAX_EPOCHS,
        eta_min=TrainConfig.LEARNING_RATE / 10
    )
    print(f"  Scheduler: CosineAnnealingLR (T_max={TrainConfig.MAX_EPOCHS})")
    
    # Loss function (Cross-entropy, ignoring padding)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print(f"  Loss: CrossEntropyLoss (ignore_index={tokenizer.pad_token_id})")
    
    # Ensure checkpoint directory exists
    TrainConfig.ensure_save_dir()
    

    # Training Loop

    
    print(f"\n[Training]")
    print(f"  Max epochs: {TrainConfig.MAX_EPOCHS}")
    print(f"  Early stopping patience: {TrainConfig.PATIENCE}")
    print("-" * 60)
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'answer_accuracy': [],
        'learning_rate': []
    }
    
    try:
        for epoch in range(1, TrainConfig.MAX_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{TrainConfig.MAX_EPOCHS}")
            
            # Train one epoch
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Evaluate on validation set
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, tokenizer)
            
            # Compute answer accuracy (more meaningful metric)
            answer_acc = compute_answer_accuracy(
                model, val_loader, device, tokenizer, num_samples=500
            )
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['answer_accuracy'].append(answer_acc)
            history['learning_rate'].append(current_lr)
            
            # Print epoch stats
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2%} | Answer Acc: {answer_acc:.2%}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model checkpoint
                checkpoint_path = os.path.join(TrainConfig.SAVE_DIR, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'answer_accuracy': answer_acc,
                    'config': {
                        'model_config': {
                            'N_LAYERS': model_config.N_LAYERS,
                            'D_MODEL': model_config.D_MODEL,
                            'N_HEADS': model_config.N_HEADS,
                            'D_FF': model_config.D_FF,
                            'DROPOUT': model_config.DROPOUT,
                            'MAX_SEQ_LEN': model_config.MAX_SEQ_LEN,
                            'VOCAB_SIZE': model_config.VOCAB_SIZE
                        }
                    }
                }, checkpoint_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{TrainConfig.PATIENCE})")
            
            # Early stopping check
            if patience_counter >= TrainConfig.PATIENCE:
                print(f"\n  Early stopping triggered after {epoch} epochs")
                break
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current state...")
    

    # Save Final Results

    
    print("\n" + "-" * 60)
    print("[Saving Results]")
    
    # Save final model
    final_path = os.path.join(TrainConfig.SAVE_DIR, "final_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }, final_path)
    print(f"  Saved final model to {final_path}")
    
    # Save training history
    history_path = os.path.join(TrainConfig.SAVE_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history to {history_path}")
    

    # Training Summary

    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Total epochs: {len(history['train_loss'])}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final validation accuracy: {history['val_accuracy'][-1]:.2%}")
    print(f"  Final answer accuracy: {history['answer_accuracy'][-1]:.2%}")
    print(f"\n  Checkpoints saved to: {TrainConfig.SAVE_DIR}/")
    print("=" * 60)
    
    return model, history


# Entry Point


if __name__ == "__main__":
    train()

