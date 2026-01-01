# Baseline Experiment: k=3 Only Training

Trained on 50K k=3 examples only.

## Results Summary
- **k=3 Test Accuracy**: 95.1% (in-distribution)
- **k=4 Test Accuracy**: 0.0% (length generalization failure)
- **k=5 Test Accuracy**: 0.0% (length generalization failure)

## Key Findings
The model achieves excellent performance on k=3 addition problems but completely fails to generalize to longer sequences (k=4, k=5). This demonstrates that the model learned position-specific patterns rather than the underlying addition algorithm.

## Files
- `checkpoints/`: Model weights
  - `best_model.pt`: Best checkpoint based on validation loss
  - `final_model.pt`: Final checkpoint after all epochs
  - `training_history.json`: Training and validation loss curves
- `results.json`: Detailed evaluation metrics across all test sets
- `train.json`: Training data (50K k=3 examples)
- `val.json`: Validation data (5K k=3 examples)

## Training Configuration
- Architecture: Decoder-only transformer
- Parameters: ~470K
- Training examples: 50,000 (k=3 only)
- Validation examples: 5,000 (k=3 only)
- Epochs: 50
- Batch size: 128
