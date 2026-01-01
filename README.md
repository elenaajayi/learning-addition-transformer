
# Learning Addition with Transformers

This is a work task project investigating whether transformers learn arithmetic algorithms or memorize positional patterns through experiments on k-digit addition. This ReadMe was authored with Cursor Pro using the Claude's Sonnet-4.5 Model. 

## Key Question

Does a transformer trained on 3-digit addition generalize to 4-digit and 5-digit addition?

## Results Summary

| Experiment | k=3 | k=4 | k=5 | Key Finding |
|------------|-----|-----|-----|-------------|
| **Baseline** (k=3 only) | 95.1% | 0.0% | 0.0% | Positional memorization |
| **Mixed** (k=3+k=4) | 87.6% | 78.9% | 0.0% | Bounded generalization |

### Major Findings

1. **Positional Memorization**: Training only on k=3 results in 0% accuracy on k=4, revealing the model learned fixed positional patterns rather than the addition algorithm.

2. **Mixed Training Success**: Training on both k=3 and k=4 achieves 78.9% on k=4 (vs baseline 0%), successfully breaking rigid positional constraints.

3. **Bounded Generalization**: Despite k=4 success, the model still fails on k=5 (0% accuracy), suggesting it learned "flexible positions within k=3,4" rather than the true algorithm.

## Architecture

- **Model**: Decoder-only Transformer
- **Layers**: 6
- **Hidden Size**: 256
- **Attention Heads**: 8
- **Parameters**: ~4.7M
- **Tokenization**: Character-level (digits 0-9, +, =, special tokens)

## Experiments

### 1. Baseline (k=3 Only Training)
**Location**: `experiments/baseline_k3_only/`

Trained exclusively on 50,000 3-digit addition problems. Achieved high accuracy on k=3 but completely failed on k=4 and k=5, demonstrating positional memorization.

[View detailed results →](experiments/baseline_k3_only/)

### 2. Mixed Training (k=3 + k=4)
**Location**: `experiments/mixed_k3_k4/`

Trained on 25,000 k=3 + 25,000 k=4 problems. Successfully generalized to k=4 (78.9%) but still failed on k=5, revealing bounded generalization.

[View detailed results →](experiments/mixed_k3_k4/)

## Quick Start

### Setup
```bash
git clone https://github.com/elenaajayi/learning-addition-transformer.git
cd learning-addition-transformer
uv pip install -r requirements.txt
```

### Generate Data
```bash
python data_generator.py
```

### Train Models
```bash
# Baseline (k=3 only)
python train.py  # Uses data/train.json (k=3)

# Mixed (k=3+k=4) 
python train.py  # Uses data/train_mixed.json (k=3+k=4)
```

### Evaluate
```bash
python evaluate.py --model_path checkpoints/best_model.pt
```

### Interactive Testing
```bash
python evaluate.py --interactive
# Try: 123+456=
```

## Training Details

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR
- **Loss**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs**: 50
- **Early Stopping**: Patience 5

## Files

```
.
├── model.py              # Transformer architecture
├── data_generator.py     # Synthetic dataset generation
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation and testing
├── config.py             # Hyperparameters
├── experiments/          # Saved experiment results
│   ├── baseline_k3_only/
│   └── mixed_k3_k4/
├── data/                 # Generated datasets (not in git)
├── checkpoints/          # Model weights (not in git)
└── results/              # Evaluation results
```

## Key Insights

### Why Baseline didn't Generalize
When trained only on k=3:
- Model learns: "answer always starts at position 8"
- Attention focuses on fixed positions, not semantic structure
- No incentive to learn generalizable carry propagation

### Why Mixed Training Helped (Partially)
Training on k=3 and k=4:
- Variable lengths break fixed positional patterns  
- Model must attend to the `=` symbol, not absolute positions
- Forces more generalizable representations

### Why k=5 Still Fails
- Model learned "flexible positions within k=3,4 range"
- Outputs 2-3 digit answers for 5-digit problems
- Hasn't abstracted the full algorithm, just expanded its memorization

## Implications

This work demonstrates a fundamental challenge in teaching transformers arithmetic:
- **Easy**: Memorize patterns for specific lengths
- **Medium**: Handle variable lengths within training range  
- **Hard**: Learn the true algorithm that generalizes beyond training

Potential solutions to explore:
- Curriculum learning (gradually increase length)
- Training on k=3,4,5 simultaneously
- Different positional encodings (relative, learned)
- Explicit carry mechanism in architecture

## License

MIT License

## Author

Elena Ajayi - [GitHub](https://github.com/elenaajayi)

