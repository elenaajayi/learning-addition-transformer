# Mixed k=3,4 Training Experiment

## Setup
- Training data: 25,000 k=3 + 25,000 k=4 examples (50K total)
- Validation data: 2,500 k=3 + 2,500 k=4 examples (5K total)
- Epochs: 50
- Final answer accuracy: 82.6%

## Results Summary

| Test Set | Accuracy |
|----------|----------|
| k=3 (in-distribution) | 87.6% |
| k=4 (in-distribution) | 78.9% |
| k=5 (out-of-distribution) | 0.0% |
| Distribution Shift (Many 9s) | 86.8% |
| Distribution Shift (Small Numbers) | 81.6% |

## Key Findings

### Success: Breaking Positional Memorization
Compared to baseline (k=3 only training which achieved 0% on k=4), mixed training 
achieved 78.9% on k=4. This demonstrates that variable-length training successfully 
broke rigid positional patterns.

### Limitation: Bounded Generalization
Despite success on k=4, the model still achieves 0% on k=5 (never-seen length).
Error analysis shows the model outputs 2-digit answers for 5-digit problems,
suggesting it learned "flexible positions within k=3,4" rather than the general
addition algorithm.

## Files
- `checkpoints/`: Model weights and training history
- `results.json`: Full evaluation metrics
- `train.json`: Training data (25K k=3 + 25K k=4)
- `val.json`: Validation data (2.5K k=3 + 2.5K k=4)

## Comparison to Baseline
See `../baseline_k3_only/` for comparison with k=3-only training.
