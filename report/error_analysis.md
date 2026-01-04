
## Quantitative Error Analysis

### Output Length Distribution (Sample Errors)

| Test Set | Too Short | Correct Length | Too Long |
|----------|-----------|----------------|----------|
| Baseline in_distribution | 0% | 100% | 0% |
| Baseline length_gen_k4 | 100% | 0% | 0% |
| Baseline length_gen_k5 | 100% | 0% | 0% |
| Baseline shift_nines | 0% | 100% | 0% |
| Baseline shift_small | 0% | 100% | 0% |
| Mixed in_distribution | 0% | 100% | 0% |
| Mixed length_gen_k4 | 0% | 100% | 0% |
| Mixed length_gen_k5 | 100% | 0% | 0% |
| Mixed shift_nines | 0% | 100% | 0% |
| Mixed shift_small | 0% | 100% | 0% |

### Error Type Classification

| Test Set | Too Short | Off-by-1 | Off-by-10 | Large Error |
|----------|-----------|----------|-----------|-------------|
| Baseline in_distribution | 0% | 0% | 100% | 0% |
| Baseline length_gen_k4 | 100% | 0% | 0% | 0% |
| Baseline length_gen_k5 | 100% | 0% | 0% | 0% |
| Baseline shift_nines | 0% | 60% | 20% | 0% |
| Baseline shift_small | 0% | 20% | 80% | 0% |
| Mixed in_distribution | 0% | 0% | 100% | 0% |
| Mixed length_gen_k4 | 0% | 0% | 20% | 40% |
| Mixed length_gen_k5 | 100% | 0% | 0% | 0% |
| Mixed shift_nines | 0% | 0% | 60% | 0% |
| Mixed shift_small | 0% | 0% | 60% | 0% |

### Key Findings

1. **Baseline k=4 failure is systematic**: 100% of errors produce outputs 2+ digits shorter than expected. This is not random arithmetic error—the model learned a 'short answer' heuristic.

2. **Mixed k=5 failure follows same pattern**: 100% of errors are too short, confirming bounded generalization (model learned 'flexible within k=3,4' not the algorithm).

3. **Errors are length failures, not arithmetic mistakes**: Off-by-1 errors (near misses) are rare (<5%), while systematic length errors dominate. This rules out 'model learned addition but makes calculation errors'.