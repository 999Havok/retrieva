## Flat RAG vs RAPTOR on HotpotQA (N=200, model=qwen3:8b)

### Retrieval recall (%)

| Budget | System | Strict recall | Both found (strict) | Lenient recall | Both found (lenient) | Avg nodes | Avg tokens |
|---|---|---|---|---|---|---|---|
| 750 | flat | 79.5 | 61.5 | - | - | 6.8 | 743 |
| 750 | raptor | 74.5 | 53.0 | 83.8 | 69.0 | 6.3 | 743 |
| 1500 | flat | 85.5 | 72.5 | - | - | 12.2 | 1493 |
| 1500 | raptor | 84.0 | 70.0 | 90.5 | 81.5 | 11.2 | 1492 |
| 3000 | flat | 90.2 | 82.0 | - | - | 23.1 | 2993 |
| 3000 | raptor | 89.0 | 79.5 | 92.2 | 85.0 | 21.1 | 2993 |

### Answers and recall by question type (budget 1500 tokens, %)

| System | Type | n | EM | F1 | Strict recall | Both found (strict) | Lenient recall | Avg answer s |
|---|---|---|---|---|---|---|---|---|
| flat | all | 200 | 49.0 | 58.8 | 85.5 | 72.5 | - | 5.7 |
| flat | bridge | 164 | 45.1 | 56.9 | 83.2 | 68.3 | - | 5.7 |
| flat | comparison | 36 | 66.7 | 67.4 | 95.8 | 91.7 | - | 5.6 |
| raptor | all | 200 | 48.0 | 58.6 | 84.0 | 70.0 | 90.5 | 4.2 |
| raptor | bridge | 164 | 43.3 | 55.9 | 81.4 | 65.2 | 89.3 | 4.1 |
| raptor | comparison | 36 | 69.4 | 70.8 | 95.8 | 91.7 | 95.8 | 4.7 |

### RAPTOR minus flat, paired bootstrap (budget 1500, 1000 resamples, seed 42, percentage points)

Contexts differ on 183/200 questions; answer strings differ on 39/200.

| Metric | Flat | RAPTOR | Difference | 95% CI |
|---|---|---|---|---|
| EM | 49.0 | 48.0 | -1.0 | [-5.0, +3.0] |
| F1 | 58.8 | 58.6 | -0.2 | [-4.0, +3.3] |
| Strict recall | 85.5 | 84.0 | -1.5 | [-3.0, -0.2] |

### RAPTOR retrieved node levels (% of retrieved nodes)

| Budget | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| 750 | 86.5 | 12.7 | 0.5 | 0.2 |
| 1500 | 85.4 | 13.5 | 0.9 | 0.2 |
| 3000 | 86.9 | 11.8 | 1.1 | 0.2 |

### RAPTOR build cost

- Nodes per level: {0: 1977, 1: 175, 2: 13, 3: 3}
- Summarization calls: 191
- Tree build time: 57.8 min (summarization: 57.5 min, 18.0 s/call)
- Evaluation time: 33.2 min for 200 questions
