# GritSense Temporal Sequence Generation

This repository implements a temporal modeling pipeline for volleyball activity classification and forecasting.

The project now includes:

- feature preparation for heterogeneous per-frame blocks
- chronological sequence generation from frame-level features
- a GRU-based temporal classifier
- a transformer baseline for comparison
- a lightweight training and evaluation loop
- synthetic example scripts and tests

## Pipeline overview

### 1. Task 2: feature preparation

`FeaturePreparer` accepts a mapping of named blocks and concatenates them into a single feature vector per frame.

Default stable order:

1. `player`
2. `pose`
3. `ball`
4. `spatial`

Only supplied blocks are included, in that order. Each supplied block must be shaped `[num_frames, block_feature_dim]`, and all supplied blocks must share the same frame count.

The prepared output is shaped `[num_frames, feature_dim]`, where `feature_dim` is the sum of the provided block widths.

Validation errors are raised for:

- empty input
- unknown feature block names
- conflicting frame counts
- inconsistent feature dimensions
- non-numeric values
- NaN or Inf values

`FeatureStandardizer` can optionally fit on training data and transform validation, test, or inference data.

### 2. Task 1: sequence generation

`TemporalSequenceGenerator(sequence_length=16, stride=1)` creates fixed-length chronological windows.

The output shape is:

```python
[num_sequences, sequence_length, feature_dim]
```

It supports:

- a single feature matrix
- a mapping of group IDs to feature matrices
- `group_ids` with a single matrix to keep groups separate

A window never crosses a group boundary.

### 3. Task 3: GRU temporal model

`GRUTemporalModel` classifies each sequence using a PyTorch GRU.

It accepts floating-point tensor input shaped:

```python
[batch_size, sequence_length, input_dim]
```

It returns raw logits shaped:

```python
[batch_size, num_classes]
```

It also provides `predict()`, which returns:

- class indices
- class labels
- probabilities
- confidence scores

### 4. Transformer baseline

`TransformerTemporalModel` is a lightweight attention-based alternative for sequence classification.

It uses:

- input projection
- positional encoding
- stacked transformer encoder layers
- mean pooled sequence representation
- a classification head

This gives a simple baseline to compare against the GRU.

### 5. Training and evaluation

`TemporalTrainer` provides a minimal training loop with:

- optimizer setup
- batch training
- evaluation
- accuracy and macro-F1 reporting

This supports development and demo workflows before real data is available.

## Example usage

### Synthetic sequence generation

```powershell
cd C:\Users\yashk\OneDrive\Desktop\cappy_me
& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python examples/mock_sequence_generation.py
```

Example output:

```text
Input shape: [20, 4]
Output shape: [9, 4, 4]
First sequence: [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
```

### Synthetic training demo

```powershell
cd C:\Users\yashk\OneDrive\Desktop\cappy_me
& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python examples/training_demo.py
```

Example output:

```text
Training history:
  loss: [0.699481412768364, 0.6956900358200073, 0.6929677575826645, 0.6901804059743881, 0.6872852742671967]
  accuracy: [0.484375, 0.46875, 0.5, 0.515625, 0.515625]

Evaluation:
  accuracy: 0.5469
  macro_f1: 0.4797
  loss: 0.6855
```

## Tests

```powershell
cd C:\Users\yashk\OneDrive\Desktop\cappy_me
& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Verified result:

```text
Ran 29 tests in 1.042s

OK
```

## Notes

- The code is designed to accept future per-frame outputs from GNN/HRN stages without hardcoding feature dimensions.
- Synthetic examples are for development and verification only.
- Real volleyball datasets are still required for production forecasting accuracy evaluations.

The eventual GNN/HRN output can be passed directly as the matrix input once it produces one feature vector per processed frame, regardless of whether its feature dimension is 64, 128, 256, or another value.
