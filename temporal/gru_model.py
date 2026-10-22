"""Temporal models for volleyball activity classification and forecasting."""

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


ACTIVITY_CLASSES = (
    "left-pass",
    "left-spike",
    "right-set",
    "left-set",
    "right-spike",
    "right-pass",
    "left-winpoint",
    "right-winpoint",
)


class GRUTemporalModel(nn.Module):
    """Classify fixed-length feature sequences with a GRU.

    The classifier uses the final hidden state from the last GRU layer. For a
    bidirectional GRU, the final forward and backward states are concatenated
    before the linear classification head. The forward method returns raw
    logits; probabilities are produced only by :meth:`predict`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = len(ACTIVITY_CLASSES),
        bidirectional: bool = False,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_classes, "num_classes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in the range [0, 1); got {dropout!r}.")
        if not isinstance(bidirectional, bool):
            raise TypeError("bidirectional must be a boolean.")

        names = tuple(ACTIVITY_CLASSES if class_names is None else class_names)
        if len(names) != num_classes or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("class_names must contain one non-empty name per class.")
        if len(set(names)) != len(names):
            raise ValueError("class_names must not contain duplicate names.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.class_names = names
        direction_count = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.classifier = nn.Linear(hidden_dim * direction_count, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return raw class logits for ``[batch, sequence, input_dim]`` inputs."""
        self._validate_inputs(inputs)
        _, hidden = self.gru(inputs)
        if self.bidirectional:
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            representation = hidden[-1]
        return self.classifier(representation)

    @torch.inference_mode()
    def predict(self, inputs: Tensor) -> dict[str, Any]:
        """Return predicted indices, labels, probabilities, and confidence."""
        logits = self(inputs)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        labels = [self.class_names[index] for index in indices.tolist()]
        return {
            "class_indices": indices,
            "class_labels": labels,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    def _validate_inputs(self, inputs: Tensor) -> None:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor.")
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch_size, sequence_length, input_dim]; "
                f"got {tuple(inputs.shape)}."
            )
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("inputs must contain at least one batch item and one frame.")
        if inputs.shape[2] != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: model expects {self.input_dim}, "
                f"got {inputs.shape[2]}."
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype.")


class TransformerTemporalModel(nn.Module):
    """Transformer-based temporal baseline for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = len(ACTIVITY_CLASSES),
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        for value, name in (
            (input_dim, "input_dim"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (num_classes, "num_classes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in the range [0, 1); got {dropout!r}.")

        names = tuple(ACTIVITY_CLASSES if class_names is None else class_names)
        if len(names) != num_classes or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("class_names must contain one non-empty name per class.")
        if len(set(names)) != len(names):
            raise ValueError("class_names must not contain duplicate names.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.class_names = names

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=float(dropout),
                activation="relu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return raw class logits for ``[batch, sequence, input_dim]`` inputs."""
        self._validate_inputs(inputs)
        projected = self.input_projection(inputs)
        positions = self._build_positions(inputs.shape[1], inputs.device, inputs.dtype)
        encoded = self.transformer(projected + positions)
        representation = encoded.mean(dim=1)
        return self.classifier(representation)

    @torch.inference_mode()
    def predict(self, inputs: Tensor) -> dict[str, Any]:
        logits = self(inputs)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, indices = probabilities.max(dim=-1)
        labels = [self.class_names[index] for index in indices.tolist()]
        return {
            "class_indices": indices,
            "class_labels": labels,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    @staticmethod
    def _build_positions(sequence_length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 1, 2, device=device, dtype=dtype) * -(math.log(10000.0) / 1))
        pos = positions / (10000.0 ** (torch.arange(0, 1, 2, device=device, dtype=dtype) / 1))
        encoding = torch.zeros(sequence_length, 1, device=device, dtype=dtype)
        encoding[:, 0] = pos[:, 0]
        return encoding.unsqueeze(0)

    def _validate_inputs(self, inputs: Tensor) -> None:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs must be a torch.Tensor.")
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape [batch_size, sequence_length, input_dim]; "
                f"got {tuple(inputs.shape)}."
            )
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("inputs must contain at least one batch item and one frame.")
        if inputs.shape[2] != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: model expects {self.input_dim}, "
                f"got {inputs.shape[2]}."
            )
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype.")


class TemporalTrainer:
    """Simple training and evaluation loop for temporal classification models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion or nn.CrossEntropyLoss()

    def fit(
        self,
        sequences: Tensor,
        targets: Tensor,
        epochs: int = 10,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> dict[str, list[float]]:
        self._validate_training_inputs(sequences, targets)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        dataset = TensorDataset(sequences.to(self.device), targets.to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        history: dict[str, list[float]] = {"loss": [], "accuracy": [], "macro_f1": []}

        for _ in range(epochs):
            self.model.train()
            running_loss = 0.0
            predictions: list[Tensor] = []
            labels: list[Tensor] = []

            for batch_sequences, batch_targets in loader:
                self.optimizer.zero_grad()
                logits = self.model(batch_sequences)
                loss = self.criterion(logits, batch_targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_targets.size(0)
                predictions.append(logits.argmax(dim=-1).detach().cpu())
                labels.append(batch_targets.detach().cpu())

            avg_loss = running_loss / len(dataset)
            pred_tensor = torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)
            label_tensor = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
            metrics = _compute_classification_metrics(pred_tensor, label_tensor)
            history["loss"].append(float(avg_loss))
            history["accuracy"].append(float(metrics["accuracy"]))
            history["macro_f1"].append(float(metrics["macro_f1"]))

        return history

    def evaluate(self, sequences: Tensor, targets: Tensor, batch_size: int | None = None) -> dict[str, float]:
        self._validate_training_inputs(sequences, targets)
        if batch_size is None:
            batch_size = max(1, min(len(targets), 32))
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        dataset = TensorDataset(sequences.to(self.device), targets.to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        losses: list[float] = []
        predictions: list[Tensor] = []
        labels: list[Tensor] = []

        with torch.inference_mode():
            for batch_sequences, batch_targets in loader:
                logits = self.model(batch_sequences)
                loss = self.criterion(logits, batch_targets)
                losses.append(float(loss.item()) * batch_targets.size(0))
                predictions.append(logits.argmax(dim=-1).detach().cpu())
                labels.append(batch_targets.detach().cpu())

        pred_tensor = torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)
        label_tensor = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
        metrics = _compute_classification_metrics(pred_tensor, label_tensor)
        metrics["loss"] = sum(losses) / max(len(targets), 1)
        return metrics

    @staticmethod
    def _validate_training_inputs(sequences: Tensor, targets: Tensor) -> None:
        if not isinstance(sequences, Tensor):
            raise TypeError("sequences must be a torch.Tensor.")
        if not isinstance(targets, Tensor):
            raise TypeError("targets must be a torch.Tensor.")
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape [batch_size, sequence_length, input_dim].")
        if targets.ndim != 1:
            raise ValueError("targets must be a 1D tensor of class indices.")
        if sequences.shape[0] != targets.shape[0]:
            raise ValueError("sequences and targets must contain the same number of samples.")
        if sequences.shape[0] == 0:
            raise ValueError("training data cannot be empty.")
        if not sequences.is_floating_point():
            raise TypeError("sequences must use a floating-point dtype.")
        if targets.dtype not in (torch.int64, torch.int32, torch.long):
            raise TypeError("targets must be integer class labels.")


def _compute_classification_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    if predictions.numel() == 0 or targets.numel() == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    num_classes = max(int(targets.max().item()) + 1, int(predictions.max().item()) + 1)
    accuracy = float((predictions == targets).float().mean().item())

    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = ((targets == class_index) & (predictions == class_index)).sum().item()
        false_positive = ((targets != class_index) & (predictions == class_index)).sum().item()
        false_negative = ((targets == class_index) & (predictions != class_index)).sum().item()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1_score)

    macro_f1 = float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}
