"""Prepare heterogeneous per-frame features for temporal modeling."""

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any


DEFAULT_FEATURE_ORDER = ("player", "pose", "ball", "spatial")


class FeatureStandardizer:
    """Optionally standardize feature columns using training data statistics."""

    def __init__(self, epsilon: float = 1e-12) -> None:
        if not isinstance(epsilon, Real) or isinstance(epsilon, bool) or epsilon <= 0:
            raise ValueError("epsilon must be a positive number.")
        self.epsilon = float(epsilon)
        self._means: list[float] | None = None
        self._scales: list[float] | None = None

    @property
    def fitted(self) -> bool:
        return self._means is not None

    def fit(self, features: Sequence[Sequence[Real]]) -> "FeatureStandardizer":
        """Fit column statistics on training features only."""
        rows = _validate_matrix(features, "training features")
        feature_dim = len(rows[0])
        self._means = [sum(row[index] for row in rows) / len(rows) for index in range(feature_dim)]
        self._scales = []
        for index, mean in enumerate(self._means):
            variance = sum((row[index] - mean) ** 2 for row in rows) / len(rows)
            self._scales.append(max(variance**0.5, self.epsilon))
        return self

    def transform(self, features: Sequence[Sequence[Real]]) -> list[list[float]]:
        """Transform validation, test, or inference features with fitted statistics."""
        if not self.fitted:
            raise ValueError("FeatureStandardizer must be fitted before transform().")
        rows = _validate_matrix(features, "features")
        assert self._means is not None
        assert self._scales is not None
        if len(rows[0]) != len(self._means):
            raise ValueError(
                f"Feature dimension changed: expected {len(self._means)}, "
                f"got {len(rows[0])}."
            )
        return [
            [
                (float(value) - self._means[index]) / self._scales[index]
                for index, value in enumerate(row)
            ]
            for row in rows
        ]


class FeaturePreparer:
    """Combine named per-frame feature blocks into one temporal feature matrix.

    Blocks are concatenated in ``feature_order``. Each block must be shaped
    ``[num_frames, block_feature_dim]``. Missing blocks are omitted; this makes
    partial development inputs possible while retaining deterministic ordering.
    """

    def __init__(
        self,
        feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER,
        standardizer: FeatureStandardizer | None = None,
    ) -> None:
        order = tuple(feature_order)
        if not order or any(not isinstance(name, str) or not name for name in order):
            raise ValueError("feature_order must contain at least one non-empty name.")
        if len(set(order)) != len(order):
            raise ValueError("feature_order must not contain duplicate names.")
        self.feature_order = order
        self.standardizer = standardizer

    def prepare(self, feature_blocks: Mapping[str, Any]) -> list[list[Any]]:
        """Concatenate available blocks in the configured stable order."""
        if not isinstance(feature_blocks, Mapping) or not feature_blocks:
            raise ValueError("feature_blocks must be a non-empty mapping.")
        unknown = set(feature_blocks) - set(self.feature_order)
        if unknown:
            raise ValueError(f"Unknown feature block(s): {sorted(unknown)!r}.")

        prepared_blocks: list[list[list[Any]]] = []
        frame_count: int | None = None
        for block_name in self.feature_order:
            if block_name not in feature_blocks:
                continue
            block = _validate_matrix(feature_blocks[block_name], f"{block_name} block")
            if frame_count is None:
                frame_count = len(block)
            elif len(block) != frame_count:
                raise ValueError(
                    f"Frame count mismatch: {block_name!r} has {len(block)} frames; "
                    f"expected {frame_count}."
                )
            prepared_blocks.append(block)

        if not prepared_blocks:
            raise ValueError("At least one configured feature block is required.")
        combined = [
            [value for block in prepared_blocks for value in block[frame_index]]
            for frame_index in range(frame_count or 0)
        ]
        if self.standardizer is not None and self.standardizer.fitted:
            return self.standardizer.transform(combined)
        return combined

    def prepare_matrix(self, frame_features: Any) -> list[list[Any]]:
        """Validate a source-independent matrix, such as future HRN output."""
        matrix = _validate_matrix(frame_features, "frame features")
        if self.standardizer is not None and self.standardizer.fitted:
            return self.standardizer.transform(matrix)
        return matrix


def _validate_matrix(features: Any, description: str) -> list[list[Any]]:
    if isinstance(features, (str, bytes)):
        raise ValueError(f"{description} must have shape [num_frames, feature_dim].")
    try:
        rows = list(features)
    except TypeError as error:
        raise ValueError(f"{description} must have shape [num_frames, feature_dim].") from error
    if not rows:
        raise ValueError(f"{description} cannot be empty.")

    normalized: list[list[Any]] = []
    feature_dim: int | None = None
    for frame_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(f"{description} frame {frame_index} is not a feature vector.")
        try:
            vector = list(row)
        except TypeError as error:
            raise ValueError(f"{description} frame {frame_index} is not a feature vector.") from error
        if not vector:
            raise ValueError(f"{description} frame {frame_index} has no features.")
        if feature_dim is None:
            feature_dim = len(vector)
        elif len(vector) != feature_dim:
            raise ValueError(
                f"Inconsistent feature dimensions in {description}: expected "
                f"{feature_dim}, got {len(vector)} at frame {frame_index}."
            )
        for value in vector:
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"{description} contains a non-numeric value at frame {frame_index}.")
            if not isfinite(float(value)):
                raise ValueError(f"{description} contains NaN or Inf at frame {frame_index}.")
        normalized.append(vector)
    return normalized