"""Generate fixed-length chronological windows from frame features."""

from collections.abc import Mapping, Sequence
from typing import Any


class TemporalSequenceGenerator:
    """Create fixed-length windows from per-frame feature vectors.

    ``features`` may be one frame-feature matrix or a mapping of video/rally
    identifiers to frame-feature matrices. When using one matrix, ``group_ids``
    can identify the video or rally for each frame.
    """

    def __init__(self, sequence_length: int = 16, stride: int = 1) -> None:
        self.sequence_length = self._validate_positive_integer(
            sequence_length, "sequence_length"
        )
        self.stride = self._validate_positive_integer(stride, "stride")

    def generate_sequences(
        self,
        features: Any,
        sequence_length: int | None = None,
        stride: int | None = None,
        group_ids: Sequence[Any] | None = None,
    ) -> list[list[list[Any]]]:
        """Return chronological windows from frame-level feature data.

        Args:
            features: A matrix shaped ``[num_frames, feature_dim]`` or a
                mapping from video/rally IDs to such matrices.
            sequence_length: Number of frames in each output window. Defaults
                to the value configured on this generator.
            stride: Number of frames between successive window starts. Defaults
                to the configured stride.
            group_ids: Optional group ID per frame for a single matrix.

        Raises:
            ValueError: If the input is empty, malformed, inconsistent, or a
                group does not contain enough frames for one window.
            TypeError: If a parameter has the wrong type.
        """
        length = self._validate_positive_integer(
            self.sequence_length if sequence_length is None else sequence_length,
            "sequence_length",
        )
        step = self._validate_positive_integer(
            self.stride if stride is None else stride, "stride"
        )

        groups = self._split_groups(features, group_ids)
        sequences: list[list[list[Any]]] = []
        for group_name, group_features in groups:
            rows = self._validate_feature_matrix(group_features, group_name)
            if len(rows) < length:
                raise ValueError(
                    f"Group {group_name!r} has {len(rows)} frames; "
                    f"at least {length} are required for a sequence."
                )

            for start in range(0, len(rows) - length + 1, step):
                sequences.append([row[:] for row in rows[start : start + length]])

        return sequences

    @staticmethod
    def _validate_positive_integer(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer; got {value!r}.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0; got {value}.")
        return value

    @staticmethod
    def _split_groups(
        features: Any, group_ids: Sequence[Any] | None
    ) -> list[tuple[Any, Any]]:
        if isinstance(features, Mapping):
            if group_ids is not None:
                raise ValueError("group_ids cannot be used with mapping input.")
            if not features:
                raise ValueError("features cannot be empty.")
            return list(features.items())

        if group_ids is None:
            return [("input", features)]

        try:
            frame_count = len(features)
            group_count = len(group_ids)
        except (TypeError, AttributeError) as error:
            raise ValueError(
                "features and group_ids must be sized frame-level sequences."
            ) from error
        if frame_count == 0:
            raise ValueError("features cannot be empty.")
        if group_count != frame_count:
            raise ValueError(
                "group_ids must contain exactly one ID for each feature frame."
            )

        grouped: dict[Any, list[Any]] = {}
        for row, group_id in zip(features, group_ids):
            try:
                grouped.setdefault(group_id, []).append(row)
            except TypeError as error:
                raise TypeError("Each group ID must be hashable.") from error
        return list(grouped.items())

    @staticmethod
    def _validate_feature_matrix(features: Any, group_name: Any) -> list[list[Any]]:
        if isinstance(features, (str, bytes)):
            raise ValueError(
                f"Group {group_name!r} must have shape [num_frames, feature_dim]."
            )
        try:
            rows = list(features)
        except TypeError as error:
            raise ValueError(
                f"Group {group_name!r} must have shape [num_frames, feature_dim]."
            ) from error
        if not rows:
            raise ValueError(f"Group {group_name!r} contains no frames.")

        normalized: list[list[Any]] = []
        feature_dim: int | None = None
        for frame_index, row in enumerate(rows):
            if isinstance(row, (str, bytes)):
                raise ValueError(
                    f"Frame {frame_index} in group {group_name!r} is not a feature vector."
                )
            try:
                vector = list(row)
            except TypeError as error:
                raise ValueError(
                    f"Frame {frame_index} in group {group_name!r} is not a feature vector."
                ) from error
            if not vector:
                raise ValueError(
                    f"Frame {frame_index} in group {group_name!r} has no features."
                )
            if feature_dim is None:
                feature_dim = len(vector)
            elif len(vector) != feature_dim:
                raise ValueError(
                    f"Inconsistent feature dimensions in group {group_name!r}: "
                    f"expected {feature_dim}, got {len(vector)} at frame {frame_index}."
                )
            normalized.append(vector)
        return normalized
