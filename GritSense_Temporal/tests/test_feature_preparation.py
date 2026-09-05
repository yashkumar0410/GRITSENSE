import math
import unittest

from gritsense.temporal import (
    FeaturePreparer,
    FeatureStandardizer,
    TemporalSequenceGenerator,
)


class FeaturePreparerTests(unittest.TestCase):
    def test_prepares_blocks_in_configured_order(self) -> None:
        preparer = FeaturePreparer(feature_order=("player", "pose", "ball", "spatial"))
        blocks = {
            "spatial": [[7], [8]],
            "ball": [[5, 6], [9, 10]],
            "pose": [[3, 4], [11, 12]],
            "player": [[1, 2], [13, 14]],
        }

        result = preparer.prepare(blocks)

        self.assertEqual(result, [[1, 2, 3, 4, 5, 6, 7], [13, 14, 11, 12, 9, 10, 8]])

    def test_allows_available_partial_blocks_without_changing_order(self) -> None:
        preparer = FeaturePreparer()

        self.assertEqual(preparer.prepare({"ball": [[1, 2]], "player": [[3]]}), [[3, 1, 2]])

    def test_rejects_frame_count_and_dimension_mismatches(self) -> None:
        preparer = FeaturePreparer()
        with self.assertRaisesRegex(ValueError, "Frame count mismatch"):
            preparer.prepare({"player": [[1], [2]], "ball": [[3]]})
        with self.assertRaisesRegex(ValueError, "Inconsistent feature dimensions"):
            preparer.prepare({"player": [[1], [2, 3]]})

    def test_rejects_empty_unknown_and_non_numeric_values(self) -> None:
        preparer = FeaturePreparer()
        for blocks, message in [({}, "non-empty"), ({"unknown": [[1]]}, "Unknown")]:
            with self.subTest(blocks=blocks):
                with self.assertRaisesRegex(ValueError, message):
                    preparer.prepare(blocks)
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            preparer.prepare({"player": [["missing"]]})

    def test_rejects_nan_and_infinity(self) -> None:
        for invalid_value in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    FeaturePreparer().prepare({"ball": [[invalid_value]]})

    def test_supports_different_feature_dimensions(self) -> None:
        preparer = FeaturePreparer(feature_order=("hrn",))

        self.assertEqual(len(preparer.prepare({"hrn": [[0] * 64]} )[0]), 64)
        self.assertEqual(len(preparer.prepare({"hrn": [[0] * 256]} )[0]), 256)

    def test_standardizer_uses_fit_training_statistics(self) -> None:
        standardizer = FeatureStandardizer().fit([[0, 10], [2, 14]])

        result = standardizer.transform([[1, 12]])

        self.assertEqual(result[0][0], 0.0)
        self.assertEqual(result[0][1], 0.0)

    def test_standardizer_requires_fit_and_preserves_dimension(self) -> None:
        standardizer = FeatureStandardizer()
        with self.assertRaisesRegex(ValueError, "fitted"):
            standardizer.transform([[1]])
        standardizer.fit([[1, 2]])
        with self.assertRaisesRegex(ValueError, "dimension changed"):
            standardizer.transform([[1]])

    def test_preparer_connects_to_task_one(self) -> None:
        prepared = FeaturePreparer().prepare(
            {"player": [[1], [2], [3]], "ball": [[10], [20], [30]]}
        )
        sequences = TemporalSequenceGenerator(sequence_length=2).generate_sequences(prepared)

        self.assertEqual(sequences, [[[1, 10], [2, 20]], [[2, 20], [3, 30]]])

    def test_future_hrn_matrix_is_source_agnostic(self) -> None:
        hrn_output = [[frame, frame + 0.5, frame + 1] for frame in range(3)]

        result = FeaturePreparer().prepare_matrix(hrn_output)

        self.assertEqual(result, hrn_output)


if __name__ == "__main__":
    unittest.main()
