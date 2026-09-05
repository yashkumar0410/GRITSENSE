import unittest

from gritsense.temporal import TemporalSequenceGenerator


class TemporalSequenceGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [[frame, frame + 100] for frame in range(6)]

    def test_generates_normal_chronological_windows(self) -> None:
        generator = TemporalSequenceGenerator(sequence_length=3, stride=1)

        result = generator.generate_sequences(self.features)

        self.assertEqual(
            result,
            [
                [[0, 100], [1, 101], [2, 102]],
                [[1, 101], [2, 102], [3, 103]],
                [[2, 102], [3, 103], [4, 104]],
                [[3, 103], [4, 104], [5, 105]],
            ],
        )

    def test_supports_different_sequence_lengths_and_strides(self) -> None:
        generator = TemporalSequenceGenerator(sequence_length=2, stride=2)

        result = generator.generate_sequences(self.features, sequence_length=4, stride=2)

        self.assertEqual(
            result,
            [
                [[0, 100], [1, 101], [2, 102], [3, 103]],
                [[2, 102], [3, 103], [4, 104], [5, 105]],
            ],
        )

    def test_rejects_insufficient_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 7"):
            TemporalSequenceGenerator().generate_sequences(self.features, sequence_length=7)

    def test_keeps_mapping_groups_separate(self) -> None:
        grouped = {
            "video-a": [[0], [1], [2]],
            "video-b": [[100], [101], [102]],
        }

        result = TemporalSequenceGenerator(sequence_length=2).generate_sequences(grouped)

        self.assertEqual(
            result,
            [[[0], [1]], [[1], [2]], [[100], [101]], [[101], [102]]],
        )
        self.assertNotIn([[2], [100]], result)

    def test_keeps_group_ids_separate(self) -> None:
        features = [[0], [1], [10], [11]]
        group_ids = ["rally-a", "rally-a", "rally-b", "rally-b"]

        result = TemporalSequenceGenerator(sequence_length=2).generate_sequences(
            features, group_ids=group_ids
        )

        self.assertEqual(result, [[[0], [1]], [[10], [11]]])

    def test_rejects_inconsistent_feature_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "Inconsistent feature dimensions"):
            TemporalSequenceGenerator().generate_sequences([[1, 2], [3]])

    def test_rejects_empty_input_and_invalid_shapes(self) -> None:
        generator = TemporalSequenceGenerator()
        for invalid_features in ([], [1, 2], "frames"):
            with self.subTest(invalid_features=invalid_features):
                with self.assertRaises(ValueError):
                    generator.generate_sequences(invalid_features)

    def test_rejects_invalid_window_parameters(self) -> None:
        for parameter in (0, -1, True, 1.5):
            with self.subTest(parameter=parameter):
                with self.assertRaises((ValueError, TypeError)):
                    TemporalSequenceGenerator(sequence_length=parameter)
                with self.assertRaises((ValueError, TypeError)):
                    TemporalSequenceGenerator(stride=parameter)

    def test_rejects_mismatched_group_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one ID"):
            TemporalSequenceGenerator().generate_sequences(
                self.features, group_ids=["video-a"]
            )


if __name__ == "__main__":
    unittest.main()
