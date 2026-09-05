import unittest

import torch

from gritsense.temporal import (
    ACTIVITY_CLASSES,
    GRUTemporalModel,
    TemporalSequenceGenerator,
    TemporalTrainer,
    TransformerTemporalModel,
)


class GRUTemporalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_initialization_uses_configured_parameters(self) -> None:
        model = GRUTemporalModel(
            input_dim=6,
            hidden_dim=5,
            num_layers=2,
            dropout=0.2,
            num_classes=3,
            bidirectional=True,
            class_names=("a", "b", "c"),
        )

        self.assertEqual(model.input_dim, 6)
        self.assertEqual(model.gru.num_layers, 2)
        self.assertTrue(model.bidirectional)
        self.assertEqual(model.classifier.out_features, 3)
        self.assertEqual(model.classifier.in_features, 10)

    def test_forward_returns_logits_with_expected_shape(self) -> None:
        model = GRUTemporalModel(input_dim=4, hidden_dim=8)

        logits = model(torch.zeros(3, 5, 4))

        self.assertEqual(tuple(logits.shape), (3, len(ACTIVITY_CLASSES)))
        self.assertFalse(torch.all((logits >= 0) & (logits <= 1)))

    def test_supports_different_sequence_lengths_batches_and_dimensions(self) -> None:
        for batch_size, sequence_length, input_dim in ((1, 2, 3), (4, 9, 10), (2, 16, 64)):
            with self.subTest(batch_size=batch_size, sequence_length=sequence_length, input_dim=input_dim):
                model = GRUTemporalModel(input_dim=input_dim, hidden_dim=4)
                self.assertEqual(tuple(model(torch.zeros(batch_size, sequence_length, input_dim)).shape), (batch_size, 8))

    def test_bidirectional_forward_pass(self) -> None:
        model = GRUTemporalModel(input_dim=3, hidden_dim=4, bidirectional=True)

        self.assertEqual(tuple(model(torch.zeros(2, 6, 3)).shape), (2, 8))

    def test_rejects_invalid_input_shapes_dimensions_and_dtype(self) -> None:
        model = GRUTemporalModel(input_dim=4)
        for invalid in (torch.zeros(4), torch.zeros(2, 3, 5), torch.zeros(0, 3, 4), torch.zeros(2, 0, 4)):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises((ValueError, TypeError)):
                    model(invalid)
        with self.assertRaises(TypeError):
            model(torch.zeros(2, 3, 4, dtype=torch.int64))
        with self.assertRaises(TypeError):
            model([[0.0, 0.0, 0.0, 0.0]])

    def test_configurable_class_mapping_and_prediction_outputs(self) -> None:
        model = GRUTemporalModel(input_dim=2, num_classes=2, class_names=("serve", "receive"))

        result = model.predict(torch.zeros(3, 4, 2))

        self.assertEqual(result["class_indices"].shape, (3,))
        self.assertEqual(len(result["class_labels"]), 3)
        self.assertEqual(result["probabilities"].shape, (3, 2))
        self.assertTrue(torch.allclose(result["probabilities"].sum(dim=1), torch.ones(3)))
        self.assertEqual(result["confidence"].shape, (3,))

    def test_state_dict_save_and_load(self) -> None:
        model = GRUTemporalModel(input_dim=3, hidden_dim=4)
        restored = GRUTemporalModel(input_dim=3, hidden_dim=4)
        inputs = torch.randn(2, 5, 3)

        restored.load_state_dict(model.state_dict())

        self.assertTrue(torch.equal(model(inputs), restored(inputs)))

    def test_accepts_task_one_sequence_generator_output(self) -> None:
        frame_features = [[float(frame), float(frame + 1)] for frame in range(5)]
        sequences = TemporalSequenceGenerator(sequence_length=3).generate_sequences(frame_features)
        inputs = torch.tensor(sequences, dtype=torch.float32)

        self.assertEqual(tuple(GRUTemporalModel(input_dim=2)(inputs).shape), (3, 8))

    def test_transformer_baseline_returns_logits_and_prediction(self) -> None:
        model = TransformerTemporalModel(input_dim=2, hidden_dim=8, num_classes=2, class_names=("serve", "receive"))
        inputs = torch.randn(3, 4, 2)

        logits = model(inputs)
        result = model.predict(inputs)

        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(result["probabilities"].shape, (3, 2))
        self.assertEqual(len(result["class_labels"]), 3)

    def test_temporal_trainer_evaluates_and_tracks_history(self) -> None:
        model = GRUTemporalModel(input_dim=2, hidden_dim=4, num_classes=2, class_names=("serve", "receive"))
        trainer = TemporalTrainer(model)
        sequences = torch.randn(8, 4, 2)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

        history = trainer.fit(sequences, targets, epochs=2, batch_size=4)
        metrics = trainer.evaluate(sequences, targets)

        self.assertIn("loss", history)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
