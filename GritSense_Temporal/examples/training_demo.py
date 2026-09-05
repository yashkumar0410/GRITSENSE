"""Synthetic training demo for the temporal GRU model.

This script is for development and demonstration only. It uses generated random
sequence data rather than real volleyball action data.
"""

import torch

from gritsense.temporal import GRUTemporalModel, TemporalTrainer


def main() -> None:
    torch.manual_seed(7)

    num_samples = 64
    sequence_length = 8
    input_dim = 4
    num_classes = 2

    sequences = torch.randn(num_samples, sequence_length, input_dim)
    labels = torch.randint(0, num_classes, (num_samples,))

    model = GRUTemporalModel(
        input_dim=input_dim,
        hidden_dim=16,
        num_classes=num_classes,
        class_names=("serve", "receive"),
    )
    trainer = TemporalTrainer(model, learning_rate=1e-3)

    history = trainer.fit(sequences, labels, epochs=5, batch_size=16)
    metrics = trainer.evaluate(sequences, labels, batch_size=16)

    print("Training history:")
    print(f"  loss: {history['loss']}")
    print(f"  accuracy: {history['accuracy']}")
    print()
    print("Evaluation:")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  macro_f1: {metrics['macro_f1']:.4f}")
    print(f"  loss: {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
