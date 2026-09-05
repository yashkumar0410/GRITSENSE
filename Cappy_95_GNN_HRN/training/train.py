from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from models import HRNClassifier
from training.dataset import create_datasets
from graph.graph_builder import VolleyballGraphBuilder


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 8

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

DROPOUT = 0.2

DATASET_ROOT = (
    r"C:\Users\vipra\Downloads"
    r"\volleyball_\videos"
)

MODEL_DIR = Path(
    "checkpoints"
)

MODEL_PATH = (
    MODEL_DIR
    / "hrn_gat_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        print(
            "Using device: CUDA"
        )

        return torch.device(
            "cuda"
        )

    print(
        "Using device: CPU"
    )

    return torch.device(
        "cpu"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    validation_dataset,
    criterion,
    device,
):
    """
    Validate the GAT + HRN model.

    Returns:
        average validation loss
        validation accuracy
    """

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for graph in validation_dataset:

            graph = graph.to(
                device
            )

            output = model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.node_type,
                graph.node_team,
            )

            logits = output[
                "logits"
            ]

            target = graph.y

            loss = criterion(
                logits.unsqueeze(0),
                target.unsqueeze(0),
            )

            total_loss += (
                loss.item()
            )

            prediction = (
                logits.argmax()
            )

            correct += int(
                prediction.item()
                == target.item()
            )

            total += 1

    if total == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy,
    )


# ============================================================
# TEST
# ============================================================

def test(
    model,
    test_dataset,
    criterion,
    device,
):
    """
    Evaluate the trained model on the
    official Volleyball test split.
    """

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for graph in test_dataset:

            graph = graph.to(
                device
            )

            output = model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.node_type,
                graph.node_team,
            )

            logits = output[
                "logits"
            ]

            target = graph.y

            loss = criterion(
                logits.unsqueeze(0),
                target.unsqueeze(0),
            )

            total_loss += (
                loss.item()
            )

            prediction = (
                logits.argmax()
            )

            correct += int(
                prediction.item()
                == target.item()
            )

            total += 1

    if total == 0:

        raise RuntimeError(
            "Test dataset is empty."
        )

    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy,
    )


# ============================================================
# TRAINING
# ============================================================

def train(
    train_dataset,
    validation_dataset,
):
    """
    Train GAT + HRN for group activity classification.
    """

    device = get_device()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HRNClassifier(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=DROPOUT,
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_accuracy = (
        -1.0
    )

    print(
        "\n========== TASK 5: TRAINING =========="
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Number of classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "======================================="
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        # ----------------------------------------------------
        # Graph loop
        # ----------------------------------------------------

        for graph in train_dataset:

            graph = graph.to(
                device
            )

            optimizer.zero_grad()

            output = model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.node_type,
                graph.node_team,
            )

            logits = output[
                "logits"
            ]

            target = graph.y

            loss = criterion(
                logits.unsqueeze(0),
                target.unsqueeze(0),
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            prediction = (
                logits.argmax()
            )

            correct += int(
                prediction.item()
                == target.item()
            )

            total += 1

        if total == 0:

            raise RuntimeError(
                "Training dataset is empty."
            )

        training_loss = (
            total_loss / total
        )

        training_accuracy = (
            correct / total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            validation_loss,
            validation_accuracy,
        ) = validate(
            model,
            validation_dataset,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch:02d}/{EPOCHS} | "
            f"Train Loss: "
            f"{training_loss:.4f} | "
            f"Train Acc: "
            f"{training_accuracy:.4f} | "
            f"Val Loss: "
            f"{validation_loss:.4f} | "
            f"Val Acc: "
            f"{validation_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "validation_accuracy":
                        validation_accuracy,

                    "epoch":
                        epoch,

                    "class_names":
                        HRNClassifier.CLASS_NAMES,

                    "config": {
                        "in_channels": 49,
                        "hidden_channels": 32,
                        "out_channels": 32,
                        "heads": 4,
                        "dropout": DROPOUT,
                    },
                },
                MODEL_PATH,
            )

            print(
                "  Best model saved:"
                f" {MODEL_PATH}"
            )

    print(
        "\n========== TRAINING COMPLETE =========="
    )

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.4f}"
    )

    print(
        "Model:",
        MODEL_PATH
    )

    return model


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "\n========== LOADING DATASETS =========="
    )

    # --------------------------------------------------------
    # Graph builder
    # --------------------------------------------------------

    graph_builder = (
        VolleyballGraphBuilder()
    )

    # --------------------------------------------------------
    # Create official train/validation/test datasets
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(
        DATASET_ROOT,
        graph_builder,
    )

    print(
        "\nDataset loading complete."
    )

    print(
        "Training graphs:",
        len(train_dataset),
    )

    print(
        "Validation graphs:",
        len(validation_dataset),
    )

    print(
        "Test graphs:",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train(
        train_dataset,
        validation_dataset,
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    if MODEL_PATH.exists():

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "\nBest checkpoint loaded."
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    device = get_device()

    model = model.to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    (
        test_loss,
        test_accuracy,
    ) = test(
        model,
        test_dataset,
        criterion,
        device,
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print(
        "\n========== TASK 5 RESULTS =========="
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy (%): "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Best Model: "
        f"{MODEL_PATH}"
    )

    print(
        "===================================="
    )