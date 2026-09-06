"""
Train and Validate GNN/HRN Model with Real Ball Data

This script trains the GNN/HRN model using actual ball detection data
from the updated ball detection pipeline and real player keypoints.
"""

import sys
from pathlib import Path
from typing import List

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from datetime import datetime
from torch_geometric.data import Batch

from models import HRNClassifier
from graph.graph_builder import VolleyballGraphBuilder
from data.ball_player_loader import BallPlayerDataLoader


# ============================================================
# CUSTOM COLLATE FUNCTION FOR PYTORCH GEOMETRIC DATA
# ============================================================

def geo_collate_fn(batch):
    """Collate function for PyTorch Geometric Data objects."""
    return Batch.from_data_list(batch)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "ball_csv_path": "ball detection/output_test/sample3/ball.csv",
    "player_json_path": "player_keypoints.json",
    "frame_labels_path": "frame_labels.json",
    "model_checkpoint_dir": "checkpoints",
    "epochs": 50,
    "batch_size": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "num_classes": 8,  # Volleyball group activities
    "train_val_split": 0.8,
    "random_seed": 42,
}

# Group activity classes
CLASS_NAMES = [
    "r_set",
    "r_spike",
    "r_pass",
    "r_winpoint",
    "l_winpoint",
    "l_pass",
    "l_spike",
    "l_set",
]

# ============================================================
# REAL VOLLEYBALL GRAPH DATASET
# ============================================================

class RealVolleyballGraphDataset(Dataset):
    """
    Dataset that creates graphs from real ball and player data.
    """

    IMAGE_WIDTH = 1920.0
    IMAGE_HEIGHT = 1080.0
    COURT_WIDTH = IMAGE_WIDTH
    COURT_HEIGHT = IMAGE_HEIGHT

    def __init__(
        self,
        data_loader: BallPlayerDataLoader,
        frame_indices: List[int],
        graph_builder=None,
        frame_labels=None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_loader: BallPlayerDataLoader instance
            frame_indices: List of frame indices to use
            graph_builder: VolleyballGraphBuilder instance
            frame_labels: Mapping from frame index to a real class label.
        """
        self.data_loader = data_loader
        self.frame_indices = frame_indices
        self.frame_labels = frame_labels or {}

        if graph_builder is None:
            graph_builder = VolleyballGraphBuilder()
        self.graph_builder = graph_builder

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]

        # Get frame data
        frame_data = self.data_loader.get_frame(frame_idx)
        if frame_data is None:
            raise RuntimeError(
                f"Frame {frame_idx} not available in data loader"
            )

        players, ball = frame_data

        # Assign teams (left/right court position)
        players = self._assign_teams(players)

        if ball is None or not ball["detected"]:
            raise RuntimeError(
                f"Frame {frame_idx} has no ball detection; "
                "missing coordinates must not be fabricated."
            )

        # Court definition
        court = {
            "width": self.COURT_WIDTH,
            "height": self.COURT_HEIGHT,
            "net_x": self.COURT_WIDTH / 2.0,
        }

        # Build graph
        graph = self.graph_builder.build_graph(
            players=players,
            ball=ball,
            court=court,
            frame_id=frame_idx,
        )

        if frame_idx not in self.frame_labels:
            raise RuntimeError(
                f"No real activity label is available for frame {frame_idx}. "
                "Provide CONFIG['frame_labels_path'] before supervised training."
            )
        graph.y = torch.tensor(self.frame_labels[frame_idx], dtype=torch.long)

        # Store metadata
        graph.frame_id = frame_idx
        graph.num_players = sum(
            1 for p in players if p.get("team") is not None
        )

        return graph

    @staticmethod
    def _assign_teams(players):
        """
        Assign players to two teams based on horizontal court position.
        """
        if len(players) < 2:
            return players

        sorted_players = sorted(
            players,
            key=lambda p: float(p["x"]),
        )

        midpoint = len(sorted_players) // 2
        team_0 = sorted_players[:midpoint]
        team_1 = sorted_players[midpoint:]

        for player in team_0:
            player["team"] = 0
        for player in team_1:
            player["team"] = 1

        return team_0 + team_1


# ============================================================
# DEVICE
# ============================================================

def get_device():
    """Get appropriate device for training."""
    if torch.cuda.is_available():
        print("Using device: CUDA")
        return torch.device("cuda")
    print("Using device: CPU")
    return torch.device("cpu")


def load_frame_labels(path):
    """Load real frame labels from a JSON object keyed by frame number."""
    if not path:
        raise RuntimeError(
            "Supervised training cannot start: no real frame-label file was configured."
        )
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Frame label file not found: {label_path}")
    with label_path.open("r", encoding="utf-8") as file:
        raw_labels = json.load(file)
    labels = {int(frame): int(label) for frame, label in raw_labels.items()}
    invalid = [label for label in labels.values() if label < 0 or label >= len(CLASS_NAMES)]
    if invalid:
        raise ValueError(f"Frame labels must be in [0, {len(CLASS_NAMES) - 1}]")
    return labels


# ============================================================
# TRAINING & VALIDATION
# ============================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        batch_data = batch_data.to(device)

        optimizer.zero_grad()
        
        # Model returns a dict with logits
        output = model(
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.node_type,
            batch_data.node_team,
        )
        logits = output["logits"]  # Shape: [batch_size, num_classes] or [num_classes] depending on readout
        
        # Get batch labels - take first element of each graph in batch
        # For now, use first label (they're all 0 anyway)
        batch_labels = batch_data.y  # This is a tensor of labels, one per graph
        
        # If logits is 1D (single graph case), make it 2D
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            batch_labels = batch_labels[:1]
        
        # Compute loss - batch mode
        loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 5 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                f"Loss = {loss.item():.4f}"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(device)
            
            output = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.node_type,
                batch_data.node_team,
            )
            logits = output["logits"]  # Shape: [batch_size, num_classes] or [num_classes]
            
            # Get batch             labels
            batch_labels = batch_data.y  # Tensor of labels
            
            # If logits is 1D (single graph case), make it 2D
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                batch_labels = batch_labels[:1]
            
            # Compute loss - batch mode
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / (num_batches * logits.shape[0]) if num_batches > 0 else 0.0

    return avg_loss, accuracy


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """Main training function."""

    # Set random seed for reproducibility
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("LOADING REAL BALL AND PLAYER DATA")
    print("=" * 60)

    data_loader = BallPlayerDataLoader(
        ball_csv_path=CONFIG["ball_csv_path"],
        player_json_path=CONFIG["player_json_path"],
    )

    stats = data_loader.get_statistics()
    print("\nData Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Ball detections: {stats['ball_detected']}")
    print(f"  Ball missing: {stats['ball_missing']}")
    print(f"  Ball detection rate: {stats['ball_detection_rate']:.2%}")
    print(f"  Player count (min/max/avg): "
          f"{stats['min_players']}/{stats['max_players']}/{stats['avg_players']:.1f}")

    # ========================================================
    # PREPARE DATASETS
    # ========================================================

    print("\n" + "=" * 60)
    print("PREPARING DATASETS")
    print("=" * 60)

    frame_labels = load_frame_labels(CONFIG["frame_labels_path"])

    # Only detected frames can enter the graph; missing detections are not coordinates.
    detected_frames = [
        frame for frame in data_loader.common_frames
        if data_loader.ball_data[frame]["detected"]
    ]
    missing_labels = [frame for frame in detected_frames if frame not in frame_labels]
    if missing_labels:
        raise RuntimeError(
            f"Missing real labels for {len(missing_labels)} detected frames; "
            f"first missing frame IDs: {missing_labels[:10]}. "
            "Partial labels are not allowed for supervised training."
        )
    available_frames = detected_frames
    if not available_frames:
        raise RuntimeError(
            "No common frames have both a valid ball detection and a real activity label."
        )

    # Split into train/validation
    num_train = int(len(available_frames) * CONFIG["train_val_split"])
    train_frames = available_frames[:num_train]
    val_frames = available_frames[num_train:]

    print(f"  Total available frames: {len(available_frames)}")
    print(f"  Training frames: {len(train_frames)}")
    print(f"  Validation frames: {len(val_frames)}")

    # Create datasets
    train_dataset = RealVolleyballGraphDataset(
        data_loader=data_loader,
        frame_indices=train_frames,
        frame_labels=frame_labels,
    )

    val_dataset = RealVolleyballGraphDataset(
        data_loader=data_loader,
        frame_indices=val_frames,
        frame_labels=frame_labels,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=geo_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=geo_collate_fn,
    )

    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Val loader: {len(val_loader)} batches")

    # ========================================================
    # INITIALIZE MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)

    device = get_device()

    model = HRNClassifier(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=CONFIG["dropout"],
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ========================================================
    # TRAINING SETUP
    # ========================================================

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Create checkpoint directory
    checkpoint_dir = Path(CONFIG["model_checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    training_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        # Validate
        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")

        # Record history
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(float(train_loss))
        training_history["val_loss"].append(float(val_loss))
        training_history["val_accuracy"].append(float(val_acc))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1

            best_checkpoint_path = checkpoint_dir / "hrn_gat_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                best_checkpoint_path,
            )
            print(f"  Saved best model to {best_checkpoint_path}")

    # Save final model after the last epoch as well as the best model.
    final_checkpoint_path = checkpoint_dir / "hrn_gat_final.pth"
    torch.save(
        {
            "epoch": CONFIG["epochs"] - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        },
        final_checkpoint_path,
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nBest Results:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG,
        "data_statistics": stats,
        "model": {
            "name": "HRNClassifier",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
        "training": {
            "train_samples": len(train_frames),
            "val_samples": len(val_frames),
            "epochs": CONFIG["epochs"],
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_accuracy": float(best_val_acc),
        },
        "history": training_history,
        "checkpoint_paths": {
            "best": str(checkpoint_dir / "hrn_gat_best.pth"),
            "final": str(final_checkpoint_path),
        },
    }

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved training results to {history_path}")

    print("\nTraining pipeline complete.")
    return model, results


if __name__ == "__main__":
    model, results = main()
