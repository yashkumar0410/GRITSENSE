"""
Train GNN + HRN on the actual Volleyball Dataset.

Dataset:
    C:/Users/vipra/Downloads/volleyball_/videos/
        0/annotations.txt
        1/annotations.txt
        ...
        54/annotations.txt

sample3.mp4 is NOT used for training.
"""

import sys
import re
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HRNClassifier
from graph.graph_builder import VolleyballGraphBuilder


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\vipra\Downloads\volleyball_\videos"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

TRAIN_RATIO = 0.80
SEED = 42

IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0


# ============================================================
# LABELS
# ============================================================

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

LABEL_TO_ID = {
    "r_set": 0,
    "r_spike": 1,
    "r-pass": 2,
    "r_pass": 2,
    "r_winpoint": 3,
    "l_winpoint": 4,
    "l-pass": 5,
    "l_pass": 5,
    "l-spike": 6,
    "l_spike": 6,
    "l_set": 7,
}


# ============================================================
# DATASET ANNOTATION PARSER
# ============================================================

def parse_annotation_line(line):
    """
    Parse one line from annotations.txt.

    Example:

    48075.jpg r_winpoint
    372 442 86 130 falling
    712 426 73 124 falling
    ...

    Returns:
        frame_id
        group_label
        players
    """

    parts = line.strip().split()

    if len(parts) < 2:
        return None

    image_name = parts[0]
    group_label = parts[1]

    if group_label not in LABEL_TO_ID:
        return None

    # Extract numeric image/frame identifier.
    match = re.search(r"(\d+)", image_name)

    if not match:
        return None

    frame_id = int(match.group(1))

    players = []

    # Every player after the group label has:
    #
    # x y w h action
    #
    player_values = parts[2:]

    # Five tokens per player.
    for i in range(0, len(player_values) - 4, 5):

        try:
            x = float(player_values[i])
            y = float(player_values[i + 1])
            w = float(player_values[i + 2])
            h = float(player_values[i + 3])
            action = player_values[i + 4]
        except ValueError:
            continue

        # Convert bbox top-left + width/height
        # into approximate player center.
        cx = x + w / 2.0
        cy = y + h / 2.0

        players.append({
            "id": len(players),
            "x": cx,
            "y": cy,
            "vx": 0.0,
            "vy": 0.0,
            "team": None,
            "confidence": 1.0,
            "pose": np.zeros(34, dtype=np.float32),
            "pose_available": False,
            "pose_confidence": 0.0,
            "action": action,
            "bbox": [x, y, w, h],
        })

    if len(players) == 0:
        return None

    return {
        "frame_id": frame_id,
        "label_name": group_label,
        "label": LABEL_TO_ID[group_label],
        "players": players,
    }


def load_video_annotations(video_dir):
    """
    Load one video's annotations.txt.
    """

    annotation_file = video_dir / "annotations.txt"

    if not annotation_file.exists():
        return []

    frames = []

    with annotation_file.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            parsed = parse_annotation_line(line)

            if parsed is not None:
                parsed["video_id"] = int(video_dir.name)
                frames.append(parsed)

    return frames


def load_dataset():
    """
    Load all 55 annotated videos.
    """

    all_frames = []

    video_dirs = sorted(
        [
            p for p in DATASET_ROOT.iterdir()
            if p.is_dir() and p.name.isdigit()
        ],
        key=lambda p: int(p.name)
    )

    print("=" * 70)
    print("LOADING REAL VOLLEYBALL DATASET")
    print("=" * 70)

    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Videos found: {len(video_dirs)}")

    for video_dir in video_dirs:

        frames = load_video_annotations(video_dir)

        print(
            f"Video {video_dir.name:>2}: "
            f"{len(frames):>4} annotated frames"
        )

        all_frames.extend(frames)

    print()
    print(f"Total annotated frames: {len(all_frames)}")

    return all_frames


# ============================================================
# TEAM ASSIGNMENT
# ============================================================

def assign_teams(players):
    """
    Assign teams using horizontal court position.

    Left half -> team 0
    Right half -> team 1
    """

    if len(players) < 2:
        return players

    players = sorted(
        players,
        key=lambda p: p["x"]
    )

    # Volleyball annotations generally contain
    # players from both sides of the court.
    midpoint = len(players) // 2

    for i, player in enumerate(players):

        if i < midpoint:
            player["team"] = 0
        else:
            player["team"] = 1

    return players


# ============================================================
# GRAPH DATASET
# ============================================================

class VolleyballDataset(Dataset):

    def __init__(self, frames):

        self.frames = frames
        self.graph_builder = VolleyballGraphBuilder()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):

        item = self.frames[index]

        players = assign_teams(
            item["players"]
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Current graph schema requires a ball node.
        #
        # The annotations shown do NOT contain ball coordinates.
        #
        # Therefore we use the court center only as a structural
        # placeholder.
        #
        # This should be replaced if actual ball annotations
        # are available.
        # ----------------------------------------------------

        ball = {
            "x": IMAGE_WIDTH / 2.0,
            "y": IMAGE_HEIGHT / 2.0,
            "vx": 0.0,
            "vy": 0.0,
            "radius": 0.0,
            "confidence": 0.0,
            "detected": True,
        }

        court = {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "net_x": IMAGE_WIDTH / 2.0,
        }

        graph = self.graph_builder.build_graph(
            players=players,
            ball=ball,
            court=court,
            frame_id=item["frame_id"],
        )

        graph.y = torch.tensor(
            item["label"],
            dtype=torch.long
        )

        graph.video_id = item["video_id"]

        return graph


# ============================================================
# COLLATE
# ============================================================

def collate_graphs(batch):

    return Batch.from_data_list(batch)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = HRNClassifier(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=DROPOUT,
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.node_type,
            batch.node_team,
        )

        logits = output["logits"]

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        labels = batch.y.view(-1)

        # With batch_size=1 this is straightforward.
        if logits.size(0) != labels.size(0):
            labels = labels[:logits.size(0)]

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.numel()

        if (batch_idx + 1) % 100 == 0:

            print(
                f"    Batch "
                f"{batch_idx + 1}/{len(loader)} "
                f"Loss={loss.item():.4f}"
            )

    return (
        total_loss / max(len(loader), 1),
        correct / max(total, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        batch = batch.to(device)

        output = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.node_type,
            batch.node_team,
        )

        logits = output["logits"]

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        labels = batch.y.view(-1)

        if logits.size(0) != labels.size(0):
            labels = labels[:logits.size(0)]

        loss = criterion(
            logits,
            labels
        )

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.numel()

    return (
        total_loss / max(len(loader), 1),
        correct / max(total, 1)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print()
    print("=" * 70)
    print("GNN + HRN REAL DATASET TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    frames = load_dataset()

    if not frames:
        raise RuntimeError(
            "No annotated frames were found."
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    counts = Counter(
        item["label_name"]
        for item in frames
    )

    print()
    print("Class distribution:")

    for class_name in CLASS_NAMES:

        print(
            f"  {class_name:<12}: "
            f"{counts.get(class_name, 0)}"
        )

    # --------------------------------------------------------
    # VIDEO LEVEL SPLIT
    # --------------------------------------------------------

    video_ids = sorted(
        set(
            item["video_id"]
            for item in frames
        )
    )

    random.shuffle(video_ids)

    split_index = int(
        len(video_ids) * TRAIN_RATIO
    )

    train_video_ids = set(
        video_ids[:split_index]
    )

    val_video_ids = set(
        video_ids[split_index:]
    )

    train_frames = [
        item
        for item in frames
        if item["video_id"]
        in train_video_ids
    ]

    val_frames = [
        item
        for item in frames
        if item["video_id"]
        in val_video_ids
    ]

    print()
    print("=" * 70)
    print("VIDEO-LEVEL SPLIT")
    print("=" * 70)

    print(
        f"Training videos: "
        f"{len(train_video_ids)}"
    )

    print(
        f"Validation videos: "
        f"{len(val_video_ids)}"
    )

    print(
        f"Training frames: "
        f"{len(train_frames)}"
    )

    print(
        f"Validation frames: "
        f"{len(val_frames)}"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = VolleyballDataset(
        train_frames
    )

    val_dataset = VolleyballDataset(
        val_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INITIALIZING HRN + GAT")
    print("=" * 70)

    model = create_model().to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"  Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"  Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"  Val Accuracy: "
            f"{val_acc:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_accuracy = val_acc

            checkpoint = (
                CHECKPOINT_DIR
                / "hrn_gat_best.pth"
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_loss":
                        val_loss,

                    "val_accuracy":
                        val_acc,

                    "class_names":
                        CLASS_NAMES,

                    "train_videos":
                        sorted(train_video_ids),

                    "val_videos":
                        sorted(val_video_ids),
                },
                checkpoint,
            )

            print(
                f"  ✓ Saved best model: "
                f"{checkpoint}"
            )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    final_checkpoint = (
        CHECKPOINT_DIR
        / "hrn_gat_final.pth"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "class_names":
                CLASS_NAMES,

            "val_accuracy":
                best_val_accuracy,
        },
        final_checkpoint,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print()
    print(
        f"Best checkpoint:"
    )

    print(
        CHECKPOINT_DIR
        / "hrn_gat_best.pth"
    )

    print()
    print(
        "sample3.mp4 was NOT used for training."
    )


if __name__ == "__main__":
    main()