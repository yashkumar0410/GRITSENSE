"""
GRITSENSE
============================================================
Train + Evaluate GNN/HRN using REAL Volleyball Dataset data.

INPUT:
    Volleyball Dataset annotations.txt
    Ball detector CSV

NO frame_labels.json is required.

The annotations.txt format is:

    frame.jpg activity x y w h action x y w h action ...

Example:
    48075.jpg r_winpoint 372 442 86 130 falling ...

This script:
    1. Reads real player bounding boxes from annotations.txt
    2. Reads real ball detections from ball.csv
    3. Aligns annotation frames with ball frames
    4. Builds 49-D graph nodes using VolleyballGraphBuilder
    5. Trains HRNClassifier
    6. Evaluates held-out data
    7. Runs relational ablations
    8. Saves evaluation results
============================================================
"""

import sys
import csv
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graph.graph_builder import VolleyballGraphBuilder
from models import HRNClassifier


# ============================================================
# CONFIG
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # CHANGE THESE TWO PATHS
    # --------------------------------------------------------

    "dataset_root":
        r"C:\Users\vipra\Downloads\volleyball_\videos",

    # Your real ball detection CSV.
    #
    # IMPORTANT:
    # This must correspond to the SAME sequence/video
    # as the annotations being evaluated.
    #
    # If you have one CSV per video, see the MULTI-VIDEO
    # section below.
    "ball_csv":
        r"D:\capstone\GRITSENSE\Cappy_95_GNN_HRN\ball detection\output_test\sample3\ball.csv",

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.2,

    "hidden_channels": 32,
    "out_channels": 32,
    "heads": 4,

    "train_split": 0.8,
    "seed": 42,

    # minimum ball confidence represented by Visibility=1
    "require_ball_detection": True,

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    "output_dir":
        r"evaluation_results",
}


# ============================================================
# CLASS LABELS
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

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        print("Using CUDA")

        return torch.device("cuda")

    print("Using CPU")

    return torch.device("cpu")


# ============================================================
# READ ANNOTATIONS.TXT
# ============================================================

def parse_annotations(annotation_file):

    """
    Returns:

        {
            frame_number: {
                "filename": "...",
                "activity": "...",
                "players": [...]
            }
        }

    Player:

        {
            "id": int,
            "x": float,
            "y": float,
            "w": float,
            "h": float,
            "action": str
        }
    """

    annotation_file = Path(annotation_file)

    if not annotation_file.exists():

        raise FileNotFoundError(
            f"Annotation file not found:\n{annotation_file}"
        )

    annotations = {}

    with annotation_file.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            filename = parts[0]

            # ------------------------------------------------
            # Frame number
            # ------------------------------------------------

            stem = Path(filename).stem

            try:
                frame_id = int(stem)
            except ValueError:
                continue

            # ------------------------------------------------
            # Activity
            # ------------------------------------------------

            activity_raw = parts[1]

            activity = activity_raw.replace("-", "_")

            if activity not in CLASS_TO_ID:

                continue

            # ------------------------------------------------
            # Players
            # ------------------------------------------------

            player_tokens = parts[2:]

            players = []

            # Each player:
            #
            # x y w h action
            #
            # = 5 tokens

            player_id = 0

            i = 0

            while i + 4 < len(player_tokens):

                try:

                    x = float(player_tokens[i])
                    y = float(player_tokens[i + 1])
                    w = float(player_tokens[i + 2])
                    h = float(player_tokens[i + 3])

                    action = player_tokens[i + 4]

                except ValueError:

                    i += 5
                    continue

                players.append({
                    "id": player_id,
                    "x": x + w / 2.0,
                    "y": y + h / 2.0,
                    "w": w,
                    "h": h,
                    "action": action,
                    "team": None,

                    # No pose information in annotations.txt.
                    "keypoints": [
                        [0.0, 0.0]
                        for _ in range(17)
                    ],

                    "vx": 0.0,
                    "vy": 0.0,

                    # Pose unavailable.
                    "pose_confidence": 0.0,
                })

                player_id += 1

                i += 5

            if len(players) == 0:
                continue

            annotations[frame_id] = {
                "filename": filename,
                "activity": activity,
                "label": CLASS_TO_ID[activity],
                "players": players,
            }

    return annotations


# ============================================================
# LOAD ALL DATASET ANNOTATIONS
# ============================================================

def load_dataset_annotations(dataset_root):

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():

        raise FileNotFoundError(
            f"Dataset root does not exist:\n{dataset_root}"
        )

    video_dirs = sorted([
        p for p in dataset_root.iterdir()
        if p.is_dir()
    ])

    print()
    print("=" * 70)
    print("LOADING VOLLEYBALL DATASET")
    print("=" * 70)

    print(f"Dataset root: {dataset_root}")
    print(f"Folders found: {len(video_dirs)}")

    all_annotations = {}

    for video_dir in video_dirs:

        annotation_file = video_dir / "annotations.txt"

        if not annotation_file.exists():
            continue

        parsed = parse_annotations(annotation_file)

        for frame_id, data in parsed.items():

            # Unique key:
            # video_folder + frame_id
            key = (
                video_dir.name,
                frame_id
            )

            data["video"] = video_dir.name

            all_annotations[key] = data

    print(
        f"Total annotated frames: {len(all_annotations)}"
    )

    print()

    activities = Counter(
        item["activity"]
        for item in all_annotations.values()
    )

    print("Activity distribution:")

    for activity in CLASS_NAMES:

        print(
            f"  {activity:12s}: "
            f"{activities.get(activity, 0)}"
        )

    return all_annotations


# ============================================================
# BALL CSV
# ============================================================

def load_ball_csv(csv_path):

    """
    Reads:

        Frame,Visibility,X,Y,Radius

    Returns:

        {
            frame_number: {
                "x": ...,
                "y": ...,
                "radius": ...,
                "detected": True/False
            }
        }
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Ball CSV not found:\n{csv_path}"
        )

    ball_data = {}

    with csv_path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            try:

                frame = int(float(row["Frame"]))

            except Exception:

                continue

            try:

                visibility = int(
                    float(row["Visibility"])
                )

            except Exception:

                visibility = 0

            try:

                x = float(row["X"])
                y = float(row["Y"])

            except Exception:

                x = -1.0
                y = -1.0

            try:

                radius = float(row["Radius"])

            except Exception:

                radius = 0.0

            detected = (
                visibility == 1
                and x >= 0
                and y >= 0
            )

            ball_data[frame] = {

                "x": x,
                "y": y,
                "radius": radius,
                "detected": detected,
            }

    return ball_data


# ============================================================
# TEAM ASSIGNMENT
# ============================================================

def assign_teams(players, image_width=1920.0):

    """
    Volleyball camera is treated as left/right court.

    Players left of the court center:
        Team 0

    Players right of the court center:
        Team 1

    This is deterministic and avoids the previous
    sort-and-split approach.
    """

    net_x = image_width / 2.0

    output = []

    for player in players:

        p = dict(player)

        if p["x"] < net_x:

            p["team"] = 0

        else:

            p["team"] = 1

        output.append(p)

    return output


# ============================================================
# FIND BALL/ANNOTATION ALIGNMENT
# ============================================================

def align_data(annotations, ball_data):

    """
    IMPORTANT:

    The Volleyball Dataset frame names are not necessarily
    the same numbering convention as the ball detector.

    Therefore we first check direct frame-number overlap.

    """

    annotation_frames = set(
        annotations.keys()
    )

    ball_frames = set(
        ball_data.keys()
    )

    direct_overlap = (
        annotation_frames &
        ball_frames
    )

    print()
    print("=" * 70)
    print("DATA ALIGNMENT")
    print("=" * 70)

    print(
        f"Annotation frames : {len(annotation_frames)}"
    )

    print(
        f"Ball CSV frames   : {len(ball_frames)}"
    )

    print(
        f"Direct overlap    : {len(direct_overlap)}"
    )

    return direct_overlap


# ============================================================
# GRAPH DATASET
# ============================================================

class VolleyballGraphDataset(Dataset):

    def __init__(
        self,
        samples,
        graph_builder,
    ):

        self.samples = samples

        self.graph_builder = graph_builder

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        players = assign_teams(
            sample["players"]
        )

        ball = dict(
            sample["ball"]
        )

        court = {

            "width": 1920.0,

            "height": 1080.0,

            "net_x": 960.0,
        }

        graph = self.graph_builder.build_graph(

            players=players,

            ball=ball,

            court=court,

            frame_id=sample["frame_id"],
        )

        graph.y = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        graph.activity = sample["activity"]

        graph.video = sample["video"]

        return graph


# ============================================================
# COLLATE
# ============================================================

def collate_graphs(batch):

    return Batch.from_data_list(batch)


# ============================================================
# CREATE SAMPLES
# ============================================================

def create_samples(
    annotations,
    ball_data,
):

    samples = []

    # --------------------------------------------------------
    # DIRECT FRAME ALIGNMENT
    # --------------------------------------------------------

    for key, annotation in annotations.items():

        video, frame_id = key

        if frame_id not in ball_data:
            continue

        ball = ball_data[frame_id]

        if not ball["detected"]:
            continue

        samples.append({

            "video": video,

            "frame_id": frame_id,

            "activity": annotation["activity"],

            "label": annotation["label"],

            "players": annotation["players"],

            "ball": ball,
        })

    return samples


# ============================================================
# MODEL
# ============================================================

def create_model(device):

    model = HRNClassifier(

        in_channels=49,

        hidden_channels=CONFIG["hidden_channels"],

        out_channels=CONFIG["out_channels"],

        heads=CONFIG["heads"],

        dropout=CONFIG["dropout"],
    )

    return model.to(device)


# ============================================================
# MODEL FORWARD
# ============================================================

def forward_model(
    model,
    batch,
    device,
):

    batch = batch.to(device)

    output = model(

        batch.x,

        batch.edge_index,

        batch.edge_attr,

        batch.node_type,

        batch.node_team,
    )

    return output


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    for batch in loader:

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

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() *
            labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    return {

        "loss":
            total_loss /
            max(total_samples, 1),

        "accuracy":
            total_correct /
            max(total_samples, 1),
    }


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    y_true = []

    y_pred = []

    with torch.no_grad():

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

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                batch.y.view(-1)
                .cpu()
                .numpy()
                .tolist()
            )

            y_pred.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            y_true,

            y_pred,

            labels=list(
                range(len(CLASS_NAMES))
            ),

            average="weighted",

            zero_division=0,
        )
    )

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=list(
            range(len(CLASS_NAMES))
        ),
    )

    report = classification_report(

        y_true,

        y_pred,

        labels=list(
            range(len(CLASS_NAMES))
        ),

        target_names=CLASS_NAMES,

        output_dict=True,

        zero_division=0,
    )

    return {

        "accuracy": float(accuracy),

        "precision_weighted":
            float(precision),

        "recall_weighted":
            float(recall),

        "f1_weighted":
            float(f1),

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report,
    }


# ============================================================
# ABLATION
# ============================================================

def clone_graph_with_ablation(
    graph,
    mode,
):

    """
    Three feature/relationship configurations:

    FULL:
        all node features
        all edges

    SPATIAL_ONLY:
        preserve node spatial features
        remove relational edges

    PLAYER_RELATIONS:
        keep player-player relations
        remove ball relations
    """

    graph = graph.clone()

    if mode == "full":

        return graph

    # --------------------------------------------------------
    # Identify edge types.
    #
    # edge_type:
    #   0 = player-player
    #   1 = player-ball
    # --------------------------------------------------------

    edge_type = graph.edge_type

    if mode == "spatial_only":

        # Remove all edges.
        #
        # GAT can still operate on isolated nodes.
        #
        # We keep self-free graph because the purpose is
        # measuring the contribution of relational edges.

        graph.edge_index = torch.empty(

            (2, 0),

            dtype=torch.long,

            device=graph.x.device,
        )

        graph.edge_attr = torch.empty(

            (0, 4),

            dtype=torch.float32,

            device=graph.x.device,
        )

        graph.edge_type = torch.empty(

            (0,),

            dtype=torch.long,

            device=graph.x.device,
        )

    elif mode == "player_relations":

        mask = (
            edge_type == 0
        )

        graph.edge_index = (
            graph.edge_index[:, mask]
        )

        graph.edge_attr = (
            graph.edge_attr[mask]
        )

        graph.edge_type = (
            graph.edge_type[mask]
        )

    else:

        raise ValueError(
            f"Unknown ablation mode: {mode}"
        )

    return graph


class AblationDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        mode,
    ):

        self.base_dataset = base_dataset

        self.mode = mode

    def __len__(self):

        return len(self.base_dataset)

    def __getitem__(self, index):

        graph = self.base_dataset[index]

        return clone_graph_with_ablation(
            graph,
            self.mode
        )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    train_dataset,
    val_dataset,
    device,
):

    train_loader = DataLoader(

        train_dataset,

        batch_size=1,

        shuffle=True,

        collate_fn=collate_graphs,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=1,

        shuffle=False,

        collate_fn=collate_graphs,
    )

    model = create_model(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(

        model.parameters(),

        lr=CONFIG["learning_rate"],

        weight_decay=CONFIG["weight_decay"],
    )

    best_accuracy = -1.0

    best_state = None

    history = []

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        CONFIG["epochs"] + 1
    ):

        train_metrics = train_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            device,
        )

        val_metrics = evaluate(

            model,

            val_loader,

            device,
        )

        print(

            f"Epoch {epoch:02d}/"
            f"{CONFIG['epochs']} | "

            f"Train Loss: "
            f"{train_metrics['loss']:.4f} | "

            f"Train Acc: "
            f"{train_metrics['accuracy'] * 100:.2f}% | "

            f"Val Acc: "
            f"{val_metrics['accuracy'] * 100:.2f}%"
        )

        history.append({

            "epoch": epoch,

            **train_metrics,

            "val_accuracy":
                val_metrics["accuracy"],

            "val_f1":
                val_metrics["f1_weighted"],
        })

        if (
            val_metrics["accuracy"]
            > best_accuracy
        ):

            best_accuracy = (
                val_metrics["accuracy"]
            )

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, history


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        CONFIG["seed"]
    )

    output_dir = Path(
        CONFIG["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    # --------------------------------------------------------
    # LOAD ANNOTATIONS
    # --------------------------------------------------------

    annotations = (
        load_dataset_annotations(
            CONFIG["dataset_root"]
        )
    )

    # --------------------------------------------------------
    # LOAD BALL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BALL DETECTIONS")
    print("=" * 70)

    ball_data = load_ball_csv(
        CONFIG["ball_csv"]
    )

    detected_count = sum(
        1
        for b in ball_data.values()
        if b["detected"]
    )

    print(
        f"Ball CSV frames: {len(ball_data)}"
    )

    print(
        f"Valid detections: {detected_count}"
    )

    print(
        f"Detection rate: "
        f"{100 * detected_count / max(len(ball_data), 1):.2f}%"
    )

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    # NOTE:
    # Current CSV must correspond to the same frame-number
    # sequence.
    #
    # If direct overlap is tiny, DO NOT train yet.
    # We need to generate the ball CSV for the actual
    # Volleyball Dataset sequence.

    overlap = align_data(
        annotations,
        ball_data
    )

    # --------------------------------------------------------
    # CREATE SAMPLES
    # --------------------------------------------------------

    samples = create_samples(

        annotations,

        ball_data,
    )

    print()
    print("=" * 70)
    print("USABLE GRAPH SAMPLES")
    print("=" * 70)

    print(
        f"Samples with player + ball data: "
        f"{len(samples)}"
    )

    if len(samples) < 20:

        raise RuntimeError(

            "\nNOT ENOUGH ALIGNED REAL DATA.\n\n"

            "The ball CSV does not appear to correspond "
            "to the Volleyball Dataset annotations.\n\n"

            f"Annotation/ball overlap: {len(overlap)}\n"
            f"Usable samples: {len(samples)}\n\n"

            "DO NOT train using sample3.\n"
            "Generate ball detections for the actual "
            "Volleyball Dataset sequence first."
        )

    # --------------------------------------------------------
    # VIDEO-LEVEL SPLIT
    # --------------------------------------------------------

    videos = sorted(
        set(
            s["video"]
            for s in samples
        )
    )

    random.shuffle(videos)

    split_idx = max(
        1,
        int(
            len(videos)
            * CONFIG["train_split"]
        )
    )

    train_videos = set(
        videos[:split_idx]
    )

    val_videos = set(
        videos[split_idx:]
    )

    # If only one video exists, fall back to frame split.
    if len(val_videos) == 0:

        random.shuffle(samples)

        split = int(
            len(samples)
            * CONFIG["train_split"]
        )

        train_samples = samples[:split]

        val_samples = samples[split:]

    else:

        train_samples = [
            s for s in samples
            if s["video"] in train_videos
        ]

        val_samples = [
            s for s in samples
            if s["video"] in val_videos
        ]

    print()
    print(
        f"Train samples: {len(train_samples)}"
    )

    print(
        f"Validation samples: {len(val_samples)}"
    )

    print(
        f"Train videos: {sorted(train_videos)}"
    )

    print(
        f"Validation videos: {sorted(val_videos)}"
    )

    if len(train_samples) == 0:
        raise RuntimeError(
            "No training samples."
        )

    if len(val_samples) == 0:
        raise RuntimeError(
            "No validation samples."
        )

    # --------------------------------------------------------
    # GRAPH BUILDER
    # --------------------------------------------------------

    graph_builder = (
        VolleyballGraphBuilder()
    )

    train_dataset = VolleyballGraphDataset(

        train_samples,

        graph_builder,
    )

    val_dataset = VolleyballGraphDataset(

        val_samples,

        graph_builder,
    )

    # --------------------------------------------------------
    # GRAPH SANITY CHECK
    # --------------------------------------------------------

    test_graph = train_dataset[0]

    print()
    print("=" * 70)
    print("GRAPH SANITY CHECK")
    print("=" * 70)

    print(
        f"Nodes       : {test_graph.x.shape}"
    )

    print(
        f"Edges       : {test_graph.edge_index.shape}"
    )

    print(
        f"Node feat   : {test_graph.x.shape[1]}"
    )

    print(
        f"Edge feat   : {test_graph.edge_attr.shape[1]}"
    )

    print(
        f"Node types  : "
        f"{test_graph.node_type.tolist()}"
    )

    print(
        f"Node teams  : "
        f"{test_graph.node_team.tolist()}"
    )

    print(
        f"Ball index  : "
        f"{test_graph.ball_index}"
    )

    # --------------------------------------------------------
    # TRAIN FULL MODEL
    # --------------------------------------------------------

    model, history = train_model(

        train_dataset,

        val_dataset,

        device,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        output_dir /
        "hrn_real_volleyball_best.pth"
    )

    torch.save(

        model.state_dict(),

        model_path,
    )

    print()
    print(
        f"Model saved to:\n{model_path}"
    )

    # --------------------------------------------------------
    # FULL MODEL EVALUATION
    # --------------------------------------------------------

    val_loader = DataLoader(

        val_dataset,

        batch_size=1,

        shuffle=False,

        collate_fn=collate_graphs,
    )

    full_results = evaluate(

        model,

        val_loader,

        device,
    )

    # --------------------------------------------------------
    # ABLATION EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RELATIONAL FEATURE ABLATION")
    print("=" * 70)

    ablation_results = {}

    for mode in [
        "spatial_only",
        "player_relations",
        "full",
    ]:

        print()
        print(
            f"Evaluating: {mode}"
        )

        if mode == "full":

            dataset = val_dataset

        else:

            dataset = AblationDataset(

                val_dataset,

                mode,
            )

        loader = DataLoader(

            dataset,

            batch_size=1,

            shuffle=False,

            collate_fn=collate_graphs,
        )

        result = evaluate(

            model,

            loader,

            device,
        )

        ablation_results[mode] = result

        print(
            f"Accuracy : "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{result['precision_weighted']:.4f}"
        )

        print(
            f"Recall   : "
            f"{result['recall_weighted']:.4f}"
        )

        print(
            f"F1       : "
            f"{result['f1_weighted']:.4f}"
        )

    # --------------------------------------------------------
    # RELATIONAL GAIN
    # --------------------------------------------------------

    spatial_acc = (
        ablation_results[
            "spatial_only"
        ]["accuracy"]
    )

    player_rel_acc = (
        ablation_results[
            "player_relations"
        ]["accuracy"]
    )

    full_acc = (
        ablation_results[
            "full"
        ]["accuracy"]
    )

    evaluation_summary = {

        "task":
            "Evaluate spatial and relational feature quality",

        "deliverable":
            "GNN evaluation results",

        "dataset":
            "Volleyball Dataset",

        "num_samples":
            len(samples),

        "train_samples":
            len(train_samples),

        "validation_samples":
            len(val_samples),

        "ball_detection_frames":
            len(ball_data),

        "ball_detection_valid":
            detected_count,

        "full_model":
            full_results,

        "ablation_results":
            ablation_results,

        "relational_gain_over_spatial":
            float(
                full_acc -
                spatial_acc
            ),

        "player_relation_gain_over_spatial":
            float(
                player_rel_acc -
                spatial_acc
            ),

        "training_history":
            history,
    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    results_path = (
        output_dir /
        "gnn_evaluation_results.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            evaluation_summary,

            f,

            indent=2,
        )

    # --------------------------------------------------------
    # SAVE HUMAN-READABLE TXT
    # --------------------------------------------------------

    txt_path = (
        output_dir /
        "gnn_evaluation_results.txt"
    )

    with txt_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "GRITSENSE GNN/HRN EVALUATION RESULTS\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Task: Evaluate spatial and relational feature quality\n"
        )

        f.write(
            "Deliverable: GNN evaluation results\n\n"
        )

        f.write(
            f"Total samples: {len(samples)}\n"
        )

        f.write(
            f"Training samples: {len(train_samples)}\n"
        )

        f.write(
            f"Validation samples: {len(val_samples)}\n\n"
        )

        f.write(
            "FULL MODEL\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Accuracy : "
            f"{full_results['accuracy'] * 100:.2f}%\n"
        )

        f.write(
            f"Precision: "
            f"{full_results['precision_weighted']:.4f}\n"
        )

        f.write(
            f"Recall   : "
            f"{full_results['recall_weighted']:.4f}\n"
        )

        f.write(
            f"F1       : "
            f"{full_results['f1_weighted']:.4f}\n\n"
        )

        f.write(
            "ABLATION STUDY\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        for mode, result in (
            ablation_results.items()
        ):

            f.write(
                f"\n{mode}\n"
            )

            f.write(
                f"Accuracy : "
                f"{result['accuracy'] * 100:.2f}%\n"
            )

            f.write(
                f"Precision: "
                f"{result['precision_weighted']:.4f}\n"
            )

            f.write(
                f"Recall   : "
                f"{result['recall_weighted']:.4f}\n"
            )

            f.write(
                f"F1       : "
                f"{result['f1_weighted']:.4f}\n"
            )

        f.write(
            "\nRELATIONAL GAIN\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Player relations over spatial: "
            f"{player_rel_acc - spatial_acc:+.4f}\n"
        )

        f.write(
            f"Full relations over spatial: "
            f"{full_acc - spatial_acc:+.4f}\n"
        )

        f.write(
            "\nCONFUSION MATRIX\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        for row in full_results[
            "confusion_matrix"
        ]:

            f.write(
                " ".join(
                    f"{v:5d}"
                    for v in row
                )
                + "\n"
            )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Full model accuracy : "
        f"{full_acc * 100:.2f}%"
    )

    print(
        f"Spatial-only accuracy: "
        f"{spatial_acc * 100:.2f}%"
    )

    print(
        f"Player-relation accuracy: "
        f"{player_rel_acc * 100:.2f}%"
    )

    print()
    print(
        f"Relational improvement: "
        f"{(full_acc - spatial_acc) * 100:+.2f} percentage points"
    )

    print()
    print(
        f"JSON results:\n{results_path}"
    )

    print(
        f"Text results:\n{txt_path}"
    )

    print(
        f"Checkpoint:\n{model_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()