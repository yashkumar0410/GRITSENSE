"""
Pretrained group-activity GAT (GAT/ folder, RCRG-R2-C11-conc-temporal-V1).

Runs alongside the project's own GAT/HRN (models/gat.py, models/hrn.py): it
reads the same video and player JSON and classifies each 9-frame window as one
of the Volleyball-dataset group activities (l/r set, spike, pass, winpoint).

The GAT/ folder is used as-is. Only its relational attention layer is loaded
(by file path, so its `models`/`utils` packages don't clash with ours), and the
checkpoint is unpickled without importing its training utilities, so
albumentations / sklearn / seaborn / torchinfo are not needed.

Run from the project root:
    python group_activity.py --video sample3.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import pickle
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

ROOT = Path(__file__).resolve().parent
GAT_DIR = ROOT / "GAT"
DEFAULT_CHECKPOINT = (
    GAT_DIR / "pretrained_checkpoints" / "RCRG-R2-C11-conc-temporal-V1" / "checkpoint_epoch_42.pkl"
)
DEFAULT_OUT = ROOT / "output_recent" / "group_activity" / "group_activity.json"

MAX_PLAYERS = 12
CROP_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_relational_unit():
    path = GAT_DIR / "models" / "attention_models" / "relational_attention.py"
    spec = importlib.util.spec_from_file_location("gat_pretrained_relational_attention", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # torch_geometric inspects message() via sys.modules
    spec.loader.exec_module(module)
    return module.RelationalUnit


class _Config:
    """Stand-in for GAT/utils/helper_utils/helper.Config stored in the checkpoint."""

    def __repr__(self):
        return f"Config(model={getattr(self, 'model', {})})"


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Config" and module.endswith("helper"):
            return _Config
        return super().find_class(module, name)


def _load_checkpoint(path: Path, device):
    pickle_module = types.ModuleType("gat_checkpoint_pickle")
    pickle_module.__dict__.update(pickle.__dict__)
    pickle_module.Unpickler = _CheckpointUnpickler
    return torch.load(path, map_location=device, weights_only=False, pickle_module=pickle_module)


class GroupActivityClassifer(nn.Module):
    """Same layers/names as GAT/models/attention_models/RCRG_R2_C11_conc_temporal.py."""

    def __init__(self, num_classes, relational_unit):
        super().__init__()
        self.resnet50 = nn.Sequential(*list(tv_models.resnet50(weights=None).children())[:-1])
        self.r1 = relational_unit(in_channels=2048, out_channels=2048, num_heads=4, dropout_rate=0.5)
        self.r2 = relational_unit(in_channels=2048, out_channels=2048, num_heads=4, dropout_rate=0.5)
        self.proj = nn.Linear(2048, 512)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(512)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(MAX_PLAYERS * 512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def predict_window(self, visual_features, edge_index):
        """visual_features: (seq, 12, 2048) -> logits (1, num_classes).

        Mirrors GAT/infer_video.py: the original forward() reshapes person-major
        features, so the same flattening order is kept for checkpoint parity.
        """
        seq, num_players, _ = visual_features.shape
        graph_input = visual_features.permute(1, 0, 2).contiguous().view(seq, num_players, -1)
        x_r2 = self.r2(self.r1(graph_input, edge_index), edge_index)
        x_r2 = x_r2.contiguous().view(num_players, seq, -1)
        x_t, _ = self.lstm(x_r2)
        x_t = self.ln3(x_t)
        combined = self.ln4(self.proj(x_r2[:, -1, :]) + x_t[:, -1, :])
        return self.fc(combined.reshape(1, -1))


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path, device)
    class_names = list(checkpoint["config"].model["num_clases_label"]["group_activity"])
    model = GroupActivityClassifer(len(class_names), _load_relational_unit())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, class_names, device


def _select_players(players):
    """Up to 12 players, 6 per team by confidence when crowded, sorted left to right."""
    players = list(players)
    if len(players) > MAX_PLAYERS:
        selected = []
        for team in ("left", "right"):
            team_players = [p for p in players if p.get("team_label") == team]
            team_players.sort(key=lambda p: float(p.get("confidence", 0)), reverse=True)
            selected.extend(team_players[: MAX_PLAYERS // 2])
        if len(selected) < MAX_PLAYERS:
            chosen = {id(p) for p in selected}
            rest = [p for p in players if id(p) not in chosen]
            rest.sort(key=lambda p: float(p.get("confidence", 0)), reverse=True)
            selected.extend(rest[: MAX_PLAYERS - len(selected)])
        players = selected[:MAX_PLAYERS]
    players.sort(key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2)
    return players


def _frame_crops(frame, players):
    """(12, 3, 224, 224) normalised BGR crops, zero-padded like the training collate."""
    height, width = frame.shape[:2]
    crops = torch.zeros((MAX_PLAYERS, 3, CROP_SIZE, CROP_SIZE), dtype=torch.float32)
    for slot, player in enumerate(_select_players(players)):
        x1, y1, x2, y2 = (int(round(v)) for v in player["bbox"])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = cv2.resize(frame[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        crop = (crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        crops[slot] = torch.from_numpy(crop.transpose(2, 0, 1))
    return crops


def run(video_path: Path, players_path: Path, out_path: Path = DEFAULT_OUT,
        checkpoint_path: Path = DEFAULT_CHECKPOINT, window_size: int = 9, stride: int = 9,
        frame_batch_size: int = 4):
    """Classify every window of the video and write per-window + overall predictions."""
    model, class_names, device = load_model(checkpoint_path)
    print(f"Pretrained group-activity GAT loaded on {device} ({len(class_names)} classes).")

    raw = json.loads(Path(players_path).read_text(encoding="utf-8"))
    detections = {int(k): v for k, v in raw.items()}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    feature_batches, frame_batch = [], []
    frame_index = 0

    def flush():
        crops = torch.stack(frame_batch).flatten(0, 1).to(device)
        features = model.resnet50(crops).flatten(1)
        feature_batches.append(features.reshape(len(frame_batch), MAX_PLAYERS, -1).cpu())
        frame_batch.clear()

    try:
        with torch.inference_mode():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                players = detections.get(frame_index, {})
                frame_batch.append(_frame_crops(frame, players.values() if isinstance(players, dict) else players))
                frame_index += 1
                if len(frame_batch) == frame_batch_size:
                    flush()
                    if frame_index % 40 == 0:
                        print(f"  features extracted for {frame_index} frames")
            if frame_batch:
                flush()
    finally:
        capture.release()

    if not feature_batches:
        raise ValueError(f"No frames decoded from {video_path}")
    visual_features = torch.cat(feature_batches).to(device)
    num_frames = visual_features.shape[0]
    window_size = min(window_size, num_frames)

    edge_index = torch.tensor(list(itertools.permutations(range(MAX_PLAYERS), 2)),
                              dtype=torch.long, device=device).t()
    starts = list(range(0, num_frames - window_size + 1, stride))
    if starts[-1] != num_frames - window_size:
        starts.append(num_frames - window_size)

    windows, probs_all = [], []
    with torch.inference_mode():
        for start in starts:
            end = start + window_size
            probs = torch.softmax(model.predict_window(visual_features[start:end], edge_index), dim=1)[0].cpu()
            probs_all.append(probs)
            best = int(probs.argmax())
            windows.append({
                "start_frame": start,
                "end_frame": end - 1,
                "start_s": round(start / fps, 3),
                "end_s": round(end / fps, 3),
                "label": class_names[best],
                "score": round(float(probs[best]), 4),
                "probabilities": {n: round(float(p), 4) for n, p in zip(class_names, probs.tolist())},
            })
            print(f"  frames {start}-{end - 1}: {class_names[best]} ({float(probs[best]):.3f})")

    mean = torch.stack(probs_all).mean(dim=0)
    best = int(mean.argmax())
    result = {
        "model": "RCRG-R2-C11-conc-temporal-V1 (pretrained GAT, Volleyball dataset)",
        "checkpoint": str(Path(checkpoint_path).relative_to(ROOT)) if Path(checkpoint_path).is_relative_to(ROOT) else str(checkpoint_path),
        "video": str(video_path),
        "frames": num_frames,
        "window_size": window_size,
        "stride": stride,
        "classes": class_names,
        "overall": {
            "label": class_names[best],
            "score": round(float(mean[best]), 4),
            "probabilities": {n: round(float(p), 4) for n, p in zip(class_names, mean.tolist())},
        },
        "windows": windows,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Overall group activity: {class_names[best]} ({float(mean[best]):.3f}) -> {out_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained group-activity GAT inference.")
    parser.add_argument("--video", default="sample3.mp4")
    parser.add_argument("--players", default="player_keypoints_mf3_pose.json")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=9)
    return parser.parse_args()


def main():
    args = parse_args()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    run(resolve(args.video), resolve(args.players), resolve(args.out),
        resolve(args.checkpoint), args.window_size, args.stride)


if __name__ == "__main__":
    main()
