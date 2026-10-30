import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from models.attention_models.RCRG_R2_C11_conc_temporal import (
    GroupActivityClassifer,
    PersonActivityClassifier,
)
from utils.helper_utils.helper import Config


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_input_path(path):
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict group activity from a video and its player-detection JSON."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "pretrained_checkpoints/RCRG-R2-C11-conc-temporal-V1/"
            "checkpoint_epoch_42.pkl"
        ),
    )
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument(
        "--window-size",
        type=int,
        default=9,
        help="Frames per prediction window (default: 9, matching the training clips).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=9,
        help="Frame step between windows (default: 9; use 1 for overlapping windows).",
    )
    parser.add_argument(
        "--actual-label",
        help=(
            "Optional ground-truth group-activity label for this whole video. "
            "When supplied, print single-video top-1 accuracy."
        ),
    )
    return parser.parse_args()


def load_checkpoint(path, device):
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([Config]):
            return torch.load(
                path, map_location=device, weights_only=True
            )
    return torch.load(path, map_location=device)


def prepare_frame(detections, frame, transform):
    height, width = frame.shape[:2]
    players = list(detections.values())
    if not players:
        raise ValueError("A frame has no player detections.")

    if len(players) > 12:
        by_team = {"left": [], "right": []}
        for player in players:
            team = player.get("team_label")
            if team in by_team:
                by_team[team].append(player)
        selected = []
        for team_players in by_team.values():
            selected.extend(
                sorted(
                    team_players,
                    key=lambda player: float(player.get("confidence", 0)),
                    reverse=True,
                )[:6]
            )
        if len(selected) < 12:
            selected_ids = {id(player) for player in selected}
            remaining = [
                player for player in players if id(player) not in selected_ids
            ]
            selected.extend(
                sorted(
                    remaining,
                    key=lambda player: float(player.get("confidence", 0)),
                    reverse=True,
                )[: 12 - len(selected)]
            )
        players = selected[:12]
    players.sort(key=lambda player: (player["bbox"][0] + player["bbox"][2]) / 2)

    crops = []
    for player in players:
        x1, y1, x2, y2 = (int(round(value)) for value in player["bbox"])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid player bounding box: {player['bbox']}")
        crop = frame[y1:y2, x1:x2]
        crops.append(transform(image=crop)["image"])

    while len(crops) < 12:
        crops.append(torch.zeros_like(crops[0]))
    return torch.stack(crops)


def predict_window(model, visual_features, edge_index):
    sequence_length, num_players, _ = visual_features.shape

    # Match GroupActivityClassifer.forward's person-major flattening and views
    # so windowed predictions use the same checkpoint-compatible ordering.
    person_major_features = visual_features.permute(1, 0, 2).contiguous()
    graph_input = person_major_features.view(sequence_length, num_players, -1)
    graph_embeddings = model.r2(model.r1(graph_input, edge_index), edge_index)
    player_sequences = graph_embeddings.contiguous().view(
        num_players, sequence_length, -1
    )
    temporal_features, _ = model.lstm(player_sequences)
    temporal_features = model.ln3(temporal_features)
    combined = model.ln4(
        model.proj(player_sequences[:, -1, :]) + temporal_features[:, -1, :]
    )
    return model.fc(combined.reshape(1, -1))


def format_accuracy_summary(predicted_label, actual_label):
    if actual_label is None:
        return "Single-video top-1 accuracy: N/A (ground-truth label not provided)"
    correct = predicted_label == actual_label
    accuracy = 100.0 if correct else 0.0
    result = "correct" if correct else "incorrect"
    return (
        f"Single-video top-1 accuracy: {accuracy:.1f}% "
        f"(ground truth: {actual_label}; {result})"
    )


def main():
    args = parse_args()
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be at least 1")
    if args.window_size < 1 or args.stride < 1:
        raise ValueError("--window-size and --stride must be at least 1")
    args.video = resolve_input_path(args.video)
    args.detections = resolve_input_path(args.detections)
    args.checkpoint = resolve_input_path(args.checkpoint)
    for path in (args.video, args.detections, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_detections = json.loads(args.detections.read_text(encoding="utf-8"))
    detections = {int(frame_id): players for frame_id, players in raw_detections.items()}
    expected_ids = list(range(len(detections)))
    if sorted(detections) != expected_ids:
        raise ValueError("Detection JSON frame keys must be contiguous and start at 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    class_names = config.model["num_clases_label"]["group_activity"]
    if args.actual_label is not None and args.actual_label not in class_names:
        raise ValueError(
            f"Unknown --actual-label {args.actual_label!r}. "
            f"Choose one of: {', '.join(class_names)}"
        )
    person_model = PersonActivityClassifier(
        num_classes=config.model["num_classes"]["person_activity"],
        pretrained=False,
    )
    model = GroupActivityClassifer(
        person_feature_extraction=person_model,
        num_classes=len(class_names),
        device=device,
    )
    load_result = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    model.to(device).eval()
    print(
        "Checkpoint loaded strictly "
        f"(missing keys: {len(load_result.missing_keys)}, "
        f"unexpected keys: {len(load_result.unexpected_keys)})."
    )
    print("GAT/HRN graph branch: enabled (r1 -> r2).")

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.video}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if reported_frames and reported_frames != len(detections):
        capture.release()
        raise ValueError(
            f"Video reports {reported_frames} frames but the detection JSON has "
            f"{len(detections)}. They must be frame-aligned."
        )

    feature_batches = []
    frame_batch = []
    frame_index = 0
    try:
        with torch.inference_mode():
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index >= len(detections):
                    raise ValueError("Video has more frames than the detection JSON.")
                frame_batch.append(
                    prepare_frame(detections[frame_index], frame, transform)
                )
                frame_index += 1

                if len(frame_batch) == args.frame_batch_size:
                    crops = torch.stack(frame_batch).flatten(0, 1).to(device)
                    features = model.resnet50(crops).flatten(1)
                    feature_batches.append(
                        features.reshape(len(frame_batch), 12, -1).cpu()
                    )
                    frame_batch.clear()
                    if frame_index % 40 == 0:
                        print(f"Extracted features for {frame_index}/{len(detections)} frames.")

            if frame_batch:
                crops = torch.stack(frame_batch).flatten(0, 1).to(device)
                features = model.resnet50(crops).flatten(1)
                feature_batches.append(
                    features.reshape(len(frame_batch), 12, -1).cpu()
                )
    finally:
        capture.release()

    if frame_index != len(detections):
        raise ValueError(
            f"Video decoded {frame_index} frames but the detection JSON has "
            f"{len(detections)}."
        )

    visual_features = torch.cat(feature_batches).to(device)
    sequence_length, num_players, _ = visual_features.shape
    if sequence_length < args.window_size:
        raise ValueError(
            f"Video has {sequence_length} frames, fewer than the requested "
            f"window size ({args.window_size})."
        )
    edge_index = torch.tensor(
        [
            (source, target)
            for source in range(num_players)
            for target in range(num_players)
            if source != target
        ],
        dtype=torch.long,
        device=device,
    ).t()

    window_starts = list(range(0, sequence_length - args.window_size + 1, args.stride))
    last_start = sequence_length - args.window_size
    if window_starts[-1] != last_start:
        window_starts.append(last_start)

    window_probabilities = []
    with torch.inference_mode():
        for start in window_starts:
            end = start + args.window_size
            logits = predict_window(
                model, visual_features[start:end], edge_index
            )
            probabilities = torch.softmax(logits, dim=1)[0].cpu()
            window_probabilities.append(probabilities)
            predicted_index = int(probabilities.argmax())
            start_seconds = start / fps if fps > 0 else 0.0
            end_seconds = end / fps if fps > 0 else 0.0
            print(
                f"Frames {start}-{end - 1} "
                f"({start_seconds:.2f}-{end_seconds:.2f}s): "
                f"{class_names[predicted_index]} "
                f"(score {probabilities[predicted_index].item():.4f})"
            )

    mean_probabilities = torch.stack(window_probabilities).mean(dim=0)
    prediction_index = int(mean_probabilities.argmax())
    print(f"Frames processed: {sequence_length}")
    print(f"Prediction windows: {len(window_starts)}")
    print(f"Mean-window prediction: {class_names[prediction_index]}")
    top_probabilities, top_indices = mean_probabilities.topk(
        min(2, len(class_names))
    )
    confidence = top_probabilities[0].item()
    print(f"Top-1 confidence (mean-window probability): {confidence:.4f}")
    if len(class_names) > 1:
        print(
            "Top-1 vs top-2 margin: "
            f"{(top_probabilities[0] - top_probabilities[1]).item():.4f}"
        )
    print("Mean class probabilities:")
    for name, probability in zip(class_names, mean_probabilities.tolist()):
        print(f"  {name}: {probability:.4f}")
    print(
        format_accuracy_summary(
            class_names[prediction_index], args.actual_label
        )
    )


if __name__ == "__main__":
    main()
