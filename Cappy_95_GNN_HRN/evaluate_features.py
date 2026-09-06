"""Evaluate spatial and relational graph features on the actual detector output."""

import json
from pathlib import Path

import numpy as np

from data.ball_player_loader import BallPlayerDataLoader
from graph.graph_builder import (
    PLAYER_BALL_EDGE_TYPE,
    PLAYER_PLAYER_EDGE_TYPE,
    VolleyballGraphBuilder,
)


ROOT = Path(__file__).parent
BALL_CSV = ROOT / "ball detection" / "output_test" / "sample3" / "ball.csv"
PLAYER_JSON = ROOT / "player_keypoints.json"
OUTPUT = ROOT / "outputs" / "feature_evaluation.json"
IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0


def main():
    loader = BallPlayerDataLoader(
        ball_csv_path=str(BALL_CSV),
        player_json_path=str(PLAYER_JSON),
        image_width=int(IMAGE_WIDTH),
        image_height=int(IMAGE_HEIGHT),
    )
    builder = VolleyballGraphBuilder()
    court = {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "net_x": IMAGE_WIDTH / 2}

    detected_frames = [
        frame for frame in loader.common_frames if loader.ball_data[frame]["detected"]
    ]
    spatial_distances = []
    graph_counts = []
    finite_graphs = 0
    reverse_relation_checks = 0
    reverse_relation_matches = 0
    coordinate_errors = []

    for frame in detected_frames:
        players, ball = loader.get_frame(frame)
        players = sorted(players, key=lambda player: float(player["x"]))
        midpoint = len(players) // 2
        for index, player in enumerate(players):
            player["team"] = 0 if index < midpoint else 1
        graph = builder.build_graph(players, ball, court, frame_id=frame)
        builder.validate_graph(graph)
        finite_graphs += int(np.isfinite(graph.x.numpy()).all() and np.isfinite(graph.edge_attr.numpy()).all())
        graph_counts.append({"nodes": graph.num_nodes, "edges": graph.num_edges})

        expected = np.array([ball["x"] / IMAGE_WIDTH, ball["y"] / IMAGE_HEIGHT])
        coordinate_errors.append(float(np.abs(graph.x[-1, 2:4].numpy() - expected).max()))

        ball_edges = graph.edge_type == PLAYER_BALL_EDGE_TYPE
        spatial_distances.extend(graph.edge_attr[ball_edges, 2].numpy().tolist())

        edge_map = {}
        for edge_index in range(graph.num_edges):
            source = int(graph.edge_index[0, edge_index])
            target = int(graph.edge_index[1, edge_index])
            edge_map[(source, target)] = graph.edge_attr[edge_index].numpy()
        for (source, target), attributes in edge_map.items():
            if (target, source) not in edge_map:
                continue
            reverse_relation_checks += 1
            reverse = edge_map[(target, source)]
            if np.allclose(attributes[:2], -reverse[:2]) and np.isclose(attributes[2], reverse[2]) and np.isclose(attributes[3], reverse[3]):
                reverse_relation_matches += 1

    graph_counts_array = np.asarray([[item["nodes"], item["edges"]] for item in graph_counts])
    results = {
        "data": {
            "ball_csv": str(BALL_CSV),
            "player_json": str(PLAYER_JSON),
            "aligned_frames": len(loader.common_frames),
            "detected_frames_evaluated": len(detected_frames),
            "ball_detection_coverage": loader.get_statistics()["ball_detection_rate"],
            "coordinate_system": "image pixels, normalized by 1920x1080",
            "missing_ball_frames_excluded": loader.get_statistics()["ball_missing"],
        },
        "graph": {
            "node_feature_dim": 49,
            "edge_feature_dim": 4,
            "finite_graphs": finite_graphs,
            "mean_nodes": float(graph_counts_array[:, 0].mean()),
            "mean_edges": float(graph_counts_array[:, 1].mean()),
        },
        "spatial_feature_quality": {
            "metric": "normalized player-ball edge distance",
            "mean": float(np.mean(spatial_distances)),
            "std": float(np.std(spatial_distances)),
            "ball_coordinate_max_abs_feature_error": float(max(coordinate_errors)),
            "interpretation": "Structural proxy only; no spatial ground-truth target is present.",
        },
        "relational_feature_quality": {
            "metric": "directed-edge reverse consistency",
            "checks": reverse_relation_checks,
            "matches": reverse_relation_matches,
            "rate": float(reverse_relation_matches / reverse_relation_checks) if reverse_relation_checks else 0.0,
            "interpretation": "Structural proxy only; no ground-truth relation labels are present.",
        },
        "limitations": [
            "No real activity labels are present for these frames, so supervised training and classification metrics are unavailable.",
            "No trained checkpoint was available for learned embedding evaluation.",
            "Old ball-detection data was not available, so direct old-vs-new comparison was not performed.",
        ],
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
