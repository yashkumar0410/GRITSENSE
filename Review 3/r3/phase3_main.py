
from pathlib import Path

import torch
import numpy as np

from sample_data import (
    create_sample_data,
    get_available_frames,
)

from models import GAT, HRN

from graph_builder import (
    PLAYER_BALL_EDGE_TYPE,
    PLAYER_PLAYER_EDGE_TYPE,
    VolleyballGraphBuilder,
)

from visualize_graph import (
    visualize_feature_graph,
    visualize_graph,
    visualize_interaction_graph,
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "output_graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Directory for per-frame feature/dimension outputs (one file per frame)
DIMENSIONS_DIR = (
    Path(__file__).resolve().parent
    / "dimensions_output"
)

DIMENSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # GET ALL AVAILABLE FRAMES
    # ========================================================

    frames = get_available_frames()

    print(
        "\n========================================"
    )

    print(
        f"Total frames found: {len(frames)}"
    )

    print(
        f"First frame: {frames[0]}"
    )

    print(
        f"Last frame: {frames[-1]}"
    )

    print(
        "========================================\n"
    )

    # ========================================================
    # GRAPH BUILDER
    # ========================================================

    builder = VolleyballGraphBuilder()

    # ========================================================
    # CREATE GAT ONCE
    # ========================================================

    gat = GAT(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=0.2,
    )

    # ========================================================
    # CREATE HRN ONCE
    # ========================================================

    hrn = HRN(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=0.2,
    )

    # ========================================================
    # EVALUATION MODE
    # ========================================================

    gat.eval()
    hrn.eval()

    # ========================================================
    # PROCESS EVERY FRAME
    # ========================================================

    successful_frames = 0
    skipped_frames = 0

    for frame_number, frame_id in enumerate(frames):

        print(
            "\n\n"
            "========================================"
        )

        print(
            f"PROCESSING FRAME "
            f"{frame_id} "
            f"({frame_number + 1}/{len(frames)})"
        )

        print(
            "========================================"
        )

        try:

            # =================================================
            # CREATE INPUT DATA
            # =================================================

            players, ball, court = (
                create_sample_data(frame_id)
            )

            print(
                f"Players: {len(players)}"
            )

            print(
                f"Ball: ({ball['x']:.2f}, "
                f"{ball['y']:.2f})"
            )

            # =================================================
            # BUILD GRAPH
            # =================================================

            graph = builder.build_graph(
                players,
                ball,
                court
            )

            # =================================================
            # GRAPH INFORMATION
            # =================================================

            print(
                "\n========== GRAPH INFORMATION =========="
            )

            print(
                "Number of nodes:",
                graph.num_nodes
            )

            print(
                "Number of edges:",
                graph.num_edges
            )

            print(
                "Node feature shape:",
                graph.x.shape
            )

            print(
                "Edge index shape:",
                graph.edge_index.shape
            )

            print(
                "Edge feature shape:",
                graph.edge_attr.shape
            )

            print(
                "Edge type shape:",
                graph.edge_type.shape
            )

            # =================================================
            # NODE COUNTS
            # =================================================

            num_players = int(
                (
                    graph.node_type
                    == 0
                ).sum().item()
            )

            num_balls = int(
                (
                    graph.node_type
                    == 1
                ).sum().item()
            )

            # =================================================
            # EDGE COUNTS
            # =================================================

            num_player_player_edges = int(
                (
                    graph.edge_type
                    == PLAYER_PLAYER_EDGE_TYPE
                ).sum().item()
            )

            num_player_ball_edges = int(
                (
                    graph.edge_type
                    == PLAYER_BALL_EDGE_TYPE
                ).sum().item()
            )

            print(
                "Player nodes:",
                num_players
            )

            print(
                "Ball nodes:",
                num_balls
            )

            print(
                "Player-player edges:",
                num_player_player_edges
            )

            print(
                "Player-ball edges:",
                num_player_ball_edges
            )

            # =================================================
            # FEATURE GRAPH INFORMATION
            # =================================================

            print(
                "\n========== FEATURE GRAPH =========="
            )

            print(
                "Node feature dimension:",
                graph.x.size(1)
            )

            print(
                "Expected feature dimension:",
                49
            )

            print(
                "Feature graph valid:",
                graph.x.size(1) == 49
            )

            # =================================================
            # INTERACTION GRAPH INFORMATION
            # =================================================

            print(
                "\n========== INTERACTION GRAPH =========="
            )

            print(
                "Total interactions:",
                graph.num_edges
            )

            print(
                "Player-player interactions:",
                num_player_player_edges
            )

            print(
                "Player-ball interactions:",
                num_player_ball_edges
            )

            print(
                "Edge feature dimension:",
                graph.edge_attr.size(1)
            )

            # =================================================
            # EXAMPLE EDGE
            # =================================================

            if graph.num_edges > 0:

                print(
                    "\n========== EXAMPLE EDGE =========="
                )

                source = (
                    graph.edge_index[
                        0,
                        0
                    ].item()
                )

                target = (
                    graph.edge_index[
                        1,
                        0
                    ].item()
                )

                edge_kind = (
                    graph.edge_type[
                        0
                    ].item()
                )

                attributes = (
                    graph.edge_attr[
                        0
                    ].tolist()
                )

                print(
                    "Source:",
                    source
                )

                print(
                    "Target:",
                    target
                )

                print(
                    "Edge type:",
                    edge_kind
                )

                print(
                    "Attributes:",
                    attributes
                )

            # =================================================
            # VISUALIZATIONS
            # =================================================

            print(
                "\n========== VISUALIZATIONS =========="
            )

            # -------------------------------------------------
            # IMPORTANT
            #
            # These functions currently may display/save
            # according to visualize_graph.py.
            #
            # The frame title makes it possible to distinguish
            # the generated graph.
            # -------------------------------------------------

            visualize_graph(
                graph,
                frame_id,
                title=f"Volleyball Graph - Frame {frame_id}"
            )           

            visualize_feature_graph(
                graph,
                frame_id
            )

            visualize_interaction_graph(
                graph,
                frame_id
            )

            # =================================================
            # GAT
            # =================================================

            print(
                "\n========== GRAPH ATTENTION NETWORK =========="
            )

            with torch.no_grad():

                node_embeddings = gat(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                )

            print(
                "GAT input shape:",
                graph.x.shape
            )

            print(
                "GAT edge index shape:",
                graph.edge_index.shape
            )

            print(
                "GAT edge feature shape:",
                graph.edge_attr.shape
            )

            print(
                "GAT output shape:",
                node_embeddings.shape
            )

            print(
                "Number of attention heads:",
                4
            )

            print(
                "GAT output embedding dimension:",
                node_embeddings.size(1)
            )

            print(
                "GAT output finite:",
                torch.isfinite(
                    node_embeddings
                ).all().item()
            )

            # =================================================
            # HRN
            # =================================================

            print(
                "\n========== "
                "HIERARCHICAL RELATION NETWORK "
                "=========="
            )

            with torch.no_grad():

                hrn_output = hrn(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.node_type,
                    graph.node_team,
                )

            node_embeddings_hrn = (
                hrn_output[
                    "node_embeddings"
                ]
            )

            team_embeddings = (
                hrn_output[
                    "team_embeddings"
                ]
            )

            global_embedding = (
                hrn_output[
                    "global_embedding"
                ]
            )

            # =================================================
            # PRINT HRN INFORMATION
            # =================================================

            print(
                "\nHRN hierarchy:"
            )

            print(
                "Graph:",
                graph.x.shape
            )

            print(
                "GAT node embeddings:",
                node_embeddings_hrn.shape
            )

            print(
                "Team-level embeddings:",
                team_embeddings.shape
            )

            print(
                "Global embedding:",
                global_embedding.shape
            )

            print(
                "\nHRN architecture:"
            )

            print(
                "Graph -> GAT -> "
                "Node representations -> "
                "Team representations -> "
                "Global representation"
            )

            # =================================================
            # SAVE NUMERICAL GRAPH DATA
            # =================================================

            graph_file = (
                OUTPUT_DIR
                / f"graph_frame_{frame_id}.pt"
            )

            torch.save(
                {
                    "frame_id": frame_id,
                    "graph": graph,
                    "gat_node_embeddings":
                        node_embeddings.cpu(),
                    "hrn_node_embeddings":
                        node_embeddings_hrn.cpu(),
                    "team_embeddings":
                        team_embeddings.cpu(),
                    "global_embedding":
                        global_embedding.cpu(),
                },
                graph_file
            )

            # =================================================
            # SAVE PER-FRAME DIMENSIONAL FEATURES (one file)
            # =================================================
            def _to_numpy(tensor_like):
                if isinstance(tensor_like, torch.Tensor):
                    return tensor_like.detach().cpu().numpy()
                try:
                    return np.array(tensor_like)
                except Exception:
                    return None

            dims_file = DIMENSIONS_DIR / f"frame_{frame_id}.npz"

            np.savez_compressed(
                dims_file,
                node_features=_to_numpy(graph.x),
                edge_index=_to_numpy(graph.edge_index),
                edge_attr=_to_numpy(graph.edge_attr),
                edge_type=_to_numpy(graph.edge_type),
                node_type=_to_numpy(graph.node_type),
                node_team=_to_numpy(graph.node_team),
                gat_node_embeddings=_to_numpy(node_embeddings),
                hrn_node_embeddings=_to_numpy(node_embeddings_hrn),
                team_embeddings=_to_numpy(team_embeddings),
                global_embedding=_to_numpy(global_embedding),
            )

            print("\nSaved dimensions output:")
            print(dims_file)

            print(
                "\nSaved graph:"
            )

            print(
                graph_file
            )

            successful_frames += 1

            print(
                f"\nFRAME {frame_id} COMPLETE"
            )

        except Exception as e:

            skipped_frames += 1

            print(
                f"\nERROR processing frame "
                f"{frame_id}:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            print(
                "Skipping this frame and "
                "continuing..."
            )

            continue

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        "========================================"
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "Total frames:",
        len(frames)
    )

    print(
        "Successfully processed:",
        successful_frames
    )

    print(
        "Skipped:",
        skipped_frames
    )

    print(
        "Graph files:",
        OUTPUT_DIR
    )

    print(
        "========================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
