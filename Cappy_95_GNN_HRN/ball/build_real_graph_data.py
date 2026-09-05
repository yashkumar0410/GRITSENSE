import json
import torch

from graph.graph_builder import (
    VolleyballGraphBuilder
)


# ============================================================
# CONFIGURATION
# ============================================================

MERGED_INPUT_PATH = (
    "merged_graph_input.json"
)

OUTPUT_GRAPH_PATH = (
    "real_graphs.pt"
)


# ============================================================
# COURT CONFIGURATION
#
# IMPORTANT:
# These are IMAGE dimensions because the graph builder
# expects image-space coordinates.
# ============================================================

COURT = {

    "width":
        1920.0,

    "height":
        1080.0,

    "net_x":
        960.0
}


# ============================================================
# LOAD MERGED DATA
# ============================================================

print(
    "Loading merged graph input..."
)

with open(
    MERGED_INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:

    merged_data = json.load(f)


# ============================================================
# GRAPH BUILDER
# ============================================================

builder = (
    VolleyballGraphBuilder()
)


print(
    "Graph builder initialized."
)

print(
    f"Expected node feature dimension: "
    f"{builder.feature_dim}"
)


# ============================================================
# GRAPH LIST
# ============================================================

graphs = []


# ============================================================
# STATISTICS
# ============================================================

skipped_no_ball = 0

skipped_players = 0

successful = 0

failed = 0


# ============================================================
# FRAME LOOP
# ============================================================

for frame_id in sorted(
    merged_data.keys(),
    key=lambda x: int(x)
):

    frame_data = (
        merged_data[
            frame_id
        ]
    )


    players = (
        frame_data.get(
            "players",
            []
        )
    )


    ball = (
        frame_data.get(
            "ball"
        )
    )


    # ========================================================
    # REQUIRE BALL
    #
    # We don't create fake ball coordinates.
    # ========================================================

    if ball is None:

        skipped_no_ball += 1

        continue


    # ========================================================
    # REQUIRE PLAYERS
    # ========================================================

    if len(players) == 0:

        skipped_players += 1

        continue


    # ========================================================
    # BUILD GRAPH
    # ========================================================

    try:

        graph = builder.build_graph(

            players=players,

            ball=ball,

            court=COURT,

            frame_id=int(
                frame_id
            )
        )


        graphs.append(
            graph
        )


        successful += 1


    except Exception as e:

        failed += 1


        print(
            f"Frame {frame_id} "
            f"failed: {e}"
        )


# ============================================================
# SAVE
# ============================================================

torch.save(
    graphs,
    OUTPUT_GRAPH_PATH
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print(
    "========================================"
)

print(
    "REAL GRAPH DATA CREATED"
)

print(
    "========================================"
)

print(
    f"Total input frames: "
    f"{len(merged_data)}"
)

print(
    f"Graphs created: "
    f"{successful}"
)

print(
    f"Skipped - no ball: "
    f"{skipped_no_ball}"
)

print(
    f"Skipped - no players: "
    f"{skipped_players}"
)

print(
    f"Failed: "
    f"{failed}"
)

print(
    f"Output: "
    f"{OUTPUT_GRAPH_PATH}"
)


# ============================================================
# INSPECT FIRST GRAPH
# ============================================================

if len(graphs) > 0:

    first_graph = graphs[0]


    print()
    print(
        "FIRST GRAPH"
    )

    print(
        "------------"
    )

    print(
        f"Frame ID: "
        f"{getattr(first_graph, 'frame_id', 'N/A')}"
    )

    print(
        f"Nodes: "
        f"{first_graph.num_nodes}"
    )

    print(
        f"Edges: "
        f"{first_graph.edge_index.shape[1]}"
    )

    print(
        f"Node features: "
        f"{first_graph.x.shape}"
    )

    print(
        f"Edge features: "
        f"{first_graph.edge_attr.shape}"
    )

    print(
        f"Ball index: "
        f"{first_graph.ball_index}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert (
        first_graph.x.shape[1]
        ==
        49
    )


    print()
    print(
        "49-dimensional node "
        "feature validation: PASS"
    )


print(
    "========================================"
)