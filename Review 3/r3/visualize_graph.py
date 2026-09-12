
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


PLAYER_NODE_TYPE = 0
BALL_NODE_TYPE = 1

PLAYER_PLAYER_EDGE_TYPE = 0
PLAYER_BALL_EDGE_TYPE = 1


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


# ============================================================
# MAIN GRAPH VISUALIZATION
# ============================================================

def visualize_graph(
    graph,
    frame_id,
    title=None
):
    """
    Visualize the complete volleyball graph.

    Saves:
        output_graphs/frame_<frame_id>_graph.png

    Nodes:
        Player
        Ball

    Edges:
        Solid  -> Player-Player
        Dashed -> Player-Ball
    """

    if title is None:
        title = (
            f"Volleyball Feature + "
            f"Interaction Graph - Frame {frame_id}"
        )

    # --------------------------------------------------------
    # Extract graph data
    # --------------------------------------------------------

    node_features = (
        graph.x
        .detach()
        .cpu()
        .numpy()
    )

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
        .numpy()
    )

    edge_type = (
        graph.edge_type
        .detach()
        .cpu()
        .numpy()
    )

    node_type = (
        graph.node_type
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Optional team information
    # --------------------------------------------------------

    node_team = getattr(
        graph,
        "node_team",
        None
    )

    if node_team is not None:

        node_team = (
            node_team
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Extract node positions
    # --------------------------------------------------------

    x_positions = node_features[:, 2]
    y_positions = node_features[:, 3]

    player_mask = (
        node_type
        == PLAYER_NODE_TYPE
    )

    ball_mask = (
        node_type
        == BALL_NODE_TYPE
    )

    player_indices = np.where(
        player_mask
    )[0]

    ball_indices = np.where(
        ball_mask
    )[0]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------

    for edge_idx in range(
        edge_index.shape[1]
    ):

        source = edge_index[
            0,
            edge_idx
        ]

        target = edge_index[
            1,
            edge_idx
        ]

        x1 = x_positions[source]
        y1 = y_positions[source]

        x2 = x_positions[target]
        y2 = y_positions[target]

        edge_kind = (
            edge_type[edge_idx]
        )

        if (
            edge_kind
            == PLAYER_PLAYER_EDGE_TYPE
        ):

            linestyle = "-"
            alpha = 0.25

        elif (
            edge_kind
            == PLAYER_BALL_EDGE_TYPE
        ):

            linestyle = "--"
            alpha = 0.55

        else:

            linestyle = ":"
            alpha = 0.2

        plt.plot(
            [x1, x2],
            [y1, y2],
            linestyle=linestyle,
            alpha=alpha
        )

    # --------------------------------------------------------
    # Player colors
    # --------------------------------------------------------

    if node_team is not None:

        team_color_map = {
            0: "tab:blue",
            1: "tab:orange",
            -1: "gray",
        }

        player_colors = [
            team_color_map.get(
                int(node_team[idx]),
                "gray"
            )
            for idx in player_indices
        ]

    else:

        player_colors = None

    # --------------------------------------------------------
    # Draw players
    # --------------------------------------------------------

    if len(player_indices) > 0:

        plt.scatter(
            x_positions[player_indices],
            y_positions[player_indices],
            s=100,
            c=player_colors,
            label="Player"
        )

    # --------------------------------------------------------
    # Draw ball
    # --------------------------------------------------------

    if len(ball_indices) > 0:

        plt.scatter(
            x_positions[ball_indices],
            y_positions[ball_indices],
            s=150,
            marker="o",
            color="red",
            label="Ball"
        )

    # --------------------------------------------------------
    # Player labels
    # --------------------------------------------------------

    player_ids = getattr(
        graph,
        "player_ids",
        None
    )

    if player_ids is not None:

        # Convert tensor/list if necessary
        try:
            player_ids = (
                player_ids
                .detach()
                .cpu()
                .numpy()
            )
        except AttributeError:
            pass

        for node_idx, player_id in zip(
            player_indices,
            player_ids
        ):

            plt.annotate(
                f"P{player_id}",
                (
                    x_positions[node_idx],
                    y_positions[node_idx]
                ),
                xytext=(5, 5),
                textcoords="offset points"
            )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    plt.xlabel(
        "Normalized X"
    )

    plt.ylabel(
        "Normalized Y"
    )

    plt.title(
        title
    )

    plt.xlim(
        0,
        1
    )

    # Image coordinates increase downward
    plt.ylim(
        1,
        0
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"frame_{frame_id}_graph.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved graph: {output_file}"
    )


# ============================================================
# FEATURE GRAPH
# ============================================================

def visualize_feature_graph(
    graph,
    frame_id
):
    """
    Visualize the 49-dimensional node feature graph.

    Saves:
        output_graphs/frame_<frame_id>_features.png
    """

    features = (
        graph.x
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    # --------------------------------------------------------
    # Feature matrix
    # --------------------------------------------------------

    plt.imshow(
        features,
        aspect="auto"
    )

    plt.xlabel(
        "Feature Index"
    )

    plt.ylabel(
        "Node Index"
    )

    plt.title(
        f"49-Dimensional Node Feature Graph "
        f"- Frame {frame_id}"
    )

    plt.colorbar(
        label="Normalized Feature Value"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"frame_{frame_id}_features.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved feature graph: {output_file}"
    )


# ============================================================
# INTERACTION GRAPH
# ============================================================

def visualize_interaction_graph(
    graph,
    frame_id
):
    """
    Visualize only the interaction structure.

    Player-player edges:
        Solid

    Player-ball edges:
        Dashed

    Saves:
        output_graphs/frame_<frame_id>_interaction.png
    """

    # --------------------------------------------------------
    # Extract graph data
    # --------------------------------------------------------

    node_features = (
        graph.x
        .detach()
        .cpu()
        .numpy()
    )

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
        .numpy()
    )

    edge_type = (
        graph.edge_type
        .detach()
        .cpu()
        .numpy()
    )

    node_type = (
        graph.node_type
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Team information
    # --------------------------------------------------------

    node_team = getattr(
        graph,
        "node_team",
        None
    )

    if node_team is not None:

        node_team = (
            node_team
            .detach()
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    x_positions = node_features[:, 2]
    y_positions = node_features[:, 3]

    player_indices = np.where(
        node_type
        == PLAYER_NODE_TYPE
    )[0]

    ball_indices = np.where(
        node_type
        == BALL_NODE_TYPE
    )[0]

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    # ========================================================
    # PLAYER-PLAYER INTERACTIONS
    # ========================================================

    player_player_mask = (
        edge_type
        == PLAYER_PLAYER_EDGE_TYPE
    )

    player_player_edges = (
        edge_index[
            :,
            player_player_mask
        ]
    )

    for source, target in zip(
        player_player_edges[0],
        player_player_edges[1]
    ):

        plt.plot(
            [
                x_positions[source],
                x_positions[target]
            ],
            [
                y_positions[source],
                y_positions[target]
            ],
            linestyle="-",
            alpha=0.25
        )

    # ========================================================
    # PLAYER-BALL INTERACTIONS
    # ========================================================

    player_ball_mask = (
        edge_type
        == PLAYER_BALL_EDGE_TYPE
    )

    player_ball_edges = (
        edge_index[
            :,
            player_ball_mask
        ]
    )

    for source, target in zip(
        player_ball_edges[0],
        player_ball_edges[1]
    ):

        plt.plot(
            [
                x_positions[source],
                x_positions[target]
            ],
            [
                y_positions[source],
                y_positions[target]
            ],
            linestyle="--",
            alpha=0.55
        )

    # ========================================================
    # PLAYER NODES
    # ========================================================

    if len(player_indices) > 0:

        if node_team is not None:

            player_colors = [
                (
                    "tab:blue"
                    if node_team[idx] == 0
                    else (
                        "tab:orange"
                        if node_team[idx] == 1
                        else "gray"
                    )
                )
                for idx in player_indices
            ]

        else:

            player_colors = None

        plt.scatter(
            x_positions[player_indices],
            y_positions[player_indices],
            s=120,
            c=player_colors,
            label="Players"
        )

    # ========================================================
    # BALL
    # ========================================================

    if len(ball_indices) > 0:

        plt.scatter(
            x_positions[ball_indices],
            y_positions[ball_indices],
            s=180,
            marker="o",
            color="red",
            label="Ball"
        )

    # ========================================================
    # PLAYER LABELS
    # ========================================================

    player_ids = getattr(
        graph,
        "player_ids",
        None
    )

    if player_ids is not None:

        try:
            player_ids = (
                player_ids
                .detach()
                .cpu()
                .numpy()
            )
        except AttributeError:
            pass

        for node_idx, player_id in zip(
            player_indices,
            player_ids
        ):

            plt.annotate(
                f"P{player_id}",
                (
                    x_positions[node_idx],
                    y_positions[node_idx]
                ),
                xytext=(5, 5),
                textcoords="offset points"
            )

    # ========================================================
    # FORMATTING
    # ========================================================

    plt.xlabel(
        "Normalized X"
    )

    plt.ylabel(
        "Normalized Y"
    )

    plt.title(
        f"Interaction Graph "
        f"- Frame {frame_id}"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        1,
        0
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        OUTPUT_DIR
        / f"frame_{frame_id}_interaction.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved interaction graph: {output_file}"
    )
