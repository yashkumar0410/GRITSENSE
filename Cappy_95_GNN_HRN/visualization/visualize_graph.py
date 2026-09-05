import numpy as np
import matplotlib.pyplot as plt


PLAYER_NODE_TYPE = 0
BALL_NODE_TYPE = 1

PLAYER_PLAYER_EDGE_TYPE = 0
PLAYER_BALL_EDGE_TYPE = 1


def visualize_graph(
    graph,
    title="Volleyball Feature + Interaction Graph"
):
    """
    Visualize the graph representation.

    The visualization uses:

        graph.x[:, 2] -> normalized X
        graph.x[:, 3] -> normalized Y

    Nodes:
        Player
        Ball

    Edges:
        Solid  -> Player-Player
        Dashed -> Player-Ball
    """

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
    # node_team: optional per-node team metadata (torch tensor)
    node_team = getattr(graph, "node_team", None)
    if node_team is not None:
        node_team = node_team.detach().cpu().numpy()

    # --------------------------------------------------------
    # Extract node positions
    # --------------------------------------------------------

    x_positions = node_features[:, 2]
    y_positions = node_features[:, 3]

    player_mask = (
        node_type == PLAYER_NODE_TYPE
    )

    ball_mask = (
        node_type == BALL_NODE_TYPE
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
    # Draw interaction edges first
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

        x1 = x_positions[
            source
        ]

        y1 = y_positions[
            source
        ]

        x2 = x_positions[
            target
        ]

        y2 = y_positions[
            target
        ]

        edge_kind = (
            edge_type[edge_idx]
        )

        if (
            edge_kind
            == PLAYER_PLAYER_EDGE_TYPE
        ):

            # Player-player interaction
            linestyle = "-"
            alpha = 0.25

        elif (
            edge_kind
            == PLAYER_BALL_EDGE_TYPE
        ):

            # Player-ball interaction
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
    # Draw player nodes
    # --------------------------------------------------------

    # Color players by team: team 0, team 1, unknown
    if node_team is not None:
        team_color_map = {
            0: "tab:blue",
            1: "tab:orange",
            -1: "gray",
        }

        player_colors = [
            team_color_map.get(int(node_team[idx]), "gray")
            for idx in player_indices
        ]

    else:
        player_colors = None

    plt.scatter(
        x_positions[player_indices],
        y_positions[player_indices],
        s=100,
        c=player_colors,
        label="Player"
    )

    # --------------------------------------------------------
    # Draw ball node
    # --------------------------------------------------------

    plt.scatter(
        x_positions[
            ball_indices
        ],
        y_positions[
            ball_indices
        ],
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

    # Image coordinates increase downward.
    plt.ylim(
        1,
        0
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.show()


def visualize_feature_graph(
    graph
):
    """
    Visualize the feature graph separately.

    Each node is displayed with its 49-dimensional
    feature vector represented as a horizontal image.
    """

    features = (
        graph.x
        .detach()
        .cpu()
        .numpy()
    )

    plt.figure(
        figsize=(14, 6)
    )

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
        "49-Dimensional Node Feature Graph"
    )

    plt.colorbar(
        label="Normalized Feature Value"
    )

    plt.tight_layout()

    plt.show()


def visualize_interaction_graph(
    graph
):
    """
    Visualize only the interaction structure.

    Player-player edges are shown separately
    from player-ball edges.
    """

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

    # Extract node_team for coloring if present
    node_team = getattr(graph, "node_team", None)
    if node_team is not None:
        node_team = node_team.detach().cpu().numpy()

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

    plt.figure(
        figsize=(12, 7)
    )

    # --------------------------------------------------------
    # Player-player interactions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Player-ball interactions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    plt.scatter(
        x_positions[player_indices],
        y_positions[player_indices],
        s=120,
        c=[
            ("tab:blue" if node_team[idx] == 0 else ("tab:orange" if node_team[idx] == 1 else "gray"))
            for idx in player_indices
        ] if node_team is not None else None,
        label="Players"
    )

    plt.scatter(
        x_positions[ball_indices],
        y_positions[ball_indices],
        s=180,
        marker="o",
        label="Ball"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    player_ids = getattr(
        graph,
        "player_ids",
        None
    )

    if player_ids is not None:

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

    plt.xlabel(
        "Normalized X"
    )

    plt.ylabel(
        "Normalized Y"
    )

    plt.title(
        "Interaction Graph"
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

    plt.show()