"""
VISUALIZE RECENT GRAPH OUTPUTS

Reads:
    output_recent/graphs/

Writes:
    output_recent/graph_images/

The old output_graphs/ directory is never touched.
"""

from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DEFAULT_GRAPH_DIR = (
    ROOT / "output_recent" / "graphs"
)

DEFAULT_OUTPUT_DIR = (
    ROOT / "output_recent" / "graph_images"
)


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path):

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    # New pipeline saves a dictionary.
    if isinstance(data, dict):

        if "graph" in data:

            graph = data["graph"]

        else:

            raise RuntimeError(
                f"No 'graph' object found in {path}"
            )

        return data, graph

    # Also support direct PyG graph files.
    return {}, data


# ============================================================
# GET PLAYER NODES
# ============================================================

def get_player_nodes(graph):

    if not hasattr(
        graph,
        "node_type"
    ):

        return np.arange(
            graph.x.shape[0]
        )

    node_type = (
        graph.node_type
        .detach()
        .cpu()
        .numpy()
    )

    # Existing pipeline:
    # player nodes = node_type 0

    return np.where(
        node_type == 0
    )[0]


# ============================================================
# GET PLAYER IDS
# ============================================================

def get_player_ids(
    graph,
    player_nodes
):

    # Try common attribute names.

    for attribute in [
        "player_ids",
        "player_id",
        "ids",
    ]:

        if hasattr(
            graph,
            attribute
        ):

            values = getattr(
                graph,
                attribute
            )

            try:

                values = (
                    values
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

            except Exception:

                try:

                    values = np.asarray(
                        values
                    ).reshape(-1)

                except Exception:

                    continue

            if len(values) == (
                graph.x.shape[0]
            ):

                return [
                    str(values[i])
                    for i in player_nodes
                ]

            if len(values) == len(
                player_nodes
            ):

                return [
                    str(x)
                    for x in values
                ]

    # If IDs weren't stored directly,
    # use node indices rather than inventing IDs.

    return [
        f"node_{i}"
        for i in player_nodes
    ]


# ============================================================
# GET PLAYER POSITIONS
# ============================================================

def get_positions(
    graph,
    player_nodes
):

    # --------------------------------------------------------
    # Preferred:
    # first two spatial features.
    # --------------------------------------------------------

    x = (
        graph.x
        .detach()
        .cpu()
        .numpy()
    )

    # Existing feature layout has spatial information
    # before the pose block.

    if x.shape[1] >= 2:

        positions = (
            x[
                player_nodes,
                0:2
            ]
        )

        if np.isfinite(
            positions
        ).all():

            return positions

    # --------------------------------------------------------
    # Fallback: graph.pos
    # --------------------------------------------------------

    if hasattr(
        graph,
        "pos"
    ):

        pos = (
            graph.pos
            .detach()
            .cpu()
            .numpy()
        )

        return (
            pos[player_nodes]
        )

    # --------------------------------------------------------
    # Last fallback:
    # arrange nodes.
    #
    # This is only visualization fallback.
    # --------------------------------------------------------

    n = len(
        player_nodes
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    return np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
        ]
    )


# ============================================================
# GET EDGES
# ============================================================

def get_player_edges(
    graph,
    player_nodes
):

    if not hasattr(
        graph,
        "edge_index"
    ):

        return []

    edge_index = (
        graph.edge_index
        .detach()
        .cpu()
        .numpy()
    )

    player_set = set(
        player_nodes.tolist()
    )

    edges = []

    for src, dst in zip(
        edge_index[0],
        edge_index[1]
    ):

        src = int(src)
        dst = int(dst)

        if (
            src not in player_set
            or dst not in player_set
        ):

            continue

        # Convert global node indices
        # to player-array indices.

        edges.append(
            (
                src,
                dst
            )
        )

    return edges


# ============================================================
# POSE QUALITY
# ============================================================

def get_pose_quality(
    graph,
    player_nodes
):

    x = (
        graph.x
        .detach()
        .cpu()
    )

    pose = (
        x[
            player_nodes,
            8:42
        ]
    )

    populated = (
        torch.count_nonzero(
            pose,
            dim=1
        ) > 0
    )

    return (
        populated
        .numpy()
    )


# ============================================================
# DRAW ONE GRAPH
# ============================================================

def draw_graph(
    data,
    graph,
    output_path
):

    player_nodes = (
        get_player_nodes(
            graph
        )
    )

    if len(player_nodes) == 0:

        print(
            "No player nodes found."
        )

        return

    positions = (
        get_positions(
            graph,
            player_nodes
        )
    )

    ids = (
        get_player_ids(
            graph,
            player_nodes
        )
    )

    edges = (
        get_player_edges(
            graph,
            player_nodes
        )
    )

    pose_available = (
        get_pose_quality(
            graph,
            player_nodes
        )
    )

    # --------------------------------------------------------
    # Map global node index -> plotted index
    # --------------------------------------------------------

    index_map = {
        int(node): index
        for index, node in enumerate(
            player_nodes
        )
    }

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------

    for src, dst in edges:

        if (
            src not in index_map
            or dst not in index_map
        ):

            continue

        a = index_map[src]
        b = index_map[dst]

        x1, y1 = (
            positions[a]
        )

        x2, y2 = (
            positions[b]
        )

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.2,
            alpha=0.45,
        )

    # --------------------------------------------------------
    # Draw player nodes
    # --------------------------------------------------------

    for i in range(
        len(player_nodes)
    ):

        x, y = positions[i]

        ax.scatter(
            x,
            y,
            s=220,
            alpha=0.9,
        )

        # ----------------------------------------------------
        # MF3 ID
        # ----------------------------------------------------

        ax.text(
            x,
            y,
            ids[i],
            ha="center",
            va="center",
            fontsize=8,
        )

        # ----------------------------------------------------
        # Pose status
        # ----------------------------------------------------

        if pose_available[i]:

            status = "pose ✓"

        else:

            status = "pose —"

        ax.annotate(
            status,
            (
                x,
                y
            ),
            xytext=(
                0,
                -18
            ),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    # --------------------------------------------------------
    # Frame number
    # --------------------------------------------------------

    frame_id = (
        data.get(
            "frame_id",
            "unknown"
        )
    )

    ax.set_title(
        f"Volleyball Spatial Graph — "
        f"Frame {frame_id}\n"
        f"MF3 player IDs + latest pose"
    )

    ax.set_xlabel(
        "Spatial X"
    )

    ax.set_ylabel(
        "Spatial Y"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim"
    )

    # --------------------------------------------------------
    # Info box
    # --------------------------------------------------------

    pose_count = int(
        pose_available.sum()
    )

    total_players = len(
        player_nodes
    )

    pose_percentage = (
        100.0
        * pose_count
        / total_players
        if total_players
        else 0
    )

    info = (
        f"Players: {total_players}\n"
        f"Edges: {len(edges)}\n"
        f"Pose: "
        f"{pose_count}/{total_players} "
        f"({pose_percentage:.1f}%)\n"
        f"Node features: 49-D\n"
        f"Pose features: 34-D"
    )

    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


# ============================================================
# PROCESS ALL GRAPHS
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph_dir",
        default=str(
            DEFAULT_GRAPH_DIR
        )
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None
    )

    args = parser.parse_args()

    graph_dir = Path(
        args.graph_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if "output_recent" not in (
        str(output_dir)
    ).lower():

        print(
            "WARNING:"
        )

        print(
            "Output directory does not contain "
            "'output_recent'."
        )

    if not graph_dir.exists():

        raise FileNotFoundError(
            f"Graph directory not found:\n"
            f"{graph_dir}\n\n"
            "Run integrate_and_rebuild.py first."
        )

    graph_files = sorted(
        graph_dir.glob(
            "graph_frame_*.pt"
        )
    )

    # --------------------------------------------------------
    # Optional frame filtering
    # --------------------------------------------------------

    if args.start is not None:

        graph_files = [
            p
            for p in graph_files
            if int(
                p.stem.split("_")[-1]
            ) >= args.start
        ]

    if args.end is not None:

        graph_files = [
            p
            for p in graph_files
            if int(
                p.stem.split("_")[-1]
            ) <= args.end
        ]

    print()
    print("=" * 75)
    print(
        "RECENT GRAPH VISUALIZATION"
    )
    print("=" * 75)

    print()
    print(
        "Reading:"
    )

    print(
        graph_dir
    )

    print()
    print(
        "Writing:"
    )

    print(
        output_dir
    )

    print()
    print(
        "Graphs found:",
        len(graph_files)
    )

    if not graph_files:

        print()
        print(
            "No graph files found."
        )

        print(
            "Run:"
        )

        print(
            "python integrate_and_rebuild.py"
        )

        return

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Generate PNGs
    # --------------------------------------------------------

    for index, graph_file in enumerate(
        graph_files,
        start=1
    ):

        try:

            data, graph = (
                load_graph(
                    graph_file
                )
            )

            frame_id = (
                data.get(
                    "frame_id",
                    graph_file.stem
                )
            )

            output_file = (
                output_dir
                / (
                    graph_file.stem
                    + ".png"
                )
            )

            draw_graph(
                data,
                graph,
                output_file
            )

            successful += 1

            print(
                f"[{index}/{len(graph_files)}] "
                f"✓ Frame {frame_id}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"[{index}/{len(graph_files)}] "
                f"✗ {graph_file.name}"
            )

            print(
                f"    {type(exc).__name__}: "
                f"{exc}"
            )

    print()
    print("=" * 75)
    print(
        "VISUALIZATION COMPLETE ✓"
    )
    print("=" * 75)

    print()
    print(
        "PNG graphs generated:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print()
    print(
        "Graph images:"
    )

    print(
        output_dir
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()