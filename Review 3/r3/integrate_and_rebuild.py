"""
RECENT MF3 -> POSE -> GRAPH -> GAT -> HRN -> TEMPORAL PIPELINE

Uses:
    player_keypoints_mf3_pose.json

where:
    MF3 IDs + MF3 boxes
            +
    latest pose-estimation keypoints

are converted into:

    49-D graph node features
            ->
    GAT
            ->
    HRN
            ->
    temporal sequences

IMPORTANT:
    OLD output_graphs/
    OLD dimensions_output/
    OLD temporal_output/

are NOT modified.

ALL NEW RESULTS go into:

    output_recent/
"""

from pathlib import Path
from datetime import datetime
import json
import shutil
import sys

import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# NEW MF3 + POSE SOURCE
# ------------------------------------------------------------

NEW_PLAYER_FILE = (
    ROOT / "player_keypoints_mf3_pose.json"
)

# Existing file expected by sample_data.py
CANONICAL_PLAYER_FILE = (
    ROOT / "player_keypoints.json"
)

BALL_FILE = ROOT / "ball.csv"


# ============================================================
# NEW OUTPUT ROOT
# ============================================================

RECENT_ROOT = ROOT / "output_recent"

GRAPH_DIR = (
    RECENT_ROOT / "graphs"
)

DIM_DIR = (
    RECENT_ROOT / "dimensions"
)

TEMPORAL_DIR = (
    RECENT_ROOT / "temporal"
)


# ============================================================
# IMPORTANT:
# GRAPH IMAGES WILL BE CREATED BY visualize_graph.py
# ============================================================

GRAPH_IMAGE_DIR = (
    RECENT_ROOT / "graph_images"
)


# Create ONLY the NEW directories.
GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMPORAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEMPORAL SETTINGS
# ============================================================

SEQUENCE_LENGTH = 8
SEQUENCE_STRIDE = 1


# ============================================================
# OPTIONAL CHECKPOINTS
# ============================================================

# Keep these as None unless you actually have trained
# GAT / HRN checkpoints.

GAT_CHECKPOINT = None
HRN_CHECKPOINT = None


# ============================================================
# HELPER: FLATTEN POSE
# ============================================================

def flatten_pose(keypoints):
    """
    Convert 17 x 2 pose into exactly 34 values.

    Returns:
        np.ndarray shape (34,)
        or None
    """

    if keypoints is None:
        return None

    try:
        arr = np.asarray(
            keypoints,
            dtype=np.float32
        )
    except Exception:
        return None

    if arr.size != 34:
        return None

    try:
        arr = arr.reshape(17, 2)
    except Exception:
        return None

    if not np.isfinite(arr).all():
        return None

    return arr.reshape(-1)


# ============================================================
# STEP 1
# VALIDATE NEW POSE FILE
# ============================================================

def validate_new_pose_file():

    print()
    print("=" * 75)
    print("STEP 1 - VALIDATING MF3 + POSE DATA")
    print("=" * 75)

    if not NEW_PLAYER_FILE.exists():

        raise FileNotFoundError(
            "\nMissing:\n"
            f"{NEW_PLAYER_FILE}\n\n"
            "Run mf3_pose_extract.py first."
        )

    print()
    print("Source:")
    print(NEW_PLAYER_FILE)

    with NEW_PLAYER_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    total_players = 0
    valid_poses = 0
    invalid_poses = 0

    unique_ids = set()

    for frame_id, players in data.items():

        if not isinstance(players, dict):
            continue

        for player_key, player in players.items():

            if not isinstance(player, dict):
                continue

            total_players += 1

            # MF3 ID is authoritative.
            player_id = player.get(
                "id",
                player_key
            )

            unique_ids.add(
                str(player_id)
            )

            keypoints = player.get(
                "keypoints"
            )

            pose = flatten_pose(
                keypoints
            )

            if pose is not None:
                valid_poses += 1
            else:
                invalid_poses += 1

    coverage = (
        100.0 * valid_poses / total_players
        if total_players
        else 0.0
    )

    print()
    print("Frames:", len(data))
    print("Player records:", total_players)
    print("Valid 17x2 poses:", valid_poses)
    print("Missing/invalid poses:", invalid_poses)
    print(
        f"Pose availability: {coverage:.2f}%"
    )
    print(
        "Unique MF3 IDs:",
        len(unique_ids)
    )

    if total_players == 0:

        raise RuntimeError(
            "No player records found."
        )

    if valid_poses == 0:

        raise RuntimeError(
            "No valid poses found."
        )

    print()
    print("✓ MF3 IDs detected")
    print("✓ 17-keypoint poses detected")
    print("✓ Pose source validated")

    return data


# ============================================================
# STEP 2
# BACKUP ORIGINAL PLAYER JSON
# ============================================================

def backup_original_player_file():

    print()
    print("=" * 75)
    print("STEP 2 - BACKING UP ORIGINAL PLAYER DATA")
    print("=" * 75)

    if not CANONICAL_PLAYER_FILE.exists():

        print(
            "player_keypoints.json does not exist."
        )

        return None

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        ROOT
        / "previous_pipeline_backup"
        / timestamp
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    backup_file = (
        backup_dir
        / "player_keypoints_original.json"
    )

    shutil.copy2(
        CANONICAL_PLAYER_FILE,
        backup_file
    )

    print()
    print(
        "Original player_keypoints.json backed up:"
    )

    print(backup_file)

    return backup_file


# ============================================================
# STEP 3
# INSTALL NEW DATA TEMPORARILY
# ============================================================

def install_new_player_data():

    print()
    print("=" * 75)
    print("STEP 3 - INSTALLING NEW MF3 + POSE DATA")
    print("=" * 75)

    shutil.copy2(
        NEW_PLAYER_FILE,
        CANONICAL_PLAYER_FILE
    )

    print()
    print(
        "sample_data.py will now read:"
    )

    print(
        CANONICAL_PLAYER_FILE
    )

    print()
    print(
        "This contains:"
    )

    print(
        "  MF3 player IDs"
    )

    print(
        "  MF3 bounding boxes"
    )

    print(
        "  player positions"
    )

    print(
        "  latest pose keypoints"
    )

    print()
    print("✓ New source installed")


# ============================================================
# STEP 4
# LOAD EXISTING PIPELINE
# ============================================================

def import_pipeline_modules():

    print()
    print("=" * 75)
    print("STEP 4 - LOADING EXISTING GRAPH PIPELINE")
    print("=" * 75)

    try:

        from sample_data import (
            create_sample_data,
            get_available_frames,
        )

        from graph_builder import (
            VolleyballGraphBuilder
        )

        from models import (
            GAT,
            HRN
        )

        from temporal.sequence_generator import (
            TemporalSequenceGenerator
        )

    except Exception as exc:

        print()
        print(
            "Pipeline import failed:"
        )

        print(
            type(exc).__name__,
            ":",
            exc
        )

        raise

    print()
    print("✓ sample_data.py")
    print("✓ graph_builder.py")
    print("✓ GAT")
    print("✓ HRN")
    print("✓ TemporalSequenceGenerator")

    return (
        create_sample_data,
        get_available_frames,
        VolleyballGraphBuilder,
        GAT,
        HRN,
        TemporalSequenceGenerator,
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    model_name
):

    if checkpoint_path is None:

        print()
        print(
            f"{model_name}: "
            "No trained checkpoint configured."
        )

        print(
            f"{model_name}: "
            "using current model weights."
        )

        return model

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_absolute():

        checkpoint_path = (
            ROOT / checkpoint_path
        )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"{model_name} checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(checkpoint, dict):

        state_dict = checkpoint

    else:

        raise RuntimeError(
            f"Unsupported {model_name} checkpoint."
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print()
    print(
        f"✓ {model_name} checkpoint loaded:"
    )

    print(checkpoint_path)

    return model


# ============================================================
# GRAPH FEATURE DIAGNOSTICS
# ============================================================

def feature_diagnostics(graph):

    x = (
        graph.x
        .detach()
        .cpu()
    )

    if x.ndim != 2:

        raise RuntimeError(
            "graph.x is not 2-dimensional."
        )

    if x.shape[1] != 49:

        raise RuntimeError(
            f"Expected 49-D node features, "
            f"got {x.shape[1]}"
        )

    # --------------------------------------------------------
    # Node type
    # --------------------------------------------------------

    if not hasattr(
        graph,
        "node_type"
    ):

        raise RuntimeError(
            "Graph has no node_type."
        )

    node_type = (
        graph.node_type
        .detach()
        .cpu()
    )

    # Existing pipeline uses node_type == 0
    # for player nodes.

    player_mask = (
        node_type == 0
    )

    player_features = (
        x[player_mask]
    )

    if player_features.shape[0] == 0:

        return {
            "players": 0,
            "pose_players": 0,
            "pose_percentage": 0.0,
            "pose_nonzero": 0,
            "pose_total": 0,
            "pose_mean_abs": 0.0,
        }

    # --------------------------------------------------------
    # Pose block:
    #
    # features 8:42
    #
    # 34 values
    # 17 keypoints x 2
    # --------------------------------------------------------

    pose = (
        player_features[:, 8:42]
    )

    if pose.shape[1] != 34:

        raise RuntimeError(
            "Pose feature block is not 34-D."
        )

    populated_players = (
        torch.count_nonzero(
            pose,
            dim=1
        ) > 0
    )

    pose_players = int(
        populated_players.sum().item()
    )

    pose_nonzero = int(
        torch.count_nonzero(
            pose
        ).item()
    )

    pose_total = int(
        pose.numel()
    )

    pose_percentage = (
        100.0
        * pose_players
        / player_features.shape[0]
    )

    pose_mean_abs = float(
        pose.abs().mean().item()
    )

    return {
        "players":
            int(player_features.shape[0]),

        "pose_players":
            pose_players,

        "pose_percentage":
            pose_percentage,

        "pose_nonzero":
            pose_nonzero,

        "pose_total":
            pose_total,

        "pose_mean_abs":
            pose_mean_abs,
    }


# ============================================================
# STEP 5
# BUILD NEW GRAPHS
# ============================================================

def build_graphs(
    create_sample_data,
    get_available_frames,
    VolleyballGraphBuilder,
    GAT,
    HRN,
):

    print()
    print("=" * 75)
    print("STEP 5 - BUILDING NEW GRAPHS")
    print("=" * 75)

    frames = (
        get_available_frames()
    )

    if not frames:

        raise RuntimeError(
            "No frames available."
        )

    print()
    print(
        "Frames available:",
        len(frames)
    )

    print(
        "First frame:",
        frames[0]
    )

    print(
        "Last frame:",
        frames[-1]
    )

    # --------------------------------------------------------
    # Graph builder
    # --------------------------------------------------------

    builder = (
        VolleyballGraphBuilder()
    )

    # --------------------------------------------------------
    # GAT
    #
    # Existing graph feature dimension = 49
    # --------------------------------------------------------

    gat = GAT(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=0.2,
    )

    # --------------------------------------------------------
    # HRN
    # --------------------------------------------------------

    hrn = HRN(
        in_channels=49,
        hidden_channels=32,
        out_channels=32,
        heads=4,
        dropout=0.2,
    )

    # --------------------------------------------------------
    # Optional checkpoints
    # --------------------------------------------------------

    gat = load_checkpoint(
        gat,
        GAT_CHECKPOINT,
        "GAT"
    )

    hrn = load_checkpoint(
        hrn,
        HRN_CHECKPOINT,
        "HRN"
    )

    gat.eval()
    hrn.eval()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = 0
    skipped = 0

    total_players = 0
    total_pose_players = 0

    global_embeddings = []
    global_frame_ids = []

    # --------------------------------------------------------
    # Frame processing
    # --------------------------------------------------------

    for index, frame_id in enumerate(
        frames,
        start=1
    ):

        try:

            # ------------------------------------------------
            # Get player + ball + court data
            # ------------------------------------------------

            result = (
                create_sample_data(
                    frame_id
                )
            )

            if not isinstance(
                result,
                tuple
            ) or len(result) != 3:

                raise RuntimeError(
                    "create_sample_data() "
                    "did not return "
                    "(players, ball, court)."
                )

            players, ball, court = result

            # ------------------------------------------------
            # Build graph
            # ------------------------------------------------

            graph = (
                builder.build_graph(
                    players,
                    ball,
                    court,
                    frame_id=frame_id,
                )
            )

            # ------------------------------------------------
            # Validate 49-D
            # ------------------------------------------------

            if graph.x.shape[1] != 49:

                raise RuntimeError(
                    f"Frame {frame_id}: "
                    f"expected 49-D features, "
                    f"got {graph.x.shape[1]}"
                )

            # ------------------------------------------------
            # Pose diagnostics
            # ------------------------------------------------

            diag = (
                feature_diagnostics(
                    graph
                )
            )

            total_players += (
                diag["players"]
            )

            total_pose_players += (
                diag["pose_players"]
            )

            print()
            print(
                f"[{index}/{len(frames)}] "
                f"Frame {frame_id}"
            )

            print(
                f"  Players: "
                f"{diag['players']}"
            )

            print(
                f"  Pose: "
                f"{diag['pose_players']}/"
                f"{diag['players']} "
                f"({diag['pose_percentage']:.1f}%)"
            )

            print(
                f"  Features: "
                f"{tuple(graph.x.shape)}"
            )

            print(
                f"  Pose block: "
                f"{tuple(graph.x[:, 8:42].shape)}"
            )

            print(
                f"  Non-zero pose values: "
                f"{diag['pose_nonzero']}/"
                f"{diag['pose_total']}"
            )

            # ------------------------------------------------
            # GAT
            # ------------------------------------------------

            with torch.no_grad():

                gat_output = gat(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                )

            if isinstance(
                gat_output,
                tuple
            ):

                gat_embeddings = (
                    gat_output[0]
                )

            else:

                gat_embeddings = (
                    gat_output
                )

            # ------------------------------------------------
            # HRN
            # ------------------------------------------------

            with torch.no_grad():

                hrn_output = hrn(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.node_type,
                    graph.node_team,
                )

            if not isinstance(
                hrn_output,
                dict
            ):

                raise RuntimeError(
                    "HRN output must be a dictionary."
                )

            required = [
                "node_embeddings",
                "team_embeddings",
                "global_embedding",
            ]

            for key in required:

                if key not in hrn_output:

                    raise RuntimeError(
                        f"HRN output missing "
                        f"'{key}'."
                    )

            hrn_node_embeddings = (
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

            # ------------------------------------------------
            # Save temporal global embedding
            # ------------------------------------------------

            global_embedding_np = (
                global_embedding
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )

            global_embeddings.append(
                global_embedding_np
            )

            global_frame_ids.append(
                frame_id
            )

            # ------------------------------------------------
            # SAVE GRAPH
            # ------------------------------------------------

            graph_file = (
                GRAPH_DIR
                / f"graph_frame_{int(frame_id):04d}.pt"
            )

            torch.save(
                {
                    "frame_id":
                        frame_id,

                    "graph":
                        graph,

                    "gat_node_embeddings":
                        gat_embeddings
                        .detach()
                        .cpu(),

                    "hrn_node_embeddings":
                        hrn_node_embeddings
                        .detach()
                        .cpu(),

                    "team_embeddings":
                        team_embeddings
                        .detach()
                        .cpu(),

                    "global_embedding":
                        global_embedding
                        .detach()
                        .cpu(),

                    "pose_diagnostics":
                        diag,
                },
                graph_file
            )

            # ------------------------------------------------
            # SAVE DIMENSION / FEATURE FILE
            # ------------------------------------------------

            dimension_file = (
                DIM_DIR
                / f"frame_{int(frame_id):04d}.npz"
            )

            np.savez_compressed(

                dimension_file,

                node_features=
                    graph.x
                    .detach()
                    .cpu()
                    .numpy(),

                edge_index=
                    graph.edge_index
                    .detach()
                    .cpu()
                    .numpy(),

                edge_attr=
                    graph.edge_attr
                    .detach()
                    .cpu()
                    .numpy(),

                node_type=
                    graph.node_type
                    .detach()
                    .cpu()
                    .numpy(),

                node_team=
                    graph.node_team
                    .detach()
                    .cpu()
                    .numpy(),

                gat_node_embeddings=
                    gat_embeddings
                    .detach()
                    .cpu()
                    .numpy(),

                hrn_node_embeddings=
                    hrn_node_embeddings
                    .detach()
                    .cpu()
                    .numpy(),

                team_embeddings=
                    team_embeddings
                    .detach()
                    .cpu()
                    .numpy(),

                global_embedding=
                    global_embedding
                    .detach()
                    .cpu()
                    .numpy(),

                pose_features=
                    graph.x[:, 8:42]
                    .detach()
                    .cpu()
                    .numpy(),

                pose_start=
                    np.array(8),

                pose_end=
                    np.array(42),

                pose_dimension=
                    np.array(34),

                total_feature_dimension=
                    np.array(49),
            )

            successful += 1

        except Exception as exc:

            skipped += 1

            print()
            print(
                f"  ✗ Frame {frame_id} failed:"
            )

            print(
                f"    {type(exc).__name__}: {exc}"
            )

    # ========================================================
    # GRAPH SUMMARY
    # ========================================================

    overall_pose = (
        100.0
        * total_pose_players
        / total_players
        if total_players
        else 0.0
    )

    print()
    print("=" * 75)
    print("GRAPH PROCESSING COMPLETE")
    print("=" * 75)

    print(
        "Successful frames:",
        successful
    )

    print(
        "Skipped frames:",
        skipped
    )

    print(
        f"Overall graph player-pose availability: "
        f"{overall_pose:.2f}%"
    )

    print()
    print(
        "NEW graphs:"
    )

    print(
        GRAPH_DIR
    )

    print()
    print(
        "NEW dimensions:"
    )

    print(
        DIM_DIR
    )

    return (
        global_frame_ids,
        global_embeddings
    )


# ============================================================
# STEP 6
# TEMPORAL SEQUENCES
# ============================================================

def build_temporal_sequences(
    frame_ids,
    global_embeddings,
):

    print()
    print("=" * 75)
    print("STEP 6 - BUILDING TEMPORAL SEQUENCES")
    print("=" * 75)

    if len(global_embeddings) < (
        SEQUENCE_LENGTH
    ):

        print(
            "Not enough frames for temporal sequences."
        )

        return

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    pairs = sorted(
        zip(
            frame_ids,
            global_embeddings
        ),
        key=lambda x: x[0]
    )

    ordered_frame_ids = [
        x[0]
        for x in pairs
    ]

    ordered_embeddings = np.asarray(
        [
            x[1]
            for x in pairs
        ],
        dtype=np.float32
    )

    print()
    print(
        "Frame-level embedding matrix:"
    )

    print(
        ordered_embeddings.shape
    )

    # --------------------------------------------------------
    # Generate sequences
    # --------------------------------------------------------

    sequences = []

    sequence_frame_ids = []

    for start in range(
        0,
        len(ordered_embeddings)
        - SEQUENCE_LENGTH
        + 1,
        SEQUENCE_STRIDE
    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )

        seq = (
            ordered_embeddings[
                start:end
            ]
        )

        if seq.shape[0] != (
            SEQUENCE_LENGTH
        ):

            continue

        sequences.append(
            seq
        )

        sequence_frame_ids.append(
            ordered_frame_ids[
                start:end
            ]
        )

    sequences = np.asarray(
        sequences,
        dtype=np.float32
    )

    print()
    print(
        "Temporal sequences shape:"
    )

    print(
        sequences.shape
    )

    # --------------------------------------------------------
    # Save NPY
    # --------------------------------------------------------

    npy_path = (
        TEMPORAL_DIR
        / "hrn_global_sequences.npy"
    )

    np.save(
        npy_path,
        sequences
    )

    # --------------------------------------------------------
    # Save PT
    # --------------------------------------------------------

    pt_path = (
        TEMPORAL_DIR
        / "hrn_global_sequences.pt"
    )

    torch.save(
        {
            "sequences":
                torch.from_numpy(
                    sequences
                ),

            "sequence_length":
                SEQUENCE_LENGTH,

            "stride":
                SEQUENCE_STRIDE,

            "frame_ids":
                sequence_frame_ids,

            "feature_dim":
                int(
                    sequences.shape[-1]
                ),
        },
        pt_path
    )

    # --------------------------------------------------------
    # Save frame IDs
    # --------------------------------------------------------

    ids_path = (
        TEMPORAL_DIR
        / "sequence_frame_ids.npy"
    )

    np.save(
        ids_path,
        np.asarray(
            sequence_frame_ids,
            dtype=np.int64
        )
    )

    print()
    print(
        "Saved NPY:"
    )

    print(
        npy_path
    )

    print()
    print(
        "Saved PyTorch:"
    )

    print(
        pt_path
    )

    print()
    print(
        "Saved sequence frame IDs:"
    )

    print(
        ids_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "RECENT INTEGRATED GRAPH PIPELINE"
    )
    print("=" * 75)

    print()
    print(
        "OLD OUTPUTS WILL NOT BE TOUCHED."
    )

    print()
    print(
        "NEW OUTPUT ROOT:"
    )

    print(
        RECENT_ROOT
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    validate_new_pose_file()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    backup_original_player_file()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    install_new_player_data()

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    (
        create_sample_data,
        get_available_frames,
        VolleyballGraphBuilder,
        GAT,
        HRN,
        TemporalSequenceGenerator,
    ) = import_pipeline_modules()

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    (
        frame_ids,
        global_embeddings,
    ) = build_graphs(

        create_sample_data,

        get_available_frames,

        VolleyballGraphBuilder,

        GAT,

        HRN,
    )

    # --------------------------------------------------------
    # Restore ORIGINAL player_keypoints.json
    # --------------------------------------------------------

    # Find most recent backup.
    backup_root = (
        ROOT
        / "previous_pipeline_backup"
    )

    if backup_root.exists():

        backup_dirs = sorted(
            [
                p
                for p in backup_root.iterdir()
                if p.is_dir()
            ],
            key=lambda p: p.name
        )

        if backup_dirs:

            latest_backup = (
                backup_dirs[-1]
                / "player_keypoints_original.json"
            )

            if latest_backup.exists():

                shutil.copy2(
                    latest_backup,
                    CANONICAL_PLAYER_FILE
                )

                print()
                print(
                    "✓ Original player_keypoints.json restored."
                )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    build_temporal_sequences(
        frame_ids,
        global_embeddings,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 75)
    print(
        "RECENT INTEGRATED PIPELINE COMPLETE ✓"
    )
    print("=" * 75)

    print()
    print(
        "OLD:"
    )

    print(
        "  output_graphs/"
    )

    print(
        "  dimensions_output/"
    )

    print(
        "  temporal_output/"
    )

    print()
    print(
        "NEW:"
    )

    print(
        f"  {RECENT_ROOT}/"
    )

    print()
    print(
        "  graphs/"
    )

    print(
        "  dimensions/"
    )

    print(
        "  temporal/"
    )

    print(
        "  graph_images/"
    )

    print()
    print(
        "MF3 IDs + latest pose successfully used."
    )

    print()
    print(
        "Pose feature block:"
    )

    print(
        "graph.x[:, 8:42]"
    )

    print()
    print(
        "34 values = 17 keypoints x (x,y)"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Pipeline interrupted."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 75)
        print(
            "PIPELINE FAILED"
        )
        print("=" * 75)

        print(
            type(exc).__name__,
            ":",
            exc
        )

        sys.exit(1)