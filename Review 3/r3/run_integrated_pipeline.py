import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# NEW integrated pose file:
# MF3 IDs + latest pose keypoints
PLAYER_FILE = ROOT / "player_keypoints_mf3_pose.json"

BALL_FILE = ROOT / "ball.csv"

# ------------------------------------------------------------
# NEW OUTPUTS
# ------------------------------------------------------------

RECENT_ROOT = ROOT / "output_recent"

GRAPH_DIR = RECENT_ROOT / "graphs"
DIM_DIR = RECENT_ROOT / "dimensions"
TEMPORAL_DIR = RECENT_ROOT / "temporal"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
DIM_DIR.mkdir(parents=True, exist_ok=True)
TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 8
STRIDE = 1

# ============================================================
# OPTIONAL TRAINED CHECKPOINTS
# ============================================================

# Keep None unless you actually have trained GAT/HRN weights.
GAT_CHECKPOINT = None
HRN_CHECKPOINT = None


# ============================================================
# VALIDATE POSE SOURCE
# ============================================================

def validate_pose_file():

    print("=" * 70)
    print("STEP 1 - VALIDATING MF3 + POSE DATA")
    print("=" * 70)

    if not PLAYER_FILE.exists():
        raise FileNotFoundError(
            f"\nIntegrated pose file not found:\n{PLAYER_FILE}\n\n"
            "Expected:\n"
            "player_keypoints_mf3_pose.json"
        )

    with open(PLAYER_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Source file: {PLAYER_FILE}")
    print(f"Frames: {len(data)}")

    total_players = 0
    valid_pose = 0
    invalid_pose = 0
    no_pose = 0
    ids = set()

    for frame_id, frame_data in data.items():

        # Depending on JSON structure
        if isinstance(frame_data, dict):
            players = frame_data.get("players", frame_data)
        else:
            continue

        if not isinstance(players, dict):
            continue

        for player_id, player in players.items():

            if not isinstance(player, dict):
                continue

            total_players += 1
            ids.add(str(player_id))

            keypoints = player.get("keypoints")

            if keypoints is None:
                no_pose += 1
                continue

            if not isinstance(keypoints, list):
                invalid_pose += 1
                continue

            if len(keypoints) != 17:
                invalid_pose += 1
                continue

            valid = True

            for kp in keypoints:
                if (
                    not isinstance(kp, (list, tuple))
                    or len(kp) != 2
                ):
                    valid = False
                    break

                try:
                    float(kp[0])
                    float(kp[1])
                except Exception:
                    valid = False
                    break

            if valid:
                valid_pose += 1
            else:
                invalid_pose += 1

    coverage = (
        100.0 * valid_pose / total_players
        if total_players
        else 0
    )

    print()
    print(f"Player records       : {total_players}")
    print(f"Valid 17x2 poses     : {valid_pose}")
    print(f"No pose              : {no_pose}")
    print(f"Invalid poses        : {invalid_pose}")
    print(f"Pose availability    : {coverage:.2f}%")
    print(f"Unique MF3 IDs       : {len(ids)}")

    print()
    print("Expected source coverage: ~83.73%")

    if coverage < 70:
        print("WARNING: pose coverage is unexpectedly low.")

    elif coverage >= 80:
        print("✓ Pose coverage looks healthy.")

    print()


# ============================================================
# CLEAR ONLY NEW OUTPUT
# ============================================================

def prepare_output():

    print("=" * 70)
    print("STEP 2 - PREPARING output_recent")
    print("=" * 70)

    if RECENT_ROOT.exists():

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        old_backup = ROOT / f"output_recent_backup_{timestamp}"

        print(f"Existing output_recent found.")
        print(f"Moving it to:")
        print(old_backup)

        shutil.move(str(RECENT_ROOT), str(old_backup))

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    DIM_DIR.mkdir(parents=True, exist_ok=True)
    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)

    print("✓ New output directory ready")
    print()


# ============================================================
# IMPORT EXISTING GRAPH PIPELINE
# ============================================================

def import_pipeline():

    print("=" * 70)
    print("STEP 3 - LOADING GRAPH PIPELINE")
    print("=" * 70)

    # IMPORTANT:
    #
    # sample_data.py historically expects:
    # player_keypoints.json
    #
    # We DO NOT overwrite that file.
    #
    # Instead we temporarily make a copy in a controlled way
    # and restore the original afterwards.

    canonical = ROOT / "player_keypoints.json"

    backup = None

    if canonical.exists():

        backup = ROOT / "_player_keypoints_original_backup.json"

        shutil.copy2(canonical, backup)

    shutil.copy2(PLAYER_FILE, canonical)

    try:

        from sample_data import create_sample_data
        from graph_builder import VolleyballGraphBuilder

        from build_graph import build_gat, build_hrn

    except Exception:

        # restore original
        if backup and backup.exists():
            shutil.copy2(backup, canonical)
            backup.unlink()

        raise

    # restore original immediately after imports only if
    # sample_data loads lazily this would be a problem.
    #
    # Therefore we intentionally leave the integrated file
    # in place until graph construction finishes.

    return create_sample_data, VolleyballGraphBuilder, build_gat, build_hrn, canonical, backup


# ============================================================
# GRAPH FEATURE CHECK
# ============================================================

def inspect_graph(graph, frame_id):

    if not hasattr(graph, "x"):
        print(f"WARNING frame {frame_id}: graph has no x")
        return

    x = graph.x

    print(
        f"Frame {frame_id:04d} | "
        f"nodes={x.shape[0]} | "
        f"features={x.shape[1]}",
        end=""
    )

    if x.shape[1] >= 42:

        pose = x[:, 8:42]

        nonzero = torch.count_nonzero(pose).item()
        total = pose.numel()

        percentage = (
            100.0 * nonzero / total
            if total
            else 0
        )

        print(
            f" | pose nonzero={percentage:.2f}%"
        )

    else:
        print()


# ============================================================
# BUILD GRAPHS
# ============================================================

def build_graphs(create_sample_data, VolleyballGraphBuilder,
                 build_gat, build_hrn):

    print("=" * 70)
    print("STEP 4 - BUILDING NEW GRAPHS")
    print("=" * 70)

    # Instantiate graph builder
    graph_builder = VolleyballGraphBuilder()

    # GAT / HRN
    gat = build_gat()

    hrn = build_hrn(gat)

    gat.eval()
    hrn.eval()

    if GAT_CHECKPOINT:
        print(f"Loading GAT checkpoint: {GAT_CHECKPOINT}")
        gat.load_state_dict(
            torch.load(GAT_CHECKPOINT, map_location="cpu")
        )

    if HRN_CHECKPOINT:
        print(f"Loading HRN checkpoint: {HRN_CHECKPOINT}")
        hrn.load_state_dict(
            torch.load(HRN_CHECKPOINT, map_location="cpu")
        )

    successful = 0
    skipped = 0

    embeddings = []
    frame_ids = []

    pose_total = 0
    pose_nonzero = 0

    # --------------------------------------------------------
    # Read frame IDs from source JSON
    # --------------------------------------------------------

    with open(PLAYER_FILE, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    frame_keys = sorted(
        source_data.keys(),
        key=lambda x: int(x) if str(x).isdigit() else str(x)
    )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for frame_id in frame_keys:

        try:

            # Existing project function
            sample = create_sample_data(frame_id)

            if sample is None:
                skipped += 1
                continue

            # Depending on project implementation,
            # sample may already be graph data.
            #
            # If your existing create_sample_data returns
            # player data, graph_builder handles it.

            if hasattr(sample, "x"):

                graph = sample

            else:

                graph = graph_builder.build_graph(sample)

            # ------------------------------------------------
            # Validate 49-D features
            # ------------------------------------------------

            if graph.x.shape[1] != 49:

                raise ValueError(
                    f"Frame {frame_id}: expected 49-D "
                    f"node features, got {graph.x.shape[1]}"
                )

            # ------------------------------------------------
            # Pose block
            # ------------------------------------------------

            pose = graph.x[:, 8:42]

            pose_total += pose.numel()
            pose_nonzero += torch.count_nonzero(pose).item()

            # ------------------------------------------------
            # Save graph
            # ------------------------------------------------

            graph_path = GRAPH_DIR / f"graph_frame_{int(frame_id):04d}.pt"

            torch.save(graph, graph_path)

            # ------------------------------------------------
            # GAT / HRN
            # ------------------------------------------------

            with torch.no_grad():

                gat_output = gat(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr
                    if hasattr(graph, "edge_attr")
                    else None
                )

                # Handle different GAT return formats
                if isinstance(gat_output, tuple):
                    gat_features = gat_output[0]
                else:
                    gat_features = gat_output

                # HRN
                hrn_output = hrn(graph)

                # Handle dict output
                if isinstance(hrn_output, dict):

                    global_embedding = hrn_output.get(
                        "global_embedding"
                    )

                    if global_embedding is None:
                        global_embedding = hrn_output.get(
                            "global"
                        )

                elif isinstance(hrn_output, tuple):

                    # Usually last item is global embedding
                    global_embedding = hrn_output[-1]

                else:

                    global_embedding = hrn_output

                if global_embedding is None:
                    raise RuntimeError(
                        f"Could not obtain HRN global embedding "
                        f"for frame {frame_id}"
                    )

                global_embedding = (
                    global_embedding.detach()
                    .cpu()
                    .numpy()
                )

                # Flatten [1,32] -> [32]
                global_embedding = global_embedding.reshape(-1)

                embeddings.append(global_embedding)
                frame_ids.append(int(frame_id))

            # ------------------------------------------------
            # Dimension information
            # ------------------------------------------------

            dimension_file = DIM_DIR / f"frame_{int(frame_id):04d}.npz"

            np.savez(
                dimension_file,
                num_nodes=np.array(graph.x.shape[0]),
                feature_dim=np.array(graph.x.shape[1]),
                edge_count=np.array(graph.edge_index.shape[1]),
                pose_dim=np.array(34),
                pose_start=np.array(8),
                pose_end=np.array(42),
                gat_dim=np.array(
                    gat_features.shape[-1]
                    if hasattr(gat_features, "shape")
                    else -1
                ),
                hrn_global_dim=np.array(
                    global_embedding.shape[-1]
                )
            )

            successful += 1

            if successful <= 10 or successful % 25 == 0:
                inspect_graph(graph, int(frame_id))

        except Exception as e:

            skipped += 1

            print(
                f"Skipping frame {frame_id}: {type(e).__name__}: {e}"
            )

    # --------------------------------------------------------
    # Restore original canonical player file
    # --------------------------------------------------------

    canonical = ROOT / "player_keypoints.json"
    backup = ROOT / "_player_keypoints_original_backup.json"

    if backup.exists():

        shutil.copy2(backup, canonical)
        backup.unlink()

        print()
        print("✓ Original player_keypoints.json restored.")

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage = (
        100.0 * pose_nonzero / pose_total
        if pose_total
        else 0
    )

    print()
    print("=" * 70)
    print("GRAPH PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Successful frames: {successful}")
    print(f"Skipped frames: {skipped}")
    print(
        f"Graph pose feature non-zero coverage: "
        f"{coverage:.2f}%"
    )

    print()
    print(f"Graphs saved to:")
    print(GRAPH_DIR)

    print()
    print(f"Dimension files saved to:")
    print(DIM_DIR)

    return embeddings, frame_ids


# ============================================================
# TEMPORAL SEQUENCES
# ============================================================

def build_temporal(embeddings, frame_ids):

    print()
    print("=" * 70)
    print("STEP 5 - BUILDING TEMPORAL SEQUENCES")
    print("=" * 70)

    if len(embeddings) == 0:

        print("ERROR: No embeddings generated.")
        return

    # --------------------------------------------------------
    # Make sure dimensions are consistent
    # --------------------------------------------------------

    dim = embeddings[0].shape[-1]

    valid_embeddings = []

    for emb in embeddings:

        if emb.shape[-1] != dim:

            print(
                "WARNING: inconsistent embedding dimension:",
                emb.shape
            )

            continue

        valid_embeddings.append(emb)

    embeddings = np.asarray(valid_embeddings, dtype=np.float32)

    print()
    print("Frame-level embedding matrix:")
    print(embeddings.shape)

    # --------------------------------------------------------
    # Temporal windows
    # --------------------------------------------------------

    sequences = []

    sequence_frame_ids = []

    for start in range(
        0,
        len(embeddings) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        end = start + SEQUENCE_LENGTH

        sequence = embeddings[start:end]

        if sequence.shape != (
            SEQUENCE_LENGTH,
            dim
        ):
            continue

        sequences.append(sequence)

        sequence_frame_ids.append(
            frame_ids[start:end]
        )

    sequences = np.asarray(
        sequences,
        dtype=np.float32
    )

    print()
    print("Temporal sequences shape:")
    print(sequences.shape)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    npy_path = (
        TEMPORAL_DIR /
        "hrn_global_sequences.npy"
    )

    pt_path = (
        TEMPORAL_DIR /
        "hrn_global_sequences.pt"
    )

    ids_path = (
        TEMPORAL_DIR /
        "sequence_frame_ids.npy"
    )

    np.save(npy_path, sequences)

    torch.save(
        torch.from_numpy(sequences),
        pt_path
    )

    np.save(
        ids_path,
        np.asarray(sequence_frame_ids)
    )

    print()
    print("Saved NPY:")
    print(npy_path)

    print()
    print("Saved PyTorch:")
    print(pt_path)

    print()
    print("Saved sequence frame IDs:")
    print(ids_path)

    print()
    print("Temporal representation:")
    print(
        f"{SEQUENCE_LENGTH} frames x "
        f"{dim}-D HRN global embedding"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RECENT INTEGRATED GRAPH + TEMPORAL PIPELINE")
    print("=" * 70)

    print()
    print("SOURCE:")
    print(PLAYER_FILE)

    print()
    print("NEW OUTPUT:")
    print(RECENT_ROOT)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_pose_file()

    # --------------------------------------------------------
    # Prepare output_recent
    # --------------------------------------------------------

    prepare_output()

    # --------------------------------------------------------
    # Import existing pipeline
    # --------------------------------------------------------

    (
        create_sample_data,
        VolleyballGraphBuilder,
        build_gat,
        build_hrn,
        canonical,
        backup
    ) = import_pipeline()

    # --------------------------------------------------------
    # Build graphs
    # --------------------------------------------------------

    embeddings, frame_ids = build_graphs(
        create_sample_data,
        VolleyballGraphBuilder,
        build_gat,
        build_hrn
    )

    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    build_temporal(
        embeddings,
        frame_ids
    )

    print()
    print("=" * 70)
    print("RECENT PIPELINE COMPLETE ✓")
    print("=" * 70)

    print()
    print("MF3 IDs + latest pose:")
    print("        ↓")
    print("49-D graph features")
    print("        ↓")
    print("GAT")
    print("        ↓")
    print("HRN")
    print("        ↓")
    print("8-frame temporal sequences")

    print()
    print("NEW OUTPUTS:")
    print(f"Graphs     : {GRAPH_DIR}")
    print(f"Dimensions : {DIM_DIR}")
    print(f"Temporal   : {TEMPORAL_DIR}")

    print()
    print("OLD output_graphs has NOT been modified.")


if __name__ == "__main__":
    main()