"""
Train the project's GAT and HRN without manual labels.

    Stage A - GAT, self-supervised:
        from the graph at frame t, predict each player's court displacement
        HORIZON_FRAMES later (0.5 s). The tracking data provides the targets.

    Stage B - HRN, distilled from the physics models:
        HRN starts from the trained GAT and learns to predict, from the graph,
        the physics EPV's best target zone and best shot, the attack threat,
        and each team's coverage and balance.

Split: the clip is cut into ~1 s blocks; of every 5 blocks, 3 are training,
1 is validation (used only to pick the best epoch) and 1 is test (used only for
the final metrics). Every phase of the rally appears in each set, and
near-identical neighbouring frames stay on the same side of the split.

Outputs (checkpoints/):
    gat.pt, hrn.pt            - weights (loadable by integrate_and_rebuild.py)
    training_report.json      - settings, timings, test metrics vs baselines
    training_curves.png       - train / validation loss per epoch

Run from the project root:
    python train_gat_hrn.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from analytics.epv import SHOTS
from analytics.insights import InsightEngine, zone_of
from analytics.learned import (
    CHECKPOINT_DIR, HORIZON_FRAMES, SHOT_NAMES, HRNHeads, MovementHead, build_graph, gat_config,
)
from analytics.ball_track import detect_touches, load_clean
from analytics.camera import Camera
from analytics import constants as C
from analytics.motion import MotionSmoother
from analytics.pose_signals import JumpDetector
from analytics.relations import TeamShapeModel, threat
from models.gat import GAT
from models.hrn import HRN
from tactical_dashboard import (
    BALL_FILE,
    ROOT, BallState, homography_for_frame, load_ball_rows, players_for_frame, project,
)

BLOCK = 32          # frames per split block (~1 s)
TEST_EVERY = 5      # of every 5 blocks: 1 test, 1 validation, 3 training


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------


def court_positions(player_data, annotations):
    """{frame: {player_id: (x, y) metres, plus constant-velocity prediction}}."""
    positions, H = {}, None
    for f in range(len(player_data)):
        H = homography_for_frame(annotations, f, H)
        frame = {}
        for p in player_data.get(str(f), {}).values():
            ix, iy = float(p["image_x"]), float(p["image_y"])
            vx, vy = float(p.get("vx", 0.0)), float(p.get("vy", 0.0))
            (cx, cy), (fx, fy) = project(H, [[ix, iy], [ix + vx * HORIZON_FRAMES, iy + vy * HORIZON_FRAMES]])
            frame[int(p["id"])] = {"pos": (float(cx), float(cy)), "const_vel": (float(fx - cx), float(fy - cy))}
        positions[f] = frame

    # Box bottoms jitter by tens of pixels frame to frame; average each player's
    # position over a 5-frame window so the targets measure real movement.
    smoothed = {}
    for f, frame in positions.items():
        smoothed[f] = {}
        for pid, info in frame.items():
            window = [positions[k][pid]["pos"] for k in range(f - 2, f + 3) if pid in positions.get(k, {})]
            mean = tuple(float(v) for v in np.mean(window, axis=0))
            smoothed[f][pid] = {"pos": mean, "const_vel": info["const_vel"]}
    return smoothed


def physics_labels(player_data, annotations, ball_rows, fps):
    """Run the physics models over the clip: per-frame teacher targets for the HRN."""
    engine, ball_state, shape_model = InsightEngine(), BallState(), TeamShapeModel()
    jumps = JumpDetector(Camera(annotations))
    motion = MotionSmoother(fps, C.value("top_speed"))
    touches, _ = (detect_touches(load_clean(BALL_FILE), player_data, Camera(annotations))
                  if BALL_FILE.name == "ball_clean.csv" else ({}, []))
    labels, H = {}, None
    for f in range(len(player_data)):
        H = homography_for_frame(annotations, f, H)
        players = players_for_frame(player_data.get(str(f), {}), H, fps, f, jumps, motion)
        ball = ball_state.update(f, ball_rows.get(f), H)
        ins = engine.step(f, players, ball, touches.get(f))
        if ins.surface is None:
            continue
        # The smoothed target is what the dashboard shows; the raw per-frame best
        # cell flips zone in about a quarter of frames, which makes a noisy teacher.
        tx, ty = ins.target
        best = ins.surface.sample(tx, ty)
        by_team = {t: [p for p in players if p.team == t] for t in (0, 1)}
        shapes = {t: shape_model.team_shape(by_team[t], t) for t in (0, 1)}
        labels[f] = {
            "zone": ins.target_zone - 1,
            "shot": SHOTS.index(best["shot"]),
            "threat": threat(ins.surface),
            "team_shape": [[shapes[t].coverage, shapes[t].balance] for t in (0, 1)],
        }
    return labels


def build_dataset(fps: float):
    with (ROOT / "player_keypoints_mf3_pose.json").open(encoding="utf-8") as f:
        player_data = json.load(f)
    with (ROOT / "annotations.json").open(encoding="utf-8") as f:
        annotations = json.load(f)
    ball_rows = load_ball_rows(BALL_FILE)

    print("Computing court positions and physics teacher labels...")
    positions = court_positions(player_data, annotations)
    teacher = physics_labels(player_data, annotations, ball_rows, fps)

    print("Building graphs with graph_builder.py...")
    graphs = []
    n_frames = len(player_data)
    for f in range(n_frames):
        g = build_graph(f)
        if g is None:
            continue
        n = g.x.size(0)
        move_target = torch.zeros(n, 2)
        move_mask = torch.zeros(n, dtype=torch.bool)
        const_vel = torch.zeros(n, 2)
        future = positions.get(f + HORIZON_FRAMES, {})
        for i, pid in enumerate(g.player_ids):
            now, later = positions[f].get(int(pid)), future.get(int(pid))
            if now is None or later is None:
                continue
            move_target[i] = torch.tensor([later["pos"][0] - now["pos"][0], later["pos"][1] - now["pos"][1]])
            const_vel[i] = torch.tensor(now["const_vel"])
            move_mask[i] = True
        # Discard tracking glitches (ID switches jump several metres in 0.5 s).
        move_mask &= move_target.norm(dim=1) < 4.0
        g.move_target, g.move_mask, g.const_vel = move_target, move_mask, const_vel
        g.has_teacher = f in teacher
        if g.has_teacher:
            g.teacher = teacher[f]
        g.frame = f
        graphs.append(g)

    # Interleaved block split: every TEST_EVERY-th block of BLOCK frames is test data,
    # so the test set covers every phase of the rally. Whole blocks (not single frames)
    # keep near-identical neighbouring frames on the same side of the split.
    role = lambda f: {TEST_EVERY - 1: "test", 2: "val"}.get((f // BLOCK) % TEST_EVERY, "train")
    split = {name: [g for g in graphs if role(g.frame) == name] for name in ("train", "val", "test")}
    return split["train"], split["val"], split["test"]


# ----------------------------------------------------------------------------
# Stage A: GAT next-movement prediction
# ----------------------------------------------------------------------------


def movement_loss(gat, head, batch):
    pred = head(gat(batch.x, batch.edge_index, batch.edge_attr))
    mask = batch.move_mask
    if mask.sum() == 0:
        return None, None
    loss = F.smooth_l1_loss(pred[mask], batch.move_target[mask])
    ade = (pred[mask] - batch.move_target[mask]).norm(dim=1).mean()
    return loss, ade


def evaluate_gat(gat, head, graphs):
    gat.eval(); head.eval()
    errors, cv_errors, zero_errors = [], [], []
    with torch.no_grad():
        for g in graphs:
            if g.move_mask.sum() == 0:
                continue
            pred = head(gat(g.x, g.edge_index, g.edge_attr))[g.move_mask]
            target = g.move_target[g.move_mask]
            errors.append((pred - target).norm(dim=1))
            cv_errors.append((g.const_vel[g.move_mask] - target).norm(dim=1))
            zero_errors.append(target.norm(dim=1))
    cat = lambda xs: float(torch.cat(xs).mean()) if xs else float("nan")
    return {"gat_ade_m": cat(errors), "constant_velocity_ade_m": cat(cv_errors),
            "no_movement_ade_m": cat(zero_errors), "players_evaluated": int(sum(len(e) for e in errors))}


def train_gat(train, val, epochs, lr):
    gat, head = GAT(**gat_config()), MovementHead()
    params = list(gat.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    # Batching needs plain tensor attributes only (the teacher labels are dicts).
    slim = lambda gs: [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                            move_target=g.move_target, move_mask=g.move_mask) for g in gs if g.move_mask.any()]
    train_loader = DataLoader(slim(train), batch_size=16, shuffle=True)
    val_loader = DataLoader(slim(val), batch_size=64)

    history, best = {"train": [], "val": []}, (float("inf"), None)
    for epoch in range(1, epochs + 1):
        gat.train(); head.train()
        losses = []
        for batch in train_loader:
            loss, _ = movement_loss(gat, head, batch)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        gat.eval(); head.eval()
        with torch.no_grad():
            val_losses = [l.item() for l, _ in (movement_loss(gat, head, b) for b in val_loader) if l is not None]
        tr, te = float(np.mean(losses)), float(np.mean(val_losses))
        history["train"].append(tr); history["val"].append(te)
        if te < best[0]:
            best = (te, ({k: v.clone() for k, v in gat.state_dict().items()},
                         {k: v.clone() for k, v in head.state_dict().items()}, epoch))
        if epoch % 25 == 0 or epoch == 1:
            print(f"  GAT epoch {epoch:3d}  train {tr:.4f}  val {te:.4f}")

    gat.load_state_dict(best[1][0]); head.load_state_dict(best[1][1])
    return gat, head, history, best[1][2]


# ----------------------------------------------------------------------------
# Stage B: HRN distilled from the physics models
# ----------------------------------------------------------------------------


def hrn_forward(hrn, heads, g):
    out = hrn(g.x, g.edge_index, g.edge_attr, node_type=g.node_type, node_team=g.node_team)
    return heads(out["team_embeddings"], out["global_embedding"])


def hrn_loss(pred, t):
    zone = torch.tensor(t["zone"]); shot = torch.tensor(t["shot"])
    return (F.cross_entropy(pred["zone_logits"][None], zone[None])
            + F.cross_entropy(pred["shot_logits"][None], shot[None])
            + 10.0 * F.mse_loss(pred["threat"], torch.tensor(float(t["threat"])))
            + 10.0 * F.mse_loss(pred["team_shape"], torch.tensor(t["team_shape"], dtype=torch.float32)))


def usable_for_hrn(g):
    return g.has_teacher and all((g.node_team == t).any() for t in (0, 1))


def evaluate_hrn(hrn, heads, graphs, train_graphs):
    hrn.eval(); heads.eval()
    zone_hits = shot_hits = 0
    threat_err, shape_err = [], []
    graphs = [g for g in graphs if usable_for_hrn(g)]
    with torch.no_grad():
        for g in graphs:
            p, t = hrn_forward(hrn, heads, g), g.teacher
            zone_hits += int(p["zone_logits"].argmax()) == t["zone"]
            shot_hits += int(p["shot_logits"].argmax()) == t["shot"]
            threat_err.append(abs(float(p["threat"]) - t["threat"]))
            shape_err.append(float((p["team_shape"] - torch.tensor(t["team_shape"])).abs().mean()))
    n = max(1, len(graphs))
    train_zones = Counter(g.teacher["zone"] for g in train_graphs if usable_for_hrn(g))
    train_shots = Counter(g.teacher["shot"] for g in train_graphs if usable_for_hrn(g))
    majority_zone = train_zones.most_common(1)[0][0]
    majority_shot = train_shots.most_common(1)[0][0]
    return {
        "frames_evaluated": len(graphs),
        "zone_accuracy": zone_hits / n,
        "zone_majority_baseline": sum(g.teacher["zone"] == majority_zone for g in graphs) / n,
        "shot_accuracy": shot_hits / n,
        "shot_majority_baseline": sum(g.teacher["shot"] == majority_shot for g in graphs) / n,
        "threat_mae": float(np.mean(threat_err)) if threat_err else float("nan"),
        "team_shape_mae": float(np.mean(shape_err)) if shape_err else float("nan"),
    }


def train_hrn(gat, train, val, epochs, lr):
    hrn, heads = HRN(**gat_config()), HRNHeads()
    hrn.gat.load_state_dict(gat.state_dict())          # start from the trained GAT
    opt = torch.optim.Adam(list(hrn.parameters()) + list(heads.parameters()), lr=lr, weight_decay=1e-4)
    train_g = [g for g in train if usable_for_hrn(g)]
    val_g = [g for g in val if usable_for_hrn(g)]

    history, best = {"train": [], "val": []}, (float("inf"), None)
    for epoch in range(1, epochs + 1):
        hrn.train(); heads.train()
        random.shuffle(train_g)
        losses = []
        for g in train_g:
            loss = hrn_loss(hrn_forward(hrn, heads, g), g.teacher)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        hrn.eval(); heads.eval()
        with torch.no_grad():
            val_losses = [hrn_loss(hrn_forward(hrn, heads, g), g.teacher).item() for g in val_g]
        tr, te = float(np.mean(losses)), float(np.mean(val_losses))
        history["train"].append(tr); history["val"].append(te)
        if te < best[0]:
            best = (te, ({k: v.clone() for k, v in hrn.state_dict().items()},
                         {k: v.clone() for k, v in heads.state_dict().items()}, epoch))
        if epoch % 20 == 0 or epoch == 1:
            print(f"  HRN epoch {epoch:3d}  train {tr:.4f}  val {te:.4f}")

    hrn.load_state_dict(best[1][0]); heads.load_state_dict(best[1][1])
    return hrn, heads, history, best[1][2]


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def plot_curves(gat_hist, hrn_hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, hist, title in ((axes[0], gat_hist, "GAT: next-movement prediction"),
                            (axes[1], hrn_hist, "HRN: distillation from physics")):
        ax.plot(hist["train"], label="train")
        ax.plot(hist["val"], label="validation")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train GAT and HRN (no manual labels).")
    parser.add_argument("--gat-epochs", type=int, default=150)
    parser.add_argument("--hrn-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    set_seed(args.seed)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    train, val, test = build_dataset(args.fps)
    t_data = time.time() - t0
    print(f"Graphs: {len(train)} train, {len(val)} validation, {len(test)} test  [{t_data:.0f} s]")

    print("\nStage A - training GAT (next-movement prediction)")
    t1 = time.time()
    gat, move_head, gat_hist, gat_best = train_gat(train, val, args.gat_epochs, args.lr)
    t_gat = time.time() - t1
    gat_metrics = evaluate_gat(gat, move_head, test)
    print(f"  done in {t_gat:.0f} s, best epoch {gat_best}: {gat_metrics}")

    print("\nStage B - training HRN (distillation from physics EPV + team shape)")
    t2 = time.time()
    hrn, heads, hrn_hist, hrn_best = train_hrn(gat, train, val, args.hrn_epochs, args.lr * 0.5)
    t_hrn = time.time() - t2
    hrn_metrics = evaluate_hrn(hrn, heads, test, train)
    print(f"  done in {t_hrn:.0f} s, best epoch {hrn_best}: {hrn_metrics}")

    torch.save({"model_state_dict": gat.state_dict(), "head_state_dict": move_head.state_dict(),
                "config": gat_config(), "metrics": gat_metrics, "best_epoch": gat_best},
               CHECKPOINT_DIR / "gat.pt")
    torch.save({"model_state_dict": hrn.state_dict(), "head_state_dict": heads.state_dict(),
                "config": gat_config(), "metrics": hrn_metrics, "best_epoch": hrn_best},
               CHECKPOINT_DIR / "hrn.pt")
    plot_curves(gat_hist, hrn_hist, CHECKPOINT_DIR / "training_curves.png")

    report = {
        "data": {"train_graphs": len(train), "val_graphs": len(val), "test_graphs": len(test),
                 "split": f"{BLOCK}-frame blocks, per {TEST_EVERY}: 3 train / 1 validation / 1 test",
                 "horizon_frames": HORIZON_FRAMES, "shot_classes": SHOT_NAMES},
        "settings": vars(args),
        "timing_s": {"data": round(t_data, 1), "gat": round(t_gat, 1), "hrn": round(t_hrn, 1),
                     "total": round(time.time() - t0, 1)},
        "gat": {"best_epoch": gat_best, "test": gat_metrics},
        "hrn": {"best_epoch": hrn_best, "test": hrn_metrics},
    }
    with (CHECKPOINT_DIR / "training_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved checkpoints/gat.pt, checkpoints/hrn.pt, training_report.json, training_curves.png "
          f"[total {report['timing_s']['total']:.0f} s]")


if __name__ == "__main__":
    main()
