"""
Trained GAT / HRN models: prediction heads, graph building and runtime inference.

Training (train_gat_hrn.py) needs no manual labels:

    GAT  - self-supervised next-movement prediction: from the graph at frame t,
           predict every player's court displacement 0.5 s later (the tracking
           provides the answer).
    HRN  - distillation from the physics models: from the graph, predict the
           best target zone, the best shot, the attack threat, and each team's
           coverage and balance.

At runtime the dashboard uses the GAT's attention weights for key matchups and
the HRN's team predictions for team shape. Without checkpoints it falls back to
the physics versions.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import torch
import torch.nn as nn

from .epv import SHOTS

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
HORIZON_FRAMES = 15          # 0.5 s at 30 fps
EMBED_DIM = 32
NUM_ZONES = 6
SHOT_NAMES = [s.name for s in SHOTS]


class MovementHead(nn.Module):
    """GAT node embedding -> (dx, dy) court displacement in metres."""

    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ELU(), nn.Linear(dim, 2))

    def forward(self, node_embeddings):
        return self.net(node_embeddings)


class HRNHeads(nn.Module):
    """HRN team + global embeddings -> tactical predictions."""

    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.scene = nn.Sequential(nn.Linear(3 * dim, 64), nn.ELU(), nn.Dropout(0.1))
        self.zone = nn.Linear(64, NUM_ZONES)
        self.shot = nn.Linear(64, len(SHOT_NAMES))
        self.threat = nn.Linear(64, 1)
        self.team = nn.Sequential(nn.Linear(dim, 16), nn.ELU(), nn.Linear(16, 2))

    def forward(self, team_embeddings, global_embedding):
        scene = self.scene(torch.cat([global_embedding.flatten(), team_embeddings.flatten()]))
        return {
            "zone_logits": self.zone(scene),
            "shot_logits": self.shot(scene),
            "threat": torch.sigmoid(self.threat(scene)).squeeze(-1),
            "team_shape": torch.sigmoid(self.team(team_embeddings)),   # [2, (coverage, balance)]
        }


def gat_config() -> dict:
    return {"in_channels": 49, "hidden_channels": 32, "out_channels": EMBED_DIM, "heads": 4, "dropout": 0.2}


def build_graph(frame_id: int):
    """Graph for one frame using the project's own sample_data + graph_builder.

    Returns None when the frame has no usable graph (e.g. no visible ball).
    """
    import sample_data
    from graph_builder import VolleyballGraphBuilder

    # Both modules print diagnostics on every call; keep the console clean.
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            players, ball, court = sample_data.create_sample_data(frame_id)
            return VolleyballGraphBuilder().build_graph(players, ball, court, frame_id=frame_id)
        except Exception:
            return None


def _skill(accuracy, baseline) -> float:
    if accuracy is None or baseline is None or baseline >= 1.0:
        return 0.0
    return float(max(0.0, min(1.0, (accuracy - baseline) / (1.0 - baseline))))


class LearnedModels:
    """Loads gat.pt / hrn.pt and runs them on a frame's graph."""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        from models.gat import GAT
        from models.hrn import HRN

        gat_path, hrn_path = checkpoint_dir / "gat.pt", checkpoint_dir / "hrn.pt"
        if not (gat_path.exists() and hrn_path.exists()):
            raise FileNotFoundError(f"Trained checkpoints not found in {checkpoint_dir}")

        gat_ckpt = torch.load(gat_path, map_location="cpu", weights_only=True)
        hrn_ckpt = torch.load(hrn_path, map_location="cpu", weights_only=True)

        self.gat = GAT(**gat_ckpt["config"])
        self.gat.load_state_dict(gat_ckpt["model_state_dict"])
        self.hrn = HRN(**hrn_ckpt["config"])
        self.hrn.load_state_dict(hrn_ckpt["model_state_dict"])
        self.heads = HRNHeads()
        self.heads.load_state_dict(hrn_ckpt["head_state_dict"])
        for module in (self.gat, self.hrn, self.heads):
            module.eval()
        self.metrics = {"gat": gat_ckpt.get("metrics", {}), "hrn": hrn_ckpt.get("metrics", {})}
        hrn = self.metrics["hrn"]
        # Skill on the held-out test set: 0 = no better than always guessing the most
        # common answer, 1 = perfect. Used to scale how much the HRN steers the target.
        self.zone_skill = _skill(hrn.get("zone_accuracy"), hrn.get("zone_majority_baseline"))
        self.shot_skill = _skill(hrn.get("shot_accuracy"), hrn.get("shot_majority_baseline"))

    @torch.no_grad()
    def predict(self, frame_id: int) -> dict | None:
        graph = build_graph(frame_id)
        if graph is None:
            return None
        _, (att_index, att_weights) = self.gat(graph.x, graph.edge_index, graph.edge_attr, return_attention=True)
        attention = att_weights.mean(dim=1)       # average the attention heads

        node_ids = list(graph.player_ids)          # node i (< ball_index) is player node_ids[i]
        team_counts = [(graph.node_team == t).sum().item() for t in (0, 1)]
        hrn_out = None
        if min(team_counts) > 0:
            out = self.hrn(graph.x, graph.edge_index, graph.edge_attr,
                           node_type=graph.node_type, node_team=graph.node_team)
            hrn_out = self.heads(out["team_embeddings"], out["global_embedding"])

        result = {"node_ids": node_ids, "att_index": att_index, "attention": attention}
        if hrn_out is not None:
            zone_p = torch.softmax(hrn_out["zone_logits"], dim=-1)
            shot_p = torch.softmax(hrn_out["shot_logits"], dim=-1)
            result.update(
                zone=int(zone_p.argmax()) + 1,
                zone_conf=float(zone_p.max()),
                zone_probs=[float(p) for p in zone_p],                       # index = zone - 1
                shot=SHOT_NAMES[int(shot_p.argmax())],
                shot_conf=float(shot_p.max()),
                shot_probs={name: float(p) for name, p in zip(SHOT_NAMES, shot_p)},
                threat=float(hrn_out["threat"]),
                team_shape={t: (float(hrn_out["team_shape"][t, 0]), float(hrn_out["team_shape"][t, 1]))
                            for t in (0, 1)},
            )
        return result

    @staticmethod
    def attention_to(result: dict, attacker_id: int, defender_ids) -> dict[int, float]:
        """How much the attacker's node attends to each defender's node."""
        node_ids = result["node_ids"]
        if attacker_id not in node_ids:
            return {}
        target = node_ids.index(attacker_id)
        index_of = {pid: i for i, pid in enumerate(node_ids)}
        src, dst = result["att_index"]
        weights = {}
        for d in defender_ids:
            i = index_of.get(d)
            if i is None:
                continue
            mask = (dst == target) & (src == i)
            if mask.any():
                weights[d] = float(result["attention"][mask].mean())
        return weights
