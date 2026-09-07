"""
epv/evaluate.py
------------------
Task 6 of Member 3's Phase 3 work: evaluate EPV predictions and decision
quality, and save results for the Phase 3 report.

Usage
-----
    python -m epv.evaluate --dataset epv_dataset.npz --checkpoint epv_model.pt --out-dir epv_eval
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import torch

from epv.action_generator import ACTIONS
from epv.model import EPVModel


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = EPVModel(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")  # not meaningful for a constant/degenerate split
    return {"mae": mae, "mse": mse, "rmse": rmse, "pearson_corr": corr}


def per_action_metrics(y_true, y_pred, actions):
    """
    Per-action breakdown: how many samples of each action exist and what the
    model predicts for them. This surfaces degenerate label sets (e.g. an
    action that never occurs, or an outcome that is constant per action).
    """
    rows = {}
    for i, a_idx in enumerate(actions):
        name = ACTIONS[int(a_idx)]
        rows.setdefault(name, {"count": 0, "pred": [], "actual": []})
        rows[name]["count"] += 1
        rows[name]["pred"].append(float(y_pred[i]))
        rows[name]["actual"].append(float(y_true[i]))
    return {
        name: {
            "count": r["count"],
            "mean_predicted_epv": round(float(np.mean(r["pred"])), 6),
            "mean_actual_outcome": round(float(np.mean(r["actual"])), 6),
            "mae": round(float(np.mean(np.abs(np.array(r["pred"]) - np.array(r["actual"])))), 6),
        }
        for name, r in sorted(rows.items())
    }


def action_ranking_quality(model, states, mean, std, logged_actions):
    """
    For every state, evaluate the EPV model on ALL candidate actions and
    check whether its argmax matches the action that was actually
    (heuristically) logged for that state's transition.

    NOTE: `logged_actions` come from the same zone/speed heuristic used to
    build the training labels (epv/action_generator.py), not from a real
    action classifier or true optimal-play labels. This metric is
    therefore a self-consistency check ("did the EPV model learn to prefer
    the action-zone pattern already implicit in the proxy data?"), not a
    measure of tactical correctness. It becomes meaningful once real
    action/outcome labels replace the proxy.
    """
    states_norm = (states - mean) / std
    states_t = torch.from_numpy(states_norm.astype(np.float32))
    num_actions = model.num_actions

    with torch.no_grad():
        all_values = np.zeros((states.shape[0], num_actions), dtype=np.float32)
        for a in range(num_actions):
            action_t = torch.full((states_t.shape[0],), a, dtype=torch.long)
            all_values[:, a] = model(states_t, action_t).numpy()

    predicted_best = all_values.argmax(axis=1)
    match_rate = float(np.mean(predicted_best == logged_actions))
    return match_rate, all_values


def evaluate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    npz = np.load(args.dataset, allow_pickle=True)
    states, actions, outcomes = npz["states"], npz["actions"], npz["outcomes"]

    model, ckpt = load_model(args.checkpoint)
    _plot_loss_history(args.checkpoint + ".history.json", args.out_dir)
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]

    states_norm = (states - mean) / std
    with torch.no_grad():
        preds = model(
            torch.from_numpy(states_norm.astype(np.float32)),
            torch.from_numpy(actions.astype(np.int64)),
        ).numpy()

    metrics = regression_metrics(outcomes, preds)
    match_rate, all_values = action_ranking_quality(model, states, mean, std, actions)
    metrics["action_ranking_match_rate_vs_proxy_labels"] = match_rate
    metrics["num_samples"] = int(len(outcomes))
    metrics["num_unique_outcomes"] = int(np.unique(outcomes).size)
    metrics["per_action"] = per_action_metrics(outcomes, preds, actions)
    metrics["provenance"] = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "state_dim": int(states.shape[1]),
        "num_actions": int(model.num_actions),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if metrics["num_unique_outcomes"] <= 2:
        metrics["warning"] = (
            f"Only {metrics['num_unique_outcomes']} unique outcome values in this "
            "dataset: pearson_corr and the low losses are NOT evidence of model "
            "quality (a constant-per-team predictor achieves the same). Regenerate "
            "the dataset from a longer, multi-rally log to get meaningful metrics."
        )
        print(f"[EPV evaluate][WARN] {metrics['warning']}")
    metrics["note"] = (
        "outcomes/actions are proxy labels (see epv/dataset.py, epv/action_generator.py); "
        "these metrics validate the pipeline, not tactical accuracy."
    )

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    _try_plots(outcomes, preds, ckpt, args.out_dir, actions)
    return metrics


def _plot_loss_history(history_path, out_dir):
    if not os.path.exists(history_path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    with open(history_path) as f:
        history = json.load(f)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("EPV training vs validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_val_loss.png"), dpi=150)
    plt.close()


def _try_plots(y_true, y_pred, ckpt, out_dir, actions=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[EPV evaluate] matplotlib not available, skipping plots.")
        return

    # predicted vs actual
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("Actual (proxy) outcome")
    plt.ylabel("Predicted EPV")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "predicted_vs_actual.png"), dpi=150)
    plt.close()

    # per-action breakdown: mean predicted vs mean actual per action class
    if actions is not None:
        per_action = per_action_metrics(y_true, y_pred, actions)
        names = list(per_action.keys())
        means_pred = [per_action[n]["mean_predicted_epv"] for n in names]
        means_act = [per_action[n]["mean_actual_outcome"] for n in names]
        x = np.arange(len(names))
        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.2, means_act, width=0.4, label="actual (proxy)")
        plt.bar(x + 0.2, means_pred, width=0.4, label="predicted EPV")
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel("value")
        plt.title("Per-action mean EPV vs actual")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_action_breakdown.png"), dpi=150)
        plt.close()


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate the EPV model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="epv_eval")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    evaluate(args)
