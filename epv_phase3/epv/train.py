"""
epv/train.py
-------------
Task 5 of Member 3's Phase 3 work: train the EPV MLP on the
(state, action, outcome) dataset produced by epv/dataset.py.

Splits by RALLY, not by row, so consecutive frames from the same rally
never leak across the train/val split.

Usage
-----
    python -m epv.train --dataset epv_dataset.npz --out epv_model.pt
    python -m epv.train --dataset epv_dataset.npz --out epv_model.pt --limit 200   # quick test
"""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from epv.action_generator import NUM_ACTIONS
from epv.model import EPVModel



class EPVDataset(Dataset):
    def __init__(self, states, actions, outcomes, mean=None, std=None):
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.int64)
        self.outcomes = outcomes.astype(np.float32)
        self.mean = mean
        self.std = std
        if mean is not None:
            self.states = (self.states - mean) / std

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.tensor(self.actions[idx]),
            torch.tensor(self.outcomes[idx]),
        )


def rally_split(rally_ids, val_frac=0.2, seed=0):
    """Split UNIQUE rally ids into train/val so no rally appears in both."""
    unique_rallies = np.unique(rally_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_rallies)
    n_val = max(1, int(len(unique_rallies) * val_frac)) if len(unique_rallies) > 1 else 0
    val_rallies = set(unique_rallies[:n_val].tolist())
    train_mask = np.array([rid not in val_rallies for rid in rally_ids])
    return train_mask, ~train_mask


def train(args):
    npz = np.load(args.dataset, allow_pickle=True)
    states, actions, outcomes, rally_ids = (
        npz["states"], npz["actions"], npz["outcomes"], npz["rally_ids"],
    )

    if args.limit is not None:
        states, actions, outcomes, rally_ids = (
            states[: args.limit], actions[: args.limit], outcomes[: args.limit], rally_ids[: args.limit],
        )

    train_mask, val_mask = rally_split(rally_ids, val_frac=args.val_split, seed=args.seed)
    if val_mask.sum() == 0:
        print("[WARN] Not enough distinct rallies for a val split; using train set for validation too.")
        val_mask = train_mask

    # Normalize using TRAIN split statistics only (avoid leakage).
    mean = states[train_mask].mean(axis=0, keepdims=True)
    std = states[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    train_ds = EPVDataset(states[train_mask], actions[train_mask], outcomes[train_mask], mean, std)
    val_ds = EPVDataset(states[val_mask], actions[val_mask], outcomes[val_mask], mean, std)

    print(f"[EPV train] {len(train_ds)} train samples / {len(val_ds)} val samples "
          f"({len(np.unique(rally_ids))} total rallies)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    state_dim = states.shape[1]
    num_actions = NUM_ACTIONS
    model = EPVModel(state_dim, num_actions, hidden_layers=tuple(args.hidden_layers), dropout=args.dropout)

    loss_fn = torch.nn.SmoothL1Loss() if args.loss == "smooth_l1" else torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for state, action, outcome in train_loader:
            optimizer.zero_grad()
            pred = model(state, action)
            loss = loss_fn(pred, outcome)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for state, action, outcome in val_loader:
                pred = model(state, action)
                val_losses.append(loss_fn(pred, outcome).item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": model.config(),
                "norm_mean": mean,
                "norm_std": std,
                "loss_fn": args.loss,
            }
            torch.save(checkpoint, args.out)

    with open(args.out + ".history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[EPV train] best val_loss={best_val:.4f} -> saved checkpoint to {args.out}")
    if len(np.unique(rally_ids)) < 5:
        print("[LIMITATION] Very few rallies were available for training/validation "
              "(dataset is small / proxy-labelled). Loss values above should be read "
              "as a pipeline sanity check, not as evidence of a strong model -- see "
              "epv/dataset.py and the Phase 3 report limitations section.")

    return model, history


def build_argparser():
    p = argparse.ArgumentParser(description="Train the EPV MLP model")
    p.add_argument("--dataset", required=True, help="path to .npz produced by epv/dataset.py")
    p.add_argument("--out", default="epv_model.pt", help="output checkpoint path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[128, 64])
    p.add_argument("--loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="use only first N rows (quick test mode)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
