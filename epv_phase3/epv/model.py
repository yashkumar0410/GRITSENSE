"""
epv/model.py
-------------
Task 4 of Member 3's Phase 3 work: a clean, modular MLP baseline for the
EPV (Expected Possession Value) model.

Input  = state vector (see epv/state_builder.py) concatenated with a
         one-hot encoding of the candidate action (see epv/action_generator.py)
Output = a single scalar "expected value" for taking that action in that
         state. Kept unconstrained (linear output) rather than squashed to
         [-1, 1], since the proxy outcome labels are already roughly in
         that range but real labels later might not be.
"""

import torch
import torch.nn as nn


class EPVModel(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_layers=(128, 64), dropout=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        input_dim = state_dim + num_actions
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output layer: scalar EPV
        self.net = nn.Sequential(*layers)

    def forward(self, state, action_idx):
        """
        state : FloatTensor (B, state_dim)
        action_idx : LongTensor (B,)
        returns FloatTensor (B,) EPV predictions
        """
        action_onehot = torch.zeros(
            state.shape[0], self.num_actions, device=state.device, dtype=state.dtype
        )
        action_onehot.scatter_(1, action_idx.view(-1, 1), 1.0)
        x = torch.cat([state, action_onehot], dim=1)
        return self.net(x).squeeze(-1)

    def config(self):
        """Config dict needed to reconstruct this model (saved alongside checkpoints)."""
        hidden = [m.out_features for m in self.net if isinstance(m, nn.Linear)][:-1]
        dropout = next((m.p for m in self.net if isinstance(m, nn.Dropout)), 0.0)
        return {
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "hidden_layers": hidden,
            "dropout": dropout,
        }
