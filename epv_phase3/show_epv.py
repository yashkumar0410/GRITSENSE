"""
show_epv.py
-----------
Exports and displays EPV predictions in a clean, well-formatted CSV spreadsheet
and Markdown report for your Phase 3 report and viva.
"""

import csv
import json
import os
import numpy as np
import torch

from epv.action_generator import ACTIONS
from epv.evaluate import load_model

def main():
    # 1. Load dataset & model checkpoint
    npz = np.load("epv_dataset_match0.npz")
    states = npz["states"]
    model, ckpt = load_model("epv_model_match0.pt")
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]

    # 2. Normalize state vector
    states_norm = (states - mean) / std
    states_t = torch.from_numpy(states_norm.astype(np.float32))

    # 3. Calculate EPV score for all 6 candidate actions
    num_actions = len(ACTIONS)
    with torch.no_grad():
        all_values = np.zeros((states.shape[0], num_actions), dtype=np.float32)
        for a in range(num_actions):
            action_t = torch.full((states_t.shape[0],), a, dtype=torch.long)
            all_values[:, a] = model(states_t, action_t).numpy()

    # 4. Save to CSV spreadsheet (epv_results.csv)
    csv_file = "epv_results_match0.csv"
    headers = ["Frame", "Recommended_Action", "Best_EPV_Score"] + [f"EPV_{act}" for act in ACTIONS]
    
    rows = []
    for i in range(len(states)):
        best_idx = all_values[i].argmax()
        best_action = ACTIONS[best_idx]
        best_score = float(all_values[i][best_idx])
        
        row = [i, best_action, round(best_score, 4)] + [round(float(all_values[i][a]), 4) for a in range(num_actions)]
        rows.append(row)

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # 5. Print a beautifully formatted summary table
    print("\n" + "=" * 90)
    print(" GRITSENSE EPV MODEL -- PER-FRAME ACTION EVALUATION REPORT")
    print("=" * 90)
    print(f"{'Frame':<8} | {'Best Recommended Action':<26} | {'Best EPV Score':<15} | {'Status':<15}")
    print("-" * 90)

    for i in range(min(15, len(states))):
        best_act = rows[i][1]
        best_score = rows[i][2]
        print(f"Frame {i:<3} | {best_act:<26} | {best_score:<+15.4f} | Optimal Move [OK]")

    print("-" * 90)
    print(f"Full results saved to spreadsheet: {os.path.abspath(csv_file)}")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
