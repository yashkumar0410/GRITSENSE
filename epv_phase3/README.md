# GritSense — EPV Module (Member 3, Phase 3)

Expected Possession Value (EPV) modelling for volleyball: turn the
perception pipeline's per-frame tracking output into (state, action,
outcome) samples, train an MLP that estimates the value of each candidate
action in each game state, and evaluate prediction + decision quality.

```
main.py --log-out [--ball-log-out]      (perception pipeline, additive EPV hooks)
        │
        ▼  player log (JSON)            (+ optional court-space ball log)
epv/rally_segmentation.py   ── split log into rallies + proxy game states
epv/state_builder.py        ── fixed-size state vector per frame (69-D base / 76-D with ball)
epv/action_generator.py     ── 6-action space + proxy action labels
epv/dataset.py              ── (state, action, outcome) dataset (.npz + .meta.json)
epv/train.py                ── MLP training, rally-level train/val split
epv/model.py                ── EPVModel: (state ⊕ one-hot action) → 128 → 64 → scalar EPV
epv/evaluate.py             ── regression + per-action + ranking metrics, plots
epv/log_report.py           ── input-log quality report (catches broken homography)
epv/make_ball_log.py        ── pixel ball detections → court-space ball log
```

## Quick start

```bash
# 0. check log quality FIRST (quantifies how much of the log is in-court)
python -m epv.log_report --log match0_log.json

# 1. build the dataset from a player log (add --ball-log-out log for 76-D states)
python -c "from epv.dataset import generate_dataset; generate_dataset('match0_log.json', 'epv_dataset_match0.npz')"

# 2. train (rally-level split; CPU-friendly)
python -m epv.train --dataset epv_dataset_match0.npz --out epv_model_match0.pt --epochs 30

# 3. evaluate (metrics.json + plots)
python -m epv.evaluate --dataset epv_dataset_match0.npz --checkpoint epv_model_match0.pt --out-dir epv_eval_match0

# 4. export per-frame action recommendations to CSV
python show_epv.py
```

Generate a new player/ball log from video (needs the court-points `cp.json`
and the YOLO weights; `--ball-log-out` additionally loads `ball.pt`):

```bash
python main.py --video full_match_0.mp4 --log-out match0_log.json --ball-log-out match0_ball_log.json --no-display
```

If you already have ball detections in *pixel* space (e.g. the Cappy
`ball_coordinates.json`), convert them instead of re-running detection:

```bash
python -m epv.make_ball_log --ball-pixels ball_coordinates.json --court-json cp.json --fps 30 --out ball_log.json
```

## State vector (epv/state_builder.py)

- **Base (69-D):** per team (left/right), 7 player slots × 4 features
  (present, x, y, speed — normalised) + 6 team aggregates (centroid, spread,
  mean speed, count) = 68, + 1 rally-progress = **69**.
- **Ball-augmented (76-D):** base + 7 ball features (present, x, y, speed,
  mean distance of each team to the ball, ball side). Enabled by passing a
  ball log to `generate_dataset(..., ball_log_path=...)` or by attaching a
  `"ball"` key to frame records.
- Pose features are **not** included yet (pose.pt is not wired into
  main.py); the extension point is documented at the top of
  `state_builder.py`.

## Honest limitations (read before quoting numbers)

1. **Proxy outcomes.** There are no real rally/point labels; outcomes are
   per-team net court-territory gain (`outcome_is_proxy: true` in every
   `.meta.json`). Swap point: `epv.dataset.compute_proxy_outcome`.
2. **Proxy actions.** Action labels come from a zone/speed heuristic
   (`epv/action_generator.py`), not a trained classifier.
3. **Current logs are short and degenerate** (≈150 frames, 1 rally, 2
   unique outcome values). `evaluate.py` prints an explicit warning when
   the outcome set is degenerate; low loss / high Pearson on such data is
   a pipeline sanity check, NOT model quality.
4. **Homography quality.** The committed logs have broken court mapping
   (only ~1% of records land inside the court) — see
   `python -m epv.log_report`. Fix the court-point mapping in main.py and
   re-export logs before drawing any conclusions.
5. Train/val split is by rally; with a single rally the code validates on
   the training set and says so (`[LIMITATION]` at end of training).

## Tests

```bash
python -m unittest discover -s tests -v
```

18 tests cover the state layout (69/76-D), ball features, rally
segmentation, game-state labels, ball-log loading, log quality report,
proxy outcome direction, action inference and the model forward pass.

## Artifacts

| File | Meaning |
|---|---|
| `epv_dataset_match0.npz` (+`.meta.json`) | dataset from `match0_log.json` (148 samples, 69-D) |
| `epv_dataset.npz` (+`.meta.json`) | dataset from `real_log.json` (151 samples, 69-D) |
| `epv_model_match0.pt` / `epv_model.pt` | trained MLP checkpoints (state_dim, norm stats, weights) |
| `epv_eval_match0/` / `epv_eval/` | `metrics.json` (regression + per-action + ranking + provenance), loss/predicted-vs-actual/per-action plots |
| `epv_results_match0.csv` | per-frame recommended action + EPV of all 6 actions |
| `epv_eval_ball_demo/` | end-to-end demo of the ball-augmented 76-D path (synthetic ball log) |
