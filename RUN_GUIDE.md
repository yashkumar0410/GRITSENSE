# GRITSENSE: How to Run

This guide covers setting up, running the full pipeline, and producing the tactical
dashboard with **LLM-written commentary** (Claude or a local model).

---

## 1. Requirements

- Windows 10/11, macOS or Linux
- Python 3.10 or newer (tested on 3.13)
- About 5 GB of free disk space (models plus the optional local LLM)
- A GPU is optional. Detection and the local LLM both run on CPU, just slower.

## 2. Project files that must be present

Run everything from the project folder (the one containing `master_main.py`).

| File | What it is |
|---|---|
| `sample3.mp4` | Input match video |
| `player_detection.pt` | YOLO player detector |
| `pose.pt` (or `yolov8n-pose.pt`) | YOLO pose model (17 keypoints) |
| `ball.pt` | YOLO ball detector |
| `annotations.json` | Four court corners for every frame (used for the homography) |
| `court.png` | Mini-court image for the detection preview |
| `analytics/` | EPV, matchups, team shape and commentary code |
| `tactical_dashboard.py` | Dashboard renderer |

## 3. Install

Open a terminal in the project folder.

```powershell
# (optional but recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

`requirements.txt` covers the detection pipeline (opencv-python, ultralytics, torch,
supervision, numpy), the graph models (torch_geometric), training plots (matplotlib),
the dashboard (pillow) and the commentary LLMs (anthropic, transformers).

## 4. Set up the LLM for commentary

The dashboard picks its commentary engine with `--nlp`:

| `--nlp` | Engine | Needs | Notes |
|---|---|---|---|
| `claude` | Claude (`claude-opus-5-5`) via the Anthropic API | API key + internet | **Recommended**: most accurate and fastest |
| `local` | `Qwen/Qwen2.5-1.5B-Instruct` on your machine | Nothing (downloads ~3 GB once) | Works offline; slow on CPU (10+ min per video) |
| `template` | Fixed sentences with the numbers filled in | Nothing | Instant, always accurate, but not an LLM |
| `auto` (default) | Tries Claude, then local, then template | — | — |

### Option A: Claude (recommended)

1. Create an API key at <https://console.anthropic.com> (Settings → API Keys).
2. Set it in your terminal:

   ```powershell
   # Windows PowerShell, current window only
   $env:ANTHROPIC_API_KEY = "sk-ant-..."

   # Windows, permanently (open a NEW terminal afterwards)
   setx ANTHROPIC_API_KEY "sk-ant-..."
   ```

   ```bash
   # macOS / Linux
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

3. Never put the key in the code or commit it anywhere.

Cost: the dashboard writes roughly 8–12 commentary lines per video (one per event,
not one per frame), so a run costs very little, well under $1.

### Option B: Local LLM (offline)

Nothing to set up. The first run downloads the model from Hugging Face (~3 GB)
into your Hugging Face cache. To use a different model:

```powershell
python tactical_dashboard.py --nlp local --local-model Qwen/Qwen2.5-3B-Instruct
```

Bigger models write better lines but are slower on CPU.

## 5. Run it

### A. Dashboard only (fast: uses the detection results already in the folder)

```powershell
python tactical_dashboard.py --nlp claude
```

Same thing through the master runner:

```powershell
python master_main.py --dashboard-only
```

### B. Full pipeline: video → detection → graphs → dashboard

```powershell
python master_main.py --video sample3.mp4 --dashboard
```

This runs these stages:

1. **Detection:** player tracking, pose, ball tracking and homography. Writes
   `player_keypoints_mf3_pose.json`, `ball.csv` and `master_detection_preview.mp4`.
   - **1b. Ball re-detection and cleaning:** writes `ball_candidates.json` and
     `ball_clean.csv` (see section C).
2. **Graph pipeline:** 49-feature graphs, GAT → HRN embeddings, temporal sequences.
3. **Graph visualisations**
4. **Tactical dashboard**

Stage 4 uses `--nlp auto`. To force Claude for the full pipeline, set the API key
first (auto picks Claude whenever the key is set).

### C. Improve the ball track

```powershell
python redetect_ball.py      # needs ultralytics + ball.pt; about 2-3 minutes on CPU
```

Re-runs the ball detector at three image sizes and keeps several candidates per frame,
removes static objects (referee's head, cursor, antenna), picks the physically possible
path and fills short gaps. Writes `ball_candidates.json` and `ball_clean.csv`, which the
graph pipeline, calibration, training and dashboard then use automatically.
`master_main.py` runs this step after detection. Without ultralytics,
`python clean_ball_track.py` cleans the original `ball.csv` instead.

### C2. Calibrate the physics from your tracking

```powershell
python calibrate_physics.py
```

Recovers the camera from your court homography, then measures from the video: player
top speed, standing height, and block-jump height. It also checks the model against
published attack success. Results go to `analytics/physics_calibration.json`, which the
physics model loads automatically. Run it again after adding videos.

Every constant and its source (measured / published / derived / estimated) is listed in
`analytics/constants.py`.

### D. Train the GAT and HRN

```powershell
python train_gat_hrn.py
```

About 2–3 minutes on CPU for the current clip. No manual labels are needed:

- **GAT** learns to predict every player's movement 0.5 s ahead (the tracking provides the answers).
- **HRN** starts from the trained GAT and learns the physics model's best zone, best shot, threat and team shape.

It writes to `checkpoints/`:

| File | Contents |
|---|---|
| `gat.pt`, `hrn.pt` | Trained weights (loaded automatically by `integrate_and_rebuild.py`) |
| `training_report.json` | Data split, timings, test results vs baselines |
| `training_curves.png` | Train / validation loss per epoch |

Once `checkpoints/gat.pt` and `hrn.pt` exist, the dashboard **combines** the physics
models with the trained GAT / HRN automatically:

| Output | Combination |
|---|---|
| Where to hit (map, target, move) | Physics EPV re-ranked by the HRN's shot and zone probabilities |
| Key matchups | 70% physics "threat stopped" + 30% GAT attention |
| Team shape | 70% physics coverage/balance + 30% HRN prediction |

The panels are labelled "PHYSICS + GAT" and "PHYSICS + HRN", and the HRN's own zone and
shot prediction is shown in the Recommended Play card. "Point chance" stays the physics
probability for the chosen target.

```powershell
python tactical_dashboard.py --nlp claude                       # combined (default)
python tactical_dashboard.py --nlp claude --learned-weight 0.5  # give the models more say
python tactical_dashboard.py --nlp claude --physics-only        # physics alone
```

To retrain after adding more videos, run `master_main.py` on each video first, then
`train_gat_hrn.py` again.

### Useful options

| Option | Example | What it does |
|---|---|---|
| `--video` | `--video match2.mp4` | Input video |
| `--teams` | `--teams ITA POL` | Team names for the left and right halves |
| `--max-frames` | `--max-frames 150` | Render only the first N frames (quick test) |
| `--show` | `--show` | Live preview window while rendering (press `q` to stop) |
| `--nlp` | `--nlp claude` | Commentary engine (see section 4) |
| `--claude-model` | `--claude-model claude-sonnet-5-5` | Different Claude model |
| `--learned-weight` | `--learned-weight 0.5` | Share of the trained GAT / HRN in the combined outputs (default 0.3) |
| `--physics-only` | `--physics-only` | Ignore the trained GAT / HRN |
| `--keep-existing-output` | (master_main only) | Don't clear `output_recent/` first |

Quick test before a full render:

```powershell
python tactical_dashboard.py --nlp claude --max-frames 90 --show
```

## 6. Outputs

All dashboard outputs go to `output_recent/dashboard/`:

| File | Contents |
|---|---|
| `gritsense_dashboard.mp4` | The dashboard video (1920 × 1080) |
| `commentary.txt` | Every commentary line, with time and which engine wrote it |
| `dashboard_insights.json` | Every frame's numbers (point chance, target, matchups, team shape…) |
| `dashboard_0080.png` … | Still snapshots every 80 frames |

### Check the LLM is really being used

The console prints the engine at the start:

```
Commentary engine: CLAUDE
```

and every line in `commentary.txt` is tagged:

```
[00:07.2] (move, CLAUDE) #92 has the angle for BRA...
```

A line tagged `TEMPLATE` means the LLM's answer was rejected (it didn't mention the
real zone and player) or the call failed, so the safe template was used for that line.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Claude unavailable - Could not resolve authentication method` | API key not set in *this* terminal. Set `$env:ANTHROPIC_API_KEY` again, or open a new terminal after `setx`. |
| `Claude unavailable - AuthenticationError` | Key is wrong or revoked. Create a new one. |
| `anthropic SDK not installed` | `pip install anthropic` |
| `transformers not installed` | `pip install transformers` |
| Local LLM is very slow | Expected on CPU. Use `--nlp claude`, or `--max-frames` for tests. |
| `No module named 'ultralytics'` | `pip install ultralytics` (needed for detection only) |
| `No module named 'torch_geometric'` | `pip install torch_geometric` (needed for the graph stage only) |
| Windows symlink warning from Hugging Face | Harmless. Enable Developer Mode to silence it. |
| `UnicodeEncodeError: 'charmap' codec can't encode character '✓'` | The console can't print ✓. Run `$env:PYTHONIOENCODING = "utf-8"` first. |
| `No trained checkpoints - using physics only` | Run `python train_gat_hrn.py` first. |
| Dashboard looks wrong for a new video | `annotations.json` must contain the four court corners for that video's frames. |
