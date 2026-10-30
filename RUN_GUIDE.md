# GRITSENSE: How to Run

This guide covers setting up, running the full pipeline, and producing the tactical
dashboard with **LLM-written commentary** from an offline local model (no API needed).

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
the dashboard (pillow) and the offline commentary LLM (transformers).

## 4. Commentary LLM (offline)

Commentary is written by a local model, `Qwen/Qwen2.5-3B-Instruct`, running on your
machine (on the GPU if there is one, otherwise CPU). No API key or internet is needed once the model is cached: the first run
downloads it (~6 GB) into your Hugging Face cache, and every run after that loads it
from the cache without touching the network.

| `--nlp` | Engine | Notes |
|---|---|---|
| `local` (default) | Offline local LLM | Falls back to templates if the model can't load; slow on CPU |
| `template` | Fixed sentences with the numbers filled in | Instant, always accurate, but not an LLM |

To use a different model:

```powershell
python tactical_dashboard.py --local-model Qwen/Qwen2.5-7B-Instruct
```

Bigger models write better lines but are slower on CPU.

## 5. Run it

### Web app (recommended)

```powershell
python webapp/server.py
```

Open <http://127.0.0.1:5000>. From there you can:

- **Upload video:** drop a match video, click the four court corners on a frame
  (or upload an `annotations.json`), set team names and press **Run analysis**.
  This runs the full pipeline (detection → ball cleaning → graphs → dashboard).
- **Use sample match:** renders the dashboard from the bundled sample in a few minutes.

You can follow progress live. The results page shows the dashboard video with synced
commentary, a point-chance timeline (click it to seek), recommended plays, key
defenders, target zones and snapshots, plus downloads. Runs are saved in
`webapp/jobs/`. Videos that aren't 1080p are scaled to 1080p automatically.

A full run writes its detections into the project root, the same way
`master_main.py` does. The app backs up the sample's detection files to
`webapp/sample_backup/` first, so sample runs stay correct.

### Command line

### A. Dashboard only (fast: uses the detection results already in the folder)

```powershell
python tactical_dashboard.py
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

Stage 4 uses the offline local LLM for commentary.

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

### C3. Pretrained group-activity GAT

```powershell
python group_activity.py --video sample3.mp4
```

Runs the pretrained GAT in `GAT/` (RCRG-R2-C11-conc-temporal-V1, Volleyball dataset)
next to the project's own GAT/HRN. Every 9-frame window is classified as one of
`l/r set`, `l/r spike`, `l/r pass` or `l/r winpoint`. The results go to
`output_recent/group_activity/group_activity.json`. `master_main.py` runs this
automatically after the graph stage. It only needs torch, torchvision and
torch_geometric; the extra packages in `GAT/requirements.txt` are not needed.

The dashboard shows it on the video feed: a **Group activity** card (current action,
team, confidence) at the top right and a colour-coded **rally timeline** along the
bottom (blue pass, teal set, amber spike, red winpoint; the thin top edge is the team).
If no predictions exist for the video, the dashboard runs the model first. To turn
the overlay off, use `python tactical_dashboard.py --no-group-activity`. In the web
app, results pages also get a **Rally phases** card: a timeline you can click to
seek, kept in sync with playback, plus each team's split of actions.

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
python tactical_dashboard.py                       # combined (default)
python tactical_dashboard.py --learned-weight 0.5  # give the models more say
python tactical_dashboard.py --physics-only        # physics alone
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
| `--nlp` | `--nlp template` | Commentary engine (see section 4) |
| `--local-model` | `--local-model Qwen/Qwen2.5-7B-Instruct` | Different local model |
| `--learned-weight` | `--learned-weight 0.5` | Share of the trained GAT / HRN in the combined outputs (default 0.3) |
| `--physics-only` | `--physics-only` | Ignore the trained GAT / HRN |
| `--keep-existing-output` | (master_main only) | Don't clear `output_recent/` first |

Quick test before a full render:

```powershell
python tactical_dashboard.py --max-frames 90 --show
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
Commentary engine: LOCAL LLM
```

and every line in `commentary.txt` is tagged:

```
[00:07.2] (move, LOCAL LLM) #92 has the angle for BRA...
```

A line tagged `TEMPLATE` means the LLM's answer was rejected (it didn't mention the
real zone and player) or generation failed, so the safe template was used for that line.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `transformers not installed` | `pip install transformers` |
| Local LLM is very slow | Expected on CPU. Use `--nlp template` or `--max-frames` for tests. |
| `No module named 'ultralytics'` | `pip install ultralytics` (needed for detection only) |
| `No module named 'torch_geometric'` | `pip install torch_geometric` (needed for the graph stage only) |
| Windows symlink warning from Hugging Face | Harmless. Enable Developer Mode to silence it. |
| `UnicodeEncodeError: 'charmap' codec can't encode character '✓'` | The console can't print ✓. Run `$env:PYTHONIOENCODING = "utf-8"` first. |
| `No trained checkpoints - using physics only` | Run `python train_gat_hrn.py` first. |
| Dashboard looks wrong for a new video | `annotations.json` must contain the four court corners for that video's frames. |
