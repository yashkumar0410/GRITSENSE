# GRITSENSE – COMPLETE SYSTEM ANALYSIS, IMPLEMENTATION DOCUMENTATION & TECHNICAL AUDIT

**Project:** GritSense – Real-Time Volleyball Game Group Activity Recognition and Forecasting via Pose and Graph-Based Temporal Learning

**Course:** UE23CS320B – Capstone Project, Phase 2 and Phase 3 Review 1
**Institution:** Department of Computer Science and Engineering, PES University, Bengaluru
**Team:** Viprakshi Gupta (PES2UG23CS697), Vishwas H Gowda (PES2UG23CS705), Yash Kumar (PES2UG23CS711), Y Kusumita (PES2UG23CS718)
**Guide:** Prof. Kundhavai K R

**Document type:** Reverse-engineered technical documentation and implementation audit
**Audit date:** 13 November 2026
**Repository audited:** `E:\capstone-4\GRITSENSE` (git HEAD `15f05e2`, working tree mid-reorganisation)

---

## How to read this document

This report was produced by reading the repository itself, not by restating the project proposal. Every architectural claim is tied to a file that exists, a checkpoint that was inspected, or a log/artifact that the pipeline produced. Where a Phase 2 or Phase 3 document promises something that is absent, incomplete or broken in the code, the report says so explicitly in Chapter 7 and in the status column of every component in Chapter 6.

Three facts should be kept in mind while reading:

1. **The repository is currently mid-restructure.** Files have been moved into `pipeline/`, `data/`, `weights/`, `docs/` and `tests/`, but the moved modules still resolve their data paths relative to their own directory. As a result the pipeline cannot run end-to-end *in the tree as it stands today*, even though it demonstrably ran six times on 4 November 2026 (Chapter 10). Chapter 7 documents this precisely.
2. **Only one clip has ever been processed.** `data/sample3.mp4` – a 319-frame, 30 fps, 10.6 s broadcast excerpt of Brazil v Japan (2022 FIVB Men's World Championship). Every number in this report comes from that clip or from the model trained on it.
3. **The learned models are trained but not useful yet.** The project's GAT and HRN have checkpoints and a held-out evaluation, and they *lose* to trivial baselines. The report reproduces those numbers exactly instead of hiding them.

### Evidence basis

| Evidence class | Sources used | Weight in this report |
|---|---|---|
| Phase 2 ESA report | `95_Phase2_ESA_Report.pdf` (text extracted) | Intent, dataset plan, original architecture |
| Phase 3 Review 1 deck | `final ppt for phase 3 review 1.pdf` (text extracted) | Phase 3 planned scope |
| Project documentation | `docs/RUN_GUIDE.md`, `docs/SYSTEM_EXPLAINED.md` | Operational intent, author-reported observations |
| Source code | 51 Python modules, ~17.9k lines outside `.venv` | **Primary evidence** for all implementation claims |
| Trained artifacts | `checkpoints/training_report.json`, `checkpoints/gat.pt`, `checkpoints/hrn.pt` | Metric provenance |
| Measured constants | `analytics/physics_calibration.json` | Measured physics values |
| Run artifacts | 6 completed jobs in `webapp/jobs/`, dashboard JSON/MP4/commentary | Proof of end-to-end execution |
| Live verification | Import/path probes, unit-test run, weight-file introspection (13 Nov 2026) | Current-state defects |

**Extraction limitation.** Both supplied PDFs are partly image-based: several Phase 2 chapters (Design Details 4.1–4.12, System Architecture 5.x, Design Description 6.x, Technologies 7.x, Implementation 8.x, Conclusion 9.x) extracted as headings only, and the Phase 3 deck extracted as slide titles plus bullets. Diagram, pseudocode and table content inside those pages could not be machine-read. Consequently, where a document claim is checked, the code is treated as authoritative and the document as intent.

---

# 1. EXECUTIVE SUMMARY

GritSense is a computer-vision and graph-learning pipeline that converts a broadcast volleyball video into court-space tactical analytics: per-player court positions and velocities, 17-keypoint poses, a cleaned ball trajectory, ball-touch events, rally phase, interaction graphs processed by a Graph Attention Network and a Hierarchical Relational Network, a physics-based Expected Point Value surface over 1,296 target cells, blended "physics + learned" recommendations, LLM-written commentary, and a rendered 1920x1080 dashboard video. A Flask web application wraps the pipeline in a job queue with upload, live progress, insight API and downloadable results.

**What is genuinely implemented and runs.** Player tracking (Ultralytics YOLO + BoT-SORT), 17-keypoint pose estimation and pose-to-track association, homography from hand-annotated court corners, camera recovery from that homography (focal length and 3-D pose), court-metre smoothing and velocity, ball detection plus multi-scale re-detection, dynamic-programming ball path selection with static-object rejection and quadratic gap filling, ball-touch detection with set/pass/attack classification, the 49-feature interaction graph, the project's own GAT and HRN with training and checkpoints, 32-D per-frame global embeddings, 8-frame temporal windows with a GRU and a Transformer implementation, a physics EPV engine (ball flight with drag, blocker model, defender race, aim error), zone/matchup/team-shape analytics, an inference path for an external pretrained group-activity model, a template/LLM commentary generator, a dashboard renderer, and a local web application.

**What is not implemented, is only partially implemented, or is broken.** Automatic court keypoint detection does not exist (corners are hand-annotated for 319 specific frames). BlazePose is not used (the code uses YOLOv8-pose with 17 COCO keypoints). There is no rally *segmentation* dataset; only a per-frame phase label derived from touches. There is no trained GRU/Transformer checkpoint and no script that trains one, so *activity classification by the project's own temporal models has never produced an output*; the pipeline prints that it skipped it. "Forecasting" exists only as GAT 0.5 s movement prediction, not as future group-activity prediction. The GRU-versus-Transformer comparison promised in the Phase 3 deck was never run. The external pretrained group-activity stage now fails because the `GAT/` directory it loads from was deleted in the working tree. The test suite fails to import. And every pipeline module resolves its data files to `pipeline/` instead of `data/`, so the current tree cannot execute.

**Measured quality.** On a held-out 63-frame test split the project's GAT predicts player movement with 0.873 m average displacement error, against 0.765 m for the "nobody moves" baseline and 1.557 m for constant velocity. The HRN reaches 46.0% zone accuracy against a 49.2% majority-class baseline and 92.1% shot accuracy against a 92.1% baseline; its team-shape error is 0.066. These are honest negative results caused by having 10.6 seconds of data, and the code itself scales the HRN's influence to zero because its measured skill is zero.

**Maturity verdict.** A well-engineered, unusually self-critical research prototype with a complete physics/analytics pipeline and a complete but unexercised learning pipeline. It is not real-time (offline batch), not multi-video (one hand-annotated clip), not statistically evaluated, and temporarily non-executable at HEAD. It is, however, honest: the system contains its own baselines, provenance-tagged constants, and explicit guards that prevent unskilled learned models from overriding the physics.

---

# 2. PROBLEM STATEMENT AND WHAT THE SYSTEM ACTUALLY SOLVES

## 2.1 The stated problem (Phase 2)

Phase 2 defines the problem as designing a real-time group activity recognition system that combines fast object detection, pose estimation, temporal modelling and graph-based reasoning to understand and predict collective behaviour from video. Its motivating example is precise and technically correct: recognising that several players are jumping does not distinguish a coordinated block from a celebration, because the *relations* between players carry the meaning. The report also argues – correctly – that most group-activity systems are heavy, slow, and stop at recognition without evaluation or recommendation.

## 2.2 Why individual action recognition is insufficient in volleyball

Volleyball is a three-touch, phase-structured game. A spike is not defined by one body: it needs a pass (reception), a set (a specific player moving to a specific point), an approach, a jump, and a block forming on the other side. The same pixel-level motion – two players leaving the floor near the net – is a block on one side and an attack on the other; the distinguisher is who is on which side of the net, where the ball is, and which team has touched it last. GritSense's implemented answer to this is a three-part mechanism:

1. **Spatial relations** – the graph encodes every player-player and player-ball pair with relative displacement, distance and same-team flags, so a node's representation depends on its neighbours, not only itself.
2. **Team hierarchy** – the HRN pools player embeddings into two team embeddings and one scene embedding, which is what makes "our shape versus their shape" representable.
3. **Phase state** – touches (detected from ball-trajectory kinks plus reach geometry) assign possession and drive a serve → pass → set → attack state machine, so the same formation is interpreted differently in different phases.

## 2.3 What the system actually delivers today

| Claimed capability | Delivered reality |
|---|---|
| Understand group activity from video | Court-space state estimation + physics analytics are real; learned group-activity labels come from an external checkpoint, not from the project's own models |
| Predict collective future behaviour | Only per-player 0.5 s displacement prediction (GAT), and it is worse than assuming no movement |
| Real-time | No. Offline batch over a whole clip, with optional profiling; no FPS target met or measured in the repo |
| Tactical recommendation | Yes, and this is the strongest part: EPV surface, target, shot type, blockers, defensive matchups, team shape |
| Multi-sport generalisation | Design is modular, but nothing outside volleyball exists |

The honest framing is therefore: **GritSense is a volleyball tactical-analytics engine whose perception and decision layers are implemented and working, wrapped around a group-activity learning layer that is architecturally complete but data-starved.**

---

# 3. CONCEPTS FROM FIRST PRINCIPLES

Each concept below is defined only as far as GritSense needs it, then tied to the module that implements it.

## 3.1 Computer vision layer

**Computer vision** is the extraction of structured information from pixels. GritSense never reasons on raw pixels beyond the detection stage; everything downstream is numbers (metres, m/s, keypoints, probabilities).

**Object detection** localises instances with axis-aligned boxes plus a class and a confidence. **YOLO** ("You Only Look Once") predicts all boxes in a single forward pass over a grid, which is why it is used here for throughput. GritSense loads three Ultralytics YOLO checkpoints: `weights/player_detection.pt` (classes `ball`, `player`), `weights/ball.pt` (classes `Volleyball`, `ball`), and `weights/pose.pt` (task `pose`, class `person`). A **bounding box** is stored as `[x1, y1, x2, y2]` in pixels.

**Object tracking** links detections across frames so each person keeps an identity. GritSense calls `model.track(..., tracker="botsort.yaml", persist=True)`, i.e. BoT-SORT with appearance and motion cues, and stores the resulting integer `id`. These IDs are *tracker* IDs (e.g. 3, 5, 76, 92) and are displayed as if they were shirt numbers.

**Pose estimation** regresses body keypoints. **Skeletal keypoints** here are the 17 COCO joints (`kpt_shape = [17, 3]`: x, y, confidence). **BlazePose** – cited in both Phase 2 and Phase 3 – produces 33 landmarks and is *not* what the code uses; the implemented estimator is YOLOv8-pose. Because the detector box and the pose box for the same person differ (pose boxes are tighter around visible limbs), the pipeline does not use IoU alone: `_match_pose_to_track` in `master_main.py` scores each candidate pose box by `0.35 * IoU + 0.40 * pose_coverage + 0.25 * centre_proximity`, plus a bonus when the pose centre lies inside a slightly expanded track box, and accepts the best match above 0.18.

**Court detection** is finding the playing surface. GritSense does *not* detect it: `data/annotations.json` contains four image-space corners for every one of the 319 frames, hand-annotated once. **Homography** is the 3x3 projective transform that maps the court plane in the image to real court metres; it is computed per frame by `cv2.findHomography(image_corners, [(0,0),(18,0),(18,9),(0,9)])`. **Image coordinates** are pixels; **real-world coordinates** are court metres with the net at x = 9 m, length 18 m, width 9 m.

**Velocity and speed.** A player's *feet* (box bottom-centre) are the only point guaranteed to lie on the floor, so they are the only point that maps correctly through a floor homography. Raw per-frame differences are unusable (box bottoms jitter by tens of pixels), so `analytics/motion.py` fits a straight line to the last 6 court positions (0.2 s) and takes the slope as velocity, averages the last 3 frames for position, resets a track after a 3-frame gap, and caps speed at the measured top speed (4.75 m/s). The camera model (§6.5) also recovers heights, which is how jump height, reach and touch height are measured from a single view.

## 3.2 Graph learning layer

A **graph** is a set of **nodes** (here: players and the ball) and **edges** (relationships). A **node feature** is a vector describing a node (49 numbers here); an **edge feature** describes a relation (relative x, relative y, distance, same-team). A **Graph Neural Network** updates each node's vector by aggregating its neighbours', so information flows along relations. A **Graph Attention Network (GAT)** learns *how much* each neighbour matters: attention coefficients are computed per edge from the two endpoint vectors (and, in this implementation, from the 4-D edge features via `edge_dim=4`). **Attention** is therefore a learned, normalised importance weight used both for aggregation and – usefully – as an interpretable "who is looking at whom" signal.

A **Hierarchical Relation Network (HRN)**, following Ibrahim & Mori (ECCV 2018), adds levels: player → team → scene. GritSense implements it as GAT node embeddings, then mean-pooling per team (team 0 / team 1), then a learned team projection, then a global vector that averages the two team vectors with the ball embedding, then a learned global projection. **Relational reasoning matters in volleyball** because a defender's importance is not a property of the defender alone – it is a property of the attacker-defender-blocker triangle, which is exactly what message passing over the interaction graph computes.

## 3.3 Temporal layer

**Temporal modelling** means reasoning over a sequence rather than a frame; **sequential data** here is one 32-D vector per frame. A **time window** is a fixed-length slice (8 frames, stride 1) treated as one sample. A **GRU** (Gated Recurrent Unit) is a recurrent network with update and reset gates that carries a hidden state forward, which lets it accumulate rally context cheaply; GritSense's `GRUTemporalModel` consumes the window and classifies from the final hidden state. A **Transformer** replaces recurrence with **self-attention**, letting every timestep attend to every other in one layer; `TransformerTemporalModel` projects the 32-D input to 128-D, adds a positional encoding, runs two encoder layers with 4 heads, mean-pools and classifies. Comparing the two is a legitimate research question (recurrence is cheaper and better for short, ordered windows; attention scales better and captures long-range dependencies), but **GritSense has not run that comparison** – both models exist, neither is trained.

**Future prediction** exists in exactly one place: the GAT's self-supervised head predicts each player's court displacement 15 frames (0.5 s) ahead from the current graph.

## 3.4 Decision layer

The **game state** is the tuple (player positions and velocities, poses, ball position and velocity, possession, phase). **Rally segmentation** – cutting a match into rallies with outcomes – is *not* implemented; what exists is a **rally phase state machine** (serve / pass / set / attack / in flight) driven by detected touches and a three-touch counter.

An **action** is a shot the attacker can legally play: spike, tip/roll shot, back-row attack, jump serve, float serve. **Expected Possession Value (EPV)** here means the probability that a shot to a given court cell wins the point outright:

```
EPV(cell, shot) = P(lands in) x (1 - P(blocked)) x P(no defender reaches it)
```

The **Markov Decision Process** framing from the Phase 3 deck is *not* implemented – there is no transition model, no reward learning and no policy optimisation. What exists is a one-step, model-based evaluation of candidate actions followed by greedy selection with temporal debouncing (a new recommendation must persist 10 frames). **Action recommendation** is therefore best described as physics-based one-step lookahead, not reinforcement learning.

---

# 4. ARCHITECTURE AS IMPLEMENTED

## 4.1 Entry points

There is exactly one project-level entry point in the working tree:

| Entry point | Purpose | Status |
|---|---|---|
| `master_main.py` (785 lines) | Full pipeline orchestrator: detection → ball cleaning → graphs/GAT/HRN → temporal sequences → pretrained group activity → dashboard | Present, currently blocked by path defects (Ch. 7) |
| `webapp/server.py` (495 lines) | Flask app; queues jobs, launches the pipeline as a subprocess, transcodes video, serves insights | Present, same blocker |
| `pipeline/*.py` (14 modules) | Stage implementations, each with its own `__main__` | Present but path-broken when run from root |
| `main.py` | Original Phase 2 script | **Deleted in working tree**; exists only in git HEAD |

Stage order as coded in `master_main.main()`:

1. `run_video_detections()` – YOLO player tracking + pose + ball, homography, writes `data/player_keypoints_mf3_pose.json`, `data/ball.csv`, `docs/demo/master_detection_preview.mp4`.
2. `redetect_ball.main()` – multi-scale ball re-detection → `ball_candidates.json` → `clean_ball_track.main()` → `ball_clean.csv`.
3. `integrate_and_rebuild.main()` – graphs, GAT, HRN, 32-D global embeddings, 8-frame sequences, optional temporal inference.
4. `run_pretrained_group_activity()` – external RCRG checkpoint over 9-frame windows → `output_recent/group_activity/group_activity.json`.
5. `tactical_dashboard.render_dashboard()` – per-frame InsightEngine, EPV rendering, commentary, MP4.

## 4.2 High-level system architecture (implemented)

```
 INPUT                PERCEPTION                    STATE                       DECISION                    PRESENTATION
+-----------+    +------------------------+    +----------------------+    +---------------------+    +----------------------+
| broadcast |    | YOLO player + BoT-SORT |    | 49-D interaction     |    | physics EPV surface |    | 1920x1080 dashboard  |
| video     |--->| YOLO pose (17 kpts)    |--->| graph (11 players +  |--->| (1296 cells x 5     |--->| video + top-down     |
| (319 fr)  |    | YOLO ball + DP path    |    | 1 ball, 4-D edges)   |    | shots)              |    | court + heat map     |
+-----------+    | homography (manual     |    |                      |    |                     |    |                      |
                 | corners) + camera fit  |    | GAT -> HRN -> 32-D   |    | + HRN zone/shot     |    | commentary (LLM or   |
                 | court-metre smoothing  |    | global embedding     |    |   prior (skill-     |    | template) + insights |
                 +------------------------+    | 8-frame windows      |    |   scaled)           |    | JSON + snapshots     |
                          |                    | GRU / Transformer    |    | + GAT attention for |    |                      |
                          v                    | (untrained)          |    |   matchups          |    | Flask web app        |
                 ball_clean.csv,               +----------------------+    +---------------------+    +----------------------+
                 player_keypoints_*.json             |                            ^
                                                    v                            |
                                        pretrained group-activity GAT     touches / possession /
                                        (external RckG checkpoint)        phase state machine
```

## 4.3 Component-level architecture (files and interfaces)

```
master_main.py
 |-- pipeline/device_selection.py      auto|xpu|cuda|cpu resolution, ultralytics device string
 |-- pipeline/pipeline_profiling.py    timed spans, GPU sync, throughput report
 |-- pipeline/ball_tracker.py          BallTracker: YOLO ball candidates + nearest-previous selection
 |-- pipeline/redetect_ball.py         multi-scale (640,1280) candidate generation
 |-- pipeline/clean_ball_track.py      invokes analytics.ball_track.select_path -> ball_clean.csv
 |-- pipeline/integrate_and_rebuild.py
 |     |-- pipeline/sample_data.py          frame -> (players, ball, court) normalisation + cache
 |     |-- pipeline/graph_builder.py         frame -> PyG Data (x: [N,49], edge_attr: [E,4])
 |     |-- models/gat.py                     GATConv(49->32x4) -> ELU + residual -> GATConv(128->32)
 |     |-- models/hrn.py                     GAT -> per-team mean -> team proj -> global proj
 |     |-- temporal/sequence_generator.py    TemporalSequenceGenerator(window=8, stride=1)
 |     |-- temporal/gru_model.py             GRUTemporalModel / TransformerTemporalModel / TemporalTrainer
 |-- pipeline/group_activity.py         external RCRG-R2-C11-conc-temporal-V1 (ResNet50 + relational units)
 |-- pipeline/tactical_dashboard.py
       |-- analytics/insights.py          InsightEngine: possession, phase, blending, debouncing
       |-- analytics/epv.py               PhysicsEPV: flight tables, blockers, reach race, EPV surface
       |-- analytics/relations.py         key_matchups (leave-one-out), TeamShapeModel
       |-- analytics/camera.py            focal fit, (R,t), project/height_above
       |-- analytics/motion.py            MotionSmoother (court m, m/s)
       |-- analytics/pose_signals.py      JumpDetector (jump, reach, arms_up)
       |-- analytics/ball_track.py        DP path selection, touch detection
       |-- analytics/commentary.py        rule-based event planning + LLM/template rendering
       |-- analytics/nlp.py               Qwen2.5-3B-Instruct local backend (fp16 or NF4)
       |-- analytics/learned.py           LearnedModels: gat.pt/hrn.pt load, attention extraction
       |-- analytics/constants.py         provenance-tagged physical constants
```

## 4.4 Data-flow architecture (artifacts on disk)

```
data/sample3.mp4 --------------------------------+
data/annotations.json (319 x 4 corners) ---------+--> master_main stage 1
                                                       |
              +----------------------------------------+-----------------------------+
              v                                        v                             v
 data/player_keypoints_mf3_pose.json        data/ball.csv            docs/demo/master_detection_preview.mp4
 (frame -> {id -> 18 fields})                       |
              |                                     v
              |                       data/ball_candidates.json  (multi-scale)
              |                                     |
              |                                     v
              |                       data/ball_clean.csv  (+ Source column)
              v                                     v
        pipeline/sample_data.py  <------------------+
              |  (players, ball, court) per frame
              v
   output_recent/graphs/graph_frame_XXXX.pt     (PyG graph + embeddings)
   output_recent/dimensions/frame_XXXX.npz       (node/edge/team/global arrays)
   output_recent/temporal/hrn_global_sequences.npy / .pt / sequence_frame_ids.npy
   output_recent/temporal/temporal_predictions.csv   (only with a checkpoint)
              |
              v
   output_recent/dashboard/gritsense_dashboard.mp4
   output_recent/dashboard/dashboard_insights.json   (one row per frame)
   output_recent/dashboard/commentary.txt
   output_recent/group_activity/group_activity.json
```

## 4.5 Model-processing pipeline

```
 frame t (1920x1080)
   |  player track boxes (x1,y1,x2,y2,id,conf) ; pose keypoints (17,3) ; ball (x,y)
   v
 sample_data.create_sample_data(t)  -> normalised players/ball/court (image space, /1920, /1080)
   v
 VolleyballGraphBuilder.build_graph() -> x[N,49] ; edge_index[2,E] ; edge_attr[E,4] ; node_team[N]
   |   N = 12 (11 players + ball) ; E = 11*10 + 2*11 = 132 ; 49 = 2+2+2+2+34+1+1+5
   v
 GAT: 49 -> (32 x 4 heads) -> 128 + residual(49->128) -> ELU -> dropout -> GAT head-mean -> 32
   v
 HRN: node emb [12,32] -> player mask -> team0 mean, team1 mean -> [2,32] -> team_proj
      -> mean(teams) + mean(ball) -> /2 -> [1,32] -> global_proj   (global embedding)
   v
 temporal: 8 consecutive global embeddings (32-D) -> [8,32] window, stride 1
   v
 GRU (untrained in repo)                       Transformer (untrained in repo)
   v                                             v
 8-class group activity (l/r pass, set, spike, winpoint) -- requires a checkpoint that does not exist

 SIMULTANEOUS PHYSICS PATH (per frame, dashboard):
 players in court metres + velocities + jump heights  --> EPV surface (defender race, block, aim)
 touches --> possession --> phase (serve/pass/set/attack/in-flight) --> attack chance?
 HRN prediction (if checkpoint) --> zone prior (weighted by measured skill) + shot prior
   v
 target cell + shot + EPV + block/out risk + matchups + team shape --> dashboard + commentary
```

## 4.6 Documented versus implemented architecture

| Layer | Phase 2 / Phase 3 document | Implemented | Note |
|---|---|---|---|
| Object detection | YOLOv8 (Phase 2), YOLOv3 citation (deck) | Ultralytics YOLO, 3 custom checkpoints | Deck reference lists are inherited, not descriptive |
| Pose | BlazePose | YOLOv8-pose, 17 COCO keypoints | Substitution not documented anywhere |
| Court keypoints | "Court keypoint detection" | Hand-annotated 4 corners per frame | Automatic detection absent |
| Homography | Yes | Yes, per frame, 18x9 m | Plus camera recovery (not in Phase 2) |
| Speed | Yes | Court-metre velocity via regression; JSON `speed` field always 0.0 | Redundant/inconsistent field |
| Graph | Players + ball nodes, pose/position/ball features | Exactly that, 49-D, image-normalised | Phase 3 says "ball features"; ball gets 5 court-context features, no 3-D velocity |
| GAT | Yes | Yes, 2 layers, 4 heads, residual | Trained self-supervised |
| HRN | Yes | Yes, player/team/global | Trained by distillation from physics |
| Temporal features | Positions, poses, ball, spatial, HRN features | **Only the 32-D HRN global embedding** | `FeaturePreparer` exists but is never used |
| GRU | Yes | Implemented, untrained | No training entry point |
| Transformer | Baseline, compared to GRU | Implemented, untrained, no comparison | "Experimental, unexecuted" |
| Rally segmentation | Rally segmentation + state-action-outcome tuples | Phase state machine only; no rally dataset | EPV training tuples absent |
| EPV | Yes | Physics EPV, 5 shots, 1296 cells | Not learned; the Phase 3 deck implies learning |
| Action forecasting | Future group activity | 0.5 s per-player displacement only | No activity forecasting |
| Dashboard / web app | Not in Phase 2; implied by Phase 3 plan | Fully implemented | Largest single addition beyond documentation |

---

# 5. EXECUTION FLOW, TRACED THROUGH THE CODE

This chapter follows real calls, not intended ones. Numbers in brackets are the file that owns the step.

**5.1 Start.** `python master_main.py --video <path> [--dashboard] [--temporal-checkpoint ...]`. `parse_args` resolves a relative video path against the project root; unless `--keep-existing-output` is given, `_clear_generated_outputs()` deletes `output_recent/{graphs,dimensions,temporal,graph_images,graph_features}` so stale frames cannot mix with a new clip.

**5.2 Video load (`run_video_detections`).** `cv2.VideoCapture`; fps falls back to 30 if the container reports <= 1; width/height/frame-count read. `data/court.png` is resized to 300x150 as the mini-court overlay; destination corners are its four pixel corners, mapped to 18 x 9 m for display only.

**5.3 Per-frame loop.** For each frame:
- `annotations.json["<idx>"]` gives four corners; `cv2.findHomography(corners, court_pts)` yields **H** or `None`.
- `player_model.track(frame, persist=True, conf=0.4, tracker="botsort.yaml", half=<cuda>, quantize=16)`. Boxes, IDs and confidences are extracted; **all classes are kept** even though the checkpoint has a `ball` class; three track IDs (`{20, 63, 69}`) are hard-coded as exclusions.
- `pose_model.predict(frame, conf=0.10)` returns 17-keypoint poses; `_match_pose_to_track` associates them one-to-one with tracked players.
- For each track: `image_x = (x1+x2)/2`, `image_y = y2` (feet); `vx, vy` = per-frame image displacement since the previous sighting; foot point transformed by **H** to court pixels, clipped, converted to metres with `18 m / 299 px` and `9 m / 149 px`; team is assigned by court half (`court_x < 9 m` -> 0 else 1) with an image-midpoint fallback when **H** is unavailable.
- `ball_tracker.update(frame, idx)` returns at most one ball per frame (`BallTracker.select_candidate` scores `0.6*conf + 0.4*(1 - distance/300 px)` among candidates within 300 px of the previous ball, else the most confident). `None` is written as `Visibility=0`.
- Rendering: boxes, ID labels, pose dots, court outline, mini-court compositing; frame appended to `master_detection_preview.mp4`.

**5.4 Output of stage 1.** `data/player_keypoints_mf3_pose.json` is a dict `frame -> {track_id -> {id, team, team_label, image_x, image_y, vx, vy, confidence, bbox, court_x, court_y, speed, pose_available, pose_confidence, keypoints[17][2]}}`; `data/ball.csv` has `Frame, Visibility, X, Y, Radius, Confidence, Velocity_X, Velocity_Y`.

**5.5 Ball re-detection (`redetect_ball.main`).** `ball.pt` is run on batches of 8 frames at 640 px and 1280 px with `conf=0.05`; candidates within 12 px across scales are merged; every candidate is kept per frame in `ball_candidates.json`. `clean_ball_track.main()` then calls `analytics.ball_track.select_path`.

**5.6 Ball path selection (`analytics/ball_track.py`).** Three mechanisms act in order:
- *Static-object rejection*: a candidate whose pixel neighbourhood is detected in 8+ frames while no player is within reach (`touch_score < 1.0`) is marked static (referee's head, cursor, antenna).
- *Physical gate*: `max_step_px(camera)` converts the fastest published spike (mean + SD ≈ 31 m/s) at the camera's closest court distance into a per-frame pixel limit.
- *Dynamic programming*: nodes are all candidates; score of a node is `conf - 0.12`; transitions are allowed only within the gate, penalised by `0.02` per skipped frame and `0.2*(step/gate)^2`, with a `-0.6` restart penalty; the best-scoring chain is backtracked, reduced to one point per frame, then gaps up to 8 frames are filled by a quadratic-in-time fit (degree 1 if fewer than 4 neighbours). Output: `ball_clean.csv` with an extra `Source` column (`detected` / `interpolated`).

**5.7 Graph construction (`integrate_and_rebuild.build_graphs`).** `get_available_frames()` reads frame IDs from the ball CSV; for each frame `sample_data.create_sample_data(f)` normalises players (accepting dict or list layouts, several key aliases, bbox-derived positions, string or numeric team labels, 34-value flattened poses) and deep-copies the court with `width=1920`, `height=1080`, `net_x=960`; `VolleyballGraphBuilder.build_graph` produces the PyG graph and validates every invariant with asserts. A GAT and an HRN are instantiated (49 -> 32, 4 heads) and checkpoints are loaded if `checkpoints/gat.pt` / `hrn.pt` exist. Per frame, both models run, and the graph plus `gat_node_embeddings`, `hrn_node_embeddings`, `team_embeddings`, `global_embedding` and pose diagnostics are saved to `output_recent/graphs/graph_frame_XXXX.pt`.

**5.8 Temporal sequences.** `SEQUENCE_LENGTH = 8`, `SEQUENCE_STRIDE = 1`. The 32-D global embeddings are stacked chronologically and sliced into `[8, 32]` windows; saved as `hrn_global_sequences.npy`, `hrn_global_sequences.pt` and `sequence_frame_ids.npy`.

**5.9 Optional temporal inference (`run_temporal_predictions`).** With `--temporal-checkpoint`, the checkpoint is loaded (`weights_only=False`), `model_config.input_dim` is forced to the actual feature width, `GRUTemporalModel` or `TransformerTemporalModel` is built from an allow-listed config subset, `state_dict` prefix `module.` is stripped, and predictions are written to `temporal_predictions.csv` with per-class probabilities. **Without a checkpoint the function returns after printing that classification was skipped** – which is exactly what every real run in `webapp/jobs/` recorded.

**5.10 External group-activity inference (`group_activity.run`).** Up to 12 players per frame are cropped to 224x224, normalised with ImageNet statistics, resized and batched (8 frames at a time) through a frozen ResNet50 to 2048-D per player; 9-frame windows (stride 9) go through two relational-attention units and an LSTM to an 8-class softmax. Windows and the mean distribution are written to `output_recent/group_activity/group_activity.json`.

**5.11 Dashboard (`tactical_dashboard.render_dashboard`).** For each frame: `players_for_frame` projects feet through **H**, drops players outside the court + slack, keeps at most 6 per team by confidence, computes jump/reach/arms-up via `JumpDetector`, and smooths position/velocity in court metres; `BallState.update` projects the ball, decides the side by the net's image x, holds the last ball for 12 frames through dropouts; `detect_touches` supplies `(player, kind)` for the frame; `InsightEngine.step` returns a `FrameInsight`; the renderer draws the video card, the top-down court with the EPV heat map and target, the recommended play, matchups, team shape, phase badge, group-activity card and rally timeline, and writes frames to `gritsense_dashboard.mp4`. All per-frame insights are serialised to `dashboard_insights.json`, and timed commentary lines to `commentary.txt`.

**5.12 Where the run terminates in the current tree.** At step 5.7: `integrate_and_rebuild` resolves `ROOT` to `<repo>/pipeline`, so `player_keypoints_mf3_pose.json`, `ball.csv` and `checkpoints/` are looked for *inside* `pipeline/` and do not exist. Chapter 7 records the proof and the one-line fix, which is deliberately **not applied**.

---

# 6. COMPONENT-BY-COMPONENT IMPLEMENTATION

Each component is described with: what it is, the problem it solves, why GritSense needs it, how it is implemented, its files/functions, inputs and outputs, internal working, connections, and status/limitations. Volleyball examples are used where they clarify the mechanics.

---

## 6.1 Player detection and multi-object tracking

| | |
|---|---|
| **Files** | `master_main.py:run_video_detections`, `pipeline/ball_tracker.py` (ball only), `pipeline/device_selection.py` |
| **Input** | BGR frame (1920x1080) |
| **Output** | `[x1,y1,x2,y2]`, integer track id, detector confidence, per frame |
| **Status** | **Completed** (working), with fragility noted below |

**Concept and need.** Detection gives *where* bodies are; tracking gives *who* they are across time. Without persistent identities, velocity, possession, jump history and matchups are all undefined.

**Implementation.** `YOLO("weights/player_detection.pt")` is called with `.track(frame, persist=True, conf=0.4, tracker="botsort.yaml", half=<True on CUDA>, quantize=16)`. The checkpoint contains two classes (`ball`, `player`) and no keypoint head. Track IDs come from `boxes.id`; frames where BoT-SORT has not yet confirmed an identity simply contribute no boxes. At most 6 players per team are kept downstream (`MAX_PER_TEAM` in the dashboard), and three IDs (`{20, 63, 69}`) are excluded in `master_main` – a clip-specific hand-tune.

**Connection.** Feeds the graph (node position/confidence), the analyst (court position, velocity, jump), the group-activity stage (12 crops per frame) and the renderer.

**Limitations.** (i) **No class filtering**: boxes from the `ball` class are accepted as players. (ii) IDs are not shirt numbers, so a fan or coach sees "player 76". (iii) `excluded_ids` is hard-coded for this clip. (iv) No re-identification across camera cuts; the broadcast clip contains cuts, which is one source of ID switches and the reason the GAT training masks out displacements above 4 m per 0.5 s. (v) Detection runs at the video's native resolution; a 4K source is not downscaled before inference.

---

## 6.2 Pose estimation and pose-to-track association

| | |
|---|---|
| **Files** | `master_main.py:_match_pose_to_track`, `analytics/pose_signals.py` |
| **Input** | Frame; tracked player boxes |
| **Output** | 17 keypoints `[x, y]` per player, mean keypoint confidence, `pose_available` flag |
| **Status** | **Completed**; achieved coverage is low (see below) |

**Concept and need.** Pose supplies limb configuration, which is what distinguishes a jump-set from a jump-block when both bodies are airborne. GritSense uses it in three ways: as 34 of the 49 node features, for the `arms_up` signal, and for jump/reach measurements through the camera.

**Implementation.** `pose_model.predict(frame, conf=0.10)` (YOLOv8-pose, `kpt_shape=[17,3]`). Because the player detector and the pose model produce different boxes, association uses the composite score described in §3.1 with a 0.18 acceptance threshold and a one-to-one `used_pose` set. The 17 (x, y) pairs are flattened to 34 values; `sample_data` requires exactly 34 to set `pose_available = True`, so a partial skeleton is treated as *no* skeleton. Keypoints with zero confidence are stored as (0,0) rather than omitted, which is why `normalize_pose` clips rather than rejects them.

**Measured coverage.** In the recorded run log: `Players: 11, Pose: 5/11 (45.5%)`, `Pose: 3/11 (27.3%)`, with `Non-zero pose values: 102/374` on frame 273. Roughly 85% of the 374 pose numbers for that frame are zero, meaning most players contribute an empty pose block. The graph still functions (pose availability is a feature), but pose contributes little signal at this resolution and camera distance.

**Limitations.** Small players (far side of the court, ~40 px tall) are frequently missed at `conf=0.10`; there is no temporal pose smoothing or gap filling; pose is not used to improve tracking, only to describe it.

---

## 6.3 Court homography and camera recovery

| | |
|---|---|
| **Files** | `pipeline/tactical_dashboard.py:homography_for_frame/project`, `analytics/camera.py` |
| **Input** | `annotations.json` (4 image corners per frame) |
| **Output** | 3x3 **H** (image -> court metres); camera intrinsics/extrinsics; `project()`, `height_above()`, `ground_point()` |
| **Status** | Homography **completed** (manual input); automatic court detection **not implemented**; camera recovery **completed and validated** |

**Concept and need.** Everything metric depends on this: metres, m/s, jump height, touch height, blocker geometry. A floor homography is exact only for points on the floor, which is why feet are used.

**Homography.** `findHomography(corners, [[0,0],[18,0],[18,9],[0,9]])` per frame; if a frame is unannotated, the previous **H** is reused (dashboard) or the frame has no court mapping (detection preview). `project()` applies `cv2.perspectiveTransform`.

**Camera recovery.** With square pixels and the principal point at the image centre, the floor homography determines the camera up to focal length. `Camera._fit_focal` grid-searches focal length in 800–6000 px (step 10) minimising, across all annotated frames, the orthogonality and equal-length residuals of the first two rotation columns after `K^-1 H`. The fitted value is **2500 px**. Per frame, `pose()` recovers (R, t) by normalising the columns, fixing the sign so the court is in front, and orthonormalising via SVD. `height_above()` uses the image rows per metre at a given ground point to convert a pixel row into a height in metres – a single-view measurement that is valid because the ground point is known.

**Validation (author-reported, reproducible from `calibrate_physics.py`).** Camera centre median (8.62, 33.25, 10.06) m with standard deviation (0.95, 1.29, 0.71) m over 319 panning frames; 2,636 standing-height samples with median 1.92 m; 81 detected jumps with 90th-percentile near-net peak 0.70 m and maximum 1.00 m; maximum near-net reach 3.19 m against a published block reach of 3.18 m.

**Limitations.** (i) Corners are hand-made for one video; a new match needs re-annotation, and the web app explicitly reuses the sample's annotations (`server.py`: "the web app never writes this file"). (ii) `docs/RUN_GUIDE.md` tells the user to "click the four court corners on a frame", but no such endpoint exists in `webapp/server.py`; a video with different geometry therefore inherits the wrong homography and all metric outputs become meaningless, silently. (iii) A single global focal length assumes no zoom; broadcast zooms would violate it.

---

## 6.4 Ball detection, re-detection and path cleaning

| | |
|---|---|
| **Files** | `pipeline/ball_tracker.py`, `pipeline/redetect_ball.py`, `pipeline/clean_ball_track.py`, `analytics/ball_track.py` |
| **Input** | Frames; `weights/ball.pt` |
| **Output** | `ball_candidates.json`, `ball_clean.csv` (Frame, Visibility, X, Y, Radius, Confidence, Velocity_X, Velocity_Y, Source) |
| **Status** | **Completed**, with a measured improvement over the naive track |

**Problem.** At 640 px input a volleyball is a few pixels wide; the single-detection-per-frame tracker locked onto the first referee's bald head, the mouse cursor of the screen recording, and the net antenna (author's manual check, `docs/SYSTEM_EXPLAINED.md` §C).

**Implementation, three passes.**
1. `BallTracker` (used inside detection) – YOLO at default size, `conf=0.15`, candidate selection by `0.6*conf + 0.4*proximity` within 300 px, else the most confident; velocity from consecutive positions; never invents a position.
2. `redetect_ball` – two image sizes (640, 1280) at `conf=0.05`, batched 8 frames, cross-scale merge within 12 px, all candidates retained. *(Docstring and RUN_GUIDE say three sizes including 1920; `SIZES = (640, 1280)` – documentation drift.)*
3. `analytics.ball_track.select_path` – static rejection, then DP over candidates with a physically derived per-frame pixel gate, then quadratic gap filling (§5.6).

**Reported outcome.** 259 real detections plus 58 filled frames = 317 of 319 frames with a ball, versus 229 before, "many of them false" (author-reported). The system therefore produced **317 graphs** rather than 229, which directly raises the amount of training data for the GAT.

**Limitations.** Ball 3-D position is not recovered: the ball is mapped through the floor homography, which is wrong for a ball in the air, so ball speed is deliberately **not** claimed (calibration notes: "too few clean ball detections in one clip"). The `Radius` column is a detection artefact, not a metric size. No spin, no trajectory model beyond short-span quadratic fits.

---

## 6.5 Touch detection, possession and rally phase

| | |
|---|---|
| **Files** | `analytics/ball_track.py:detect_touches`, `analytics/insights.py:InsightEngine.step` |
| **Input** | Cleaned ball track; player boxes; `Camera` |
| **Output** | `{frame: (player_id, kind)}` with `kind in {pass, set, attack, hold}`; possession; phase |
| **Status** | **Completed**; author-reported precision 7/10 |

**Concept.** Possession, the hitter, the rally phase and the "when to show advice" logic all require knowing *who touched the ball and how*. Volleyball gives a physical signature: a touch is a sharp bend in the trajectory, and its type is decided by the ball's height and where it goes next.

**Implementation.** For each frame with enough ball points, incoming and outgoing image velocities are computed over 3-frame windows; their difference is the "kink" magnitude, and frames below 20 px/frame are rejected. The strongest bend in a ±3-frame neighbourhood is kept. A player is accepted as the toucher if `touch_score(ball, bbox) < 1.5` **and**, through the camera, the ball is between −0.2 m and 3.67 m (highest published reach + 0.3 m margin) above that player and within 1.5 m sideways. Classification: ball sent upward from above 2.2 m -> `set`; upward from below -> `pass`; downward/across from above 2.2 m -> `attack`, unless the ball does not cross the net within 1.5 s (in which case `pass`). A ball that stays within 3 px/frame for 5 frames next to a player is `hold` (the pre-serve case).

**Consumers.** `InsightEngine` turns `(id, kind)` into: possession change (immediate for a touch, debounced over 8 frames for a side change), a three-touch counter, the hitter, the phase label, and `attack_chance` – which gates whether a target and heat map are drawn at all. During passes and digs the dashboard shows "PASS · SET TO #52" and no target.

**Example from the data.** Ten touches at frames 34, 50, 71, 130, 176, 213, 255, 273, 296, 308; heights 3.05, 2.09, 2.38, 0.86, 3.19, 1.82, 2.59, 1.38, 3.64, 1.29 m. Sets at 2.38/2.59 m match a 2.48 m standing reach; the 3.64 m "attack" is 0.27 m above the published 3.37 m reach and is one of the three false positives the author identified (frame 296), because a ball high over the far side lines up with a nearer player in the image. The recorded run detected 11 touches, i.e. one more than the calibration snapshot.

**Limitations.** Classification looks ~0.1 s ahead (needs future frames) – acceptable offline, a delay if live. False touches near the net are systematic in single-view geometry. There is no notion of a block touch or a net touch.

---

## 6.6 Motion, speed and jump signals

| | |
|---|---|
| **Files** | `analytics/motion.py`, `analytics/pose_signals.py`, `pipeline/tactical_dashboard.py:players_for_frame` |
| **Input** | Court-metre positions; boxes; keypoints |
| **Output** | Smoothed (x, y, vx, vy) in m and m/s; `jump_m`, `reach_m`, `arms_up`, take-off point |
| **Status** | **Completed** |

**Motion.** Per player: history window of 6 frames, position = mean of last 3, velocity = slope of a least-squares line over 0.2 s, history reset after a 3-frame absence, speed capped at the measured top speed (4.75 m/s). `players_for_frame` uses `MotionSmoother` when available and otherwise derives speed from a 10-frame image-velocity lookahead projected to court metres, capped at the same limit.

**Jump detection.** Within a 10-frame lookback (~0.33 s), if the feet rise and the head rises by at least half as much while the box height changes by less than ±10%, the player is judged airborne; the reference frame is the one with the lowest feet. `Camera.height_above` converts the pixel lift into metres. Walking away from the camera lifts feet in the image but shrinks the box, so it is rejected. While airborne, the player's court position is replaced by the take-off point, because airborne feet project too far from the camera.

**Why it matters.** Jump height feeds the EPV blocker model (hands reach `standing_block_reach + jump`), and reach feeds touch-height plausibility. This is the mechanism that lets a single fixed camera measure heights at all.

**Limitations.** A jump is inferred, not labelled; the ±10% size gate will fail for players moving toward the camera while jumping. `arms_up` is a single geometric test on nose-versus-wrist rows.

---

## 6.7 Interaction graph construction

| | |
|---|---|
| **Files** | `pipeline/graph_builder.py` (1,277 lines), `pipeline/sample_data.py` (1,130 lines) |
| **Input** | Normalised players, ball, court dict |
| **Output** | `torch_geometric.data.Data` with `x [N,49]`, `edge_index [2,E]`, `edge_attr [E,4]`, `edge_type [E]`, `node_type [N]`, `node_team [N]`, `player_ids`, `ball_index` |
| **Status** | **Completed**, with five documented feature blocks |

**Node features (49).** Indices 0–1 node type one-hot (player / ball); 2–3 normalised position; 4–5 normalised velocity; 6–7 team one-hot (unknown -> `[0,0]`); 8–41 17 keypoints x 2, normalised; 42 pose availability; 43 detector confidence; 44–48 court context (distance from left, right, top, bottom and net, all normalised). The ball node reuses the same layout with team and pose zeros and `pose_available = 0`.

**Edges.** Directed player->player for all ordered pairs (N(N−1) = 110 for 11 players), plus player->ball and ball->player (22), total **132** for a 12-node graph. Edge features: relative x, relative y, distance, same-team flag. Ball edges carry `same_team = 0` and reversed relative offsets.

**Validation.** `validate_frame_inputs` and `validate_graph` assert shapes, finiteness, team values in {−1, 0, 1}, node types in {0, 1}, edge types in {0, 1} and edge-index bounds. This is the most defensive module in the project and the reason malformed frames fail loudly rather than silently poisoning training.

**Important detail.** The graph works in **normalised image space** (x/1920, y/1080), not court metres, and its "distance from net" feature uses `net_x = 960 px`, not the true net line. The GAT's targets, however, are court-metre displacements. So the model must learn perspective implicitly. This is a genuine architectural weakness and the single highest-value graph improvement (Ch. 12).

---

## 6.8 Graph Attention Network (project's own)

| | |
|---|---|
| **Files** | `models/gat.py` (206 lines), `analytics/learned.py` |
| **Input** | `x [N,49]`, `edge_index`, `edge_attr [E,4]` |
| **Output** | `[N,32]` node embeddings; optionally `(edge_index, attention [E,4])` |
| **Status** | **Completed and trained**; predictive quality worse than a trivial baseline |

**Architecture.** `GATConv(49 -> 32, heads=4, concat=True, edge_dim=4)` -> ELU -> `+ Linear(49 -> 128)` residual -> dropout 0.2 -> `GATConv(128 -> 32, heads=4, concat=False, edge_dim=4)`. Two layers means information reaches 2-hop neighbours: a player sees teammates, opponents and the ball, and through them the ball's neighbours. 32-D output is deliberate: it matches the temporal input width and keeps the checkpoint small (~134 kB).

**Training signal (self-supervised).** No manual labels. For every frame the target is each player's court displacement 15 frames (0.5 s) later, obtained by projecting the current and future feet through the homography; targets are averaged over a 5-frame window because box bottoms jitter, and displacements above 4 m are masked out as ID switches. Loss: Smooth-L1 on masked players; Adam, lr 1e-3, weight decay 1e-4, batch 16, 150 epochs, best epoch chosen on validation.

**Result (test split, 63 frames, 487 players).** GAT ADE **0.873 m**; constant-velocity ADE 1.557 m; "no movement" ADE **0.765 m**. The model beats constant velocity by 44% but loses to assuming nobody moves, i.e. it over-predicts motion. Best epoch 99 of 150; training took 25 s.

**Attention as an output.** `return_attention=True` exposes layer-2 attention, averaged over heads, which `LearnedModels.attention_to(result, attacker, defenders)` turns into per-defender weights. This is the only place where a learned relation is surfaced to the user, and it is blended at 30% weight into "key matchups".

---

## 6.9 Hierarchical Relational Network

| | |
|---|---|
| **Files** | `models/hrn.py` (357 lines), `analytics/learned.py:HRNHeads` |
| **Input** | Same graph as the GAT, plus `node_type` and `node_team` |
| **Output** | dict: `node_embeddings [N,32]`, `team_embeddings [2,32]`, `global_embedding [1,32]` |
| **Status** | **Completed and trained**; skill on held-out data is zero |

**Implementation.** Level 1 is the GAT. Level 2 masks players, splits by `node_team`, **mean-pools** each team, stacks the two team vectors and applies `team_projection` (Linear 32->32, ELU, dropout). Level 3 averages the two team vectors, averages the ball embedding, averages those two and applies `global_projection`. Guards: `node_team` is mandatory; if either team is empty the module raises `ValueError` ("No players found for team 0/1"), which callers must handle (the learned-inference path checks team counts first; the graph stage catches per-frame exceptions and counts the frame as skipped).

**Training signal (distillation).** No manual labels either. The teacher is the *physics* pipeline: for each frame the `InsightEngine` computes the smoothed target zone (1–6), the best shot, the attack threat and both teams' coverage/balance; the HRN is trained to reproduce them with cross-entropy on zone and shot, plus weighted MSE (x10) on threat and team shape. It initialises its GAT from the trained GAT's weights, then trains 100 epochs at lr 5e-4 (188 s).

**Result (test split, 63 frames).** Zone accuracy **46.0%** vs majority-class baseline **49.2%**; shot accuracy **92.1%** vs baseline **92.1%**; threat MAE 0.303; team-shape MAE 0.066. Best epoch **1** – it overfits immediately and never improves.

**The guard that makes this honest.** `analytics/learned.py:_skill(accuracy, baseline)` maps performance to [0,1] above baseline. On these numbers both skills are **0.0**, so `InsightEngine` multiplies the HRN's zone and shot priors by zero and the recommendation stays pure physics. The run log prints this explicitly: "HRN target influence scaled by test skill: zone 0.00, shot 0.00". As more data improves the HRN, its influence rises automatically – a genuinely well-designed piece of engineering.

---

## 6.10 Temporal features, GRU and Transformer

| | |
|---|---|
| **Files** | `temporal/sequence_generator.py`, `temporal/feature_preparation.py`, `temporal/gru_model.py`, `pipeline/integrate_and_rebuild.py` |
| **Input** | 32-D global embedding per frame |
| **Output** | `[8,32]` windows; `temporal_predictions.csv` (only with a checkpoint) |
| **Status** | Windowing **completed**; GRU/Transformer **implemented but never trained**; feature preparation **unused** |

**Windowing.** `TemporalSequenceGenerator(sequence_length=8, stride=1)` supports grouped input (multiple videos/rallies) and validates dimensions, but the production path uses a hand-rolled loop in `integrate_and_rebuild.build_temporal_sequences` with the module imported only for its interface. Windows are `[8, 32]`, saved with their frame IDs.

**Feature set.** Only the HRN global embedding. The Phase 3 deck promises positions, poses, ball and spatial features per timestep; `FeaturePreparer` (which concatenates named blocks `player|pose|ball|spatial` in a fixed order, with `FeatureStandardizer` fitted on training data only) implements exactly that contract and is **never called**. This is the clearest gap between the documented and implemented temporal pipeline.

**GRU.** `nn.GRU(input_dim, hidden=128, layers=1, batch_first=True)`, optional bidirectional, classifier `Linear(128, 8)` on the final hidden state; `predict()` returns indices, labels, probabilities and confidence. **Transformer.** `Linear(input,128)`, learned-positional-style encoding built in `_build_positions` (note: it degenerates to a single-column encoding of `t/10000^0`, effectively linear time, not a full sinusoidal table), 2 `TransformerEncoderLayer`s (4 heads, feed-forward 256, ReLU), mean-pool, `Linear(128,8)`. `TemporalTrainer` provides fit/evaluate with accuracy and macro-F1.

**What is missing.** No script constructs a labelled dataset, no script calls `TemporalTrainer`, no checkpoint exists anywhere in the repo, and `run_temporal_predictions` returns immediately when `--temporal-checkpoint` is absent. Every recorded run printed: "Temporal sequences were generated; activity classification was skipped because no trained temporal checkpoint was supplied." The GRU-versus-Transformer comparison in the Phase 3 deck is therefore **not implemented as an experiment**.

---

## 6.11 External pretrained group-activity GAT

| | |
|---|---|
| **Files** | `pipeline/group_activity.py` (280 lines), `pretrained_checkpoints/RCRG-R2-C11-conc-temporal-V1/checkpoint_epoch_42.pkl` (693 MB) |
| **Input** | Video frames; player boxes per frame; up to 12 players |
| **Output** | `group_activity.json`: 9-frame windows with label, score, per-class probabilities, plus an overall mean |
| **Status** | **Completed earlier; broken in the current tree** |

**What it is.** A reimplementation of the *inference path only* of the third-party RCRG model family shipped in the deleted `GAT/` folder: frozen ResNet50 (2048-D per player crop), two relational-attention units over a fully connected 12-player graph (`edge_index` = all ordered pairs), an LSTM over the time axis, and a `Linear(12*512, 256)` head to 8 classes. The checkpoint is unpickled with a custom `Unpickler` that substitutes a stub for the training `Config` class, so heavy training dependencies are not needed.

**Output shape.** Classes are the Volleyball-dataset group activities as stored in the checkpoint: `r_set, r_spike, r-pass, r_winpoint, l_winpoint, l-pass, l-spike, l_set` (8). Windows: 36 for 319 frames at stride 9.

**Observed result (recorded job).** Overall distribution `l-pass 0.1973`, `r-pass 0.1707`, `l_set 0.1352`, `l_spike 0.1240`, `l_winpoint 0.1181`, `r_set 0.0889`, `r_winpoint 0.0887`, `r_spike 0.0771` – i.e. a nearly uniform distribution with a weak bias, and per-window labels that flip between `r_winpoint` and `l_winpoint`. This is what a domain-mismatched model looks like: it was trained on the Volleyball dataset's tidy 1080p clips with 12 annotated players, and is being applied to a broadcast excerpt with trackers, bench players and camera cuts.

**Current defect.** `group_activity.py` resolves `GAT_DIR = <own dir>/GAT` and `DEFAULT_CHECKPOINT = <own dir>/pretrained_checkpoints/...`. The module now lives in `pipeline/`, so both paths are wrong (`pipeline/GAT` and `pipeline/pretrained_checkpoints` do not exist), and the `GAT/` source folder referenced for `relational_attention.py` was deleted from the working tree. `master_main.run_pretrained_group_activity` catches the error **only on CPU**; on CUDA/XPU it re-raises, so a GPU run aborts at this stage.

---

## 6.12 Physics Expected Possession Value

| | |
|---|---|
| **Files** | `analytics/epv.py` (542 lines), `analytics/constants.py` (121 lines) |
| **Input** | Contact point (attacker court position), defender list with velocity and jump height, defending side |
| **Output** | `PhysicsSurface`: 36x36 grid of best-shot EPV, per-cell shot index, and layers (`flight_t`, `defender_t`, `p_in`, `p_block`, `p_unreached`, `physics_epv`, `net_clearance`) |
| **Status** | **Completed** – the strongest technical component |

**Formulation.** For every 0.25 m cell of the defending half (1,296 cells) and every legally available shot:

```
EPV = P(ball lands in) x (1 - P(blocked)) x P(no defender reaches it)
```

**Flight model.** `simulate(speed, contact_h)` integrates every launch angle from −70° to +75° in 0.25° steps with `dt = 2 ms` under gravity and quadratic drag, with drag coefficient switched between 0.47 (subcritical) and 0.17 (supercritical) at Re = 3e5; results (landing distance, flight time, height on a 5 cm path grid) are cached per (speed, contact height). For each shot the flight tables are built for the published speed range (mean ± SD) on the low branch (and additionally the high branch for tips/roll shots, which must clear the block). For every target cell the **fastest feasible** trajectory that lands there and clears the net (2.43 m + ball radius; 3.18 m + radius for tips) gives the flight time and net crossing height.

**Block model.** Attacks are evaluated at the moment of the hit: the three defenders nearest the net (within the 3 m attack line) step to the net, shuffle toward the hitter at the measured top speed after the reaction time, and are at the full published block reach (3.18 m). A shot is blocked with probability `Φ((hand_height - (z_net - r)) / 0.10)` x a Gaussian lateral overlap `exp(-(y_net - y_blocker)^2 / (2*0.5^2))`, combined multiplicatively across blockers. Before this model, judging blockers as they stood gave only a 3% block chance and advice that was always "deep middle, Zone 6" in 80% of frames.

**Defender race ("pitch control").** For each defender, reach time = reaction (0.33 s, derived as a 0.571 s measured reaction minus the 240 ms anticipation lead) + free-fall time if airborne + running time at 4.75 m/s to within the dive reach (0.96 m = half the arm span). `P(reach_i) = sigmoid((T_ball - t_i)/0.109)`, and `P(no defender) = Π(1 - P(reach_i))`.

**Aim error.** `P(lands in)` is a product of normal CDFs on distance to the sidelines and far line, with a shot-specific aim SD *fitted* so that serves miss at the published rates (20.9% jump serves, 10.8% floats), giving 1.66 m and 0.85 m; spike aim error is set equal to the jump serve's.

**Shots.** SPIKE, TIP/ROLL SHOT, BACK-ROW ATTACK (front/back based on distance to the net relative to the 3 m line) and JUMP SERVE, FLOAT SERVE (only from the baseline, with a 1 m tolerance because tracked feet drift).

**Constant provenance.** `analytics/constants.py` tags every constant MEASURED / PUBLISHED / DERIVED / ESTIMATED with a reference key, and `analytics/physics_calibration.json` overrides the measured ones (top speed 4.75, stature 1.92, block jump 0.70, set time 0.60). Four values remain explicitly ESTIMATED (tip speed cap, blocker hand half-width, hand-height spread, spike aim ratio).

**Validation without fitting.** On 247 frames with a spike available, the best-placed attack averages **0.883** point chance and an attack aimed anywhere legal averages **0.3859**, bracketing the published terminal-attack success rates (72.5% outside hitters, 55.1% opposites).

**Performance design.** Geometry, flight tables and `P(in)` do not depend on defenders, so `_geometry` caches them (LRU 64) and is shared by the display surface, the teammate-option search and the leave-one-out matchup surfaces. Without this, key matchups would recompute the full surface once per defender.

**Limitations.** The ball is treated as a point launched from the contact point with no spin; the block is a static hand plane; defenders are assumed to run straight lines; the model is one-step (no second-touch consequence); the whole thing is unvalidated beyond the aggregate bracket above.

---

## 6.13 Insight engine: blending physics with learned models

| | |
|---|---|
| **Files** | `analytics/insights.py` (528 lines), `analytics/relations.py` |
| **Input** | Per-frame players/ball/toucher; optional `LearnedModels` |
| **Output** | `FrameInsight` with target, zone, EPV, block/out risk, move, matchups, team shape, phase, sources |
| **Status** | **Completed** |

**State machine.** Possession from touches (fast path) or debounced ball side (8 frames); three-touch counter; hitter = toucher, else the planned hitter for up to 45 frames, else the last toucher for 30 frames, else the attacker nearest the ball; phase = serve/pass/set/attack/in-flight; `attack_chance = phase in (serve, set, attack)`.

**Advice logic.** On a pass or set the recommendation is *who to set*: every teammate's contact point is evaluated (peak EPV only, no display layers) and the best becomes the target; on the attack the best shot/target is shown. Blockers are judged "at the hit": 0 s away on the third touch, otherwise the measured 0.6 s set time.

**Blending.** Target: physics surface re-ranked by the HRN's shot prior (`1 - w + w*n*p`) and zone prior, with `w = 0.3 * skill` per head; rescale to the physics peak. Matchups: `(1-w)*physics threat-stopped share + w*GAT attention share`. Team shape: `(1-w)*physics + w*HRN`.

**Stability.** EMA smoothing on target (α=0.18), matchups and displayed statistics; 10-frame debounce on the move (immediate if the hitter changes); target snapped to the best reachable cell within 1 m so smoothing cannot drift onto an unreachable cell.

**Example (frame 2 of the recorded run).** attack_team 0, target (16.375, 5.375) m, zone 6, EPV 0.622, block 0.160, out-risk 0.176, `P(in)` 0.824, flight 0.412 s vs nearest defender 0.678 s, move "BACK-ROW ATTACK", key player 3, top matchup defender 1 with 0.405 impact share, team shape coverage 0.156/0.109. These are reproducible, physically interpretable numbers, not mock data.

---

## 6.14 Dashboard rendering

| | |
|---|---|
| **Files** | `pipeline/tactical_dashboard.py` (1,083 lines) |
| **Input** | Video; player JSON; ball CSV; annotations; checkpoints; group-activity JSON |
| **Output** | `gritsense_dashboard.mp4` (1920x1080), `dashboard_insights.json`, `commentary.txt`, `dashboard_XXXX.png` every 80 frames |
| **Status** | **Completed** |

**Layout.** A fixed 1920x1080 composition: video card (with boxes, IDs, projected heat map and target, group-activity badge, rally timeline), top-down court card (players, ball, heat map, zone grid, target marker), recommended-play card (move, target zone, point/block/out chances, physics numbers), text card (commentary lines with source tags), signal card (matchups, team shape, phase). Fonts fall back from Segoe UI to Arial to DejaVu across Windows/macOS/Linux.

**Heat projection.** The EPV grid is rendered to a 60 px/m image, colour-mapped through a 4-stop LUT, then inverse-homographed onto the broadcast frame so the "where to play" region sits on the real court surface – a genuinely useful visualisation choice.

**Outputs are also an API.** `dashboard_insights.json` is a list with one object per frame (all `FrameInsight.to_dict()` fields), which the web app serves for charts and timelines.

**Limitations.** CPU-bound Pillow drawing per frame; no GPU compositing; the preview MP4 (`mp4v`) is not browser-playable, which is why the web app transcodes with ffmpeg (NVENC when available).

---

## 6.15 Commentary generation

| | |
|---|---|
| **Files** | `analytics/commentary.py` (282 lines), `analytics/nlp.py` (185 lines) |
| **Input** | `FrameInsight` sequence |
| **Output** | Timed commentary lines with `kind` and `source` (`LOCAL LLM` / `TEMPLATE`) |
| **Status** | **Completed** |

**Rules decide when to speak** (possession change, new recommended move during an attack chance, periodic insight); the model only writes the sentence from computed facts. The default backend is `Qwen/Qwen2.5-3B-Instruct` running offline from the Hugging Face cache (fp16, or NF4 4-bit on CUDA), with strict validation: a generated line must mention the real zone and player number, otherwise the template is used and tagged `TEMPLATE`. Two timing modes exist: `background` (worker thread during analysis) and `serialized` (`plan(defer=True)`, generation only in `finish()`), and the test suite verifies that both produce identical lines.

**Observed output.** Six lines in the recorded run, all tagged `LOCAL LLM`, e.g. "#3's back-row is quick and tricky - hit it to the back-middle seam, Zone 6, exploiting JPN's slow reaction time to #18!". Note the plausible-but-unverified claims ("slow reaction time", "on fire tonight") – the guard checks zone and player, not semantics.

---

## 6.16 Web application

| | |
|---|---|
| **Files** | `webapp/server.py` (495 lines), `webapp/static/{index.html,app.js,app.css}` |
| **Input** | Uploaded video or the bundled sample; team names; device choice |
| **Output** | Job record, live log, insights API, downloadable MP4/JSON/commentary |
| **Status** | **Completed**; blocked by the same path defect |

**Design.** A single-worker queue (`queue.Queue`, one thread) because "the pipeline writes its detection files into the project root"; each job is a directory under `webapp/jobs/<id>/` with `job.json` state, the input, a browser-ready `video.mp4`, the dashboard MP4, insights and commentary. Jobs are launched as a **subprocess** (`VENV_PY`, `cwd` project root) and can be cancelled (`proc.kill()`). Uploads up to 4 GB; non-1080p video is scaled and padded to 1920x1080 because the camera model and court corners assume 1080p; ffmpeg comes from `imageio-ffmpeg`, with NVENC attempted first. State survives restart, with running jobs marked failed. Endpoints: `/api/system`, `/api/jobs` (list/create/delete/clear), `/api/jobs/<id>` (+cancel, +insights), `/files/<job>/<name>`, `/sample.mp4`.

**Evidence.** Six completed jobs (1 demo, 5 full) on 4 November 2026, all `device: cuda`, all `status: done`, each producing dashboard MP4, insights JSON and commentary.

**Limitations.** Flask development server, no authentication, no upload allow-list beyond extension, single-job concurrency, all runs share one set of detection files in the project root (hence the queue), and the sample's court corners are reused for every uploaded video.

---

# 7. IMPLEMENTATION AUDIT

Audit method: every row was checked against source code, a checkpoint, a log, or a live probe run on 13 November 2026. "Status" uses the vocabulary Completed / Partially implemented / Experimental / Planned / Not implemented / Unable to verify.

## 7.1 Functional and architectural requirements

| Feature | Planned functionality | Actual implementation | Relevant files | Status | Remarks |
|---|---|---|---|---|---|
| Video input | Read match video | `cv2.VideoCapture`, fps fallback 30, resolution read | `master_main.py` | Completed | `mp4v` preview is not browser-playable; web app transcodes |
| Player detection | YOLO detection of players | Ultralytics YOLO `player_detection.pt` (classes ball, player), conf 0.4 | `master_main.py` | Completed | **No class filter** – ball-class boxes can enter as players |
| Multi-object tracking | Persistent player IDs | BoT-SORT via `model.track(persist=True)` | `master_main.py` | Completed | IDs not shirt numbers; cuts cause switches; 3 IDs hard-excluded |
| Ball detection | Dedicated ball detector | `ball.pt` in-loop + multi-scale re-detection (640, 1280) | `pipeline/ball_tracker.py`, `redetect_ball.py` | Completed | Docstring/RUN_GUIDE claim three sizes incl. 1920; code uses two |
| Ball path cleaning | Reject false positives, fill gaps | Static-object rejection + speed gate + DP path + quadratic fill | `analytics/ball_track.py` | Completed | 317/319 frames with ball (author-reported) |
| Pose estimation | BlazePose | YOLOv8-pose, 17 COCO keypoints | `master_main.py`, `weights/pose.pt` | Completed (substituted) | BlazePose not used anywhere; 27–46% pose coverage observed |
| Pose-to-track association | Match poses to players | Composite IoU/coverage/proximity score, threshold 0.18 | `master_main.py` | Completed | Better than IoU-only for tight pose boxes |
| Court keypoint detection | Automatic detection of court keypoints | **None** – four corners hand-annotated for 319 frames | `data/annotations.json` | Not implemented | Documented in Phase 2 §5.2.3 as a pipeline stage |
| Homography | Image -> court metres | `findHomography` per frame to 18x9 m | `tactical_dashboard.py` | Completed | Manual corners; reused for every uploaded video |
| Camera recovery | Not planned in Phase 2 | Global focal fit (2500 px) + per-frame (R,t) | `analytics/camera.py` | Completed | Validated: centre std ≤ 1.29 m, stature median 1.92 m |
| Speed calculation | Player speed | Least-squares velocity over 0.2 s in m/s; JSON `speed` field always 0.0 | `analytics/motion.py`, `master_main.py` | Partially implemented | Metric velocity is computed downstream, not exported; field is dead |
| Jump / reach measurement | Not planned | Box-lift + camera height measurement | `analytics/pose_signals.py` | Completed | 81 jumps, 90th pct 0.70 m, max reach 3.19 m |
| Graph construction | Players + ball nodes, pose/position/ball features | 49-D nodes, player-player/player-ball edges, 4-D edge features | `pipeline/graph_builder.py` | Completed | Works in **image space**, not court metres |
| GAT | Learn relational importance | 2-layer GAT, 4 heads, residual, edge_dim=4 | `models/gat.py` | Completed | Trained self-supervised; loses to no-movement baseline |
| HRN | Player/team/scene hierarchy | GAT + team mean-pool + global projection + heads | `models/hrn.py`, `analytics/learned.py` | Completed | Trained by distillation; zero measured skill |
| GAT/HRN training | Train without labels | Self-supervised + distillation, block split, reports | `pipeline/train_gat_hrn.py` | Completed | 236 s total on one clip; overfits instantly |
| Temporal feature extraction | Positions, poses, ball, spatial, HRN features | **Only 32-D HRN global embedding** | `integrate_and_rebuild.py` | Partially implemented | `FeaturePreparer`/`FeatureStandardizer` exist, never called |
| GRU model | Temporal classifier | `GRUTemporalModel` + `TemporalTrainer` | `temporal/gru_model.py` | Experimental | No training script, no checkpoint, never produced output |
| Transformer baseline | Compare with GRU | `TransformerTemporalModel` | `temporal/gru_model.py` | Experimental | Positional encoding degenerates to linear time; no comparison run |
| Rally segmentation | Segment rallies, build state-action-outcome tuples | Phase state machine only (serve/pass/set/attack/in flight) | `analytics/insights.py` | Partially implemented | No rally boundary, no outcome label, no tuples |
| Game-state representation | Positions, poses, ball, court, relational | Frame graph + 32-D global embedding + possession/phase | `graph_builder.py`, `insights.py` | Completed | Sufficient for the implemented analytics |
| EPV | Expected possession value | Physics EPV: 1,296 cells x 5 shots, drag flight, block, defender race, aim error | `analytics/epv.py`, `constants.py` | Completed | Not learned; validated only by an aggregate bracket |
| Activity recognition | 8 group activities | External pretrained RCRG checkpoint, 9-frame windows | `pipeline/group_activity.py` | Completed earlier, **broken now** | `GAT/` folder deleted; paths resolve inside `pipeline/` |
| Action forecasting | Predict future group activity | 0.5 s per-player displacement only | `analytics/learned.py`, `gat.py` | Partially implemented | No future-activity prediction exists |
| Tactical recommendation | Recommend where/how to hit | Best-shot surface, zone, move, set-to choice, debouncing | `analytics/insights.py` | Completed | Strongest delivered feature |
| Commentary | Natural-language analysis | Rule-triggered, LLM-written with template fallback and fact validation | `analytics/commentary.py`, `nlp.py` | Completed | Guard checks zone/player, not semantic claims |
| Dashboard | Visual analytics | 1920x1080 composed video + insights JSON + snapshots | `pipeline/tactical_dashboard.py` | Completed | Strong deliverable; CPU-bound rendering |
| Web application | Upload, run, inspect | Flask, queue, subprocess, ffmpeg transcode, insights API | `webapp/server.py`, `webapp/static/*` | Completed | Dev server, no auth, single-job concurrency |
| Real-time operation | Real-time inference | Offline whole-clip batch | `master_main.py` | Not implemented | No FPS measurement stored anywhere in the repo |
| Multi-video operation | Generalise across matches | One hand-annotated clip; sample corners reused | `data/`, `webapp/server.py` | Not implemented | Correct behaviour requires new annotations per video |
| Multi-sport operation | Adaptable beyond volleyball | Volleyball-specific constants and zones only | `analytics/constants.py` | Not implemented | Modularity would allow it; no work done |

## 7.2 Non-functional requirements

| Requirement | Assessment | Evidence |
|---|---|---|
| Modularity | **Good.** Perception, graph, models, analytics, presentation are separate packages with narrow interfaces | `pipeline/`, `models/`, `analytics/`, `temporal/` |
| Code quality | **Mixed.** Analytics and models are well-documented and defensively validated; pipeline modules are verbose, heavily printed, with duplicated helpers and clip-specific constants | `integrate_and_rebuild.py` 1,613 lines mostly print statements; `graph_builder.py` 1,277 lines |
| Type hints | Used consistently in `analytics/`, `temporal/`, `models/`; almost absent in `pipeline/` | Module headers |
| Error handling | **Inconsistent.** Strong: checkpoint validation, allow-listed model config, per-frame skip counters. Weak: exceptions swallowed in `learned.build_graph` and `_LlmPreload`; `run_pretrained_group_activity` re-raises only on non-CPU | `analytics/learned.py`, `master_main.py` |
| Dependency management | **Weak.** Unpinned `requirements.txt`; `torchvision` missing though required by `group_activity`; `huggingface_hub`, `accelerate`, `bitsandbytes` missing though used by the LLM path | `requirements.txt`, `analytics/nlp.py` |
| Testing | **Weak.** One test module (291 lines) covering cache behaviour, EPV reuse, commentary determinism, quant option; **fails to import in the current tree** | `tests/test_pipeline_optimizations.py` |
| Performance instrumentation | **Good.** Profiling spans with device sync, per-frame throughput, model-device verification | `pipeline/pipeline_profiling.py` |
| Reproducibility | **Partial.** Seeds set in training; randomness in shuffle; measured constants pinned in JSON; environment versions unpinned | `train_gat_hrn.py`, `physics_calibration.json` |
| Deployment readiness | **Not ready.** Dev server, no container, no CI, no config files, no packaged artifacts | Repo root |
| Documentation accuracy | **Stale in places.** RUN_GUIDE commands reference root-level scripts that no longer exist; requirements claim coverage it lacks; corner-click UI does not exist | `docs/RUN_GUIDE.md`, `webapp/server.py` |
| Data protection | **Low risk** local-only app; uploads stored unencrypted in `webapp/jobs/`; no auth | `webapp/server.py` |

## 7.3 Defects found (verified vs. recommended)

| ID | Severity | Defect | Evidence | Class |
|---|---|---|---|---|
| D1 | **Blocker** | Pipeline modules resolve data to `pipeline/`, not `data/`, so stage 2 and the dashboard cannot read inputs | Live probe: `integrate_and_rebuild.ROOT = <repo>/pipeline`; expected player JSON, ball CSV and checkpoint dir all `exists: False`; same for `sample_data` and `tactical_dashboard` | Genuine defect (regression from the restructure) |
| D2 | **Major** | Test suite cannot import | `python -m unittest discover -s tests -v` -> `ModuleNotFoundError: No module named 'sample_data'` (1 error) | Genuine defect |
| D3 | **Major** | Pretrained group-activity stage cannot load its model or source | `GAT_DIR`/`DEFAULT_CHECKPOINT` resolve under `pipeline/`; `GAT/` deleted from working tree; `master_main` re-raises on GPU | Genuine defect |
| D4 | Medium | Player detector class not filtered; ball-class boxes can be tracked as players | `weights/player_detection.pt` names `{0: ball, 1: player}`; `master_main` iterates all `result.boxes` | Genuine defect |
| D5 | Medium | `weights/yolov8n-pose.pt` is unreadable (corrupt archive) | Live probe: `RuntimeError ... file in archive is not in a subdirectory: m3f.py` | Genuine defect (fallback weight unusable; `pose.pt` exists so it is never hit in practice) |
| D6 | Medium | Team assignment by court half; fails when teams switch sides or tracking is noisy | `master_main` (`team_id = 0 if court_x < 9 else 1`), `players_for_frame` | Fragile assumption |
| D7 | Medium | Hard-coded clip-specific constants | `excluded_ids = {20, 63, 69}`; `IMAGE_WIDTH/HEIGHT` 1920x1080; `team names BRA/JPN` | Fragile assumption |
| D8 | Medium | Uploaded videos inherit the sample's court corners | `server.py` "the web app never writes this file"; no annotation endpoint | Genuine limitation; contradicts RUN_GUIDE's corner-click description |
| D9 | Medium | Graph uses image space while targets use court metres | `graph_builder.normalize_position` (÷1920/1080), `net_x = 960`; targets via homography | Architectural weakness |
| D10 | Low | JSON `speed` field written as 0.0 while `vx/vy` are px/frame | `master_main` frame_record | Dead/inconsistent field |
| D11 | Low | Transformer positional encoding collapses to a single linear term | `_build_positions`: divisors reduce to `10000^0 = 1` | Implementation bug (harmless while untrained) |
| D12 | Low | `redetect_ball` documents three image sizes, runs two | `SIZES = (640, 1280)` | Documentation drift |
| D13 | Low | Upstream research code imports packages absent from requirements | `models/PersonActivityClassifier.py` imports `albumentations`, `torchinfo` | Unused code + dependency gap |
| D14 | Low | Duplicated/obsolete code paths | `pipeline/clean_ball_track.py` vs `analytics/ball_track.py`; 12 model variants in `models/*`; deleted `previous_pipeline_backup/` | Maintainability |
| D15 | Low | Pipeline modules print hundreds of diagnostic lines | `integrate_and_rebuild` prints per frame | Observability noise; slows web log |

No defect in this table was fixed. The report is documentation only, per the brief.

## 7.4 What a reviewer should know about the repository state

`git status` on the audited tree: **90 deletions, 11 modifications, 6 untracked paths**. The committed HEAD layout has all pipeline scripts and all data files at the repository root; the working tree has moved scripts into `pipeline/` and data into `data/`, `weights/`, `docs/`. `webapp/server.py` and `master_main.py` were partially updated for the new layout (`ROOT/"data"/annotations.json`), but the moved pipeline modules were not. The project therefore sits between two layouts. The one-line class of fix is to anchor `ROOT` at the repository parent (`Path(__file__).resolve().parent.parent`) and read `ROOT/"data"/...`, `ROOT/"checkpoints"/...`, `ROOT/"weights"/...`; that change is **recommended** in Ch. 12 and **not applied**.

---

# 8. SETUP AND EXECUTION GUIDE

## 8.1 What was verified, and what was not

| Item | Status |
|---|---|
| Interpreter and library versions | **Verified** by inspecting the project virtual environment |
| GPU/CUDA availability | **Verified**: CUDA available, NVIDIA GeForce RTX 3050 6GB Laptop GPU; Intel XPU unavailable |
| Weight files load and expose expected classes | **Verified** for `player_detection.pt`, `ball.pt`, `pose.pt`; `yolov8n-pose.pt` fails to load |
| Test suite | **Verified failing** (`ModuleNotFoundError: sample_data`) |
| Data/checkpoint path resolution | **Verified failing** (all paths point inside `pipeline/`) |
| Full pipeline execution today | **Not verified / blocked** – blocked by D1–D3 |
| Full pipeline execution on 4 Nov 2026 | **Verified from artifacts** (6 completed jobs, logs, outputs) |
| Dashboard/UI behaviour | **Not verified** in this audit (no browser session, no rendering test) |

## 8.2 Environment requirements

Observed working environment (from `.venv`):

| Component | Version |
|---|---|
| Python | 3.11.9 (docs say "3.10 or newer, tested on 3.13") |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| Ultralytics | 8.4.198 |
| PyTorch Geometric | 2.8.1 |
| OpenCV | 5.0.0 |
| NumPy | 2.4.6 |
| Flask | 3.1.3 |
| Transformers | 5.22.1 |
| supervision | 0.31.3 |
| GPU | NVIDIA RTX 3050 6 GB laptop (CUDA 12.8 build) |
| Disk | Weights ~26 MB, pretrained checkpoint ~693 MB, LLM cache ~6 GB, sample video 26 MB |

CPU-only operation is supported by the code (`--device cpu`) but the LLM commentary is very slow on CPU and detection throughput drops sharply. The Intel XPU path exists in `device_selection.py` but was untestable here (no XPU device).

## 8.3 Installation

```bash
# 1. From the repository root, create and activate an environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install the declared dependencies
pip install -r requirements.txt

# 3. Install the dependencies the declared list is missing but the code imports
pip install torchvision huggingface_hub accelerate     # group_activity.py, analytics/nlp.py
#    optional, for --llm-quant 4bit:
pip install bitsandbytes

# 4. Confirm the GPU is visible to the installed torch build
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` contains: `opencv-python, ultralytics, torch, supervision, numpy, torch_geometric, pillow, matplotlib, transformers, flask, imageio-ffmpeg` – unpinned, and missing `torchvision`, `huggingface_hub`, `accelerate`, `bitsandbytes`.

## 8.4 Required files

| Path | Required by | Present |
|---|---|---|
| `data/sample3.mp4` | sample run | Yes (26 MB) |
| `data/annotations.json` | homography, camera, all metrics | Yes (319 frames) |
| `data/court.png` | detection preview mini-court | Yes |
| `data/player_keypoints_mf3_pose.json` | graph stage | Yes (3.6 MB, 4 Nov) |
| `data/ball.csv`, `data/ball_clean.csv`, `data/ball_candidates.json` | graph stage, dashboard | Yes |
| `weights/player_detection.pt`, `weights/ball.pt`, `weights/pose.pt` | detection | Yes |
| `weights/yolov8n-pose.pt` | fallback pose | Yes but **corrupt** |
| `checkpoints/gat.pt`, `checkpoints/hrn.pt` | learned blending | Yes |
| `pretrained_checkpoints/RCRG-R2-C11-conc-temporal-V1/checkpoint_epoch_42.pkl` | group activity | Yes (693 MB) |
| `GAT/models/attention_models/relational_attention.py` | group activity | **No – deleted in working tree** |

## 8.5 Running the project

**Web application** (intended primary path):

```bash
python webapp/server.py           # http://127.0.0.1:5000
```

Internally the server launches the pipeline as a subprocess from the repository root and writes job artifacts to `webapp/jobs/<job-id>/`. Jobs are serialised because the pipeline overwrites shared detection files in the project root.

**Command line** (single entry point that exists today):

```bash
python master_main.py --video data/sample3.mp4 --dashboard            # full pipeline + dashboard
python master_main.py --dashboard-only                                # dashboard from existing detections
python master_main.py --video data/sample3.mp4 --profile              # add timing report
python master_main.py --video data/sample3.mp4 --temporal-model transformer \
                      --temporal-checkpoint <path-to-checkpoint.pt>    # temporal inference (needs a checkpoint)
python master_main.py --video data/sample3.mp4 --visualize-graphs     # + debug PNGs (slow)
```

Selection flags: `--device {auto,cpu,cuda,xpu}`, `--teams LEFT RIGHT` (dashboard labels), `--commentary-mode {background,serialized}`, `--llm-quant {fp16,4bit}`, `--preload-llm`, `--keep-existing-output`.

**Stage scripts.** `pipeline/*.py` each have a `__main__`, but they must be run with the package directory on the path (e.g. `python -m pipeline.train_gat_hrn` is **not** supported because the modules use absolute sibling imports; the working pattern is `python pipeline/train_gat_hrn.py` **only after** fixing the data-path anchoring of D1). The commands printed in `docs/RUN_GUIDE.md` (`python tactical_dashboard.py`, `python train_gat_hrn.py`, `python redetect_ball.py`, `python group_activity.py`, `python calibrate_physics.py`) **do not work today**: those files live in `pipeline/`, and `python train_gat_hrn.py` additionally imports `tactical_dashboard` as a top-level module.

**Training the GAT/HRN** (once D1 is fixed):

```bash
python pipeline/train_gat_hrn.py --gat-epochs 150 --hrn-epochs 100 --lr 1e-3 --seed 7 --fps 30
```

It reads the existing detections and annotations, builds graphs, and writes `checkpoints/{gat.pt,hrn.pt,training_report.json,training_curves.png}`. Recorded cost: 236 s total (22 s data, 25 s GAT, 188 s HRN) on the RTX 3050.

## 8.6 Outputs and expected results

| Output | Location | Content |
|---|---|---|
| Detection JSON | `data/player_keypoints_mf3_pose.json` | 319 frames x (players x 18 fields) |
| Ball tracks | `data/ball.csv`, `ball_clean.csv`, `ball_candidates.json` | One row per frame; cleaned track adds `Source` |
| Detection preview | `docs/demo/master_detection_preview.mp4` | Boxes, IDs, pose dots, court outline, mini-court |
| Graphs | `output_recent/graphs/graph_frame_XXXX.pt` | PyG graph + GAT/HRN embeddings + diagnostics |
| Dimensions | `output_recent/dimensions/frame_XXXX.npz` | Node/edge/team/global arrays |
| Sequences | `output_recent/temporal/hrn_global_sequences.{npy,pt}`, `sequence_frame_ids.npy` | 8-frame windows of 32-D embeddings |
| Group activity | `output_recent/group_activity/group_activity.json` | 36 windows + overall distribution (8 classes) |
| Dashboard | `output_recent/dashboard/gritsense_dashboard.mp4`, `dashboard_insights.json`, `commentary.txt`, `dashboard_XXXX.png` | Rendered analytics |

Expected scale for the bundled clip: 319 frames, typically 11 tracked players per frame, 12-node graphs, ~311 temporal windows (319 − 8 + 1), 36 group-activity windows, 10–11 touches.

## 8.7 Troubleshooting

**Verified causes** (observed in this audit or in the recorded logs):

| Symptom | Verified cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sample_data'` when running tests | `tests/` is discovered from the root but the module lives in `pipeline/` | Add `pipeline/` to `sys.path` in the test module, or move the test's import target |
| `FileNotFoundError: player_keypoints_mf3_pose.json not found: <repo>\pipeline\player_keypoints_mf3_pose.json` | D1 – `ROOT` anchored to `pipeline/` | Re-anchor `ROOT` to the repository parent (Ch. 12, item 1) |
| Group-activity stage aborts on GPU with a file-not-found for `relational_attention.py` | D3 – `GAT/` deleted, paths wrong | Restore `GAT/models/attention_models/relational_attention.py` or vendor it into the repo and re-point `GAT_DIR` |
| `RuntimeError ... file in archive is not in a subdirectory: m3f.py` | `weights/yolov8n-pose.pt` is not a valid Ultralytics archive | Delete/ignore it; `weights/pose.pt` is used first and is valid |
| `WARNING: Redirects are currently not supported ... redirects.py` in job logs | PyTorch distributed redirect warning during LLM load | Harmless |
| `torch_dtype is deprecated! Use dtype instead` | Transformers 5.x naming change in `analytics/nlp.py` | Cosmetic; update the kwarg |
| `UnicodeEncodeError: 'charmap' codec can't encode character '✓'` | Windows console code page vs. `print("✓ ...")` | `set PYTHONIOENCODING=utf-8` (verified working in this audit) |
| Dashboard shows wrong court geometry for a new video | Uploaded video reuses `data/annotations.json` | Annotate the new video's corners; there is no UI for it |

**General suggestions** (not verified against this repo): CUDA/python wheel mismatch (install the torch build matching the driver), GPU OOM (reduce batch sizes in `redetect_ball.detect` / `group_activity.run`, or use `--llm-quant 4bit`), and slow first LLM load (Hugging Face download of ~6 GB into the cache).

---

# 9. DATASET AND MODEL ANALYSIS

## 9.1 What the Phase 2 report specifies

The Volleyball Dataset (Ibrahim et al., CVPR 2016): 55 YouTube match videos, 4,830 labelled frames, 3,493 train / 1,337 test, 9 player actions (standing, moving, waiting, blocking, digging, setting, falling, spiking, jumping) and 8 group activities (left/right pass, set, spike, winpoint), with bounding boxes, player IDs and team labels; Phase 2 also plans YOLO-format conversion, de-duplication and a 70–80/10–15/10–15 split.

## 9.2 What the repository actually contains

| Artifact | Reality |
|---|---|
| Volleyball Dataset files | **Absent.** No annotation files, no frame directories, no label pickles |
| Dataset loader code | **Present but unused:** `utils/data_utils/data_loader.py` (422 lines), `volleyball annot_loader.py`, `eval_metrics.py`, `sampler_weights.py` – inherited from the third-party RCRG repository; imports would need `albumentations`, `torchinfo`, `sklearn` |
| Actual data used for all GritSense training | **One clip:** `data/sample3.mp4`, 319 frames, 30 fps, 10.6 s, Brazil v Japan, 2022 FIVB Men's World Championship |
| Labels used for training | **None manual.** GAT targets are future tracked positions; HRN targets are physics-model outputs |
| Annotation used | `data/annotations.json` – four court corners x 319 frames (geometry, not semantics) |

The distinction matters: the Phase 2 document's dataset statistics describe an *intended* benchmark, while the implemented system trains and evaluates on a single 10-second broadcast excerpt. The report's own logs confirm this (`train_graphs 190 / val 64 / test 63` from 319 frames; 36 group-activity windows).

## 9.3 Preprocessing and split

- **Detection output normalisation:** `sample_data.py` accepts dict/list frames, many key aliases, bbox-derived positions, string or numeric team labels, and 34-value flattened poses; it deep-copies court data and caches parses on `(mtime, size)` so repeated per-frame reads are cheap.
- **Ball track:** cleaned as described in §6.4; `Source` distinguishes detected from interpolated points.
- **Split (GAT/HRN):** the clip is cut into 32-frame blocks; of every 5 blocks, 3 are training, 1 validation (epoch selection), 1 test (final metrics). This interleaved block split keeps every rally phase on all sides while preventing near-identical adjacent frames from leaking across the split – a sensible choice for tiny data. Counts: 190 / 64 / 63.
- **Temporal windows:** 8 frames, stride 1, on the 32-D global embedding; no standardisation is applied in this path (`FeatureStandardizer` unused).
- **Group activity windows:** 9 frames, stride 9, RGB crops normalised with ImageNet statistics, max 12 players sorted left-to-right and padded with zeros.

## 9.4 Models, checkpoints, and evaluation methodology

| Model | Params (approx.) | Checkpoint | Training | Evaluation |
|---|---|---|---|---|
| GAT + movement head | 49x32x4 + head ≈ 40 k | `checkpoints/gat.pt` (134 kB) | Self-supervised, 150 epochs, 25 s | ADE vs two baselines on 63 frames / 487 players |
| HRN + tactical heads | GAT + projections + heads ≈ 60 k | `checkpoints/hrn.pt` (172 kB) | Distillation, 100 epochs, 188 s | Zone/shot accuracy vs majority baseline, threat and shape MAE |
| GRU temporal | input 32, hidden 128 ≈ 62 k | **None** | **Never trained** | None |
| Transformer temporal | 2 layers, 128-D, 4 heads ≈ 200 k | **None** | **Never trained** | None |
| RCRG group activity (external) | ResNet50 + 2 relational units + LSTM ≈ 34 M | 693 MB `.pkl` | Pre-trained by its authors on the Volleyball Dataset | Not evaluated here; applied zero-shot |
| YOLO detectors | nano/small scale (~6 MB each) | `weights/*.pt` | Upstream provenance unknown | Not evaluated; no mAP reported |

**Training vs validation vs inference.** Training updates weights; validation only selects the best epoch; the test split is touched once for the reported numbers. That discipline is correctly implemented. Inference paths (`integrate_and_rebuild`, `LearnedModels.predict`, `group_activity.run`) never compute gradients (`torch.no_grad` / `inference_mode`).

---

# 10. RESULTS AND EVALUATION

All numbers below are copied from artifacts in the repository, with the file named. Nothing is estimated or extrapolated.

## 10.1 Learned model evaluation (held-out test split)

From `checkpoints/training_report.json`:

| Metric | Value | Baseline | Interpretation |
|---|---|---|---|
| GAT movement ADE | **0.873 m** | constant velocity 1.557 m; no movement 0.765 m | Beats constant velocity by 44%, loses to "nobody moves" |
| GAT players evaluated | 487 | – | 63 test frames |
| GAT best epoch | 99 / 150 | – | Slow improvement, then plateau |
| HRN zone accuracy | **46.0%** | majority class 49.2% | Worse than always guessing the most common zone |
| HRN shot accuracy | **92.1%** | majority class 92.1% | Exactly the baseline – the teacher is nearly constant |
| HRN threat MAE | 0.303 | – | On a 0–1 threat scale |
| HRN team-shape MAE | 0.066 | – | Coverage/balance in 0–1 |
| HRN best epoch | **1** / 100 | – | Overfits immediately |

Timings: data 22 s, GAT 25 s, HRN 188 s, total 236 s. Settings: 150/100 epochs, lr 1e-3 (HRN 5e-4), seed 7, horizon 15 frames.

**How to read these results.** They are a data problem, not (primarily) a modelling problem. With 317 usable frames, the GAT sees ~190 training graphs; a 12-node graph with 49 features and 4-head attention has more capacity than that can constrain. The HRN's teacher (the physics model) is itself nearly constant in shot choice (hence 92.1% = baseline), so the HRN has almost nothing to learn. Both training curves show training loss falling while validation stays flat, which the authors correctly describe in `docs/SYSTEM_EXPLAINED.md`.

## 10.2 Physics and perception results

From `analytics/physics_calibration.json` (measured with `calibrate_physics.py`):

| Quantity | Value |
|---|---|
| Camera focal length | 2500 px (fitted over all 319 frames) |
| Camera centre (median) | (8.62, 33.25, 10.06) m; std (0.95, 1.29, 0.71) m |
| Player speed samples | 3,136; median 1.58 m/s; p99 6.97 m/s (capped to 4.75) |
| Standing-height samples | 2,636; median stature 1.92 m |
| Jumps detected | 81; near-net peak median 0.46 m, 90th pct 0.70 m, max 1.00 m |
| Maximum near-net reach | 3.19 m (published block reach 3.18 m) |
| Ball horizontal speed pairs | 60; maximum 25.2 m/s (sanity check only) |
| Touches detected | 10 (frames 34, 50, 71, 130, 176, 213, 255, 273, 296, 308) |
| Set-to-attack time | 0.60 s |
| Highest touch | 3.64 m |
| EPV validation | best-placed spike 0.883; average legal aim 0.3859; published attack success 0.725 / 0.551 |

**Ball tracking improvement (author-reported, `docs/SYSTEM_EXPLAINED.md`):** 259 real detections + 58 interpolated = 317 of 319 frames, versus 229 frames before, "many of them false". Touch precision: 10 detected, 7 verified by eye, 3 false; every real touch classified correctly.

## 10.3 End-to-end run evidence

From `webapp/jobs/20261104-181032-a17e/` (one of five full runs on 4 November 2026, `device: cuda`):

| Observation | Value |
|---|---|
| Frames processed | 319 |
| Players per frame (example) | 11 |
| Pose coverage (example frames) | 3/11 and 5/11; non-zero pose values 102–170 of 374 |
| Graph nodes per frame | 12 (11 players + ball) |
| Group-activity windows | 36; overall distribution nearly uniform, argmax `l-pass` at 0.197 |
| Ball touches detected in pipeline | 11 |
| Commentary lines | 6, all `LOCAL LLM` |
| Temporal classification | **Skipped** – no trained checkpoint supplied |
| Learned blending | "learned weight 30%; HRN target influence scaled by test skill: zone 0.00, shot 0.00" |

## 10.4 What is missing from evaluation

| Metric | Availability |
|---|---|
| Detection mAP / precision-recall | **Not available** – no evaluation of the YOLO weights anywhere in the repo |
| Tracking consistency (MOTA/IDF1/ID switches) | **Not available** |
| Pose accuracy (PCK/OKS) | **Not available** |
| Activity classification accuracy (project's own GRU/Transformer) | **Not available** – no checkpoint, no run |
| Forecasting accuracy for future group activity | **Not available** – capability not implemented |
| GRU vs Transformer comparison | **Not available** – promised in Phase 3, never run |
| Inference latency / FPS | **Not available** – profiling exists but no stored report; the run logs contain no throughput summary |
| GPU utilisation / memory | **Not available** |

## 10.5 How the project *should* be evaluated

1. **Detection/tracking:** mAP@0.5 and MOTA/IDF1 on 200+ annotated frames of 3–5 matches, including occlusions and camera cuts.
2. **Pose:** PCK@0.2 per keypoint on the far-court players, where coverage is currently weakest.
3. **Ball:** per-frame precision/recall and trajectory RMSE, plus touch-level precision/recall/F1 with a shot-type confusion matrix.
4. **Graph/temporal:** group-activity top-1 and macro-F1 on the Volleyball Dataset test split (1,337 frames), with the majority baseline reported beside every number – the repository already has the baseline machinery.
5. **Forecasting:** ADE/FDE for 0.5 s and 1 s horizons, with no-movement and constant-velocity baselines, reported per rally phase.
6. **EPV:** calibration (reliability diagram) against rally outcomes once a labelled dataset exists; until then, only the aggregate bracket in §6.12 is defensible.
7. **Systems:** end-to-end latency decomposed by stage (the profiling spans already implement this) and steady-state FPS on a fixed GPU.

---

# 11. PROBLEMS SOLVED AND PRACTICAL APPLICATIONS

## 11.1 The original problem, restated precisely

Coaches and analysts watch match video and mentally reconstruct what happened: who passed, who set, where the block formed, where the attack should have gone. That reconstruction is slow, subjective, and impossible to aggregate across matches. Automatic analysis is hard for three reasons: (i) perception in broadcast footage is noisy (small objects, occlusion, camera cuts, non-players); (ii) meaning lives in relations between players, not in any one player; (iii) the *decision* – where should the ball have gone – is not observable in the pixels at all.

## 11.2 What GritSense actually solves

- **Metric state extraction from a single camera.** Using the court homography plus a recovered camera, the system produces court positions, velocities, jump heights and touch heights from ordinary broadcast video, with no special hardware. That is a real capability, validated against published net height, block reach and standing stature.
- **Ball truth from a bad tracker.** Re-detection plus a physically constrained dynamic programme converts a track that followed a referee's head into a 317/319-frame trajectory.
- **Relational representation.** The 49-D interaction graph and HRN hierarchy give a principled, extensible representation of a volleyball rally that a group-activity model can be trained on – the scaffolding the project set out to build.
- **A decision layer that does not need labels.** The physics EPV provides interpretable, sourced tactical advice immediately, and simultaneously acts as a teacher that can label data for a future learned model. This is the project's most original engineering decision.
- **Self-aware learned components.** Baselines are computed and printed next to every learned metric, and learned influence is scaled by measured skill, so a bad model cannot degrade the product.

## 11.3 Applications

**Demonstrated today** (supported by artifacts in this repository):

| User | Capability |
|---|---|
| Coach, match review | Top-down positions, player speeds, jump counts and heights, per-touch phase timeline |
| Coach, tactical decision support | Where to hit, with which shot, expected point chance, block and error risk, per-rally-moment |
| Analyst, defensive assessment | Which defender removes the most threat (leave-one-out), team coverage and balance |
| Analyst, communication | Auto-generated, fact-constrained commentary and per-frame insight JSON |
| Researcher | A reproducible pipeline and honest baselines for group-activity work on volleyball |

**Potential, not demonstrated:**

| Area | Why it is plausible | What is missing |
|---|---|---|
| Formation and rotation analysis | Court positions and team shape are already computed | Rotation detection, per-rotation aggregation, multi-rally statistics |
| Player workload and fatigue | Speeds and jumps are measured | Session-level aggregation, minutes played, load models |
| Live broadcast augmentation | Pipeline is modular and device-selectable | True streaming, latency budget, automatic court detection |
| Other sports (basketball, handball) | Graph + EPV structure is generic | Sport-specific constants, court geometry, rules, and data |
| Crowd/healthcare monitoring (Phase 2 §1.5) | Generic group-activity framing | Everything: no dataset, no adaptation, and the current models are volleyball-specific |

---

# 12. FUTURE IMPROVEMENTS (PRIORITISED)

| # | Limitation | Proposed solution | Why it helps | Expected impact | Complexity |
|---|---|---|---|---|---|
| 1 | Pipeline cannot run at HEAD (D1–D3) | Anchor `ROOT` at the repo parent; central `paths.py`; restore/vendor `GAT/` relational unit; add `pipeline/` to test path | Unblocks every other improvement | **Critical** | Low |
| 2 | One hand-annotated clip | Automatic court keypoint detection (e.g. a small keypoint model on court lines) + temporal smoothing of corners; accept manual override | Removes the per-video manual step; enables multi-match data collection | **Very high** | High |
| 3 | Data starvation (GAT loses to no-movement) | Process 10–20 matches; train on the Volleyball Dataset pretrained features; expand to ~50 k graphs | Learning only becomes meaningful beyond a few thousand graphs | **Very high** | Medium (compute) |
| 4 | Graph features are image-space | Express positions/velocities in court metres, add relative-to-net and relative-to-ball features, include per-player 3-D height | Removes the need to learn perspective; should cut ADE substantially | High | Medium |
| 5 | Temporal features are only the 32-D global embedding | Wire `FeaturePreparer` to concatenate player/pose/ball/spatial blocks; add the 49-D node summary | Matches the documented design; gives the GRU/Transformer real input | High | Low–Medium |
| 6 | No trained temporal model | Build a labelled window dataset (phase labels from touches, plus Volleyball Dataset group labels), add a training CLI around `TemporalTrainer`, run the GRU-vs-Transformer comparison | Delivers the Phase 3 evaluation that is currently missing | High | Medium |
| 7 | Activity labels come from a mismatched external model | Fine-tune the RCRG-style model on this pipeline's own graphs/crops, or train the project's HRN heads on real labels | Turns near-uniform outputs into usable activity recognition | High | High |
| 8 | Tracking breaks at cuts/occlusions | Re-identification embedding (appearance) + track stitching across cuts; jersey-number OCR for stable identities | Fixes velocity spikes, team mix-ups and matchup errors | Medium–High | Medium |
| 9 | Pose coverage 27–46% | Larger pose model or tiled inference on the far court; temporal keypoint interpolation; use pose confidence as a feature | Better pose block utilisation, better jump/arm signals | Medium | Low |
| 10 | Ball 3-D position unknown | Two-camera or court-plane-constrained trajectory fitting; parabolic fit with known contact heights to recover height | Enables real ball speed, better touch and attack detection | Medium | High |
| 11 | Team labels by court half | Cluster jersey colours once per match; persist team identity | Correct attacker/defender assignment after side switches | Medium | Low |
| 12 | EPV unvalidated at rally level | Collect rally outcomes; fit/verify `P(in)`, block and reach models; report calibration curves | Converts a physically motivated score into a calibrated probability | Medium | Medium |
| 13 | Offline batch only | Streaming ingestion, frame-skipping policy, async detection, ONNX/TensorRT export of detectors and GAT | Moving toward the project's "real-time" claim | Medium | High |
| 14 | Test coverage is one module and currently broken | Fix the import, add graph-builder invariants, EPV monotonicity, touch classification and path-selection tests; add CI | Prevents silent regressions like D1 | Medium | Low |
| 15 | No structured logging or metrics | Replace `print` with `logging`, emit per-stage timings and per-frame counters to JSON | Makes profiling and regression detection routine | Low–Medium | Low |
| 16 | Deployment is a dev server | Containerise, pin dependencies with a lock file, add config for paths/devices, authenticate the UI | Reproducible installs and safe sharing | Low (for research use) | Medium |

Recommended order: **1 → 2 → 3 → 4 → 5 → 6**, because each unlocks the next, and because the project's central claim (learned group-activity recognition and forecasting) cannot be evaluated at all until data and features are in place.

---

# 13. CONCLUSION

## 13.1 What was achieved

**Phase 2 (perception and court mapping).** Implemented and working: YOLO player tracking, 17-keypoint pose with association, dedicated ball detection with a cleaned trajectory, per-frame homography, court-metre positions and velocities, plus—beyond the Phase 2 plan—full camera recovery enabling height measurement (jumps, reach, touch heights) from a single view. The one Phase 2 stage that was *not* implemented is automatic court keypoint detection; corners are hand-annotated.

**Phase 3 (relational and temporal modelling, EPV).** Implemented: the 49-D interaction graph, a two-layer GAT, an HRN with player/team/global levels, self-supervised and distillation training with checkpoints, 32-D per-frame embeddings, 8-frame temporal windows, GRU and Transformer implementations, a full physics EPV engine with provenance-tagged constants, blending logic with skill-scaled learned influence, an external pretrained group-activity inference path, LLM commentary, a dashboard renderer and a web application. Not implemented or not operational: rally segmentation as a dataset, temporal activity classification (no trained model), the GRU-vs-Transformer comparison, future group-activity forecasting, and real-time operation.

## 13.2 Maturity assessment

| Dimension | Rating | Basis |
|---|---|---|
| Perception | **Working, single-clip** | Validated heights and reach; 27–46% pose coverage; manual court corners |
| Analytics / decision support | **Strong** | Interpretable EPV, matchups, team shape, phase gating, debouncing |
| Learned relational models | **Prototype** | Trained, evaluated, honest, and currently worse than trivial baselines |
| Temporal models | **Scaffold** | Code complete, never trained, never run |
| Engineering | **Good structure, fragile state** | Clean package separation; path regression blocks execution at HEAD; tests broken |
| Documentation | **Rich but drifting** | Excellent internal rationale (`SYSTEM_EXPLAINED.md`); `RUN_GUIDE.md` commands are stale |
| Scientific honesty | **Unusually high** | Baselines beside every metric, sourced constants, skill-scaled model influence, self-reported false positives |

## 13.3 Major strengths

1. **The physics EPV with full provenance.** Every constant is tagged measured/published/derived/estimated, and the model was checked against published attack success without being fitted to it.
2. **The honesty architecture.** `_skill()` scaling means a learned model with no measured skill literally cannot influence the recommendation. Few student projects build that guard.
3. **Real, verifiable perception engineering.** The ball-cleaning work (static rejection, physical gate, DP path, gap filling) addresses a genuine failure mode with a measurable improvement.
4. **Breadth delivered end-to-end.** From raw video to a rendered, narrated dashboard and a working web app, with an insight JSON API.

## 13.4 Major limitations

1. **Data.** One 10.6-second clip; the learned models cannot work, and the project says so.
2. **Generalisation.** Court corners are hand-made for one video; uploaded videos silently inherit them.
3. **Repository state.** The working tree cannot execute the pipeline; the test suite cannot import; the pretrained group-activity stage is broken; one weight file is corrupt.
4. **Claims vs reality.** "Real-time", "activity forecasting" and "GRU vs Transformer comparison" are promised in the deck and not delivered; BlazePose is cited but not used; automatic court detection is described but absent.
5. **Evaluation gaps.** No detection/tracking/pose metrics, no FPS measurement, no activity-classification result from the project's own temporal models.

## 13.5 Remaining work, in one sentence each

Fix the path anchoring and the test import (hours). Restore or vendor the group-activity dependency (hours). Replace manual court annotation with automatic detection (weeks). Collect and process multiple matches (weeks, mostly compute). Move graph features into court space and wire the full temporal feature set (days). Train and compare the GRU and Transformer on labelled windows (weeks). Validate EPV at rally level once outcomes exist (weeks).

## 13.6 Final assessment

Judged as a Phase 3 research prototype, GritSense is **substantially implemented on the perception and decision sides, architecturally complete but empirically empty on the learning side, and temporarily broken in its packaging**. The repository contains a genuinely useful tactical analytics engine, a defensible graph/temporal design, and an evaluation culture that reports its own failures. It does not yet demonstrate group activity recognition or forecasting with its own trained models, and it cannot run end-to-end in the state audited. On the evidence available, the correct summary is: **a strong systems-and-analytics implementation awaiting data, plus a short list of concrete, low-risk repairs**, not a finished group-activity recognition system.

---

# APPENDIX A. FILE INVENTORY AND ROLE MAP

| Path | Lines | Role | Used at runtime |
|---|---|---|---|
| `master_main.py` | 785 | Orchestrator; detection stage; stage sequencing | Yes |
| `pipeline/integrate_and_rebuild.py` | 1,613 | Graph/GAT/HRN/sequence stage, artifact writing | Yes |
| `pipeline/graph_builder.py` | 1,277 | 49-D node / 4-D edge graph + validation | Yes |
| `pipeline/sample_data.py` | 1,130 | Frame -> (players, ball, court); cached parsing | Yes |
| `pipeline/tactical_dashboard.py` | 1,083 | Dashboard rendering; court-space player/ball helpers | Yes |
| `pipeline/visualize_recent_graphs.py` | 1,024 | Debug PNG dumps of graphs | Optional |
| `pipeline/train_gat_hrn.py` | 400 | Self-supervised GAT + distilled HRN training | On demand |
| `pipeline/group_activity.py` | 280 | External pretrained group-activity inference | Yes (broken) |
| `pipeline/calibrate_physics.py` | 232 | Measures physical constants from the clip | On demand |
| `pipeline/ball_tracker.py` | 200 | In-loop ball detection/selection; HSV helper | Yes |
| `pipeline/redetect_ball.py` | 143 | Multi-scale candidate generation | Yes |
| `pipeline/clean_ball_track.py` | ~90 | DP path selection wrapper -> `ball_clean.csv` | Yes |
| `pipeline/pipeline_profiling.py` | 115 | Timing spans, GPU sync, report | Yes |
| `pipeline/device_selection.py` | 50 | Device resolution (xpu/cuda/cpu) | Yes |
| `analytics/epv.py` | 542 | Flight simulation, blockers, reach race, EPV surface | Yes |
| `analytics/insights.py` | 528 | Possession/phase state machine, blending, debouncing | Yes |
| `analytics/ball_track.py` | 403 | Static rejection, DP path, gap fill, touch detection | Yes |
| `analytics/commentary.py` | 282 | Trigger rules, prompting, template fallback | Yes |
| `analytics/nlp.py` | 185 | Qwen2.5-3B local backend (fp16/NF4) | Yes |
| `analytics/learned.py` | 178 | Checkpoint loading, heads, attention extraction, skill | Yes |
| `analytics/constants.py` | 121 | Provenance-tagged constants + calibration override | Yes |
| `analytics/relations.py` | 109 | Leave-one-out matchups, team shape | Yes |
| `analytics/camera.py` | 109 | Focal fit, (R,t), project/height_above | Yes |
| `analytics/pose_signals.py` | 72 | Jump/reach/arms-up detection | Yes |
| `analytics/motion.py` | 42 | Court position/velocity smoothing | Yes |
| `models/gat.py` | 206 | Two-layer GAT with residual | Yes |
| `models/hrn.py` | 357 | Player/team/global hierarchy | Yes |
| `models/__init__.py` | 34 | Lazy GAT/HRN proxies | Yes |
| `models/{single_frame,temporal,attention}_models/*` | ~3,300 | Third-party RCRG variants (training code) | **No** |
| `models/PersonActivityClassifier.py` | 36 | Upstream ResNet50 person classifier | **No** |
| `temporal/gru_model.py` | 363 | GRU/Transformer classifiers + trainer | Optional |
| `temporal/sequence_generator.py` | 151 | Generic window generator | Imported interface |
| `temporal/feature_preparation.py` | 153 | Named feature blocks + standardiser | **No** |
| `utils/**` | ~900 | Upstream dataset loaders/eval/helpers | **No** |
| `webapp/server.py` | 495 | Flask job API, transcode, insights | Yes |
| `tests/test_pipeline_optimizations.py` | 291 | Cache, EPV-reuse, commentary, quant tests | Currently broken |
| `docs/RUN_GUIDE.md`, `docs/SYSTEM_EXPLAINED.md` | – | Operational guide; viva Q&A | Documentation |
| `data/*`, `weights/*`, `checkpoints/*`, `pretrained_checkpoints/*` | – | Inputs, weights, checkpoints | Yes |

# APPENDIX B. DIAGRAM SOURCES

Mermaid sources are provided as separate files under `docs/report/diagrams/`:

| File | Diagram |
|---|---|
| `01_system_architecture.mmd` | High-level implemented system architecture |
| `02_component_architecture.mmd` | Component/file-level architecture |
| `03_data_flow.mmd` | Artifact data flow |
| `04_model_pipeline.mmd` | Graph → GAT → HRN → temporal → EPV processing |
| `05_planned_vs_implemented.mmd` | Phase 2/3 intent vs implemented status |
| `06_rally_state_epv.mmd` | Possession/phase state machine and EPV decision loop |

# APPENDIX C. VERIFICATION LOG (13 NOVEMBER 2026)

| Check | Command / probe | Result |
|---|---|---|
| Repository layout | `git ls-tree HEAD`, `git status --short` | HEAD has root-level scripts/data; working tree has 90 deletions, 11 modifications, 6 untracked paths |
| Python/library versions | import probe in `.venv` | Python 3.11.9, torch 2.11.0+cu128, ultralytics 8.4.198, PyG 2.8.1, cv2 5.0.0, numpy 2.4.6, flask 3.1.3, transformers 5.22.1 |
| GPU | `torch.cuda.is_available()` | True – NVIDIA GeForce RTX 3050 6GB Laptop GPU; XPU False |
| Weight classes | `YOLO(...)` introspection | `player_detection.pt` {ball, player}; `ball.pt` {Volleyball, ball}; `pose.pt` pose, kpt_shape [17,3]; `yolov8n-pose.pt` **fails to load** |
| Path resolution | import `integrate_and_rebuild`, `sample_data`, `tactical_dashboard` | All anchor `ROOT` to `<repo>\pipeline`; player JSON, ball CSV and checkpoint dir all absent there |
| Unit tests | `python -m unittest discover -s tests -v` | **FAILED (errors=1)** – `ModuleNotFoundError: No module named 'sample_data'` |
| Run artifacts | 6 `webapp/jobs/*/job.json` | All `status: done`, `device: cuda`, dated 4 Nov 2026 |
| Metrics provenance | `checkpoints/training_report.json` | Values reproduced verbatim in §10.1 |
| Physics provenance | `analytics/physics_calibration.json` | Values reproduced verbatim in §10.2 |
| Dashboard outputs | `dashboard_insights.json`, `group_activity.json`, `commentary.json` | Frame-level EPV example and group-activity distribution quoted in §6.13, §6.11 |
| Files modified by this audit | – | **None.** Only new files were added under `docs/report/` |
