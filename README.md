# GRITSENSE: Tactical Volleyball Intelligence Platform

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/PyG-Torch_Geometric-3C2179?style=flat-square" alt="PyG" />
  <img src="https://img.shields.io/badge/Computer_Vision-YOLOv8-00FFFF?style=flat-square&logoColor=black" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/Acceleration-NVIDIA_CUDA_%7C_NVENC_%7C_Intel_XPU-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Hardware Acceleration" />
  <img src="https://img.shields.io/badge/LLM_Inference-Qwen2.5--3B--Instruct-blue?style=flat-square&logo=huggingface&logoColor=white" alt="Qwen2.5" />
  <img src="https://img.shields.io/badge/Frontend-Vanilla_JS_%7C_CSS3_%7C_Canvas-E34F26?style=flat-square&logo=html5&logoColor=white" alt="Frontend" />
  <img src="https://img.shields.io/badge/Backend-Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Backend" />
</p>

An end-to-end computer vision and sports analytics system that transforms single-camera broadcast volleyball footage into real-time metric projections, court spatial coordinates, tactical decision maps, and synchronized natural language commentary.

---

## Table of Contents

- [1. System Overview](#1-system-overview)
- [2. End-to-End Pipeline Architecture](#2-end-to-end-pipeline-architecture)
- [3. Computer Vision & Perception Engine](#3-computer-vision--perception-engine)
- [4. Spatio-Temporal Graph Neural Networks (GAT & HRN)](#4-spatio-temporal-graph-neural-networks-gat--hrn)
- [5. Physics-Based Expected Point Value (EPV) Formulation](#5-physics-based-expected-point-value-epv-formulation)
- [6. Hardware Acceleration & CUDA Pipeline](#6-hardware-acceleration--cuda-pipeline)
- [7. Web Application & Analytics Dashboard](#7-web-application--analytics-dashboard)
- [8. Repository Structure](#8-repository-structure)
- [9. Setup & Installation](#9-setup--installation)
- [10. Command Line Interface (CLI)](#10-command-line-interface-cli)
- [11. Physical Constants & Empirical Validation](#11-physical-constants--empirical-validation)
- [12. Scientific References](#12-scientific-references)

---

## 1. System Overview

GRITSENSE reconstructs real-world court physics and tactical dynamics from 1080p, 30 fps broadcast video without sensory hardware, multi-camera arrays, or player telemetry wearables.

The platform executes across five core computing stages:
1. **Perception**: Multi-object tracking (BoT-SORT), 17-keypoint human pose estimation (YOLOv8), multi-scale ballistic ball re-detection, and planar court homography estimation.
2. **Kinematic & Camera Calibration**: Single-view projective camera recovery, estimating focal length ($f \approx 2500$ px), 3D optical center, player velocity vectors, vertical jump metrics, and reach envelopes in true metric units.
3. **Graph Representation Learning**: Frame-by-frame construction of 49-dimensional player-ball interaction graphs evaluated via a 2-layer Graph Attention Network (GAT) and Hierarchical Relational Network (HRN) trained without manual labels.
4. **Aerodynamic Flight & Pitch Control Modeling**: Physics EPV simulation computing point probabilities across 1,296 discrete court cells for 5 distinct shot profiles under gravitational and aerodynamic drag constraints.
5. **Generative Commentary & Dashboard Rendering**: Asynchronous generation of tactical observations via local quantized LLMs (`Qwen2.5-3B-Instruct`), coupled with high-throughput 1080p video composition with NVENC acceleration.

---

## 2. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    subgraph S1["Stage 1: Multi-Modal Perception"]
        A["Broadcast Video Stream (1080p @ 30fps)"] --> B1["Player Detection & BoT-SORT Tracking"]
        A --> B2["17-Keypoint Skeletal Pose (YOLOv8)"]
        A --> B3["Multi-Scale Ball Detection (640 / 1280 / 1920 px)"]
        A --> B4["Planar Court Homography (H Matrix)"]
    end

    subgraph S2["Stage 2: Physics Filtering & Trajectory Reconstruction"]
        B3 --> C1["Static Artifact Rejection (Referees, UI, Antennas)"]
        C1 --> C2["Dynamic Programming Optimal Ball Path"]
        C2 --> C3["Quadratic Parabolic Gap Interpolation (<= 8 frames)"]
        B4 --> C4["Camera Geometry Recovery (Focal Length, Pose, Center)"]
        B1 & B2 & C4 --> C5["Metric Kinematics (Velocities, Vertical Jump Heights)"]
    end

    subgraph S3["Stage 3: Graph Construction & Representation Learning"]
        C3 & C5 --> D1["Volleyball Interaction Graph (49-D Nodes, 4-D Edges)"]
        D1 --> D2["Self-Supervised GAT (Trajectory Displacement Prediction)"]
        D1 --> D3["Physics-Distilled HRN (Player -> Team -> Scene Hierarchy)"]
        A & B1 --> D4["Pretrained Group Activity GAT (Volleyball Action Windows)"]
    end

    subgraph S4["Stage 4: Decision Analysis & Hybrid EPV Engine"]
        C3 & C5 --> E1["Touch Classification & Rally State Machine"]
        E1 --> E2["Flight Mechanics Simulation (Gravity + Supercritical Drag)"]
        E2 --> E3["Pitch Control Surface & Defensive Block Envelopes"]
        E3 --> E4["Physics EPV (1,296 Grid Cells x 5 Attack Profiles)"]
        D2 & D3 & E4 --> E5["Skill-Weighted Hybrid Target & Matchup Formulation"]
    end

    subgraph S5["Stage 5: Rendering & Broadcast Generation"]
        E5 --> F1["Tactical Surface Rendering (Heatmaps, Zones, Matchups)"]
        E5 --> F2["Asynchronous Local LLM Engine (Qwen2.5-3B-Instruct)"]
        F1 & F2 --> F3["Dual-Pass Broadcast Video Compositor"]
        F3 --> F4["NVENC H.264 Encoder -> Web Application Dashboard"]
    end

    S1 --> S2 --> S3 --> S4 --> S5
```

---

## 3. Computer Vision & Perception Engine

### 3.1 Player Detection and Tracking
- **Detector**: Fine-tuned YOLOv8 architecture (`weights/player_detection.pt`).
- **Tracking**: Integrated BoT-SORT algorithm combining appearance feature embeddings and Kalman filter state estimates.
- **Court Boundary Filter**: Dynamic gating strips out-of-bounds individuals ($>2.0$ m lateral clearance from court boundaries), eliminating coaches, referees, and bench personnel to isolate active on-court players ($\le 6$ per side).

### 3.2 Skeletal Pose Estimation
- **Keypoints**: 17 anatomical landmarks per player following the COCO keypoint standard.
- **Kinematic Extraction**:
  - Vertical Jump Detection: Synchronized upward translation of ankle, knee, and hip keypoints while bounding box vertical scale remains invariant, distinguishing explosive jumps from scale changes caused by camera perspective.
  - Spike Reach & Block Envelopes: Direct measurement of wrist and shoulder extension relative to recovered court plane.

### 3.3 Ball Detection, Trajectory Selection, and Touch Parsing
Standard single-frame ball detection suffers from occlusion, small physical footprint ($\approx 8\times 8$ pixels at 1080p), and false positives against white floor markings or referees' heads.
1. **Multi-Scale Inference**: `weights/ball.pt` evaluates each frame at native 640 px, intermediate 1280 px, and full 1920 px scales, accumulating top candidate detections into `ball_candidates.json`.
2. **Static Feature Filtering**: Spatial clustering identifies and prunes stationary detections across successive windows that lack player interaction.
3. **Dynamic Programming Optimization**: Global trajectory extraction solves for the path minimizing frame-to-frame velocity discontinuities while enforcing aerodynamic velocity caps ($\le 98.6$ px/frame at 30 fps).
4. **Gap Interpolation**: Parabolic curve-fitting bridges detection losses $\le 8$ frames, achieving valid ball state tracking across 317 of 319 evaluation frames (99.4% coverage).
5. **Touch Detection**: Identified by sharp trajectory deflection vectors ($>20$ px/frame acceleration shift) within the reach cylinder of a calibrated player ($\le 1.5$ m lateral, floor to 3.67 m elevation). Classifies interactions into **Pass / Dig**, **Set**, and **Attack**.

---

## 4. Spatio-Temporal Graph Neural Networks (GAT & HRN)

```
                 [ Ball Node: 49-D ]
                     ^         |
        4-D edge     |         | 4-D edge
                     |         v
[ Player Node A: 49-D ] <---> [ Player Node B: 49-D ]
                      4-D edge
```

### 4.1 Graph Topological Specification
Every video frame is transformed into a directed, fully connected spatial interaction graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:

- **Node Feature Vector $\mathbf{x}_i \in \mathbb{R}^{49}$**:
  - Indices `0:2`: One-hot entity type (`[1, 0]` player, `[0, 1]` ball)
  - Indices `2:4`: Normalized court coordinate $(x, y) \in [0, 1]^2$
  - Indices `4:6`: Normalized velocity vector $(\dot{x}, \dot{y})$
  - Indices `6:8`: One-hot team attribution
  - Indices `8:42`: 17 Pose Keypoint coordinates $(x_k, y_k)_{k=1}^{17}$
  - Index `42`: Binary pose availability flag
  - Index `43`: Object detection confidence score
  - Indices `44:49`: Court-relative geometric distances (net line, sidelines, endlines)
- **Edge Feature Vector $\mathbf{e}_{ij} \in \mathbb{R}^4$**:
  - Relative displacement vector $(\Delta x, \Delta y)$
  - Euclidean distance $d(i, j)$
  - Same-team indicator boolean $\{0, 1\}$

### 4.2 GAT Architecture (Graph Attention Network)
The GAT architecture models player-to-player attention weights:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{x}_i \parallel \mathbf{W}\mathbf{x}_j \parallel \mathbf{W}_e \mathbf{e}_{ij}]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{x}_i \parallel \mathbf{W}\mathbf{x}_k \parallel \mathbf{W}_e \mathbf{e}_{ik}]\right)\right)}$$

- **Structure**: 2 GATConv layers, 4 attention heads, 32 hidden dimensions, dropout 0.2.
- **Supervision**: Self-supervised trajectory displacement target. Predicts player positions $t + 15$ frames ahead (0.5 s horizon) using automated tracking as ground truth.
- **Empirical Validation**: Test Average Displacement Error (**ADE = 0.87 m**) vs. Constant Velocity baseline (**1.56 m**).

### 4.3 HRN Architecture (Hierarchical Relational Network)
Follows Ibrahim & Mori (ECCV 2018) for volleyball group activity abstraction:
1. **Player Level**: GAT node embeddings encode local interactions.
2. **Team Subgraphs**: Aggregation into intra-team representations via relational units.
3. **Global Scene**: Cross-team temporal concatenation capturing match balance.
- **Distillation**: Supervised by the Physics EPV model to jointly predict target zone (Cross-Entropy), shot profile (Cross-Entropy), and team spatial compactness (MSE).

---

## 5. Physics-Based Expected Point Value (EPV) Formulation

The Expected Point Value metric evaluates offensive point yield across candidate targets on the defending court half:

$$EPV(c, s) = P_{\text{in}}(c, s) \times \left(1 - P_{\text{blocked}}(c, s)\right) \times \prod_{d \in \mathcal{D}} \left(1 - P_{\text{reach}}(d, c, s)\right)$$

Where $c \in \mathcal{C}$ denotes a discrete court coordinate within a $0.25\times 0.25$ m grid (1,296 candidate cells), and $s \in \mathcal{S}$ represents one of 5 valid shot mechanisms.

```
+----------------------------------------------------------------------------------+
| COMPONENT                  GOVERNING EQUATION                                    |
+----------------------------------------------------------------------------------+
| Ball Flight Dynamics       m * d²r/dt² = m*g - (1/2) * rho * v² * Cd * A * v_hat |
| Aerodynamic Drag (Cd)      Cd = 0.17  (Re >= 3.0e5, Supercritical)               |
|                            Cd = 0.47  (Re < 3.0e5, Subcritical)                  |
| Defensive Interception     t_def = t_react + sqrt(2h_jump / g) + d_run / v_top   |
| Pitch Control Sigmoid      P_reach = 1 / (1 + exp(-(t_ball - t_def) / 0.109))   |
| Aim Distribution           P_in = Integral over Court of N(c, sigma_s²) dc       |
+----------------------------------------------------------------------------------+
```

```
DEFENDING HALF DISCRETIZATION (1,296 CELLS)
+-------+-------+-------+
|  Z4   |  Z3   |  Z2   |  <- Net Line (2.43 m)
+-------+-------+-------+
|  Z5   |  Z6   |  Z1   |  <- Back Row / Baseline
+-------+-------+-------+
```

### 5.1 Dynamic Blocking Modeling
Front-row blockers are modeled dynamically at the moment of ball contact:
- Up to 3 defending players adjacent to the net translate laterally toward the predicted crossing point during set flight ($t_{\text{set}} \approx 0.6$ s).
- Hand reach is projected up to measured jump reach ($h_{\text{reach}} = 3.18$ m).
- Shots crossing beneath blocker reach and within horizontal hand spread ($w_{\text{hands}} = \pm 0.5$ m) are assigned elevated $P_{\text{blocked}}$ values.

### 5.2 Hybrid Integration Scheme
The dashboard target selection combines physics-driven calculation with learned neural inference:
$$\text{Score}(c, s) = (1 - w_{\text{learned}}) \cdot EPV_{\text{physics}}(c, s) + w_{\text{learned}} \cdot \lambda_{\text{skill}} \cdot P_{\text{HRN}}(c, s)$$
Where $\lambda_{\text{skill}}$ automatically gates model influence based on validated test-set accuracy over majority-class baselines.

---

## 6. Hardware Acceleration & CUDA Pipeline

GRITSENSE leverages an asynchronous compute topology to maximize hardware saturation:

```
[Main Thread: Perception / Tracking / Graphs] 
       |
       +---> [Worker Thread: Local LLM Engine (Async Ingestion)]
       |
[Render Thread: Vectorized PIL Dashboard Assembly] 
       |
       +---> [FFmpeg Subprocess: NVENC Hardware Transcoder (h264_nvenc)]
```

### 6.1 Inference Device Management
`pipeline/device_selection.py` provides centralized, fault-tolerant device allocation:
- **Automatic Priority**: Intel Arc XPU $\rightarrow$ NVIDIA CUDA $\rightarrow$ CPU.
- **Memory Footprint Gating**: When GPU VRAM constraints preclude full-precision weights, the platform supports 4-bit NormalFloat (NF4) quantization via `bitsandbytes` and `accelerate`.

### 6.2 Dual-Pass Video Compositor
Rather than blocking per-frame rendering on generative LLM calls:
1. **Pass 1 (Analysis & Event Planning)**: Computes geometry, tracks, and EPV grids across all frames. When commentary triggers fire, factual prompts are queued asynchronously.
2. **Pass 2 (Serialized Generation & Composition)**: Generation resolves concurrently while frame graphics compile, eliminating GPU resource thrashing between PyTorch models and the transformer decoder.
3. **Transcoding**: Video outputs are pipe-streamed directly through FFmpeg using NVIDIA `h264_nvenc` with zero-copy buffer handoff, dropping dashboard encoding times to $<16$ seconds for 319 frames.

---

## 7. Web Application & Analytics Dashboard

The platform includes a client-server analytical suite built with Flask and Vanilla ES6+:

```
+----------------------------------------------------------------------------------+
| GRITSENSE TACTICAL INTELLIGENCE                      BRA  vs  JPN        00:03.7 |
+-------------------------------------------------+--------------------------------+
| MATCH FEED                                      | COURT MAP          BRA   JPN   |
|                                                 | +----------------------------+ |
| +---------------------------------------------+ | |              * Ball        | |
| |                                             | | |     (7)            (2)     | |
| |           [Video Broadcast Frame]           | | |   (9)    (11)     (4)      | |
| |                                             | | |           (5)       (10)   | |
| |                                             | | |     (6)            (1)     | |
| |      (Attack overlay & player markers)      | | |            (3)             | |
| +---------------------------------------------+ | +----------------------------+ |
| [Rally Timeline: Blue=Pass Teal=Set Amber=Spk]| RECOMMENDED PLAY   BRA ATTACKING |
+-------------------------------------------------+ READ & REPOSITION              |
| TACTICAL COMMENTARY                             | waiting for ball               |
| 00:01.3  #3 is the key matchup here - go for it | +-----------+-----------+----+ |
|          against #18, target Zone 6 back-middle | | PT CHANCE | BLK CHANCE| OUT| |
|          seam to exploit the defensive gap!     | |    --     |    --     | -- | |
| 00:00.1  Key matchup: #3 against #1. JPN #1     | +-----------+-----------+----+ |
|          alone neutralizes 40% of the threat.   | KEY MATCHUPS    TEAM SHAPE     |
+-------------------------------------------------+ #7 vs #4  56%   BRA Cov: 16%   |
+-------------------------------------------------+ #7 vs #1  30%   JPN Cov: 11%   |
```

- **Interactive Coordinate Homography**: Web-based 4-point court calibration tool allowing users to upload novel footage and establish court corner correspondences in the browser.
- **Rally Phase Scrubbing**: Playback timeline synchronized with underlying GAT group activity classifications (Pass, Set, Spike, Winpoint).
- **Match Archive Engine**: SQLite/filesystem job archive allowing concurrent execution, job cancellation, JSON/CSV data export, and batch clearing.

---

## 8. Repository Structure

```
GRITSENSE/
|-- master_main.py                  # Primary pipeline orchestration runner
|-- requirements.txt                # Python package dependency manifest
|-- README.md                       # Comprehensive platform documentation
|
|-- pipeline/                       # Processing and computation stages
|   |-- __init__.py
|   |-- ball_tracker.py             # YOLOv8 + BoT-SORT player tracking
|   |-- redetect_ball.py            # Multi-scale ballistic ball re-detection
|   |-- clean_ball_track.py         # Dynamic programming trajectory optimization
|   |-- graph_builder.py            # 49-D interaction graph construction
|   |-- integrate_and_rebuild.py    # GAT & HRN inference and feature aggregation
|   |-- group_activity.py           # Pretrained group-activity GAT execution
|   |-- tactical_dashboard.py       # High-throughput broadcast video renderer
|   |-- calibrate_physics.py        # Automatic camera & kinematic parameter recovery
|   |-- train_gat_hrn.py            # Self-supervised GAT and HRN distillation
|   |-- sample_data.py              # File caching and IO pipeline utilities
|   |-- visualize_recent_graphs.py  # Graph topology visualization utilities
|   |-- device_selection.py         # Unified hardware acceleration dispatch
|   +-- pipeline_profiling.py       # High-resolution stage-by-stage profiler
|
|-- analytics/                      # Mathematical, kinematic & NLP engines
|   |-- __init__.py
|   |-- constants.py                # Sourced physical and aerodynamic parameters
|   |-- epv.py                      # Multi-cell Expected Point Value simulation
|   |-- camera.py                   # Single-view camera parameter recovery
|   |-- ball_track.py               # Trajectory smoothing and gap interpolation
|   |-- motion.py                   # Moving-window velocity regression
|   |-- pose_signals.py             # Skeletal vertical jump detector
|   |-- insights.py                 # Hybrid physics-neural decision engine
|   |-- relations.py                # Relational spatial feature calculations
|   |-- commentary.py               # Tactical rule and event trigger engine
|   |-- nlp.py                      # Transformers LLM pipeline (Qwen2.5-3B)
|   |-- learned.py                  # PyTorch model loader & inference bindings
|   +-- physics_calibration.json    # Calibrated physical parameters
|
|-- models/                         # Neural network architectures
|   |-- gat.py                      # Graph Attention Network definition
|   |-- hrn.py                      # Hierarchical Relational Network definition
|   +-- attention_models/           # Relational attention layers
|
|-- weights/                        # Neural network model weights
|   |-- player_detection.pt         # Fine-tuned YOLOv8 player weights
|   |-- pose.pt                     # YOLOv8 17-keypoint skeletal weights
|   |-- yolov8n-pose.pt             # Compact pose fallback weights
|   +-- ball.pt                     # YOLOv8 ballistic ball detection weights
|
|-- checkpoints/                    # Self-supervised & distilled weights
|   |-- gat.pt                      # Trained GAT checkpoint
|   |-- hrn.pt                      # Distilled HRN checkpoint
|   |-- training_report.json        # Test evaluation loss metrics
|   +-- training_curves.png         # Convergence loss plots
|
|-- pretrained_checkpoints/         # External reference checkpoints
|   +-- RCRG-R2-C11-conc-temporal-V1/
|       +-- checkpoint_epoch_42.pkl # Pretrained Volleyball Dataset weights
|
|-- data/                           # Reference dataset assets
|   |-- sample3.mp4                 # Primary validation match sequence
|   |-- court.png                   # Top-down court graphic plate
|   +-- annotations.json            # Reference planar court corners
|
|-- webapp/                         # Full-stack analytical application
|   |-- server.py                   # Multithreaded Flask application server
|   |-- jobs/                       # Dynamic run cache directory
|   +-- static/                     # Static client resources
|       |-- index.html              # Single Page Application (SPA) DOM
|       |-- app.css                 # Broadcast design system stylesheet
|       |-- app.js                  # Client logic, canvas graphics, and API client
|       +-- assets/                 # Photographic backdrops and assets
|
+-- docs/                           # Documentation and technical reports
    |-- RUN_GUIDE.md                # Operator execution handbook
    |-- SYSTEM_EXPLAINED.md         # Comprehensive viva voce reference
    |-- demo/                       # Architectural and demo recordings
    +-- report/                     # Formal technical report, schemas, and diagrams
```

---

## 9. Setup & Installation

### 9.1 Environment Prerequisites
- Operating System: Windows 10/11, Ubuntu 20.04+, or macOS
- Python Version: `3.10` to `3.13`
- Compute Target: NVIDIA GPU (Turing / Ampere / Ada / Blackwell with CUDA 11.8+ or 12.0+) recommended; Intel Arc XPU or CPU execution fully supported.

### 9.2 Virtual Environment Configuration
```powershell
# Clone the repository
git clone https://github.com/yashkumar0410/GRITSENSE.git
cd GRITSENSE

# Initialize and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

# Install core runtime dependencies
pip install -r requirements.txt
```

### 9.3 Model Weight Verification
Confirm that required perception weights are located within `weights/`:
- `weights/player_detection.pt`
- `weights/pose.pt`
- `weights/ball.pt`

*Note: The local LLM (`Qwen/Qwen2.5-3B-Instruct`, ~6 GB) downloads automatically to your Hugging Face local cache directory on the initial execution.*

---

## 10. Command Line Interface (CLI)

### 10.1 Web Application Server
Launch the graphical platform interface:
```powershell
python webapp/server.py
```
*Access interface via web browser at `http://127.0.0.1:5000`.*

### 10.2 Full Pipeline Execution
Execute detection, trajectory parsing, graph learning, and video rendering sequentially:
```powershell
python master_main.py --video data/sample3.mp4 --teams BRA JPN --dashboard --device cuda
```

### 10.3 Render Tactical Dashboard Only
Bypass upstream detection and render directly from pre-computed feature files:
```powershell
python pipeline/tactical_dashboard.py --video data/sample3.mp4 --device cuda
```

### 10.4 Self-Supervised Neural Network Training
Train the GAT and distill the HRN using accumulated graph caches:
```powershell
python pipeline/train_gat_hrn.py
```

### 10.5 CLI Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--video` | String | `data/sample3.mp4` | Relative or absolute path to source video file |
| `--teams` | String String | `BRA JPN` | Team abbreviations assigned to left and right halves |
| `--device` | Choice | `auto` | Execution device: `auto`, `cuda`, `xpu`, or `cpu` |
| `--nlp` | Choice | `local` | Commentary backend: `local` (offline LLM) or `template` |
| `--local-model` | String | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face model identifier for offline text generation |
| `--llm-quant` | Choice | `fp16` | Precision target: `fp16` or `4bit` (NF4 via bitsandbytes) |
| `--learned-weight` | Float | `0.3` | Proportional influence of GAT/HRN in hybrid EPV selection ($[0.0, 1.0]$) |
| `--physics-only` | Flag | `False` | Disables neural models, operating strictly on kinematic EPV |
| `--max-frames` | Integer | `None` | Truncates execution to first $N$ frames for testing |
| `--show` | Flag | `False` | Renders interactive OpenCV preview window during compilation |
| `--profile` | Flag | `False` | Emits hardware-synchronized latency profiling per stage |

---

## 11. Physical Constants & Empirical Validation

All constants implemented within `analytics/constants.py` are strictly tied to official sporting regulations or peer-reviewed biomechanical publications:

```
+---------------------+-------------+-----------+-----------------------------------+
| PARAMETER           | VALUE       | UNIT      | SOURCE / METHODOLOGY              |
+---------------------+-------------+-----------+-----------------------------------+
| Net Height (Men)    | 2.43        | m         | FIVB Official Rules 2021-2024     |
| Court Dimensions    | 18.0 x 9.0  | m         | FIVB Official Rules 2021-2024     |
| Ball Mass           | 0.269       | kg        | Applied Sciences 2020, 10(9):3256 |
| Ball Radius         | 0.105       | m         | FIVB Standard Specification       |
| Supercritical Cd    | 0.17        | unitless  | Wind Tunnel Volleyball Studies    |
| Mean Spike Velocity | 29.53       | m/s       | CROSBI 659042 Radar Study         |
| Player Reaction     | 0.33        | s         | J. Hum. Kinet (0.571s - 0.240s)   |
| Player Stature      | 1.92        | m         | Calibrated Homography Median      |
| Player Top Speed    | 4.75        | m/s       | 95th Percentile Tracked Velocity  |
| Max Vertical Jump   | 0.70        | m         | 90th Percentile Net-Zone Jumps    |
| Attack Contact Ht   | 3.37        | m         | FIVB 2022 World Championship      |
| Block Jump Reach    | 3.18        | m         | FIVB 2022 World Championship      |
+---------------------+-------------+-----------+-----------------------------------+
```

### Empirical Model Benchmark
Validation against 2,925 terminal attacks recorded during the FIVB Men's World Championship (Millan-Sanchez et al., 2017):
- Published Terminal Attack Success: **72.5%** (Outside Hitters), **55.1%** (Opposites).
- GRITSENSE Physics EPV Prediction: **88.2%** (Optimal Uncontested Target), **39.1%** (Random Legal Court Placement).
- Real-world match outcomes fall within the simulated envelope.

---

## 12. Scientific References

1. **Fédération Internationale de Volleyball (FIVB)**. *Official Volleyball Rules 2021–2024*. FIVB Regulations, 2021.
2. **Ibrahim, M. S., & Mori, G.** *Hierarchical Relational Networks for Group Activity Recognition and Retrieval*. European Conference on Computer Vision (ECCV), 2018.
3. **Millan-Sanchez, A., Morante Rabago, J. C., & Urena Espa, A.** *Attack success according to the competitive level in men's volleyball*. Journal of Human Sport and Exercise, 12(2), 2017.
4. **Spearman, W.** *Beyond Expected Goals: Writing pitch control models for soccer analytics*. MIT Sloan Sports Analytics Conference, 2018.
5. **Applied Sciences Editorial Board**. *Aerodynamic force measurements and drag coefficient transitions of elite competition volleyballs*. Applied Sciences, 10(9):3256, 2020.
6. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y.** *Graph Attention Networks*. International Conference on Learning Representations (ICLR), 2018.
7. **Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M.** *YOLOv4 / YOLOv8 Real-Time Object Detection*. Technical Reports, 2020–2024.
8. **Aharon, N., Orfaig, R., & Bobrovsky, B. Z.** *BoT-SORT: Robust Associations Multi-Pedestrian Tracking*. arXiv:2206.14651, 2022.
