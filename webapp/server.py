"""
GRITSENSE web app: upload a match video, run the pipeline, explore the dashboard.

    python webapp/server.py            # then open http://127.0.0.1:5000

Each run is a job in webapp/jobs/<id>/ (input, browser-ready dashboard video,
commentary, per-frame insights, snapshots). Jobs run one at a time because the
pipeline writes its detection files into the project root.

Modes
    demo  - render the dashboard from the bundled sample's detections (minutes)
    full  - detection -> ball cleaning -> graphs -> dashboard on a new video
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
JOBS = WEB / "jobs"
SAMPLE_BACKUP = WEB / "sample_backup"
SAMPLE_VIDEO = ROOT / "sample3.mp4"
sys.path.insert(0, str(ROOT))
# Detection files the pipeline overwrites in the project root.
SAMPLE_FILES = ("player_keypoints_mf3_pose.json", "ball.csv", "ball_clean.csv", "ball_candidates.json")
# Court keypoint detection is unreliable, so every run uses the hand-made per-frame
# court corners in the project root as-is. The web app never writes this file.
ANNOTATIONS = ROOT / "annotations.json"
TARGET_W, TARGET_H = 1920, 1080  # the camera model and court corners assume 1080p

JOBS.mkdir(exist_ok=True)
app = Flask(__name__, static_folder=str(WEB / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_procs: dict[str, subprocess.Popen] = {}


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------------------
# Job state
# ----------------------------------------------------------------------------

def _save(job: dict) -> None:
    path = JOBS / job["id"] / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    tmp.replace(path)


def _update(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(fields)
        _save(job)


def _log(job_id: str, line: str) -> None:
    with _lock:
        job = _jobs[job_id]
        job["log"].append(line)
        job["log"] = job["log"][-400:]


def _load_jobs() -> None:
    for meta in JOBS.glob("*/job.json"):
        try:
            job = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("status") in ("queued", "running"):
            job["status"], job["error"] = "failed", "Server restarted while this job was running."
        _jobs[job["id"]] = job


def _public(job: dict, with_log: bool = False) -> dict:
    out = {k: v for k, v in job.items() if k != "log"}
    if with_log:
        out["log"] = job["log"][-120:]
    return out


# ----------------------------------------------------------------------------
# Video helpers (ffmpeg ships with imageio-ffmpeg, so nothing to install by hand)
# ----------------------------------------------------------------------------

def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return info


def _transcode(src: Path, dst: Path, scale: bool = False) -> None:
    """H.264 + faststart so browsers can play and seek; optionally force 1080p."""
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src)]
    if scale:
        cmd += ["-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
                       f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2"]
    cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


# ----------------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------------

# (pattern, stage key, progress floor, progress ceiling) for full runs.
_FULL_STAGES = [
    ("STAGE 1 —", "detect", 0.04, 0.45),
    ("STAGE 1b", "ball", 0.45, 0.56),
    ("STAGE 2", "graphs", 0.56, 0.66),
    ("STAGE 3", "graphs", 0.66, 0.70),
]
_PROCESSED = re.compile(r"processed (\d+)/(\d+)")
_RENDERED = re.compile(r"rendered (\d+)/(\d+)")
_BALL_FRAMES = re.compile(r"^\s+(\d+) frames$")


def _ensure_sample_backup() -> None:
    """Keep a copy of the sample's detections so demo runs stay consistent after full runs."""
    if SAMPLE_BACKUP.exists():
        return
    SAMPLE_BACKUP.mkdir()
    for name in SAMPLE_FILES:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, SAMPLE_BACKUP / name)


def _restore_sample() -> None:
    for name in SAMPLE_FILES:
        if (SAMPLE_BACKUP / name).exists():
            shutil.copy2(SAMPLE_BACKUP / name, ROOT / name)


def _run(job_id: str, cmd: list[str], stages: list | None, base: tuple[float, float]) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    _procs[job_id] = proc
    lo, hi = base
    total = _jobs[job_id].get("frames") or 1
    for raw in proc.stdout:
        line = raw.rstrip()
        if not line or "Loading weights" in line:
            continue
        _log(job_id, line)
        for pattern, key, s_lo, s_hi in stages or []:
            if pattern in line:
                lo, hi = s_lo, s_hi
                _update(job_id, stage=key, progress=lo)
        m = _PROCESSED.search(line) or _RENDERED.search(line)
        if m and int(m.group(2)):
            frac = int(m.group(1)) / int(m.group(2))
            _update(job_id, progress=round(lo + (hi - lo) * min(frac, 1.0), 4))
        m = _BALL_FRAMES.match(line)
        if m:
            _update(job_id, progress=round(lo + (hi - lo) * min(int(m.group(1)) / total, 1.0), 4))
        if "Commentary engine:" in line:
            _update(job_id, engine=line.split(":", 1)[1].strip())
    code = proc.wait()
    _procs.pop(job_id, None)
    if _jobs[job_id].get("cancel"):
        raise Cancelled()
    if code != 0:
        tail = next((l for l in reversed(_jobs[job_id]["log"]) if l.strip()), "unknown error")
        raise RuntimeError(f"{Path(cmd[2]).name} exited with code {proc.returncode}: {tail}")


def _parse_commentary(path: Path) -> list[dict]:
    lines = []
    pattern = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\] \((\w+), ([^)]+)\) (.*)$")
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            m = pattern.match(raw)
            if m:
                lines.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "kind": m.group(3),
                              "source": m.group(4), "text": m.group(5)})
    return lines


def _process(job_id: str) -> None:
    job = _jobs[job_id]
    folder = JOBS / job_id
    out_dir = folder / "dashboard"
    py = sys.executable
    _update(job_id, status="running", started=time.time(), stage="prepare", progress=0.01)

    if job["mode"] == "demo":
        _restore_sample()
        video = SAMPLE_VIDEO
        _update(job_id, frames=_probe(video)["frames"])
        base = (0.03, 0.95)
    else:
        _ensure_sample_backup()
        src = folder / job["input_name"]
        info = _probe(src)
        video = folder / "video.mp4"
        _log(job_id, f"Input: {info['width']}x{info['height']} @ {info['fps']:.1f} fps, {info['frames']} frames")
        if (info["width"], info["height"]) != (TARGET_W, TARGET_H):
            _log(job_id, f"Rescaling to {TARGET_W}x{TARGET_H} (the court model expects 1080p)")
        _transcode(src, video, scale=True)
        frames = _probe(video)["frames"] or info["frames"]
        _update(job_id, frames=frames, progress=0.03)

        annotated = len(json.loads(ANNOTATIONS.read_text(encoding="utf-8")))
        _log(job_id, f"Court corners from annotations.json ({annotated} frames annotated)")
        if frames > annotated:
            _log(job_id, f"WARNING: video has {frames} frames; frames past {annotated} have no court mapping")

        _run(job_id, [py, "-u", "master_main.py", "--video", str(video), "--teams", *job["teams"]],
             _FULL_STAGES, (0.04, 0.45))
        base = (0.70, 0.96)

    _update(job_id, stage="dashboard", progress=base[0])
    cmd = [py, "-u", "tactical_dashboard.py", "--video", str(video), "--out", str(out_dir),
           "--teams", *job["teams"], "--nlp", job["nlp"]]
    if job.get("max_frames"):
        cmd += ["--max-frames", str(job["max_frames"])]
    _log(job_id, "=== DASHBOARD ===")
    _run(job_id, cmd, None, base)

    _update(job_id, stage="finalize", progress=0.97)
    _transcode(out_dir / "gritsense_dashboard.mp4", folder / "dashboard.mp4")
    commentary = _parse_commentary(out_dir / "commentary.txt")
    (folder / "commentary.json").write_text(json.dumps(commentary), encoding="utf-8")
    snapshots = sorted(p.name for p in out_dir.glob("dashboard_*.png"))
    fps = _probe(folder / "dashboard.mp4")["fps"]
    _update(job_id, status="done", stage="done", progress=1.0, finished=time.time(),
            fps=fps, snapshots=snapshots, commentary_count=len(commentary))


def _worker() -> None:
    while True:
        job_id = _queue.get()
        try:
            if _jobs.get(job_id, {}).get("cancel"):
                continue
            _process(job_id)
        except Cancelled:
            _log(job_id, "Cancelled by user")
            _update(job_id, status="failed", error="Cancelled by user", finished=time.time())
        except Exception as exc:
            _log(job_id, f"ERROR: {type(exc).__name__}: {exc}")
            _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500],
                    finished=time.time())
        finally:
            _queue.task_done()


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(WEB / "static", "index.html")


_system_cache: dict = {}


@app.get("/api/system")
def system():
    if not _system_cache:
        try:
            import torch

            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            gpu = None
        from analytics.nlp import DEFAULT_LOCAL_MODEL

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(DEFAULT_LOCAL_MODEL, local_files_only=True)
            cached = True
        except Exception:
            cached = False
        _system_cache.update(gpu=gpu, model=DEFAULT_LOCAL_MODEL, model_cached=cached,
                             sample=SAMPLE_VIDEO.exists())
    busy = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))
    return jsonify({**_system_cache, "busy": busy})


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)
    return jsonify([_public(j) for j in jobs])


@app.post("/api/jobs")
def create_job():
    form = request.form
    mode = form.get("mode", "full")
    if mode not in ("full", "demo"):
        abort(400, "mode must be full or demo")
    teams = [form.get("team_left", "BRA").strip()[:4].upper() or "BRA",
             form.get("team_right", "JPN").strip()[:4].upper() or "JPN"]
    nlp = form.get("nlp", "local") if form.get("nlp") in ("local", "template") else "local"
    max_frames = int(form["max_frames"]) if form.get("max_frames", "").isdigit() else None

    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    folder = JOBS / job_id
    folder.mkdir()
    job = {"id": job_id, "mode": mode, "teams": teams, "nlp": nlp, "max_frames": max_frames,
           "status": "queued", "stage": "queued", "progress": 0.0, "created": time.time(),
           "title": form.get("title", "").strip()[:80], "log": []}

    if not ANNOTATIONS.exists():
        abort(500, f"Missing {ANNOTATIONS.name} in the project root (court corners per frame)")

    if mode == "full":
        video = request.files.get("video")
        if not video or not video.filename:
            shutil.rmtree(folder)
            abort(400, "Upload a video for a full analysis")
        suffix = Path(video.filename).suffix.lower() or ".mp4"
        job["input_name"] = "input" + suffix
        job["title"] = job["title"] or Path(video.filename).stem
        video.save(folder / job["input_name"])
    else:
        job["title"] = job["title"] or "Sample match (demo)"

    with _lock:
        _jobs[job_id] = job
        _save(job)
    _queue.put(job_id)
    return jsonify(_public(job)), 201


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    job = _jobs.get(job_id) or abort(404)
    return jsonify(_public(job, with_log=True))


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    job = _jobs.get(job_id) or abort(404)
    if job["status"] in ("queued", "running"):
        abort(409, "Job is still running")
    with _lock:
        _jobs.pop(job_id)
    shutil.rmtree(JOBS / job_id, ignore_errors=True)
    return "", 204


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    job = _jobs.get(job_id) or abort(404)
    if job["status"] not in ("queued", "running"):
        abort(409, "Job is not running")
    _update(job_id, cancel=True)
    if job["status"] == "queued":
        _update(job_id, status="failed", error="Cancelled by user", finished=time.time())
    proc = _procs.get(job_id)
    if proc and proc.poll() is None:
        proc.kill()
    return jsonify(_public(job))


@app.get("/api/jobs/<job_id>/insights")
def insights(job_id: str):
    path = JOBS / job_id / "dashboard" / "dashboard_insights.json"
    if not path.exists():
        abort(404)
    rows = json.loads(path.read_text(encoding="utf-8"))
    keep = ("frame", "attack_team", "target_zone", "epv", "block_prob", "gap_m", "move", "key_player",
            "matchups", "team_shape", "attack_chance", "phase", "group_activity")
    return jsonify([{k: r.get(k) for k in keep} for r in rows])


@app.get("/files/<job_id>/<path:name>")
def files(job_id: str, name: str):
    folder = JOBS / job_id
    if not folder.is_dir():
        abort(404)
    return send_from_directory(folder, name, as_attachment=request.args.get("download") == "1")


@app.get("/sample.mp4")
def sample_video():
    return send_from_directory(ROOT, SAMPLE_VIDEO.name)


_load_jobs()
threading.Thread(target=_worker, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"GRITSENSE web app -> http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
