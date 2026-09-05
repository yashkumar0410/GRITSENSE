"""
make_video_from_frames.py
-------------------------
Stitches frame images into an MP4 video file for main.py and the GritSense pipeline.
Supports combining an entire match folder (e.g. --clip "videos/0") into a long multi-minute video!
"""

import argparse
import glob
import os
import zipfile
import cv2
import numpy as np

def get_frame_num(filepath):
    base = os.path.basename(filepath)
    name_no_ext = os.path.splitext(base)[0]
    try:
        return int(name_no_ext)
    except ValueError:
        return filepath

def convert_zip_clip_to_video(zip_path, clip_pattern, out_path, fps=30):
    print(f"Opening ZIP archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        clean_pattern = clip_pattern.replace("\\", "/").strip("/")
        
        all_files = z.namelist()
        img_names = [
            f for f in all_files 
            if clean_pattern in f and (f.lower().endswith(".jpg") or f.lower().endswith(".png") or f.lower().endswith(".jpeg"))
        ]

        if not img_names:
            raise ValueError(f"No image files matching '{clip_pattern}' were found inside {zip_path}.\n"
                             f"Available sample paths in zip look like: {all_files[:5]}")

        # Deduplicate and sort frames chronologically
        seen_frame_nums = set()
        unique_img_names = []
        
        # Sort by numerical frame ID in filename
        sorted_raw = sorted(img_names, key=get_frame_num)
        for f in sorted_raw:
            f_num = get_frame_num(f)
            if isinstance(f_num, int):
                if f_num not in seen_frame_nums:
                    seen_frame_nums.add(f_num)
                    unique_img_names.append(f)
            else:
                unique_img_names.append(f)

        img_names = unique_img_names if unique_img_names else sorted_raw
        duration_sec = len(img_names) / fps
        print(f"Found {len(img_names)} unique frame images for '{clip_pattern}' (approx {duration_sec/60:.2f} minutes). Building video...")

        first_data = z.read(img_names[0])
        first_img = cv2.imdecode(np.frombuffer(first_data, np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = first_img.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        for idx, name in enumerate(img_names):
            img_bytes = z.read(name)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            out.write(frame)
            if (idx + 1) % 500 == 0 or idx + 1 == len(img_names):
                print(f"  Processed {idx + 1}/{len(img_names)} frames...")

        out.release()
        print(f"Successfully created long video: {out_path} ({len(img_names)} frames, {duration_sec/60:.2f} mins @ {fps}fps)")

def convert_frames_dir_to_video(frames_dir, out_path, fps=30):
    img_files = sorted(glob.glob(os.path.join(frames_dir, "**", "*.jpg"), recursive=True))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(frames_dir, "**", "*.png"), recursive=True))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(frames_dir, "**", "*.jpeg"), recursive=True))
        
    if not img_files:
        raise ValueError(f"No image files (.jpg, .png) found in: {frames_dir}")

    # Deduplicate and sort
    seen_nums = set()
    unique_files = []
    for f in sorted(img_files, key=get_frame_num):
        num = get_frame_num(f)
        if isinstance(num, int):
            if num not in seen_nums:
                seen_nums.add(num)
                unique_files.append(f)
        else:
            unique_files.append(f)

    img_files = unique_files if unique_files else img_files
    duration_sec = len(img_files) / fps
    print(f"Found {len(img_files)} frame images in directory (approx {duration_sec/60:.2f} mins). Building video...")

    first_img = cv2.imread(img_files[0])
    h, w, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for idx, img_path in enumerate(img_files):
        frame = cv2.imread(img_path)
        out.write(frame)
        if (idx + 1) % 500 == 0 or idx + 1 == len(img_files):
            print(f"  Processed {idx + 1}/{len(img_files)} frames...")

    out.release()
    print(f"Successfully created long video: {out_path} ({len(img_files)} frames, {duration_sec/60:.2f} mins @ {fps}fps)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert frame images to MP4 video")
    parser.add_argument("--zip-file", help="Path to volleyball_.zip or videos_sample.zip")
    parser.add_argument("--clip", help="Clip or match folder, e.g. 'videos/0' for entire match 0, or 'videos/0/3596' for clip 3596")
    parser.add_argument("--frames-dir", help="Path to extracted folder containing frame images")
    parser.add_argument("--out", default="volleyball_match.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    if args.zip_file and args.clip:
        convert_zip_clip_to_video(args.zip_file, args.clip, args.out, args.fps)
    elif args.frames_dir:
        convert_frames_dir_to_video(args.frames_dir, args.out, args.fps)
    else:
        print("Usage error: Please provide either --zip-file and --clip OR --frames-dir.")
