import cv2
from pathlib import Path
import argparse
import re


# ============================================================
# PNG -> MP4
# ============================================================

def get_frame_number(path):
    """
    Extract frame number from filename.

    Example:
        frame_0001_graph.png -> 1
        graph_frame_0020.png -> 20
    """

    numbers = re.findall(r"\d+", path.stem)

    if not numbers:
        return 999999999

    return int(numbers[-1])


def create_video(
    input_dir,
    output_file,
    fps
):

    input_dir = Path(input_dir)
    output_file = Path(output_file)

    # --------------------------------------------------------
    # Find PNGs
    # --------------------------------------------------------

    png_files = list(
        input_dir.glob("*.png")
    )

    if not png_files:

        raise FileNotFoundError(
            f"No PNG files found in:\n{input_dir}"
        )

    # --------------------------------------------------------
    # Sort by frame number
    # --------------------------------------------------------

    png_files.sort(
        key=get_frame_number
    )

    print("=" * 70)
    print("PNG GRAPH → MP4")
    print("=" * 70)

    print()
    print("Input folder:")
    print(input_dir)

    print()
    print("PNG files found:")
    print(len(png_files))

    print()
    print("First:")
    print(png_files[0].name)

    print()
    print("Last:")
    print(png_files[-1].name)

    # --------------------------------------------------------
    # Read first PNG
    # --------------------------------------------------------

    first = cv2.imread(
        str(png_files[0])
    )

    if first is None:

        raise RuntimeError(
            f"Could not read:\n{png_files[0]}"
        )

    height, width = first.shape[:2]

    print()
    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps}"
    )

    # --------------------------------------------------------
    # Create MP4 writer
    # --------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create MP4 video."
        )

    # --------------------------------------------------------
    # Add PNGs one by one
    # --------------------------------------------------------

    for i, png_file in enumerate(
        png_files
    ):

        frame = cv2.imread(
            str(png_file)
        )

        if frame is None:

            print(
                f"WARNING: skipping {png_file.name}"
            )

            continue

        # ----------------------------------------------------
        # Make sure dimensions match first image
        # ----------------------------------------------------

        if (
            frame.shape[1] != width
            or frame.shape[0] != height
        ):

            frame = cv2.resize(
                frame,
                (width, height)
            )

        writer.write(frame)

        if (
            i < 5
            or (i + 1) % 25 == 0
            or i == len(png_files) - 1
        ):

            print(
                f"[{i + 1}/{len(png_files)}] "
                f"{png_file.name}"
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    writer.release()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Video created:")
    print(output_file)

    print()
    print(
        f"Frames: {len(png_files)}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Approx duration: "
        f"{len(png_files) / fps:.2f} seconds"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert graph PNG frames directly "
            "into an MP4 video."
        )
    )

    parser.add_argument(
        "--input_dir",
        default="output_recent/feature_images",
        help="Folder containing graph PNGs"
    )

    parser.add_argument(
        "--output",
        default="output_recent/feature_video.mp4",
        help="Output MP4 path"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Video FPS"
    )

    args = parser.parse_args()

    create_video(
        input_dir=args.input_dir,
        output_file=args.output,
        fps=args.fps
    )


if __name__ == "__main__":
    main()