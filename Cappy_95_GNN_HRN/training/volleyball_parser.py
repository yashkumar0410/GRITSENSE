from pathlib import Path


class VolleyballDatasetParser:
    """
    Parses the original Volleyball Dataset annotations.txt files.

    Each annotation line has the form:

        frame.jpg
        group_activity
        x y width height action
        x y width height action
        ...

    There are 12 player annotations per frame.

    The group activity is the supervised target for Task 5.
    """

    CLASS_NAMES = [
        "r_set",
        "r_spike",
        "r_pass",
        "r_winpoint",
        "l_winpoint",
        "l_pass",
        "l_spike",
        "l_set",
    ]

    CLASS_TO_INDEX = {
        name: index
        for index, name in enumerate(CLASS_NAMES)
    }

    # Official split from Info.txt
    TRAIN_VIDEOS = {
        1, 3, 6, 7, 10, 13, 15, 16,
        18, 22, 23, 31, 32, 36, 38,
        39, 40, 41, 42, 48, 50, 52,
        53, 54,
    }

    VALIDATION_VIDEOS = {
        0, 2, 8, 12, 17, 19, 24, 26,
        27, 28, 30, 33, 46, 49, 51,
    }

    TEST_VIDEOS = {
        4, 5, 9, 11, 14, 20, 21, 25,
        29, 34, 35, 37, 43, 44, 45,
        47,
    }

    def __init__(self, dataset_root):

        self.dataset_root = Path(
            dataset_root
        )

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found:\n"
                f"{self.dataset_root}"
            )

    # ==========================================================
    # PARSE ONE LINE
    # ==========================================================

    @classmethod
    def parse_line(cls, line):

        parts = line.strip().split()

        if len(parts) < 7:
            return None

        frame_name = parts[0]

        group_activity = (
            parts[1]
            .lower()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if group_activity not in cls.CLASS_TO_INDEX:
            return None

        # ------------------------------------------------------
        # Player annotations
        # ------------------------------------------------------

        players = []

        values = parts[2:]

        # Each player = 5 values:
        #
        # x
        # y
        # width
        # height
        # action

        for i in range(
            0,
            len(values) - 4,
            5,
        ):

            try:

                x = float(values[i])
                y = float(values[i + 1])

                width = float(
                    values[i + 2]
                )

                height = float(
                    values[i + 3]
                )

                action = values[
                    i + 4
                ].lower()

            except ValueError:

                continue

            players.append(
                {
                    "id": len(players),

                    "x": (
                        x + width / 2.0
                    ),

                    "y": (
                        y + height / 2.0
                    ),

                    "xmin": x,
                    "ymin": y,

                    "xmax": (
                        x + width
                    ),

                    "ymax": (
                        y + height
                    ),

                    "width": width,
                    "height": height,

                    "action": action,

                    "grouping": 1,
                }
            )

        if not players:
            return None

        return {
            "frame_name": frame_name,

            "frame_id": int(
                Path(frame_name).stem
            ),

            "group_activity":
                group_activity,

            "label":
                cls.CLASS_TO_INDEX[
                    group_activity
                ],

            "players": players,
        }

    # ==========================================================
    # PARSE VIDEO
    # ==========================================================

    def parse_video(
        self,
        video_id,
    ):

        video_dir = (
            self.dataset_root
            / str(video_id)
        )

        annotation_file = (
            video_dir
            / "annotations.txt"
        )

        if not annotation_file.exists():

            raise FileNotFoundError(
                f"Missing annotations:\n"
                f"{annotation_file}"
            )

        samples = []

        with annotation_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                sample = self.parse_line(
                    line
                )

                if sample is None:
                    continue

                sample["video_id"] = (
                    video_id
                )

                samples.append(
                    sample
                )

        return samples

    # ==========================================================
    # PARSE SPLIT
    # ==========================================================

    def parse_split(
        self,
        split,
    ):

        split = split.lower()

        if split == "train":

            video_ids = sorted(
                self.TRAIN_VIDEOS
            )

        elif split in {
            "val",
            "validation",
        }:

            video_ids = sorted(
                self.VALIDATION_VIDEOS
            )

        elif split == "test":

            video_ids = sorted(
                self.TEST_VIDEOS
            )

        else:

            raise ValueError(
                "split must be "
                "'train', 'validation', "
                "or 'test'"
            )

        samples = []

        for video_id in video_ids:

            video_samples = (
                self.parse_video(
                    video_id
                )
            )

            samples.extend(
                video_samples
            )

        return samples

    # ==========================================================
    # STATISTICS
    # ==========================================================

    @classmethod
    def print_statistics(
        cls,
        samples,
        split_name,
    ):

        counts = {
            label: 0
            for label in cls.CLASS_NAMES
        }

        for sample in samples:

            label = sample[
                "group_activity"
            ]

            counts[label] += 1

        print(
            f"\n========== "
            f"{split_name.upper()} DATASET "
            f"=========="
        )

        print(
            "Samples:",
            len(samples),
        )

        for label in cls.CLASS_NAMES:

            print(
                f"{label:15s}: "
                f"{counts[label]}"
            )

        print(
            "================================"
        )


# ==============================================================
# TEST
# ==============================================================

if __name__ == "__main__":

    root = (
        r"C:\Users\vipra\Downloads"
        r"\volleyball_\videos"
    )

    parser = (
        VolleyballDatasetParser(root)
    )

    train_samples = (
        parser.parse_split(
            "train"
        )
    )

    validation_samples = (
        parser.parse_split(
            "validation"
        )
    )

    test_samples = (
        parser.parse_split(
            "test"
        )
    )

    parser.print_statistics(
        train_samples,
        "train",
    )

    parser.print_statistics(
        validation_samples,
        "validation",
    )

    parser.print_statistics(
        test_samples,
        "test",
    )

    if train_samples:

        print(
            "\n========== EXAMPLE =========="
        )

        print(
            train_samples[0]
        )

        print(
            "=============================="
        )