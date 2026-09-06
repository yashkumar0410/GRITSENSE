"""
Real Ball and Player Data Loader

Loads actual ball detection data from CSV and player keypoints from JSON.
This replaces the synthetic/sample data placeholder.
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BallPlayerDataLoader:
    """
    Loads actual ball coordinates and player keypoints for the GNN/HRN pipeline.
    
    Data sources:
    - Ball coordinates: ball.csv (from updated ball detection pipeline)
    - Player keypoints: player_keypoints.json
    """

    def __init__(
        self,
        ball_csv_path: str,
        player_json_path: str,
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        """
        Initialize the data loader.
        
        Args:
            ball_csv_path: Path to ball.csv containing ball detections
            player_json_path: Path to player_keypoints.json
            image_width: Image width for coordinate normalization
            image_height: Image height for coordinate normalization
        """
        self.ball_csv_path = Path(ball_csv_path)
        self.player_json_path = Path(player_json_path)
        self.image_width = image_width
        self.image_height = image_height

        if not self.ball_csv_path.exists():
            raise FileNotFoundError(
                f"Ball CSV not found: {self.ball_csv_path}"
            )
        if not self.player_json_path.exists():
            raise FileNotFoundError(
                f"Player JSON not found: {self.player_json_path}"
            )

        # Load data
        self.ball_data = self._load_ball_data()
        self.player_data = self._load_player_data()

        # Validate alignment
        self._validate_frame_alignment()

    def _load_ball_data(self) -> Dict[int, Dict]:
        """
        Load ball data from CSV.
        
        CSV format:
            Frame,Visibility,X,Y,Radius
            
        Missing detections have Visibility=0 and X=-1, Y=-1.
        """
        ball_data = {}

        with open(self.ball_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row["Frame"])
                visibility = int(row["Visibility"])
                x = float(row["X"])
                y = float(row["Y"])
                radius = float(row["Radius"])

                if visibility == 1 and x >= 0 and y >= 0:
                    # Valid detection
                    ball_data[frame_idx] = {
                        "x": x,
                        "y": y,
                        "radius": radius,
                        "confidence": 1.0,  # From detection
                        "detected": True,
                    }
                else:
                    # Missing detection
                    ball_data[frame_idx] = {
                        "x": None,
                        "y": None,
                        "radius": None,
                        "confidence": 0.0,
                        "detected": False,
                    }

        return ball_data

    def _load_player_data(self) -> Dict[int, List[Dict]]:
        """
        Load player keypoints from JSON.
        
        Expected format (from player_keypoints.json):
        {
            "frame_idx": {
                "player_id": {
                    "image_x": float,
                    "image_y": float,
                    "keypoints": [...],  # 17 keypoints
                    "pose_confidence": float,
                    ...
                }
            }
        }
        """
        with open(self.player_json_path, "r") as f:
            return json.load(f)

    def _validate_frame_alignment(self):
        """
        Verify that ball and player data are aligned.
        """
        ball_frames = set(self.ball_data.keys())
        player_frames = set(int(k) for k in self.player_data.keys())

        if ball_frames != player_frames:
            missing_in_ball = player_frames - ball_frames
            missing_in_player = ball_frames - player_frames
            if missing_in_ball:
                print(
                    f"WARNING: Frames in player data but not ball data: "
                    f"{sorted(list(missing_in_ball))[:10]}..."
                )
            if missing_in_player:
                print(
                    f"WARNING: Frames in ball data but not player data: "
                    f"{sorted(list(missing_in_player))[:10]}..."
                )

        self.common_frames = sorted(ball_frames & player_frames)
        print(
            f"Data alignment: {len(self.common_frames)} common frames "
            f"out of {len(ball_frames)} ball frames "
            f"and {len(player_frames)} player frames"
        )

    def get_ball(self, frame_idx: int) -> Optional[Dict]:
        """
        Get ball data for a specific frame.
        
        Returns:
            Dict with keys: x, y, radius, confidence, detected
            Or None if frame not available
        """
        if frame_idx not in self.ball_data:
            return None
        return self.ball_data[frame_idx]

    def get_players(self, frame_idx: int) -> List[Dict]:
        """
        Get player data for a specific frame.
        
        Returns:
            List of player dicts with normalized coordinates
        """
        if str(frame_idx) not in self.player_data:
            return []

        frame_players_raw = self.player_data[str(frame_idx)]
        players = []

        for player_id_str, player_info in frame_players_raw.items():
            player_id = int(player_id_str)

            # Extract keypoints
            keypoints = player_info.get("keypoints", [])
            if isinstance(keypoints, list):
                pose = np.asarray(keypoints, dtype=np.float32)
            else:
                pose = np.zeros(34, dtype=np.float32)

            # Extract position
            x = float(player_info.get("image_x", 0.0))
            y = float(player_info.get("image_y", 0.0))

            # Velocity (compute from temporal data if available)
            vx = float(player_info.get("vx", 0.0))
            vy = float(player_info.get("vy", 0.0))

            # Confidence
            confidence = float(player_info.get("pose_confidence", 0.0))

            # Team (not available in player_keypoints.json, will be assigned later)
            team = None

            # Compile player data
            player = {
                "id": player_id,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "team": team,
                "confidence": confidence,
                "pose": pose,
                "pose_available": len(keypoints) > 0,
                "pose_confidence": confidence,
            }

            players.append(player)

        return players

    def get_frame(self, frame_idx: int) -> Optional[Tuple[List[Dict], Dict]]:
        """
        Get both players and ball for a frame.
        
        Returns:
            Tuple of (players_list, ball_dict) or None if frame unavailable
        """
        if frame_idx not in self.common_frames:
            return None

        players = self.get_players(frame_idx)
        ball = self.get_ball(frame_idx)

        if not players or ball is None:
            return None

        return players, ball

    def get_frames_range(
        self, start_idx: int = 0, end_idx: Optional[int] = None
    ) -> List[Tuple[int, List[Dict], Dict]]:
        """
        Get multiple frames.
        
        Returns:
            List of tuples: (frame_idx, players_list, ball_dict)
        """
        if end_idx is None:
            end_idx = max(self.common_frames) + 1

        frames = []
        for frame_idx in range(start_idx, end_idx):
            if frame_idx not in self.common_frames:
                continue
            data = self.get_frame(frame_idx)
            if data is not None:
                players, ball = data
                frames.append((frame_idx, players, ball))

        return frames

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded data.
        """
        total_frames = len(self.common_frames)
        ball_detections = sum(
            1 for idx in self.common_frames
            if self.ball_data[idx]["detected"]
        )
        ball_missing = total_frames - ball_detections

        player_counts = []
        for frame_idx in self.common_frames:
            players = self.get_players(frame_idx)
            player_counts.append(len(players))

        return {
            "total_frames": total_frames,
            "ball_detected": ball_detections,
            "ball_missing": ball_missing,
            "ball_detection_rate": (
                ball_detections / total_frames if total_frames > 0 else 0.0
            ),
            "min_players": min(player_counts) if player_counts else 0,
            "max_players": max(player_counts) if player_counts else 0,
            "avg_players": (
                sum(player_counts) / len(player_counts)
                if player_counts
                else 0.0
            ),
        }

    def get_sequence(
        self, start_frame: int, length: int
    ) -> List[Tuple[int, List[Dict], Dict]]:
        """
        Get a contiguous sequence of frames.
        
        Returns:
            List of (frame_idx, players, ball) tuples
        """
        sequence = []
        for i in range(length):
            frame_idx = start_frame + i
            if frame_idx in self.common_frames:
                data = self.get_frame(frame_idx)
                if data is not None:
                    players, ball = data
                    sequence.append((frame_idx, players, ball))
        return sequence
