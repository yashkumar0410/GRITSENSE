#!/usr/bin/env python
"""Test the data loader."""

from data.ball_player_loader import BallPlayerDataLoader

loader = BallPlayerDataLoader(
    ball_csv_path='ball detection/output_test/sample3/ball.csv',
    player_json_path='player_keypoints.json'
)

stats = loader.get_statistics()
print('Statistics:')
for k, v in stats.items():
    print(f'  {k}: {v}')

# Get a sample frame
frame_data = loader.get_frame(0)
if frame_data:
    players, ball = frame_data
    print(f'\nFrame 0:')
    print(f'  Players: {len(players)}')
    print(f'  Ball detected: {ball["detected"]}')
    if len(players) > 0:
        print(f'  First player: id={players[0]["id"]}, x={players[0]["x"]}, y={players[0]["y"]}')
else:
    print("Frame 0 not available")
