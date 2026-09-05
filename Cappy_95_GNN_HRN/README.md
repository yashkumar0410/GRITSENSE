# Cappy_95_GNN_HRN

## Problem
A volleyball frame is converted into a graph so that player-level interactions, ball dynamics, and court context can be processed by a graph neural network. The graph representation is the current Task 1 module and is intentionally independent of GAT/HRN implementation.

## Graph Definition
The graph is defined as:

G = (V, E)

where:
- V = player nodes + ball node
- E = directed player-player relationships + directed player-ball relationships

## Nodes
The graph always contains:
- player nodes for each detected athlete
- one ball node

For N players:
- nodes = N + 1

## Player Node Schema
Each player node is a 49-dimensional vector with the exact ordering below:

| Index | Feature | Dimension | Meaning |
|---|---|---:|---|
| 0-1 | node type | 2 | [1, 0] for player, [0, 1] for ball |
| 2-3 | position | 2 | normalized x/y coordinates |
| 4-5 | velocity | 2 | normalized vx/vy |
| 6-7 | team one-hot | 2 | team 0 or team 1 |
| 8-41 | pose | 34 | 17 keypoints x (x, y) |
| 42 | pose_available | 1 | 1.0 if pose is valid, otherwise 0.0 |
| 43 | confidence | 1 | detection confidence |
| 44-48 | court features | 5 | distance_left, distance_right, distance_top, distance_bottom, distance_net |

The pose is explicitly interpreted as 17 keypoints × (x, y) = 34 values.

## Ball Node Schema
The ball uses the same 49-dimensional feature layout:

| Index | Feature | Dimension | Meaning |
|---|---|---:|---|
| 0-1 | node type | 2 | [0, 1] for ball |
| 2-3 | position | 2 | normalized x/y |
| 4-5 | velocity | 2 | normalized vx/vy |
| 6-7 | team one-hot | 2 | zeros because ball is not a team |
| 8-41 | pose | 34 | zeros placeholder |
| 42 | pose_available | 1 | 0.0 by default |
| 43 | confidence | 1 | ball confidence |
| 44-48 | court features | 5 | distance_left, distance_right, distance_top, distance_bottom, distance_net |

## Edge Schema
Edge attributes are 4-dimensional and normalized:

- relative_x
- relative_y
- distance
- same_team

The graph supports two explicit edge types:

- PLAYER_PLAYER = 0
- PLAYER_BALL = 1

## Connectivity
For N players:
- player-player directed edges = N * (N - 1)
- player-ball directed edges = 2N
- total edges = N * (N - 1) + 2N

For the synthetic 6-player frame:
- player-player edges = 30
- player-ball edges = 12
- total edges = 42

## Coordinate Normalization
Image coordinates are normalized to the range [0, 1]:

x_norm = x / image_width
y_norm = y / image_height

Velocity is normalized similarly:

vx_norm = vx / image_width
vy_norm = vy / image_height

Spatial edge differences are also normalized by the court width and height before computing the distance.

## Missing Data Handling
Optional values are handled with explicit defaults:

- missing velocity => 0.0, 0.0
- missing pose => zeros(34), pose_available = 0.0
- missing confidence => 0.0
- missing team => [0.0, 0.0]

The graph builder does not fabricate a ball if the ball is missing; it raises a clear error unless a documented missing-ball strategy is introduced later.

## Example
For a synthetic frame with 6 players and 1 ball:

- nodes = 7
- edges = 42
- x shape = [7, 49]
- edge_index shape = [2, 42]
- edge_attr shape = [42, 4]
- edge_type shape = [42]

## Future Integration
This graph representation is designed to be consumed by:

Graph Representation -> GAT -> HRN

At this stage, the task is only to produce a valid graph representation. GAT, HRN, temporal modeling, and EPV are future modules and are not implemented here.

## Visualization
The repository includes a simple graph visualization helper to verify player nodes, ball node, and edge connectivity. It is intended for debugging and demonstration only.
