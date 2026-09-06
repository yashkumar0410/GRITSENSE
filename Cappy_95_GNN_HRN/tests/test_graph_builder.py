import math

import numpy as np
import torch

from data.sample_data import create_sample_data

from graph.graph_builder import (
    NODE_FEATURE_DIM,
    PLAYER_BALL_EDGE_TYPE,
    PLAYER_PLAYER_EDGE_TYPE,
    VolleyballGraphBuilder,
)


# ============================================================
# FEATURE GRAPH TESTS
# ============================================================

def test_sample_players_and_ball_create_expected_nodes():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert graph.num_nodes == len(players) + 1


def test_node_feature_dimension_is_49():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert (
        graph.x.shape
        == (len(players) + 1, 49)
    )

    assert (
        graph.x.size(1)
        == NODE_FEATURE_DIM
    )


def test_all_node_features_are_finite():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert torch.isfinite(
        graph.x
    ).all()


def test_normalized_positions_are_valid():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert torch.all(
        graph.x[:, 2] >= 0
    )

    assert torch.all(
        graph.x[:, 2] <= 1
    )

    assert torch.all(
        graph.x[:, 3] >= 0
    )

    assert torch.all(
        graph.x[:, 3] <= 1
    )


def test_missing_pose_is_handled():

    players, ball, court = (
        create_sample_data()
    )

    players[0]["pose"] = None

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert (
        graph.x.shape[1]
        == 49
    )

    # Feature index 42 = pose availability
    assert (
        graph.x[0, 42].item()
        == 0.0
    )


def test_missing_velocity_is_handled():

    players, ball, court = (
        create_sample_data()
    )

    players[0].pop(
        "vx",
        None
    )

    players[0].pop(
        "vy",
        None
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert torch.isfinite(
        graph.x
    ).all()


# ============================================================
# INTERACTION GRAPH TESTS
# ============================================================

def test_players_create_expected_player_player_edges():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    player_player_edges = (
        graph.edge_type
        == PLAYER_PLAYER_EDGE_TYPE
    )

    assert (
        int(
            player_player_edges.sum()
        )
        == len(players) * (len(players) - 1)
    )


def test_players_create_expected_player_ball_edges():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    player_ball_edges = (
        graph.edge_type
        == PLAYER_BALL_EDGE_TYPE
    )

    assert (
        int(
            player_ball_edges.sum()
        )
        == 2 * len(players)
    )


def test_total_edge_count_matches_player_count():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert (
        graph.num_edges
        == len(players) * (len(players) - 1) + 2 * len(players)
    )


def test_edge_index_shape():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert (
        graph.edge_index.shape
        == (2, graph.num_edges)
    )


def test_edge_feature_shape():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert (
        graph.edge_attr.shape
        == (graph.num_edges, 4)
    )


def test_edge_attributes_are_finite():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert torch.isfinite(
        graph.edge_attr
    ).all()


def test_edge_indices_are_valid():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert torch.all(
        graph.edge_index >= 0
    )

    assert torch.all(
        graph.edge_index
        < graph.num_nodes
    )


def test_edge_distance_is_non_negative():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    assert torch.all(
        graph.edge_attr[:, 2]
        >= 0
    )


def test_reverse_player_edges_have_opposite_direction():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    for i in range(
        graph.edge_index.size(1)
    ):

        source = int(
            graph.edge_index[
                0,
                i
            ]
        )

        target = int(
            graph.edge_index[
                1,
                i
            ]
        )

        if (
            source >= len(players)
            or target >= len(players)
            or source == target
        ):
            continue

        for j in range(
            graph.edge_index.size(1)
        ):

            reverse_source = int(
                graph.edge_index[
                    0,
                    j
                ]
            )

            reverse_target = int(
                graph.edge_index[
                    1,
                    j
                ]
            )

            if (
                reverse_source == target
                and reverse_target == source
            ):

                assert torch.allclose(
                    graph.edge_attr[
                        i,
                        :2
                    ],
                    -graph.edge_attr[
                        j,
                        :2
                    ]
                )

                break


def test_same_team_edge_feature():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court
        )
    )

    player_player_mask = (
        graph.edge_type
        == PLAYER_PLAYER_EDGE_TYPE
    )

    same_team_values = (
        graph.edge_attr[
            player_player_mask,
            3
        ]
    )

    assert torch.any(
        same_team_values == 1.0
    )

    assert torch.all(
        same_team_values >= 0
    )

    assert torch.all(
        same_team_values <= 1
    )