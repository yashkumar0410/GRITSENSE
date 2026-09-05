import torch

from data.sample_data import create_sample_data
from graph.graph_builder import VolleyballGraphBuilder
from models import HRN


def build_test_graph():

    players, ball, court = (
        create_sample_data()
    )

    graph = (
        VolleyballGraphBuilder()
        .build_graph(
            players,
            ball,
            court,
        )
    )

    return graph


def test_hrn_contains_gat():

    hrn = HRN()

    assert hasattr(
        hrn,
        "gat",
    )


def test_hrn_node_output_shape():

    graph = build_test_graph()

    hrn = HRN()

    output = hrn(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.node_type,
    )

    assert (
        output["node_embeddings"].shape
        == (7, 32)
    )


def test_hrn_team_output_shape():

    graph = build_test_graph()

    hrn = HRN()

    output = hrn(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.node_type,
    )

    assert (
        output["team_embeddings"].shape
        == (2, 32)
    )


def test_hrn_global_output_shape():

    graph = build_test_graph()

    hrn = HRN()

    output = hrn(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.node_type,
    )

    assert (
        output["global_embedding"].shape
        == (1, 32)
    )


def test_hrn_outputs_are_finite():

    graph = build_test_graph()

    hrn = HRN()

    output = hrn(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.node_type,
    )

    assert torch.isfinite(
        output["node_embeddings"]
    ).all()

    assert torch.isfinite(
        output["team_embeddings"]
    ).all()

    assert torch.isfinite(
        output["global_embedding"]
    ).all()


def test_gradient_flows_through_gat():

    graph = build_test_graph()

    hrn = HRN()

    output = hrn(
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.node_type,
    )

    loss = (
        output["global_embedding"]
        .sum()
    )

    loss.backward()

    gat_has_gradient = False

    for parameter in hrn.gat.parameters():

        if parameter.grad is not None:

            gat_has_gradient = True

            break

    assert gat_has_gradient