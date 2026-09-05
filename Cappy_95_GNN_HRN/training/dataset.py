import torch
from torch.utils.data import Dataset

from training.volleyball_parser import VolleyballDatasetParser
from graph.graph_builder import VolleyballGraphBuilder


class VolleyballGraphDataset(Dataset):
    """
    Converts Volleyball Dataset annotations into graphs
    for GAT + HRN training.

    Graph structure:

        6 players Team 0
        6 players Team 1
        1 ball

        = 13 nodes

    Each node has 49 features.
    """

    COURT_WIDTH = 1280.0
    COURT_HEIGHT = 720.0

    def __init__(
        self,
        samples,
        graph_builder=None,
    ):

        self.samples = samples

        if graph_builder is None:
            graph_builder = VolleyballGraphBuilder()

        self.graph_builder = graph_builder

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # TEAM ASSIGNMENT
    # ========================================================

    @staticmethod
    def assign_teams(players):
        """
        Assign detected players to two teams using
        horizontal court position.

        The Volleyball Dataset can contain a variable
        number of detected players per frame, so we
        keep ALL detected players.

        Left side  -> Team 0
        Right side -> Team 1
        """

        if len(players) < 2:
            raise ValueError(
                f"Need at least 2 players, "
                f"but found {len(players)}."
            )

        # Sort by horizontal position.
        sorted_players = sorted(
            players,
            key=lambda p: float(p["x"])
        )

        # Split the detected players approximately
        # equally between the two court sides.
        midpoint = len(sorted_players) // 2

        team_0 = sorted_players[:midpoint]
        team_1 = sorted_players[midpoint:]

        for player in team_0:
            player["team"] = 0

        for player in team_1:
            player["team"] = 1

        return team_0 + team_1

    # ========================================================
    # BALL PLACEHOLDER
    # ========================================================

    @staticmethod
    def create_ball():

        return {
            "x": 640.0,
            "y": 360.0,
        }

    # ========================================================
    # COURT
    # ========================================================

    @staticmethod
    def create_court():

        return {
            "width":
                VolleyballGraphDataset.COURT_WIDTH,

            "height":
                VolleyballGraphDataset.COURT_HEIGHT,

            "net_x":
                VolleyballGraphDataset.COURT_WIDTH / 2.0,
        }

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        sample = self.samples[
            index
        ]

        # ----------------------------------------------------
        # Copy player dictionaries
        #
        # This prevents modifying the parser's original data.
        # ----------------------------------------------------

        players = [
            dict(player)
            for player in sample[
                "players"
            ]
        ]

        # ----------------------------------------------------
        # Assign teams
        # ----------------------------------------------------

        players = self.assign_teams(
            players
        )

        # ----------------------------------------------------
        # Temporary ball
        #
        # Real ball detection will be integrated later.
        # ----------------------------------------------------

        ball = self.create_ball()

        # ----------------------------------------------------
        # Court
        # ----------------------------------------------------

        court = self.create_court()

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        graph = (
            self.graph_builder.build_graph(
                players=players,
                ball=ball,
                court=court,
                frame_id=sample[
                    "frame_id"
                ],
            )
        )

        # ----------------------------------------------------
        # Target label
        # ----------------------------------------------------

        graph.y = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        graph.video_id = (
            sample["video_id"]
        )

        graph.frame_id = (
            sample["frame_id"]
        )

        graph.group_activity = (
            sample["group_activity"]
        )

        return graph


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets(
    dataset_root,
    graph_builder=None,
):

    parser = VolleyballDatasetParser(
        dataset_root
    )

    # --------------------------------------------------------
    # Parse official dataset splits
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create graph datasets
    # --------------------------------------------------------

    train_dataset = (
        VolleyballGraphDataset(
            train_samples,
            graph_builder,
        )
    )

    validation_dataset = (
        VolleyballGraphDataset(
            validation_samples,
            graph_builder,
        )
    )

    test_dataset = (
        VolleyballGraphDataset(
            test_samples,
            graph_builder,
        )
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    DATASET_ROOT = (
        r"C:\Users\vipra\Downloads"
        r"\volleyball_\videos"
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(
        DATASET_ROOT
    )

    print(
        "\n========== GRAPH DATASET =========="
    )

    print(
        "Training graphs:",
        len(train_dataset),
    )

    print(
        "Validation graphs:",
        len(validation_dataset),
    )

    print(
        "Test graphs:",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # Build one real graph
    # --------------------------------------------------------

    graph = train_dataset[0]

    print(
        "\n========== EXAMPLE GRAPH =========="
    )

    print(
        "Node feature shape:",
        graph.x.shape,
    )

    print(
        "Edge index shape:",
        graph.edge_index.shape,
    )

    print(
        "Edge feature shape:",
        graph.edge_attr.shape,
    )

    print(
        "Node type shape:",
        graph.node_type.shape,
    )

    print(
        "Node team shape:",
        graph.node_team.shape,
    )

    print(
        "Number of nodes:",
        graph.num_nodes,
    )

    print(
        "Number of edges:",
        graph.edge_index.size(1),
    )

    print(
        "Label:",
        graph.y.item(),
    )

    print(
        "Activity:",
        graph.group_activity,
    )

    print(
        "Node teams:",
        graph.node_team.tolist(),
    )

    print(
        "===================================="
    )