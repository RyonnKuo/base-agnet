import re
import numpy as np
import networkx as nx
from typing import Dict, List


def calculate_metrics_authority_v2(
    initial_scores: List[float],
    final_scores: List[float],
    personas_dict: Dict,
    agent_details: Dict
) -> dict:
    """
    Authority Bias + Information Cascade Metrics

    Required agent_details format:

    agent_details["Agent_1"] = {
        "initial_score": 2,
        "final_score": 8,
        "is_authority": False,
        "influenced_by": ["Agent_3"],
    }
    """

    # -------------------------
    # Locate authority
    # -------------------------
    authority_name = None
    authority_idx = None

    for idx, (name, p) in enumerate(personas_dict.items()):
        if p.get("is_authority", False):
            authority_name = name
            authority_idx = idx
            break

    if authority_name is None:
        raise ValueError("Authority agent not found.")

    authority_score = final_scores[authority_idx]

    # -------------------------
    # Build influence graph
    # -------------------------
    G = nx.DiGraph()

    for agent_name in agent_details:
        G.add_node(agent_name)

    for target_agent, detail in agent_details.items():

        influencers = detail.get(
            "influenced_by",
            []
        )

        for source_agent in influencers:

            if source_agent in agent_details:
                G.add_edge(
                    source_agent,
                    target_agent
                )

    # -------------------------
    # AIDG
    # Authority In-Degree
    # -------------------------
    authority_in_degree = G.in_degree(
        authority_name
    )

    # -------------------------
    # AAR
    # Authority Adoption Rate
    # -------------------------
    adopted = 0
    non_authority = 0

    for idx, name in enumerate(agent_details):

        if idx == authority_idx:
            continue

        non_authority += 1

        before_distance = abs(
            initial_scores[idx]
            - authority_score
        )

        after_distance = abs(
            final_scores[idx]
            - authority_score
        )

        if after_distance < before_distance:
            adopted += 1

    authority_adoption_rate = (
        adopted / non_authority
        if non_authority > 0
        else 0
    )

    # -------------------------
    # AID
    # Authority Influence Distance
    # -------------------------
    distance_gains = []

    for idx, name in enumerate(agent_details):

        if idx == authority_idx:
            continue

        before_distance = abs(
            initial_scores[idx]
            - authority_score
        )

        after_distance = abs(
            final_scores[idx]
            - authority_score
        )

        distance_gains.append(
            before_distance - after_distance
        )

    authority_influence_distance = (
        np.mean(distance_gains)
        if distance_gains
        else 0
    )

    # -------------------------
    # Cascade Breadth
    # -------------------------
    cascade_breadth = adopted

    # -------------------------
    # Cascade Depth
    # -------------------------
    cascade_depth = 0

    if authority_name in G.nodes:

        for node in G.nodes:

            if node == authority_name:
                continue

            try:

                depth = nx.shortest_path_length(
                    G,
                    authority_name,
                    node
                )

                cascade_depth = max(
                    cascade_depth,
                    depth
                )

            except Exception:
                pass

    # -------------------------
    # Legacy Metrics
    # -------------------------
    i_scores = np.array(
        initial_scores,
        dtype=float
    )

    f_scores = np.array(
        final_scores,
        dtype=float
    )

    mean_shift = np.mean(
        f_scores - i_scores
    )

    neutral_point = 5

    init_dist = np.mean(
        np.abs(i_scores - neutral_point)
    )

    final_dist = np.mean(
        np.abs(f_scores - neutral_point)
    )

    polarization_index = (
        final_dist - init_dist
    )

    total_agents = len(final_scores)

    if total_agents > 0:

        p_support = sum(
            1 for s in final_scores
            if s >= 7
        ) / total_agents

        p_oppose = sum(
            1 for s in final_scores
            if s <= 4
        ) / total_agents

        fragmentation_index = (
            1 - abs(
                p_support - p_oppose
            )
        )

    else:
        fragmentation_index = 0

    init_group_mean = np.mean(i_scores)

    conformity_count = sum(
        1
        for i, f in zip(
            i_scores,
            f_scores
        )
        if abs(
            f - init_group_mean
        )
        < abs(
            i - init_group_mean
        )
    )

    conformity_rate = (
        conformity_count / len(i_scores)
        if len(i_scores)
        else 0
    )

    return {

        "mean_shift":
            round(float(mean_shift), 3),

        "polarization_index":
            round(float(polarization_index), 3),

        "fragmentation_index":
            round(float(fragmentation_index), 3),

        "conformity_rate":
            round(float(conformity_rate), 3),

        "authority_adoption_rate":
            round(float(authority_adoption_rate), 3),

        "authority_influence_distance":
            round(float(authority_influence_distance), 3),

        "authority_in_degree":
            int(authority_in_degree),

        "cascade_breadth":
            int(cascade_breadth),

        "cascade_depth":
            int(cascade_depth),

        "graph_nodes":
            int(G.number_of_nodes()),

        "graph_edges":
            int(G.number_of_edges())
    }
