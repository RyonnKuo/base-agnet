import networkx as nx

def calculate_adjacency_matrix(personas_dump, conversation_history) -> dict:
    G = nx.DiGraph()
    for name in personas_dump.keys():
        G.add_node(name)

    for entry in conversation_history:
        # 假設 entry 格式為 "Agent_A: 回應 Agent_B ..."
        speaker, text = entry.split(":", 1)
        for other_name in personas_dump.keys():
            if other_name in text and speaker != other_name:
                # 建立一條從 speaker 指向被提及者的有向邊
                if G.has_edge(speaker, other_name):
                    G[speaker][other_name]['weight'] += 1
                else:
                    G.add_edge(speaker, other_name, weight=1)