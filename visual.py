import networkx as nx
from net import Net
import matplotlib.pyplot as plt
import math


def draw_pop(nets):
    rows = 4
    cols = math.ceil(len(nets) / 4)
    graphs = [convert2nx(net) for net in nets]
    for idx, g in enumerate(graphs):
        plt.subplot(rows, cols, idx+1)
        nx.draw(g)

    plt.tight_layout()
    plt.show()


def convert2nx(net):
    g = nx.DiGraph()

    for node in net.get_all_nodes():
        for child in node.children:
            g.add_edge(child.id, node.id)

    return g
