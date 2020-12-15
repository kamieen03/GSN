import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import gym

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


def showcase(net, env):
    observation = env.reset()
    for _ in range(1000):
        env.render()
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            action = np.argmax(net(net.best_w, observation))
        else:
            action = net(net.best_w, observation)
        observation, reward, done, info = env.step(action)
        if done: break
    env.close()
