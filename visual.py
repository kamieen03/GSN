import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import gym
from matplotlib import animation


def draw_net(net, env_name):
    G = convert2nx(net, env_name)
    color = [data["color"] for v, data in G.nodes(data=True)]
    label = {v: data["label"] for v, data in G.nodes(data=True)}
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=False, node_color=color)
    for p in pos:  # raise text positions
        pos[p][1] += 0.09
    nx.draw_networkx_labels(G, pos, labels=label)
    plt.show()


def convert2nx(net, env_name):
    g = nx.DiGraph()
    layers = [net.inputs, *net.layers, net.outputs]

    for i in range(len(layers)):
        for j, node in enumerate(layers[i]):
            if i == 0:
                label = OBSERVATIONS[env_name][j]
            elif i == len(layers)-1:
                label = ACTIONS[env_name][j]
            else:
                label = node.fun_name

            props = {
                "color": FUN2COL[node.fun_name],
                "label": label,
                "layer": i
            }
            g.add_nodes_from([(node.id, props)])

    for node in net.get_all_nodes():
        for child in node.children:
            g.add_edge(child.id, node.id)

    return g


def showcase(net, env):
    observation = env.reset()
    total_reward = 0
    frames = []
    for _ in range(1000):
        frames.append(env.render(mode="rgb_array"))
        if type(env.action_space) == gym.spaces.discrete.Discrete:
            action = np.argmax(net(net.best_w, observation))
        else:
            action = net(net.best_w, observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done: break
    env.close()
    print(total_reward)
    return frames

def plot_evo(logbook, env_name):
    gen, max_, avg, min_ = [np.array(arr) for arr in logbook.select("gen", "max", "avg", "min")]
    fig, axs = plt.subplots(3)
    fig.suptitle("Evolution")

    axs[0].plot(gen, max_[:, 0])
    axs[0].plot(gen, avg[:, 0])
    axs[0].plot(gen, min_[:, 0])
    axs[0].set_ylabel("Avg fitness")
    axs[0].legend(["max", "avg", "min"])

    axs[1].plot(gen, max_[:, 1])
    axs[1].plot(gen, avg[:, 1])
    axs[1].plot(gen, min_[:, 1])
    axs[1].set_ylabel("Max fitness")
    axs[1].legend(["max", "avg", "min"])

    axs[2].plot(gen, max_[:, 2])
    axs[2].plot(gen, avg[:, 2])
    axs[2].plot(gen, min_[:, 2])
    axs[2].set_xlabel("Generation")
    axs[2].set_ylabel("Connections")
    axs[2].legend(["max", "avg", "min"])
    fig.savefig(f"plots/evo_{env_name}.jpg")
    fig.show()




# Ensure you have imagemagick installed with
# sudo apt-get install imagemagick
def save_frames_as_gif(frames, path):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path, writer='imagemagick', fps=60)
    plt.close()


FUN2COL = {
    'ID'      : "gray",
    'SIN'     : "gold",
    'COS'     : "violet",
    'ABS'     : "darkgreen",
    'TANH'    : "darkorange",
    'RELU'    : "cyan",
    'GAUSS'   : "lime",
    'SIGMOID' : "royalblue",
    '2TANH'   : "darkmagenta"
}


OBSERVATIONS = {
    "MountainCar-v0": [
        "Position",
        "Velocity"
    ]
}

ACTIONS = {
    "MountainCar-v0": [
        "Acc left",
        "No acc",
        "Acc right"
    ]
}
