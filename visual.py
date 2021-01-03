import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import gym
from matplotlib import animation


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


