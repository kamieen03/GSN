#!/usr/bin/env python3
from functions import FUN, FUN_NAMES
from numpy import array as arr
import random
from node import Node
import numpy as np
from matplotlib import pyplot as plt
from visual import showcase, plot_evo, save_frames_as_gif
import pickle
import sys
import gym
from tqdm import tqdm


class Net:
    def __init__(self, n_inputs, n_outputs, out_fun_name='ID'):
        self.curr_node_id = 0
        self.best_w = None
        self.nodes = []
        self.layers = []  # list of lists of nodes
        self.inputs = [Node('ID', self.get_node_id()) for _ in range(n_inputs)]
        self.outputs = [Node(out_fun_name, self.get_node_id()) for _ in range(n_outputs)]
        for o in self.outputs:
            o.children = random.sample(self.inputs, k=2)

    def __call__(self, W, X):
        for i, x in zip(self.inputs, X):
            i.val = x

        for layer in self.layers:
            for node in layer:
                node(W)

        return arr([out(W) for out in self.outputs])

    def change_activation(self):
        if len(self.nodes) > 0:
            node = random.choice(self.nodes)
            node.fun_name = random.choice(FUN_NAMES)
            node.fun = FUN[node.fun_name]
        return self

    def insert_node(self):
        # get random node in network and number of it's layer
        weights = [len(layer) for layer in self.layers] + [len(self.outputs)]
        lparent = random.choices(range(len(self.layers) + 1), weights=weights)[0]
        if lparent < len(self.layers):
            parent = random.choice(self.layers[lparent])
        else:
            parent = random.choice(self.outputs)

        # get one of it's children and produce new node between parent and child
        child = random.choice(parent.children)
        node = Node(random.choice(FUN_NAMES), self.get_node_id())
        node.children.append(child)
        parent.children.remove(child)
        parent.children.append(node)

        # insert newly created node to list of nodes and add it to proper layer
        # if parent and child were in consecutive layer, then we have to create a new layerr between them
        self.nodes.append(node)
        if lparent == 0 or child in self.layers[lparent - 1]:
            self.layers.insert(lparent, [])
            self.layers[lparent].append(node)
        else:
            self.layers[lparent - 1].append(node)
        return self

    def add_connection(self):
        # draw random layers according to numbers of nodes in them
        weights1 = [len(self.inputs)] + [len(layer) for layer in self.layers] + [len(self.outputs)]
        l1 = random.choices(range(len(self.layers) + 2), weights=weights1)[0]
        possible_layers = list(range(len(self.layers) + 2))
        possible_layers.pop(l1)
        weights2 = weights1
        weights2.pop(l1)
        l2 = random.choices(possible_layers, weights=weights2)[0]

        # draw random nodes from chosen layers; node n1 must be in layer earlier than node n2
        l1, l2 = sorted([l1, l2])
        if l1 == 0:
            n1 = random.choice(self.inputs)
        else:
            n1 = random.choice(self.layers[l1 - 1])
        if l2 == len(self.layers) + 1:
            n2 = random.choice(self.outputs)
        else:
            n2 = random.choice(self.layers[l2 - 1])
        if n1 not in n2.children:
            n2.children.append(n1)
        return self

    def get_all_nodes(self):
        return self.inputs + self.nodes + self.outputs

    def get_num_connections(self):
        return sum([len(node.children) for node in self.get_all_nodes()])

    def get_node_id(self):
        id_ = self.curr_node_id
        self.curr_node_id += 1
        return id_

    def test_range(self, env):
        print("Testing range of weights...")
        xs, ys = [], []
        for w in tqdm(np.arange(-3.0,3.0,0.03)):
            total_reward = 0
            observation = env.reset()
            for i in range(1000):
                if type(env.action_space) == gym.spaces.discrete.Discrete:
                    action = np.argmax(self(w, observation))
                else:
                    action = self(w, observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            xs.append(w)
            ys.append(total_reward)
        plt.plot(xs, ys)
        plt.xlabel("Weight")
        plt.ylabel("Score")
        plt.title("Weight tuning")
        plt.savefig(f"plots/weights_{env.unwrapped.spec.id}.jpg")
        plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 net.py (walker|cartpole|pendulum|lander|gora)")
    else:
        if sys.argv[1] == 'walker':
            ENV_NAME = "BipedalWalker-v3"
        elif sys.argv[1] == "cartpole":
            ENV_NAME = "CartPole-v1"
        elif sys.argv[1] == "pendulum":
            ENV_NAME = "Pendulum-v0"
        elif sys.argv[1] == "gora":
            ENV_NAME = "MountainCar-v0"
        elif sys.argv[1] == "lander":
            ENV_NAME = "LunarLanderContinuous-v2"
        else:
            print("Usage: py net.py (walker|cartpole|pendulum|lander|gora)")

    with open(f"models/best_net_{ENV_NAME}.pickle", "rb") as f:
        n = pickle.load(f)
    with open(f"log/logbook_{ENV_NAME}.pkl", 'rb') as f:
        logbook = pickle.load(f)

    env = gym.make(ENV_NAME)
    frames = showcase(n, env)
    save_frames_as_gif(frames, f"renders/render_{ENV_NAME}.gif")
    n.test_range(env)
    plot_evo(logbook, ENV_NAME)
    env.close()



if __name__ == '__main__':
    main()

