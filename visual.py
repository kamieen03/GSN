import matplotlib.pyplot as plt
import numpy as np
import gym
from matplotlib import animation
import pygraphviz as pgv


def draw_net(net, env_name, path):
    G = pgv.AGraph(strict=False, directed=True, rankdir="LR")

    # Add nodes
    layers = [net.inputs, *net.layers, net.outputs]
    for i in range(len(layers)):
        for j, node in enumerate(layers[i]):
            if i == 0:
                label = OBSERVATIONS[env_name][j]
            elif i == len(layers) - 1:
                label = ACTIONS[env_name][j]
            else:
                label = node.fun_name

            G.add_node(node.id,
                       color=FUN2COL[node.fun_name],
                       label=label,
                       shape="box",
                       )
    # Add edges
    for node in net.get_all_nodes():
        for child in node.children:
            G.add_edge(child.id, node.id)

    # Create layers
    for i in range(len(layers)):
        G.add_subgraph([node.id for node in layers[i]], rank="same")

    G.draw(path, prog='dot')


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

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path, writer='imagemagick', fps=60)
    plt.close()


FUN2COL = {
    'ID': "black",
    'SIN': "gold",
    'COS': "violet",
    'ABS': "darkgreen",
    'TANH': "darkorange",
    'RELU': "cyan",
    'GAUSS': "lime",
    'SIGMOID': "royalblue",
    '2TANH': "darkmagenta"
}

OBSERVATIONS = {
    "CartPole-v1": [
        "Cart Pos",
        "Cart Vel",
        "Pole Angle",
        "Pole Ang Vel",
    ],
    "MountainCar-v0": [
        "Position",
        "Velocity"
    ],
    "LunarLanderContinuous-v2": [
        "Pos X",
        "Pos Y",
        "Velo X",
        "Velo Y",
        "Angle",
        "Ang Velo",
        "Left ground",
        "Right ground"
    ],
    "BipedalWalker-v3": [
        "hull_angle",
        "hull_angularVelocity",
        "vel_x",
        "vel_y",
        "hip_joint_1_angle",
        "hip_joint_1_speed",
        "knee_joint_1_angle",
        "knee_joint_1_speed",
        "leg_1_ground_contact_flag",
        "hip_joint_2_angle",
        "hip_joint_2_speed",
        "knee_joint_2_angle",
        "knee_joint_2_speed",
        "leg_2_ground_contact_flag",
        "1 lidar",
        "2 lidar",
        "3 lidar",
        "4 lidar",
        "5 lidar",
        "6 lidar",
        "7 lidar",
        "8 lidar",
        "9 lidar",
        "10 lidar"
    ]
}

ACTIONS = {
    "CartPole-v1": [
        "Left",
        "Right"
    ],
    "MountainCar-v0": [
        "Acc left",
        "No acc",
        "Acc right"
    ],
    "LunarLanderContinuous-v2": [
        "Left Engine",
        "Off",
        "Right Engine"
    ],
    "BipedalWalker-v3": [
        "Hip_1",
        "Knee_1",
        "Hip_2",
        "Knee_2"
    ]
}
