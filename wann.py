#!/usr/bin/env python3
import random
import gym
import numpy as np
from net import Net
from mutations import Mutation
from deap import base
from deap import creator
from deap import tools
from visual import draw_pop, showcase, save_frames_as_gif
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt

np.set_printoptions(precision=1)

# Net initialization params
ENV_NAME = ['CartPole-v1',
            'BipedalWalker-v3',
            'Pendulum-v0',
            'MountainCar-v0',
            'LunarLanderContinuous-v2'][4]
env = gym.make(ENV_NAME)
NET_IN = env.observation_space.shape[0]
try:
    NET_OUT = env.action_space.n
except:
    NET_OUT = env.action_space.shape[0]
env.close(); del env
out_fun = {'CartPole-v1': 'ID',
           'BipedalWalker-v3': 'TANH',
           'Pendulum-v0': '2TANH',
           'MountainCar-v0': 'ID',
           'LunarLanderContinuous-v2': 'TANH'
           }[ENV_NAME]

creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", Net, fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("individual", creator.Individual, NET_IN, NET_OUT, out_fun)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mutate(individual):
    mutation = random.choices(list(Mutation), weights=[1,1,1])[0]
    if mutation == Mutation.CHANGE_ACTIV:
        return individual.change_activation()
    elif mutation == Mutation.INSERT_NODE:
        return individual.insert_node()
    elif mutation == Mutation.ADD_CONN:
        return individual.add_connection()
    else:
        raise Exception("Nie wybrano żadnej mutacji")


def out2action(out, env):
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        return np.argmax(out)
    return out


def evaluate(individual):
    total_fit = 0
    max_fit = -np.inf
    env = gym.make(ENV_NAME)
    weights=[-2, -1, -0.5, 0.5, 1, 2]
    for w in weights:
        total_reward = 0
        for _ in range(5):
            observation = env.reset()
            for i in range(1000):
                action = out2action(individual(w, observation), env)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
        total_reward /=5

        if total_reward > max_fit:
            max_fit = total_reward
            individual.best_w = w
        total_fit += total_reward
    env.close()
    avg_fit = total_fit / len(weights)
    complexity = individual.get_num_connections()
    individual.fitness.values = avg_fit, max_fit, complexity
    return individual


toolbox.register("evaluate", evaluate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)


def choose_best(pop):
    best_ind = None
    best_max = -np.inf
    best_avg = -np.inf
    for ind in pop:
        iavg, imax, _ = ind.fitness.values
        if iavg + imax > best_avg + best_max:
            best_ind = ind
            best_max = imax
            best_avg = iavg
    return best_ind, best_avg+best_max


def serialize(ind):
    n = Net(NET_IN, NET_OUT, out_fun)
    n.best_w = ind.best_w
    n.nodes = ind.nodes
    n.layers = ind.layers
    n.inputs = ind.inputs
    n.outputs = ind.outputs
    with open(f'models/best_net_{ENV_NAME}.pickle', 'wb') as f:
        pickle.dump(n, f)


def plot_evo(logbook):
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
    fig.savefig(f"plots/evo_{ENV_NAME}.jpg")
    fig.show()


def main():
    MAX_GEN = 250
    POP_SIZE = 20
    all_time_best_ind = None
    all_time_best_score = -np.inf

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # skip avg part of the fitness
    stats.register("max", np.max, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "max", "avg", "min"

    pool = Pool()

    pop = toolbox.population(n=POP_SIZE)
    pop = pool.map(toolbox.evaluate, pop)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=POP_SIZE, **record)
    print(logbook.stream)

    for gen in range(1, MAX_GEN):
        offspring = list(map(toolbox.clone, pop))
        offspring = pool.map(toolbox.mutate, offspring)
        offspring = pool.map(toolbox.evaluate, offspring)
        offspring = toolbox.select(pop + offspring, POP_SIZE)
        pop[:] = offspring

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=POP_SIZE, **record)
        with open(f"log/logbook_{ENV_NAME}.pkl", 'wb') as f:
            pickle.dump(logbook, f)
        print(logbook.stream)
        best_ind, best_score = choose_best(pop)
        if best_score >= all_time_best_score:
            all_time_best_score = best_score
            all_time_best_ind = best_ind
#        env = gym.make(ENV_NAME)
#        print(all_time_best_ind.fitness)
#        showcase(all_time_best_ind, env)
#        env.close()
        serialize(best_ind)

    best_ind = choose_best(pop)
    serialize(best_ind)
    for i in range(POP_SIZE):
        print(pop[i].fitness.values)
    print(f"\nBest individual is {best_ind.fitness.values}")
    env = gym.make(ENV_NAME)
    frames = showcase(best_ind, env)
    save_frames_as_gif(frames, f"renders/render_{ENV_NAME}.gif")
    env.close()
    best_ind.test_range(env)
    plot_evo(logbook)


if __name__ == "__main__":
    main()

