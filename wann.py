#!/usr/bin/env python3
import random
import gym
import numpy as np
from net import Net
from mutations import Mutation
from deap import base
from deap import creator
from deap import tools
from visual import draw_pop, showcase
from multiprocessing import Pool
import pickle
from functions import FUN
import matplotlib.pyplot as plt

np.set_printoptions(precision=1)

WS = (1,1,1)
# Net initialization params
ENV_NAME = ['CartPole-v1', 'BipedalWalker-v3', 'Pendulum-v0'][2]
env = gym.make(ENV_NAME)
NET_IN = env.observation_space.shape[0]
try:
    NET_OUT = env.action_space.n
except:
    NET_OUT = env.action_space.shape[0]
env.close(); del env
out_fun = {'CartPole-v1': FUN['ID'],
           'BipedalWalker-v3': FUN['TANH'],
           'Pendulum-v0': FUN['2TANH']
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
        observation = env.reset()
        for i in range(1000):
            action = out2action(individual(w, observation), env)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        if total_reward > max_fit:
            max_fit = total_reward
            individual.best_w = w
        total_fit += total_reward
    env.close()
    avg_fit = total_fit / len(weights)
    complexity = individual.get_num_connections()
    individual.fitness.values = (WS[0]*avg_fit, WS[1]*max_fit, WS[2]*complexity)
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
    return best_ind


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
    fig, axs = plt.subplots(2)
    axs[0].plot(gen, max_[:, 0])
    axs[0].plot(gen, avg[:, 0])
    axs[0].plot(gen, min_[:, 0])
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Fitness")
    axs[0].legend(["max", "avg", "min"])

    axs[1].plot(gen, max_[:, 1])
    axs[1].plot(gen, avg[:, 1])
    axs[1].plot(gen, min_[:, 1])
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Connections")
    axs[0].legend(["max", "avg", "min"])
    fig.show()


def main():
    global WS
    MAX_GEN = 300
    POP_SIZE = 100

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
        WS = (1,1,0)
        if random.random() < 0.8:
            WS = (1,0,1)
        offspring = list(map(toolbox.clone, pop))
        offspring = pool.map(toolbox.mutate, offspring)
        offspring = pool.map(toolbox.evaluate, offspring)
        offspring = toolbox.select(pop + offspring, POP_SIZE)
        pop[:] = offspring

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=POP_SIZE, **record)
        print(logbook.stream)
        best_ind = choose_best(pop)
        serialize(best_ind)

    best_ind = choose_best(pop)
    serialize(best_ind)
    for i in range(POP_SIZE):
        print(pop[i].fitness.values)
    print(f"\nBest individual is {best_ind.fitness.values}")
    env = gym.make(ENV_NAME)
    showcase(best_ind, env)
    env.close()
    best_ind.test_range(env)
    plot_evo(logbook)


if __name__ == "__main__":
    main()

