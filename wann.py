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

creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", Net, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Net initialization params
NET_IN, NET_OUT = 4, 2
ENV_NAME = 'CartPole-v1'

toolbox.register("individual", creator.Individual, NET_IN, NET_OUT)
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


def evaluate(individual):
    total_fit = 0
    max_fit = 0
    env = gym.make(ENV_NAME)
    weights=[-2, -1, -0.5, 0.5, 1, 2]
    for w in weights:
        total_reward = 0
        observation = env.reset()
        for i in range(1000):
            action = np.argmax(individual(w, observation))
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
    individual.fitness.values = (avg_fit, max_fit, complexity)
    return individual


toolbox.register("evaluate", evaluate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)


def main():
    MAX_GEN = 30
    POP_SIZE = 8
    SELECT_K = int(POP_SIZE * 0.8)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats.register("max", np.max, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "max", "min", "avg"

    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        toolbox.evaluate(ind)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=POP_SIZE, **record)
    pool = Pool()

    for gen in range(1, MAX_GEN):
        offspring = toolbox.select(pop, SELECT_K)
        offspring = [offspring[random.randint(0, len(offspring)-1)] for _ in range(POP_SIZE)]
        offspring = list(map(toolbox.clone, offspring))
        offspring = pool.map(toolbox.mutate, offspring)
        offspring = pool.map(toolbox.evaluate, offspring)
        pop[:] = offspring

#        draw_pop(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=POP_SIZE, **record)
        print(logbook.stream)

    best_ind = tools.selBest(pop, 1)[0]
    with open(f'models/best_net_{ENV_NAME}.pickle', 'wb') as f:
        pickle.dump(best_ind, f)
    print(f"\nBest individual is {best_ind.fitness.values}")
    env = gym.make(ENV_NAME)
    showcase(best_ind, env)
    env.close()
    best_ind.test_range(env)



if __name__ == "__main__":
    main()

