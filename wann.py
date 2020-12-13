import random
import gym
from net import Net
from mutations import Mutation
from deap import base
from deap import creator
from deap import tools


creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", Net, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Net initialization params
NET_IN, NET_OUT = 2, 1

toolbox.register("individual", creator.Individual, NET_IN, NET_OUT)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def mutate(individual, weights):
    mutation = random.choices(list(Mutation), weights=weights)[0]

    if mutation == Mutation.CHANGE_ACTIV:
        individual.change_activation()
    elif mutation == Mutation.INSERT_NODE:
        individual.insert_node()
    elif mutation == Mutation.ADD_CONN:
        individual.add_connection()


def select_action_cart_pole(net_output):
    return 1 if net_output >= 0.5 else 0


def evaluate(individual, env, weights):
    total_fit = 0
    max_fit = 0

    for w in weights:
        total_reward = 0
        observation = env.reset()
        for i in range(1000):
            action = select_action_cart_pole(individual(w, observation))
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        max_fit = total_reward if total_reward > max_fit else max_fit
        total_fit += total_reward

    avg_fit = total_fit / len(weights)
    complexity = individual.get_num_connections()

    return avg_fit, max_fit, complexity


toolbox.register("evaluate", evaluate, weights=[-2, -1, -0.5, 0.5, 1, 2])
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)


def main():
    MAX_ITER = 100
    POP_SIZE = 10
    SELECT_K = int(POP_SIZE * 0.8)
    MUT_WEIGHTS = [1, 1, 1]

    env = gym.make('CartPole-v1')

    pop = toolbox.population(n=POP_SIZE)

    fitnesses = [toolbox.evaluate(ind, env) for ind in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for i in range(MAX_ITER):
        offspring = toolbox.select(pop, SELECT_K)
        offspring = random.choices(offspring, k=POP_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        for mutant in offspring:
            toolbox.mutate(mutant, MUT_WEIGHTS)
            del mutant.fitness.values

        fitnesses = [toolbox.evaluate(ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
    env.close()


if __name__ == "__main__":
    main()

