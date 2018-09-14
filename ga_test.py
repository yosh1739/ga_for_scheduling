'''
http://darden.hatenablog.com/entry/2017/03/29/213948
このサイトを参考に実装
遺伝子が取り得る値を0、1とし、個体内の遺伝子の和を最大化するテストスクリプトを書いてみました。
個体内の全ての遺伝子が1になる物が最適解です。
'''

import random
import numpy as np
from operator import attrgetter


def main():
    n_gene = 100  # The number of genes.
    n_ind = 300  # The number of individuals in a population.
    CXPB = 0.5  # The probability of crossover.
    MUTPB = 0.2  # The probability of individdual mutation.
    MUTINDPB = 0.05  # The probability of gene mutation.
    NGEN = 40  # The number of generation loop.

    random.seed(64)
    # --- Step1 : Create initial generation.
    pop = create_pop(n_ind, n_gene) #100個の遺伝子を持つ個体を300個生成
    set_fitness(evalOneMax, pop) #適合度計算
    best_ind = max(pop, key=attrgetter("fitness")) #popの中から最も良いものを抽出

    # --- Generation loop.
    print("Generation loop start.")
    print("Generation: 0. Best fitness: " + str(best_ind.fitness))
    for g in range(NGEN):

        # --- Step2 : Selection.
        offspring = selTournament(pop, n_ind, tournsize=3) #ランダムに3個選択する

        # --- Step3 : Crossover.
        crossover = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1, child2 = cxTwoPointCopy(child1, child2)
                child1.fitness = None
                child2.fitness = None
            crossover.append(child1)
            crossover.append(child2)

        offspring = crossover[:]

        # --- Step4 : Mutation.
        mutant = []
        for mut in offspring:
            if random.random() < MUTPB:
                mut = mutFlipBit(mut, indpb=MUTINDPB)
                mut.fitness = None
            mutant.append(mut)

        offspring = mutant[:]

        # --- Update next population.
        pop = offspring[:]
        set_fitness(evalOneMax, pop)

        # --- Print best fitness in the population.
        best_ind = max(pop, key=attrgetter("fitness"))
        print("Generation: " + str(g + 1) + ". Best fitness: " + str(best_ind.fitness))

    print("Generation loop ended. The best individual: ")
    print(best_ind)


class Individual(np.ndarray):
    """Container of a individual."""
    fitness = None

    def __new__(cls, a):
        return np.asarray(a).view(cls)


def create_ind(n_gene):
    """Create a individual."""
    return Individual([random.randint(0, 1) for i in range(n_gene)])


def create_pop(n_ind, n_gene):
    """Create a population."""
    pop = []
    for i in range(n_ind):
        ind = create_ind(n_gene)
        pop.append(ind)
    return pop


def set_fitness(eval_func, pop):
    """Set fitnesses of each individual in a population."""
    for i, fit in zip(range(len(pop)), map(eval_func, pop)):
        pop[i].fitness = fit


def evalOneMax(ind): #適応度の中で最も大きなもの
    """Objective function."""
    return sum(ind)


def selTournament(pop, n_ind, tournsize):
    """Selection function."""
    chosen = []
    for i in range(n_ind):
        aspirants = [random.choice(pop) for j in range(tournsize)]
        chosen.append(max(aspirants, key=attrgetter("fitness")))
    return chosen


def cxTwoPointCopy(ind1, ind2):
    """Crossover function."""
    size = len(ind1)
    tmp1 = ind1.copy()
    tmp2 = ind2.copy()
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2].copy(), tmp1[cxpoint1:cxpoint2].copy()
    return tmp1, tmp2


def mutFlipBit(ind, indpb):
    """Mutation function."""
    tmp = ind.copy()
    for i in range(len(ind)):
        if random.random() < indpb:
            tmp[i] = type(ind[i])(not ind[i])
    return tmp


if __name__ == "__main__":
    main()