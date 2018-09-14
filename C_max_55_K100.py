# coding:utf-8
# 遺伝的アルゴリズムでパラレルショップの不良率を最小化したもの

import random
import numpy as np
import csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from math import modf

# 遺伝子のセットである個体を表現するのに利用
# 第2引数で指定したクラスを継承して、第1引数で指定する名前のクラスをcreatorモジュール内に新しく生成します。第3引数以降は、生成する子クラスに追加するメンバ変数を指定します。
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 適応度の定義
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)  # 個体の定義

# base.Toolbox.register()関数を使うと、引数のデフォルト値が無い関数に、デフォルト値を設定することができます。
toolbox = base.Toolbox()

# 遺伝子の取り得る範囲を指定しています
n_gene = 100
I_num = 2  # ステージ数
min_ind = np.ones(n_gene) * 1.00
max_ind = np.ones(n_gene) * 5.99  # 可変
mcn_num1 = 5
mcn_num2 = 5

'''
# 処理時間の生成
p1 = np.array([])
p2 = np.array([])
for i in range(0,n_gene):
    p1 = np.append(p1,random.randint(5,15))
    p2 = np.append(p2,random.randint(5,15))
print(p1,p2)
'''
p1 = np.array([15,5,14,14,8,15,6,6,11,11,11,11,13,14,14,14,12,7,7,13,8,14,9,6,10,14,9,6,15,9,5,15,6,13,15,13,14,14,13,12,13,7,8,15,8,10,9,14,15,15,13,8,10,10,9,10,5,12,7,14,14,13,6,11,6,12,14,9,7,12,7,11,15,11,14,13,12,15,7,15,12,8,13,13,15,15,9,6,7,6,6,11,8,13,15,14,7,6,9,8])
p2 = np.array([14,10,14,6,13,7,9,7,7,12,9,6,6,5,10,7,9,8,10,8,8,11,7,12,12,13,10,6,11,6,14,8,7,6,14,5,9,8,12,15,14,10,12,6,9,8,14,6,6,7,15,12,9,12,10,13,14,13,5,12,5,12,13,14,8,5,10,10,5,14,10,12,5,6,10,13,5,15,14,6,11,14,11,5,14,8,12,8,9,5,10,6,5,7,15,8,14,7,14,13])

# 個体を生成する関数を定義しています。
def create_ind_uniform(min_ind, max_ind):
    ind = []
    for min, max in zip(min_ind, max_ind):
        ind.append(random.uniform(min, max))
    return ind


toolbox.register("create_ind", create_ind_uniform, min_ind, max_ind)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def evalOneMax(individual):  # 評価関数の計算
    finish = [[0 for i in range(n_gene)] for j in range(I_num)]
    quality = [0 for i in range(n_gene)]
    mcn = np.zeros([10, n_gene])

    size = len(individual)
    m1 = np.empty((0, 2), int)
    m2 = np.empty((0, 2), int)
    m3 = np.empty((0, 2), int)
    m4 = np.empty((0, 2), int)
    m5 = np.empty((0, 2), int)

    for i in range(size):
        decimal, integer = modf(individual[i])
        if integer == 1:
            m1 = np.append(m1, np.array([[i, individual[i]]]), axis=0)
        elif integer == 2:
            m2 = np.append(m2, np.array([[i, individual[i]]]), axis=0)
        elif integer == 3:
            m3 = np.append(m3, np.array([[i, individual[i]]]), axis=0)
        elif integer == 4:
            m4 = np.append(m4, np.array([[i, individual[i]]]), axis=0)
        elif integer == 5:
            m5 = np.append(m5, np.array([[i, individual[i]]]), axis=0)

    m1 = m1[m1[:, -1].argsort()]  # 数字が小さい順に並べ替え
    m2 = m2[m2[:, -1].argsort()]
    m3 = m3[m3[:, -1].argsort()]  # 数字が小さい順に並べ替え
    m4 = m4[m4[:, -1].argsort()]
    m5 = m5[m5[:, -1].argsort()]
    size1 = len(m1)
    size2 = len(m2)
    size3 = len(m3)
    size4 = len(m4)
    size5 = len(m5)
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    t5 = 0

    if m1.size != 0:
        for i in range(size1):
            a = m1[i, 0]
            t1 += (p1[int(a)])
            finish[0][int(a)] = t1
            mcn[0][int(a)] = 1

    if m2.size != 0:
        for i in range(size2):
            a = m2[i, 0]
            t2 += (p1[int(a)])
            finish[0][int(a)] = t2
            mcn[1][int(a)] = 1

    if m3.size != 0:
        for i in range(size3):
            a = m3[i, 0]
            t3 += (p1[int(a)])
            finish[0][int(a)] = t3
            mcn[2][int(a)] = 1

    if m4.size != 0:
        for i in range(size4):
            a = m4[i, 0]
            t4 += (p1[int(a)])
            finish[0][int(a)] = t4
            mcn[3][int(a)] = 1

    if m5.size != 0:
        for i in range(size5):
            a = m5[i, 0]
            t5 += (p1[int(a)])
            finish[0][int(a)] = t5
            mcn[4][int(a)] = 1

    fin_seq1 = argsort(finish[0])  # 装置の処理順序

    # 初期値設定**************
    # 処理時刻の入れ物
    t21 = p2[fin_seq1[0]]
    t22 = p2[fin_seq1[1]]
    t23 = p2[fin_seq1[2]]
    t24 = p2[fin_seq1[3]]
    t25 = p2[fin_seq1[4]]

    # 装置の開始時刻
    c21 = finish[0][fin_seq1[0]]
    c22 = finish[0][fin_seq1[1]]
    c23 = finish[0][fin_seq1[2]]
    c24 = finish[0][fin_seq1[3]]
    c25 = finish[0][fin_seq1[4]]

    # 終了時刻
    finish[1][fin_seq1[0]] = c21 + t21
    finish[1][fin_seq1[1]] = c22 + t22
    finish[1][fin_seq1[2]] = c23 + t23
    finish[1][fin_seq1[3]] = c24 + t24
    finish[1][fin_seq1[4]] = c25 + t25

    # 割り当て
    mcn[5][fin_seq1[0]] = 1
    mcn[6][fin_seq1[1]] = 1
    mcn[7][fin_seq1[3]] = 1
    mcn[8][fin_seq1[4]] = 1
    mcn[9][fin_seq1[5]] = 1

    # 初期値設定ここまで**************

    for i in range(mcn_num2, n_gene):
        # 次のジョブが処理可能になる時刻
        # min(c21+t21,c22+t22,c23+t23) > fin_time1[fin_seq1[i]] ///処理可能な装置が空くまでしばらく処理を待つ状態
        # min(c21+t21,c22+t22,c23+t23) < fin_time1[fin_seq1[i]] ///前の処理が終わればすぐ次の処理にかかれる状態
        max_time = max(min(c21 + t21, c22 + t22, c23 + t23, c24 + t24, c25 + t25), finish[0][fin_seq1[i]])

        if max_time == c21 + t21:
            c21 = max_time
            t21 = p2[fin_seq1[i]]
            finish[1][fin_seq1[i]] = c21 + t21
            mcn[5][fin_seq1[i]] = 1
        elif max_time == c22 + t22:
            c22 = max_time
            t22 = p2[fin_seq1[i]]
            finish[1][fin_seq1[i]] = c22 + t22
            mcn[6][fin_seq1[i]] = 1
        elif max_time == c23 + t23:
            c23 = max_time
            t23 = p2[fin_seq1[i]]
            finish[1][fin_seq1[i]] = c23 + t23
            mcn[7][fin_seq1[i]] = 1
        elif max_time == c24 + t24:
            c24 = max_time
            t24 = p2[fin_seq1[i]]
            finish[1][fin_seq1[i]] = c24 + t24
            mcn[8][fin_seq1[i]] = 1
        elif max_time == c25 + t25:
            c25 = max_time
            t25 = p2[fin_seq1[i]]
            finish[1][fin_seq1[i]] = c25 + t25
            mcn[9][fin_seq1[i]] = 1
        elif max_time == finish[0][fin_seq1[i]]:
            max_sub_time = min(c21 + t21, c22 + t22, c23 + t23, c24 + t24, c25 + t25)
            if max_sub_time == c21 + t21:
                c21 = max_time
                t21 = p2[fin_seq1[i]]
                finish[1][fin_seq1[i]] = c21 + t21
                mcn[5][fin_seq1[i]] = 1
            elif max_sub_time == c22 + t22:
                c22 = max_time
                t22 = p2[fin_seq1[i]]
                finish[1][fin_seq1[i]] = c22 + t22
                mcn[6][fin_seq1[i]] = 1
            elif max_sub_time == c23 + t23:
                c23 = max_time
                t23 = p2[fin_seq1[i]]
                finish[1][fin_seq1[i]] = c23 + t23
                mcn[7][fin_seq1[i]] = 1
            elif max_sub_time == c24 + t24:
                c24 = max_time
                t24 = p2[fin_seq1[i]]
                finish[1][fin_seq1[i]] = c24 + t24
                mcn[8][fin_seq1[i]] = 1
            elif max_sub_time == c25 + t25:
                c25 = max_time
                t25 = p2[fin_seq1[i]]
                finish[1][fin_seq1[i]] = c25 + t25
                mcn[9][fin_seq1[i]] = 1

    fin_seq2 = argsort(finish[1])

    C_max = max(finish[1])

    return C_max,finish, mcn, p1, p2, fin_seq1, fin_seq2,


def cxTwoPointCopy(ind1, ind2):  # 2点交叉
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2


# 突然変異する関数
def mutUniformDbl(individual, min_ind, max_ind, indpb):  # indpbは突然変異率
    size = len(individual)
    for i, min, max in zip(range(size), min_ind, max_ind):
        if random.random() < indpb:
            individual[i] = random.uniform(min, max)
    return individual,


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)  # 交叉の戦略の指定
toolbox.register("mutate", mutUniformDbl, min_ind=min_ind, max_ind=max_ind, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed()

    pop = toolbox.population(n=30)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.01, ngen=1000, stats=stats,
                        halloffame=hof)  # 計算を実行する．cxpb:交叉率，mutpb:個体突然変異率

    # popの中で最も良い解を見つけて表示させる
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    best_completetime = evalOneMax(best_ind)[1]
    best_mcnassign = evalOneMax(best_ind)[2]

    with open('test.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(range(n_gene))
        writer.writerows(best_completetime)
        writer.writerows(best_mcnassign)
        writer.writerow(p1.tolist())
        writer.writerow(p2.tolist())

    return pop, stats, hof


if __name__ == "__main__":
    main()