# coding:utf-8
#遺伝的アルゴリズムでパラレルショップの総所要時間最小化したもの

import random
import numpy as np
import csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from math import modf

#遺伝子のセットである個体を表現するのに利用
#第2引数で指定したクラスを継承して、第1引数で指定する名前のクラスをcreatorモジュール内に新しく生成します。第3引数以降は、生成する子クラスに追加するメンバ変数を指定します。
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #適応度の定義
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin) #個体の定義

#base.Toolbox.register()関数を使うと、引数のデフォルト値が無い関数に、デフォルト値を設定することができます。
toolbox = base.Toolbox()

#遺伝子の取り得る範囲を指定しています
n_gene = 30
I_num = 3 #ステージ数
test_number = 0
min_ind = np.ones(n_gene) * 1.00
max_ind = np.ones(n_gene) * 2.99  #可変
mcn_num1 = 2
mcn_num2 = 3
mcn_num3 = 2


#処理時間の生成
'''
p1 = np.array([])
p2 = np.array([])
p3 = np.array([])
for i in range(0,n_gene):
    p1 = np.append(p1,random.randint(3,7))
    p2 = np.append(p2,random.randint(5,15))
    p3 = np.append(p3,random.randint(3,10))
print(p1,p2,p3)
'''
p1 = np.array([7, 6, 3, 6, 3, 4, 5, 4, 4, 7, 7, 7, 7, 7, 3, 7, 7, 5, 5, 5, 4, 7, 5, 6, 3, 5, 3, 7, 5, 5] )
p2 = np.array([10, 8, 12, 10, 14, 6, 8, 5, 11, 6, 5, 11, 7, 5, 13, 11, 6, 12, 7, 10, 12, 12, 7,  7, 14, 11, 11, 11, 10, 14])
p3 = np.array([6, 7, 4, 9, 9, 9, 4, 3, 10, 8, 8, 10, 4, 6, 4, 7, 10, 10, 10, 8, 9, 5, 6, 5, 8, 7, 4, 3, 10, 5])

#個体を生成する関数を定義しています。
def create_ind_uniform(min_ind, max_ind):
    ind = []
    for min, max in zip(min_ind, max_ind):
        ind.append(random.uniform(min, max))
    return ind

toolbox.register("create_ind", create_ind_uniform, min_ind, max_ind)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual): #評価関数の計算
    finish = [[0 for i in range(n_gene)] for j in range(I_num)]
    mcn = np.zeros([7,n_gene])

    size = len(individual)
    m1 = np.empty((0,2), int)
    m2 = np.empty((0,2), int)
    fin_time1 = np.ones(n_gene) #工程1の終了時刻

    for i in range(size):
        decimal, integer = modf(individual[i])
        if integer == 1:
            m1 = np.append(m1, np.array([[i, individual[i]]]), axis=0)
        elif integer == 2:
            m2 = np.append(m2, np.array([[i, individual[i]]]), axis=0)

    m1 = m1[m1[:,-1].argsort()] #数字が小さい順に並べ替え
    m2 = m2[m2[:,-1].argsort()]
    size1 = len(m1)
    size2 = len(m2)
    t1 = 0
    t2 = 0

    if m1.size != 0:
        for i in range(size1):
            a = m1[i,0]
            t1 += (p1[int(a)])
            fin_time1[int(a)] =t1
            finish[0][int(a)] = t1
            mcn[0][int(a)] = 1

    if m2.size != 0:
        for i in range(size2):
            a = m2[i,0]
            t2 += (p1[int(a)])
            fin_time1[int(a)] =t2
            finish[0][int(a)] = t2
            mcn[1][int(a)] = 1
   
    fin_seq1 = fin_time1.argsort() #装置の処理順序

    fin_time2 = np.array([]) #工程2の終了時間

    #初期値設定**************
    # 処理時刻の入れ物
    t21 = p2[fin_seq1[0]]
    t22 = p2[fin_seq1[1]]
    t23 = p2[fin_seq1[2]]

    # 装置の開始時刻
    c21 = fin_time1[fin_seq1[0]]
    c22 = fin_time1[fin_seq1[1]]
    c23 = fin_time1[fin_seq1[2]]

    # 終了時刻
    finish[1][fin_seq1[0]] = c21 + t21
    finish[1][fin_seq1[1]] = c22 + t22
    finish[1][fin_seq1[2]] = c23 + t23

    # 割り当て
    mcn[2][fin_seq1[0]] = 1
    mcn[3][fin_seq1[1]] = 1
    mcn[4][fin_seq1[2]] = 1

    fin_time2 = np.append(fin_time2, [c21+ t21])
    fin_time2= np.append(fin_time2, [c22 + t22])
    fin_time2 = np.append(fin_time2, [c23 + t23])
    # 初期値設定ここまで**************

    for i in range(mcn_num2,n_gene):
        #次のジョブが処理可能になる時刻
        #min(c21+t21,c22+t22,c23+t23) > fin_time1[fin_seq1[i]] ///処理可能な装置が空くまでしばらく処理を待つ状態
        # min(c21+t21,c22+t22,c23+t23) < fin_time1[fin_seq1[i]] ///前の処理が終わればすぐ次の処理にかかれる状態
        max_time = max(min(c21+t21,c22+t22,c23+t23),fin_time1[fin_seq1[i]])
        
        if max_time == c21+t21:
            c21 = max_time
            t21 = p2[fin_seq1[i]]
            fin_time2 = np.append(fin_time2, [c21+ t21])
            finish[1][fin_seq1[i]] = c21+t21
            mcn[2][fin_seq1[i]] = 1
        elif max_time ==c22+t22:
            c22 = max_time
            t22 = p2[fin_seq1[i]]
            fin_time2 = np.append(fin_time2, [c22 + t22])
            finish[1][fin_seq1[i]] = c22+t22
            mcn[3][fin_seq1[i]] = 1
        elif max_time ==c23+t23:
            c23 = max_time
            t23 = p2[fin_seq1[i]]
            fin_time2 = np.append(fin_time2, [c23 + t23])
            finish[1][fin_seq1[i]] = c23+t23
            mcn[4][fin_seq1[i]] = 1
        elif max_time == fin_time1[fin_seq1[i]]:
            max_sub_time = min(c21+t21,c22+t22,c23+t23)
            if max_sub_time == c21+t21:
                c21 = max_time
                t21 = p2[fin_seq1[i]]
                fin_time2 = np.append(fin_time2, [c21+ t21])
                finish[1][fin_seq1[i]] = c21+t21
                mcn[2][fin_seq1[i]] = 1
            elif max_sub_time == c22+t22:
                c22 = max_time
                t22 = p2[fin_seq1[i]]
                fin_time2 = np.append(fin_time2, [c22 + t22])
                finish[1][fin_seq1[i]] = c22+t22
                mcn[3][fin_seq1[i]] = 1
            elif max_sub_time == c23+t23:
                c23 = max_time
                t23 = p2[fin_seq1[i]]
                fin_time2 = np.append(fin_time2, [c23 + t23])
                finish[1][fin_seq1[i]] = c23+t23
                mcn[4][fin_seq1[i]] = 1

    fin_seq2 = fin_time2.argsort()


    fin_time3 = np.array([]) #工程2の終了時間
    t31 = p3[fin_seq2[0]] #処理時刻の入れ物
    t32 = p3[fin_seq2[1]] #処理時刻の入れ物
    c31 = fin_time2[fin_seq2[0]] #装置1の開始時刻
    c32 = fin_time2[fin_seq2[1]] #装置2の開始時刻
    finish[2][fin_seq2[0]] = c31 + t31
    finish[2][fin_seq2[1]] = c32 + t32
    mcn[5][fin_seq2[0]] = 1
    mcn[6][fin_seq2[1]] = 1
    fin_time3 = np.append(fin_time3, [c31 + t31])
    fin_time3= np.append(fin_time3, [c32 + t32])

    for i in range(mcn_num3,n_gene):
        max_time = max(min(c31+t31,c32+t32),fin_time2[fin_seq2[i]]) #次のジョブが処理可能になる時刻
        
        if max_time == c31+t31:
            c31 = max_time
            t31 = p3[fin_seq2[i]]
            fin_time3 = np.append(fin_time3, [c31+ t31])
            finish[2][fin_seq2[i]] = c31+t31
            mcn[5][fin_seq2[i]] = 1

        elif max_time ==c32+t32:
            c32 = max_time
            t32 = p3[fin_seq2[i]]
            fin_time3 = np.append(fin_time3, [c32 + t32])
            finish[2][fin_seq2[i]] = c32+t32
            mcn[6][fin_seq2[i]] = 1

        elif max_time == fin_time2[fin_seq2[i]]:
            max_sub_time = min(c31+t31,c32+t32)
            if max_sub_time == c31+t31:
                c31 = max_time
                t31 = p3[fin_seq2[i]]
                fin_time3 = np.append(fin_time3, [c31+ t31])
                finish[2][fin_seq2[i]] = c31+t31
                mcn[5][fin_seq2[i]] = 1
            else:
                c32 = max_time
                t32 = p3[fin_seq2[i]]
                fin_time3 = np.append(fin_time3, [c32 + t32])
                finish[2][fin_seq2[i]] = c32+t32
                mcn[6][fin_seq2[i]] = 1

    fin_seq3 = fin_time3.argsort()

    C_max = max(fin_time3)


    return C_max,finish,mcn,p1,p2,p3,fin_time1,fin_time2,fin_time3,

def cxTwoPointCopy(ind1, ind2): #2点交叉
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2

#突然変異する関数
def mutUniformDbl(individual, min_ind, max_ind, indpb): #indpbは突然変異率
    size = len(individual)
    for i, min, max  in zip(range(size), min_ind, max_ind):
        if random.random() < indpb:
            individual[i] = random.uniform(min, max)
    return individual,
    
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy) #交叉の戦略の指定
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
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.01, ngen=100, stats=stats,halloffame=hof) #計算を実行する．cxpb:交叉率，mutpb:個体突然変異率

    #popの中で最も良い解を見つけて表示させる
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    best_completetime = evalOneMax(best_ind)[1]
    best_mcnassign = evalOneMax(best_ind)[2]


    with open('test.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(range(30))
        writer.writerows(best_completetime)
        writer.writerows(best_mcnassign)
        writer.writerow(p1.tolist())
        writer.writerow(p2.tolist())
        writer.writerow(p3.tolist())


    return pop, stats, hof

if __name__ == "__main__":
    main()