#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:52:42 2018

@author: kongxiangyi
"""

import numpy as np
from matplotlib import pyplot as plt


class GA(object):
    def __init__(self, cities, gene_num, recombination_prob, 
                    mutation_prob):
        self.cities = cities
        self.N = len(cities)
        self.distance = self.get_distance()
        self.gene_num = gene_num
        #self.child_num = child_num
        #self.alpha = alpha
        self.recombination_prob = recombination_prob
        self.mutation_prob = mutation_prob
    
    def get_price(self, vec1, vec2):
        return np.linalg.norm(np.array(vec1) - np.array(vec2))
    
    def get_distance(self):
        distance = np.zeros((self.N, self.N))
        for i, c1 in enumerate(self.cities):
            line = []
            for j, c2 in enumerate(self.cities):
                if i != j:
                    line.append(self.get_price(c1, c2))
                else:
                    line.append(0.0)
            distance[i] = line
        return distance
    
    def init_genes(self):
        self.genes = []
        for i in range(self.gene_num):
            gene = list(range(self.N))  # 染色体基因编码
            np.random.shuffle(gene)
            # 调换染色体的头部基因为给定的基因
            if self.start != -1:
                for j, g in enumerate(gene):
                    if g == self.start:
                        gene[0], gene[j] = gene[j], gene[0]
            self.genes.append(gene)
    
    # 适应度打分函数，返回该染色体的适应度(优秀)分数
    def calc_fitness(self, gens):
        gens = np.copy(gens)
        gens = np.append(gens, gens[0])  # 在染色体的末尾添加上头部基因
        D = np.sum([self.get_price(self.cities[gens[i]], self.cities[gens[i+1]]) for i in range(len(gens) - 1)])
        return 1.0 / D
    
    # 赌徒转盘(精英染色体被选中的概率与适应度函数打分的结果成正比)
    def roulette_gambler(self, fit_pros):
        pick = np.random.random()
        for j in range(self.N):
            pick -= fit_pros[j]
            if pick <= 0:
                return j
        return 0
            
    def select(self):
        fit_pros = []
        [fit_pros.append(self.calc_fitness(self.genes[i])) for i in range(self.gene_num)]
        choice_gens = []
        for i in range(self.gene_num):
            j = self.roulette_gambler(fit_pros)  # 采用赌徒转盘选择出一个更好的染色体
            choice_gens.append(self.genes[j])  # 选中一个染色体
        for i in range(self.gene_num):
            self.genes[i] = choice_gens[i]  # 优胜劣汰，替换出更精英的染色体
            
    def recombination(self):
        move = 0  # 当前基因移动的位置
        while move < self.gene_num - 1:
            cur_pro = np.random.random()  # 决定是否进行交叉操作
            # 本次不进行交叉操作
            if cur_pro > self.recombination_prob:
                move += 2
                continue
            parent1, parent2 = move, move + 1  # 准备杂交的两个染色体(种群)
            index1 = np.random.randint(1, self.N - 2)
            index2 = np.random.randint(index1, self.N - 2)
            if index1 == index2:
                continue
            temp_gen1 = self.genes[parent1][index1:index2+1]  # 交叉的基因片段1
            temp_gen2 = self.genes[parent2][index1:index2+1]  # 交叉的基因片段2
            # 杂交插入染色体片段
            temp_parent1, temp_parent2 = np.copy(self.genes[parent1]).tolist(), np.copy(self.genes[parent2]).tolist()
            temp_parent1[index1:index2+1] = temp_gen2
            temp_parent2[index1:index2+1] = temp_gen1
            # 消去冲突
            pos = index1 + len(temp_gen1)  # 插入杂交基因片段的结束位置
            conflict1_ids, conflict2_ids = [], []
            [conflict1_ids.append(i) for i, v in enumerate(temp_parent1) if v in temp_parent1[index1:pos]
             and i not in list(range(index1, pos))]
            [conflict2_ids.append(i) for i, v in enumerate(temp_parent2) if v in temp_parent2[index1:pos]
             and i not in list(range(index1, pos))]
            for i, j in zip(conflict1_ids, conflict2_ids):
                temp_parent1[i], temp_parent2[j] = temp_parent2[j], temp_parent1[i]
            self.genes[parent1] = temp_parent1
            self.genes[parent2] = temp_parent2
            move += 2
    
    def mutation(self):
        for i in range(self.gene_num):
            cur_pro = np.random.random()  # 决定是否进行变异操作
            # 本次不进行变异操作
            if cur_pro > self.mutation_prob:
                continue
            index1 = np.random.randint(1, self.N - 2)
            index2 = np.random.randint(1, self.N - 2)
            self.genes[i][index1], self.genes[i][index2] = self.genes[i][index2], self.genes[i][index1]

    
    def solve(self, max_evolution_num):
        self.best_gens = [-1 for _ in range(self.N)]  # 精英染色体(基因排列)
        self.min_distance = np.inf  # 最短路径长度
        self.best_fit_index = 0  # 最短路径出现的代数
        self.start = 0  # 种群的初始位置
        self.length_list = []
        # 开始进化
        for step in range(max_evolution_num):
            distance_arr = []  # 每一个染色体的总路程数组
            self.init_genes()  # 种群初始化，得到所有种群
            self.select()      # 选择操作，选择出每个种群的精英染色体
            self.recombination()  # 交叉操作，两个染色体互相杂交产生新的染色体
            self.mutation()    # 变异操作，单个染色体变异
            #chroms = reverse(citys, chroms)  # 变异操作，单个染色体变得更加优秀
            [distance_arr.append(1.0 / self.calc_fitness(self.genes[i])) for i in range(self.gene_num)]
            best_gens_idx = np.argmin(distance_arr)  # 找到最短的路径位置，对应于精英染色体位置
            self.length_list.append(distance_arr[best_gens_idx])
            if distance_arr[best_gens_idx] < self.min_distance:
                self.min_distance = distance_arr[best_gens_idx]  # 更新最短路径
                self.best_gens = self.genes[best_gens_idx]  # 更新精英染色体
                self.best_gens.append(self.start)
                self.best_fit_index = step + 1  # 更新最短路径出现的代数
        print('通过{}代的基因进化，精英染色体出现在第{}代，基因序列为：'.format(max_evolution_num, self.best_fit_index))
        [print(v, end=',' if i < len(self.best_gens) - 1 else '\n') for i, v in enumerate(self.best_gens)]
        print('精英染色体映射的最短路径为：{}'.format(self.min_distance))
        
    # 可视化画出哈密顿回路
    def draw(self):
        H_path = []
        [H_path.append((cur_gen, self.best_gens[i+1])) for i, cur_gen in enumerate(self.best_gens[:-1])]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        for (from_, to_) in H_path:
            p1 = plt.Circle(self.cities[from_], 0.2, color='blue')
            p2 = plt.Circle(self.cities[to_], 0.2, color='blue')
            ax.add_patch(p1)
            ax.add_patch(p2)
            ax.plot((self.cities[from_][0], self.cities[to_][0]), 
                    (self.cities[from_][1], self.cities[to_][1]), color='blue')
            #ax.annotate(s=to_, xy=self.cities[to_], xytext=(-8, -4), 
                        #textcoords='offset points', fontsize=20)
        ax.axis('equal')
        ax.grid()
        plt.show()
        x = list(range(len(self.length_list)))
        plt.plot(x, self.length_list)
        plt.xlabel('Iterations')
        plt.ylabel('Path length')
        best_length = min(self.length_list)
        plt.title(f'Final length: {best_length:.4f}')
        plt.show()

def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [c.split(' ') for c in lines]
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities
        
if __name__ == '__main__':
    max_evolution_num = 500  # 最大进化代数
    gene_num = 100  # 种群数目
    recombination_pro = 0.9  # 交叉概率
    mutation_pro = 1.0  # 变异概率
    #cities = [(2, 6), (2, 4), (1, 3), (4, 6), (5, 5), (4, 4), (6, 4), (3, 2)]  # 城市坐标
    cities = load_data('burma14.txt')
    for i in range(1):
        ga = GA(cities, gene_num, recombination_pro, mutation_pro)
        ga.solve(max_evolution_num)
        ga.draw()
    #ga.draw()