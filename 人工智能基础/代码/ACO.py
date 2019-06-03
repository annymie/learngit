#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:47:08 2018

@author: kongxiangyi
"""

import numpy as np
import matplotlib.pyplot as plt

class ACO(object):
    
    def __init__(self, cities, ants_num, alpha=1, beta=2, rho=0.5, Q=100):
        self.cities = cities
        self.N = len(cities)
        self.distance = self.get_distance()
        self.pheromone = np.ones((self.N, self.N))
        self.ants_num = ants_num
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        #self.best_path_length = float('inf')
        #self.best_path = None
        
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
    
    def calc_distance(self, path):
        dis = 0.0
        for i in range(-1, self.N - 1):
            dis += self.distance[path[i]][path[i + 1]]
        return dis
    
    def choose_city(self, current_city, available_cities):
        city_probs = []
        for i in available_cities:
            pheromone = pow(self.pheromone[current_city][i], self.alpha)
            distance = pow(self.distance[current_city][i], self.beta)
            city_probs.append(pheromone / distance)
        sum_weight = sum(city_probs)
        city_probs = list(map(lambda x: x / sum_weight, city_probs))
        next_city = np.random.choice(available_cities, p=city_probs)
        return next_city
    
    def ant_go(self):
        available_cities = list(range(self.N))
        path = []

        current_city = np.random.choice(available_cities)
        available_cities.remove(current_city)
        path.append(current_city)

        for i in range(self.N - 1):
            next_city = self.choose_city(current_city, available_cities)
            available_cities.remove(next_city)
            path.append(next_city)
            current_city = next_city
        return path
    
    def update_pheromone(self, path_list):
        pheromones = np.zeros((self.N, self.N))
        for path_length, path in path_list:
            amount_pheromone = self.Q / path_length
            for i in range(-1, self.N - 1):
                pheromones[path[i], path[i + 1]] += amount_pheromone
                pheromones[path[i + 1], path[i]] += amount_pheromone
        self.pheromone = self.pheromone * self.rho + pheromones
    
    def search_once(self):
        path_list = []
        best_length = float('inf')
        best_path = None
        for i in range(self.ants_num):
            path = self.ant_go()
            length = self.calc_distance(path)
            path_list.append((length, path))
            if length < best_length:
                best_length, best_path = length, path
        self.update_pheromone(path_list)
        return best_length, best_path


    def solve(self, max_iter=200):
        self.best_length = float('inf')
        self.best_path = None
        self.length_list = []
        self.best_fit_index = 0
        for i in range(1, max_iter + 1):
            
            length, path = self.search_once()
            self.length_list.append(length)

            if length < self.best_length:
                self.best_fit_index = i
                self.best_length, self.best_path = length, path

        #return best_length, best_path
        #self.best_length = float('inf')  #最短路径长度
        #self.best_path = None            #最短路径
        #self.best_fit_index = 0          #最短路径出现代数
        #self.length_list = []
        
        #for step in range(max_iter):
            #thisbest = float('inf')
            #for i in range(self.ants_num):
                #path_list = []
                #path = self.ant_go()
                #length = self.calc_distance(path)
                #path_list.append((length, path))
                #if(length<thisbest):
                    #thisbest = length
                #if length < self.best_length:
                    #self.best_length = length
                    #self.best_path = path
                    #self.best_path.append(path[0])
                    #self.best_fit_index = step + 1
            #self.update_pheromone(path_list)
            #self.length_list.append(thisbest)
            #length_list.append(length)
        print('通过{}代的蚁群寻找，最短路径出现在第{}代，路径为：'.format(max_iter, self.best_fit_index))
        [print(v, end=',' if i < len(self.best_path) - 1 else '\n') for i, v in enumerate(self.best_path)]
        print('最短路径长度为：{}'.format(self.best_length))
        
    # 可视化画出哈密顿回路
    def draw(self):
        self.best_path.append(self.best_path[0])
        H_path = []
        [H_path.append((cur_gen, self.best_path[i+1])) for i, cur_gen in enumerate(self.best_path[:-1])]
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
    ants_num = 50
    alpha = 1
    beta = 2
    rho = 0.5
    Q = 100
    max_iter = 200
    #cities = [(2, 6), (2, 4), (1, 3), (4, 6), (5, 5), (4, 4), (6, 4), (3, 2)]  # 城市坐标
    cities = load_data('qa194.txt')
    for i in range(1):
        aco = ACO(cities, ants_num, alpha, beta, rho, Q)
        aco.solve(max_iter)
        aco.draw()
