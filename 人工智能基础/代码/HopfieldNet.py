#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 09:12:33 2018

@author: kongxiangyi
"""

import numpy as np
from matplotlib import pyplot as plt

class HopfieldNet(object):
    
    def __init__(self, citys, U0, step, iter_num, A, D):
        self.citys = citys
        self.N = len(citys)
        self.U0 = U0
        self.step = step
        self.iter_num = iter_num
        self.A = A
        self.D = D
        #self.U = 1 / 2 * U0 * np.log(len(citys) - 1) + (2 * (np.random((len(citys), len(citys)))) - 1)
        #self.V = self.calc_V()
        self.energy = np.array([0.0 for x in range(iter_num)])
        self.distance = self.get_distance()
        self.best_distance = np.inf
        self.best_route = []
        self.H_path = []
    
    def get_price(self, vec1, vec2):
        return np.linalg.norm(np.array(vec1) - np.array(vec2))
    
    def get_distance(self):
        distance = np.zeros((self.N, self.N))
        for i, c1 in enumerate(self.citys):
            line = []
            for j, c2 in enumerate(self.citys):
                if i != j:
                    line.append(self.get_price(c1, c2))
                else:
                    line.append(0.0)
            distance[i] = line
        return distance
    
    def calc_U(self):
        return self.U + self.du * self.step
    
    def calc_V(self):
        return 1 / 2 * (1 + np.tanh(self.U / self.U0))
    
    def calc_du(self):
        a = np.sum(self.V, axis=0) - 1  # 按列相加
        b = np.sum(self.V, axis=1) - 1  # 按行相加
        t1 = np.zeros((self.N, self.N))
        t2 = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                t1[i, j] = a[j]
        for i in range(self.N):
            for j in range(self.N):
                t2[j, i] = b[j]
        # 将第一列移动到最后一列
        c_1 = self.V[:, 1:self.N]
        c_0 = np.zeros((self.N, 1))
        c_0[:, 0] = self.V[:, 0]
        c = np.concatenate((c_1, c_0), axis=1)
        c = np.dot(self.distance, c)
        return -self.A * (t1 + t2) - self.D * c
    
    def calc_energy(self):
        t1 = np.sum(np.power(np.sum(self.V, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(self.V, axis=1) - 1, 2))
        idx = [i for i in range(1, self.N)]
        idx = idx + [0]
        Vt = self.V[:, idx]
        t3 = self.distance * Vt
        t3 = np.sum(np.sum(np.multiply(self.V, t3)))
        e = 0.5 * (self.A * (t1 + t2) + self.D*t3)
        return e
    
    def calc_distance(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.distance[path[i]][path[i + 1]]
        return dis
    
    def draw(self):
        fig = plt.figure()
        # 绘制哈密顿回路
        ax1 = fig.add_subplot(121)
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        for (from_, to_) in self.H_path:
            p1 = plt.Circle(citys[from_], 0.2, color='blue')
            p2 = plt.Circle(citys[to_], 0.2, color='blue')
            ax1.add_patch(p1)
            ax1.add_patch(p2)
            ax1.plot((citys[from_][0], citys[to_][0]), (citys[from_][1], citys[to_][1]), color='blue')
            #ax1.annotate(s=to_, xy=citys[to_], xytext=(-8, -4), textcoords='offset points', fontsize=20)
        ax1.axis('equal')
        ax1.grid()
        plt.show()
        # 绘制能量趋势图
        plt.plot(np.arange(0, len(self.energy), 1), self.energy, color='blue')
        plt.show()
    
    def solve(self):
        self.U = 1 / 2 * self.U0 * np.log(self.N - 1) 
        self.U += (2 * (np.random.random((self.N, self.N))) - 1)
        self.V = self.calc_V()
        for n in range(self.iter_num):
            self.du = self.calc_du()
            self.U = self.calc_U()
            self.V = self.calc_V()
            self.energy[n] = self.calc_energy()
            newV = np.zeros([self.N, self.N])
            route = []
            for i in range(self.N):
                mm = np.max(self.V[:, i])
                for j in range(self.N):
                    if self.V[j, i] == mm:
                        newV[j, i ] = 1
                        route += [j]
                        break
            if len(np.unique(route)) == self.N:
                route.append(route[0])
                dis = self.calc_distance(route)
                if dis < self.best_distance:
                    self.H_path = []
                    self.best_distance = dis
                    self.best_route = route
                    [self.H_path.append((route[i], route[i + 1])) for i in range(len(route) - 1)]
                    print('第{}次迭代找到的次优解距离为：{}，能量为：{}，路径为：'.format(n, self.best_distance, self.energy[n]))
                    [print(v, end=',' if i < len(self.best_route) - 1 else '\n') for i, v in enumerate(self.best_route)]  

def load_data(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = [c.split(' ') for c in lines]
    cities = [(float(c[1]), float(c[2])) for c in lines]
    return cities

if __name__ == '__main__':
    
    #citys = np.array([[2, 6], [2, 4], [1, 3], [4, 6], [5, 5], [4, 4], [6, 4], [3, 2]])
    citys = load_data('wi29.txt')
    U0 = 0.0009
    step = 0.0001
    iter_num = 10000
    A = 64
    D = 4
    HNN = HopfieldNet(citys, U0, step, iter_num, A, D)
    HNN.solve()
    if len(HNN.H_path) > 0:
        HNN.draw()
    else:
        print('no solution')
