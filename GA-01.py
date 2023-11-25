# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:19:36 2023

@author: Amin
"""

import numpy as np
import random as rand


class GA():
    
    domXi_min = -10 # مقدار min یک جواب
    domXi_max = 10 # مقدار Max یک جواب
    domXi_float = 3 # تعداد اعشار
    randSeed = -1     # seed < 0 --> Random , seed = -1 -> Fixed Random series
    typeOfCrossover = 'TwoPoint'
    
    def __init__(self, popSize, # اندازه حمعیت
                         MaxGeneration, # تعداد نسلی که باید تولید شود
                         xDim, # ابعاد مسئله (تعداد مجهولات)
                         fitnessFunc, # سوالی که باید مقدار بهینه آن پیدا شود
                         fitType, # مسئله از نوع Max است یا Min
                         pc, # احتمال ادغام هر نمونه از جمعیت - معمولا بین .9 تا .95
                         pm): #     احتمال جهش معمولا بین .1 تا .15
        
        self.popSize=popSize
        self.MaxGeneration=MaxGeneration
        self.xDim=xDim
        self.fitnessFunc=fitnessFunc
        self.fitType=fitType
        self.pc=pc
        self.pm=pm
        
        # population : ماتریس جمعیت
        self.population = np.zeros((self.popSize, xDim))
        # # fit : ماتریس جواب برای هر نمونه از جمعیت
        # self.fit = np.zeros(self.popSize)
        
    # راه اندازی اولیه جمعیت    
    def init(self):

        # Seed : آیا ست شده است یا خیر؟
        if self.randSeed > 0:
            rand.seed(self.randSeed)
            np.random.seed(self.randSeed)

        # population : به تعداد جمعیت جواب تصادفی تولید می کنیم
        # fit : ماتریس از جواب های تابع fit
        a, b = self.population.shape
        for i in range(a):
            self.population[i] = np.round(np.random.uniform(self.domXi_min, self.domXi_max, self.xDim), self.domXi_float)
            # self.fit[i] = self.fitnessFunc(self.population[i])
        
    def single_point_crossover(self, a, b):
        x = rand.randint(0, len(a))
        a = np.array(a)
        b = np.array(b)
        a_new = np.append(a[:x], b[x:])
        b_new = np.append(b[:x], a[x:])
        return a_new, b_new
    def two_point_crossover(self, a, b):
        x, y = 0, 0
        while x >= y:
            x = rand.randint(0, len(a))
            y = rand.randint(0, len(a))
        a = np.array(a)
        b = np.array(b)
        a_new = np.append(np.append(a[:x], b[x:y]), a[y:])
        b_new = np.append(np.append(b[:x], a[x:y]), b[y:])
        return a_new, b_new

    def crossOver(self):        
        poolParents = np.zeros((1, self.xDim))
        first = True
        for pop in self.population:
            if rand.random() <= self.pc:
                if first:
                    poolParents = poolParents.reshape(1,-1)
                    first = False
                else:
                    poolParents = np.concatenate((poolParents, pop.reshape(1,-1)))
                    
        np.random.shuffle(poolParents)
        
        poolNewPops= np.zeros((1, self.xDim))
        first = True
        for idx in range(0, len(poolParents)-1, 2):
            parent1 = poolParents[idx]
            parent2 = poolParents[idx+1]
            
            if self.typeOfCrossover == 'OnePoint':
                newPop1, newPop2 = self.single_point_crossover(parent1, parent2)
            elif self.typeOfCrossover == 'TwoPoint':
                newPop1, newPop2 = self.two_point_crossover(parent1, parent2)
            
                
            if first:
                newPop1 = newPop1.reshape(1,-1)
                newPop2 = newPop2.reshape(1,-1)
                poolNewPops = np.concatenate((newPop1, newPop2))
                first = False
            else:
                newPop1 = newPop1.reshape(1,-1)
                newPop2 = newPop2.reshape(1,-1)
                poolNewPops = np.concatenate((poolNewPops, newPop1, newPop2))
                
        return poolNewPops
    
    def mutation(self, pops): 
        for idx in range(len(pops)):
            if rand.random() < self.pm:
                p = np.random.choice(self.xDim, 1)
                pops[idx, p] = np.round(rand.uniform(self.domXi_min, self.domXi_max), self.domXi_float)

    
    def selection(self, newPops):
        
        totPops = np.concatenate((self.population, newPops))
        sortedPops = sorted(totPops, key=self.fitnessFunc, reverse=self.fitType=='Max')
        sortedPops = np.array(sortedPops[:self.popSize])
        
        return sortedPops


    def run(self):
        self.init()
        print(self.population[:10])

        for i in range(self.MaxGeneration):
            x = self.crossOver()
            self.mutation(x)
            self.population = self.selection(x).copy()
            print(self.population[:5])
            print(i, '-' * 30)
            print('Best fit: ', self.fitnessFunc(self.population[0]))
            print('-' * 30)

        
        print(i, '-' * 30)
        print(self.fitnessFunc(self.population[0]))
        print(self.population[0])


def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def sumation(x):
    return sum([i**2 for i in x])

    

from pymoo.problems import get_problem
problem = get_problem("rosenbrock", n_var=3)


ga = GA(popSize=100, 
        MaxGeneration=50, 
        xDim=3, 
        fitnessFunc=problem.evaluate, 
        fitType='Min', 
        pc=.9, 
        pm=.1)        

ga.domXi_float = 10
ga.randSeed = -1
ga.typeOfCrossover = 'OnePoint'

ga.run()
