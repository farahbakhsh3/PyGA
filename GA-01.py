# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:19:36 2023

@author: Amin
"""

import numpy as np
import random as rand
from tqdm import tqdm

class GA():
    
    domXi_min = -10     # مقدار min یک جواب
    domXi_max = 10      # مقدار Max یک جواب
    domXi_float = 3     # تعداد اعشار
    randSeed = -1       # seed < 0 --> Random , seed = -1 -> Fixed Random series
    typeOfCrossover = 'SinglePoint'
    floatAnswer = True  # int / float
    
    def __init__(self, popSize, # اندازه حمعیت
                       MaxGeneration, # تعداد نسلی که باید تولید شود
                       xDim, # ابعاد مسئله (تعداد مجهولات)
                       fitnessFunc, # سوالی که باید مقدار بهینه آن پیدا شود
                       fitType = 'Min', # مسئله از نوع Max است یا Min
                       pc=0.95, # احتمال ادغام هر نمونه از جمعیت - معمولا بین .9 تا .95
                       pm=0.15): #     احتمال جهش معمولا بین .1 تا .15
        
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
            if self.floatAnswer:
                self.population[i] = np.round(np.random.uniform(self.domXi_min, self.domXi_max, self.xDim), self.domXi_float)
            else:
                self.population[i] = np.random.randint(self.domXi_min, self.domXi_max, self.xDim)
            # self.fit[i] = self.fitnessFunc(self.population[i])
        
    def single_point_crossover(self, a, b):
        x = rand.randint(1, len(a)-1)
        a = np.array(a)
        b = np.array(b)
        a_new = np.append(a[:x], b[x:])
        b_new = np.append(b[:x], a[x:])
        
        return a_new, b_new
    
    def two_point_crossover(self, a, b):
        if all(a == b):
            return a, b        
        
        x, y = 0, 0
        while x == y:
            x = rand.randint(0, len(a))
            y = rand.randint(0, len(a))
        if x > y:
            x, y = y, x
        a = np.array(a)
        b = np.array(b)
        a_new = np.append(np.append(a[:x], b[x:y]), a[y:])
        b_new = np.append(np.append(b[:x], a[x:y]), b[y:])
        while all(a_new==b_new):
            a_new, b_new = self.two_point_crossover(a_new, b_new)
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
            
            if self.typeOfCrossover == 'SinglePoint':
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
    
    def mutation(self, childs): 
        mutants = np.zeros((1, self.xDim))
        First = True        
        for idx in range(len(childs)):
            if rand.random() <= self.pm:
                p = np.random.choice(self.xDim, 1)
                mutant = childs[idx].copy()
                if self.floatAnswer:
                    mutant[p] = np.round(np.random.uniform(self.domXi_min, self.domXi_max), self.domXi_float)
                else:
                    mutant[p] = np.random.randint(self.domXi_min, self.domXi_max)
                
                if First:
                    mutants = mutant.reshape((1, -1)).copy()
                    First = False
                else:
                    mutants = np.concatenate((mutants, mutant.reshape((1, -1))))
                    
        return mutants
    
    def selection(self, childs, mutants):
        
        # print('_'*10)
        # print(f'Pop: {self.population.shape}')
        # print(f'Childs: {childs.shape}')
        # print(f'Mutants: {mutants.shape}')
        # print('_'*10)
        # print('_'*10, 'population')
        # for i in range(len(self.population)):
        #     print(f'{i} : {self.population[i]} , --> Fit: {self.fitnessFunc(self.population[i])}')
        # print('_'*10, 'childs')
        # for i in range(len(childs)):
        #     print(f'{i} : {childs[i]} , --> Fit: {self.fitnessFunc(childs[i])}')
        # print('_'*10, 'mutants')
        # for i in range(len(mutants)):
        #     print(f'{i} : {mutants[i]} , --> Fit: {self.fitnessFunc(mutants[i])}')
        # print('_'*30)
        
        totPops = np.concatenate((self.population, childs, mutants))
        sortedPops = sorted(totPops, key=self.fitnessFunc, reverse=self.fitType=='Max')
        sortedPops = np.array(sortedPops[:self.popSize])

        return sortedPops


    def run(self):
        self.init()

        for i in tqdm(range(self.MaxGeneration)):
            childs = self.crossOver()
            mutants = self.mutation(childs)
            self.population = self.selection(childs, mutants).copy()
    
        print()
        print('_' * 30)
        print(f'Best Fit : {self.fitnessFunc(self.population[0])}')
        print('_' * 30)        
        print(f'Anwers: \n{self.population[:50]}')


def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def sumation(x):
    return sum([i**2 for i in x])


def sum2(x):
    return (x[0]-1)**2 + x[1]**2 + (x[2]+1)**2


def darpanjare(x):
    # |==========================|
    # | product  | choob | sood  |
    # |==========================|
    # | darb     |   10  |  30   | -> 3
    # | panjare  |   8   |  24   | -> 3
    # | miz      |   10  |  30   | -> 3
    # |==========================|

    max_choob = 1000

    darb = x[0]
    panjare = x[1]
    miz = x[2]

    masraf_darb = 10
    masraf_panjare = 8
    masraf_miz = 10
    
    sood_darb = 30
    sood_panjare = 24
    sood_miz = 30

    if darb*masraf_darb + panjare*masraf_panjare + miz*masraf_miz > max_choob:
        return 0
    
    return darb*sood_darb + panjare*sood_panjare + miz*sood_miz
    

# from pymoo.problems import get_problem
# problem = get_problem("rosenbrock", n_var=3)


ga = GA(popSize=10, 
        MaxGeneration=2000, 
        xDim=3, 
        fitnessFunc=darpanjare, 
        fitType='Max', 
        pc=.95, 
        pm=.15)        

ga.floatAnswer = False
ga.domXi_min = 0
ga.domXi_max = 500
ga.randSeed = -1
ga.typeOfCrossover = 'SinglePoint'

ga.run()
