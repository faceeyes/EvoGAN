# -*- coding: utf-8 -*-
import os
import random
import csv
import numpy as np
from tqdm import tqdm

from GAN.solvers import create_solver
from FER.recognizer import Recognizer
from EC.individual import Individual

class GA(object):
    """遗传算法类"""
    def __init__(self, opt, name, tar):

        os.mkdir(os.path.join(opt.results, name))
        os.mkdir(os.path.join(opt.results, name, "imgs"))
        self.f = open(os.path.join("results", name, "data.csv"), 'a', newline="")
        self.csvwriter = csv.writer(self.f, dialect=("excel"))
        self.csvwriter.writerow([name, "generation", "individual", "AU", "score", "max", "min", "avg"])
        self.opt = opt
        self.solver = create_solver(self.opt)
        self.solver.init_test_setting(self.opt)
        self.recognizer = Recognizer()
        self.name = name
        self.tar = tar
        self.pop = []                           # 种群
        self.max = None                          # 保存这一代中最好的个体
        self.min = None
        self.avg = None
        self.generation = 0
        self.sum = 0.0                         # 适配值之和，用于选择是计算概率

        self.initPopulation()

    def rand_new_ind(self):
        gene = []
        for j in range(self.opt.gene_length):
            gene.append(random.random())
        return Individual(gene)

    def initPopulation(self):
        self.pop = []
        self.best = None
        self.generation = 0
        for i in range(self.opt.life_count):
            self.pop.append(self.rand_new_ind())

    def sort(self):
        for i in range(len(self.pop)):
            if i >= self.opt.life_count:
                self.pop = self.pop[:self.opt.life_count]
                break
            max = self.pop[i].score
            pos = i
            for j in range(i, len(self.pop)):
                if self.pop[j].score > max:
                    max = self.pop[j].score
                    pos = j

            temp = self.pop[i]
            self.pop[i] = self.pop[pos]
            self.pop[pos] = temp



    def eval(self, individual, i):
        exp = np.array(individual)
        self.solver.expression = exp
        img = self.solver.test_ops()
        self.solver.test_save_imgs(img, self.name, str(self.generation) + "_" + str(i))
        res = self.recognizer.recognize(img).tolist()

        score = 0
        for i in range(7):
            score += abs(res[i] - self.tar[i])

        return 1 - score / 2

    def judge(self):
        self.sum = 0.0
        self.max = self.pop[0]
        self.min = self.pop[0]
        i = 0
        for individual in self.pop:
            individual.score = self.eval(individual.gene, i)
            self.sum += individual.score
            if self.max.score < individual.score:
                self.max = individual
            if self.min.score > individual.score:
                self.min = individual
            i += 1
        self.avg = self.sum / i
        i = 0

        self.sort()
        for individual in self.pop:
            self.csvwriter.writerow([self.name, self.generation, i, str(individual.gene), str(individual.score), str(self.max.score), str(self.min.score), str(self.avg)])
            i = i + 1


    def cross(self, parent1, parent2):
        index1 = random.randint(0, self.opt.gene_length - 1)
        index2 = random.randint(index1, self.opt.gene_length - 1)
        newGene = []
        for i in range(self.opt.gene_length):
            if i < index1 or i > index2:
                newGene.append(parent1.gene[i])     # 插入基因片段
            else:
                newGene.append(parent2.gene[i])
        return newGene

    def mutation(self, gene):
        index = random.randint(0, self.opt.gene_length - 1)
        index2 = random.randint(0, 1)

        newGene = gene[:]  # 产生一个新的基因序列，以免变异的时候影响父种群

        if index2 == 0:
            newGene[index] /= 2
        else:
            newGene[index] *= 2
        return newGene

    def getOne(self):
        r = random.randint(0, len(self.pop) - 1)
        return self.pop[r]

    def newChild(self, parent1, newLives):
        rate = random.random()

        pc = self.opt.pc_k * (self.max.score - parent1.score) / (self.max.score - self.avg)
        pm = self.opt.pm_k * (self.max.score - parent1.score) / (self.max.score - self.avg)
        # 按概率交叉
        if rate < pc + pm:
            # 交叉
            if rate < pc:
                parent2 = self.getOne()
                while parent2.score == parent1.score:
                    parent2 = self.getOne()

                gene = self.cross(parent1, parent2)
            else:
                gene = self.mutation(parent1.gene)
            newLives.append(Individual(gene))

        newLives.append(parent1)

    def next(self):
        newLives = []

        for individual in self.pop:
            self.newChild(individual, newLives)

        self.pop = newLives
        self.generation += 1
        self.judge()

    def run(self):

        self.initPopulation()
        self.judge()

        for i in tqdm(range(self.opt.generation_count)):
            self.next()
        self.f.close()
        return

