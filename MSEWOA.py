# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:14:14 2020

@author: ZongSing_NB

Main reference:http://www.alimirjalili.com/WOA.html
Main reference:https://www.sciencedirect.com/science/article/pii/S1568494619307185
Main reference:https://www.mdpi.com/2076-3417/10/11/3667
Main reference:http://www.ejournal.org.cn/EN/abstract/abstract11643.shtml#
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class MSEWOA():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500, 
                 b=1, x_max=1, x_min=0, a_max=2, a_min=0, l_max=1, l_min=-1, a2_max=-1, a2_min=-2, 
                 strategy_init=True, strategy_bound=True, strategy_obl=True):
        self.fit_func = fit_func        
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter     
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max
        self.l_min = l_min
        self.b = b
        self.strategy_init = strategy_init
        self.strategy_bound = strategy_bound
        self.sc = MinMaxScaler(feature_range=(-1, 1))
        self.strategy_obl = strategy_obl
        
        self._iter = 1
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.X = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min           
        self.chaotic()
        
        score = self.obl()
        
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
           
        
    def opt(self):
        bound_max = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_max[np.newaxis, :])
        bound_min = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_min[np.newaxis, :])       
        
        while(self._iter<self.max_iter):
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            a2 = self.a2_max - (self.a2_max-self.a2_min)*(self._iter/self.max_iter)               
            for i in range(self.num_particle):
                p = np.random.uniform()
                R1 = np.random.uniform()
                R2 = np.random.uniform()
                A = 2*a*R1 - a
                C = 2*R2
                l = (a2-1)*np.random.uniform() + 1
                self.b = np.random.randint(low=0, high=500)
                                
                # case3-2. 改善速度
                if p>=0.5:
                    (5)
                    D = np.abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(self.b*l)*np.cos(2*np.pi*l)+self.gBest_X
                else:
                    # (4)
                    D = C*self.gBest_X - self.X[i, :]
                    self.X[i, :] = self.gBest_X - D*np.cos(2*np.pi*l)

            self.bound(bound_max, bound_min)
            
            score = self.obl()
            
            if np.min(score) < self.gBest_score:
                self.gBest_X = self.X[score.argmin()].copy()
                self.gBest_score = score.min().copy()
            # self.chaotic()   
            self.gBest_curve[self._iter] = self.gBest_score.copy()    
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()        

    def chaotic(self):
        if self.strategy_init==True:
            temp = (self.X - self.x_min)/(self.x_max - self.x_min)
            temp = 1 - 2*( np.cos( 4*np.arccos(temp) ) )**2
            self.X = temp*(self.x_max - self.x_min)+self.x_min
    
    def obl(self):
        if self.strategy_obl:
            bound_max = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_max[np.newaxis, :])
            bound_min = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_min[np.newaxis, :])
            k = np.random.uniform()
            alpha = self.X.min(axis=0)
            beta = self.X.max(axis=0)
            new_X = k*(alpha+beta)-self.X
    
            idx_too_high = bound_max < new_X
            idx_too_low = bound_min > new_X
            
            rand_X = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min
            new_X[idx_too_high] = rand_X[idx_too_high].copy()
            new_X[idx_too_low] = rand_X[idx_too_low].copy()
            
            self.X = np.concatenate((new_X, self.X), axis=0)        
            score = self.fit_func(self.X)
            top_k = score.argsort()[:self.num_particle]
            score = score[top_k].copy()
            self.X = self.X[top_k].copy()
        else:
            score = self.fit_func(self.X)
        
        return score
    
    def bound(self, bound_max, bound_min):
        if self.strategy_bound==True:
            idx_too_high = bound_max < self.X
            idx_too_low = bound_min > self.X
            R5 = np.random.uniform()
            R6 = np.random.uniform()
            bound_max_map = bound_max[idx_too_high] + \
                            R5*bound_max[idx_too_high]*(bound_max[idx_too_high]-self.X[idx_too_high])/self.X[idx_too_high]
            bound_min_map = bound_min[idx_too_low] + \
                            R6*np.abs(bound_min[idx_too_low]*(bound_min[idx_too_low]-self.X[idx_too_low]))/np.abs(self.X[idx_too_low])
            if np.any(bound_max_map==np.inf) or np.any(bound_min_map==np.inf):                   
                print(123)
            self.X[idx_too_high] = bound_max_map.copy()
            self.X[idx_too_low] = bound_min_map.copy()
        else:
            self.X[self.X>bound_max] = bound_max[self.X>bound_max]
            self.X[self.X<bound_min] = bound_max[self.X<bound_min]
        
        