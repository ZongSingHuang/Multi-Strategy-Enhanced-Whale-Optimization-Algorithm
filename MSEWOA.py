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

np.random.seed(42)

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
        self.chaotic()
        
        score = self.obl()
        
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
           
        
    def opt(self):        
        while(self._iter<self.max_iter):
            a2 = self.a2_max - (self.a2_max-self.a2_min)*(self._iter/self.max_iter)               
            for i in range(self.num_particle):
                p = np.random.uniform()
                R2 = np.random.uniform()
                C = 2*R2
                l = (a2-1)*np.random.uniform() + 1
                self.b = np.random.randint(low=0, high=500)
                                
                if p>=0.5:
                    D = np.abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(self.b*l)*np.cos(2*np.pi*l)+self.gBest_X
                else:
                    D = C*self.gBest_X - self.X[i, :]
                    self.X[i, :] = self.gBest_X - D*np.cos(2*np.pi*l)

            self.handle_bound()
            
            score = self.obl()
            
            if np.min(score) < self.gBest_score:
                self.gBest_X = self.X[score.argmin()].copy()
                self.gBest_score = score.min().copy() 
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
            # # https://www.mdpi.com/2076-3417/10/11/3667
            # init_X = np.random.uniform(low=-1.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = 1 - 2*( np.cos( 4*np.arccos(init_X) ) )**2
            # self.X = (self.X+1) / 2

            # # Chebyshev 
            # # https://www.mathworks.com/matlabcentral/fileexchange/47215-chaos-theory-and-meta-heuristics?s_tid=mwa_osa_a
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = np.cos((i+1)*np.arccos(init_X))
            
            # # Circle
            # a = 0.5
            # b = 0.2
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = np.mod(init_X+b-(a/(2*np.pi))*np.sin(2*np.pi*init_X), 1)

            # # Gauss/mouse
            # P = 0.4
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     flag1 = init_X!=0
            #     flag2 = init_X==0
            #     init_X[flag1] = 1/(np.mod(init_X[flag1], 1))
            #     init_X[flag2] = 1

            # # Iterative
            # a = 0.7
            # init_X = np.random.uniform(low=-1.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = np.sin(a*np.pi/init_X)
            # self.X = (self.X+1) / 2
            
            # # Logistic
            # a = 4
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = a*init_X*(1-init_X)
            
            # # Piecewise
            # P = 0.4
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     for j in range(init_X.shape[0]):
            #         if 0<=init_X[j] and init_X[j]<P:
            #             init_X[j] = init_X[j]/P
            #         elif P<=init_X[j] and init_X[j]<0.5:
            #             init_X[j] = (init_X[j]-P)/(0.5-P)
            #         elif 0.5<=init_X[j] and init_X[j]<1-P:
            #             init_X[j] = (1-P-init_X[j])/(0.5-P)
            #         elif 1-P<=init_X[j] and init_X[j]<1:
            #             init_X[j] = (1-init_X[j])/P
            #         else:
            #             print(666)

            # # Sine
            # a = 4
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = (a/4)*np.sin(np.pi*init_X)

            # # Singer
            # u = 1.07
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = u*(7.86*init_X-23.31*init_X**2+28.75*init_X**3-13.302875*init_X**4)

            # # Sinusoidal
            # a = 2.3
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     init_X = a*init_X**2 * np.sin(np.pi*init_X)
            
            # # Tent
            # init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
            # self.X = np.zeros((self.num_particle, self.num_dim))
            # for i in range(self.num_particle):
            #     self.X[i] = init_X
            #     bigger = init_X>=0.7
            #     smaller = init_X<0.7
            #     init_X[bigger] = (10/3)*(1-init_X[bigger])
            #     init_X[smaller] = init_X[smaller]/0.7
            
            self.X = self.X*(self.x_max-self.x_min) + self.x_min
        else:
            self.X = np.random.uniform(low=self.x_min, high=self.x_max, size=[self.num_particle, self.num_dim])
    
    def obl(self):
        if self.strategy_obl:
            k = np.random.uniform()
            alpha = self.X.min(axis=0)
            beta = self.X.max(axis=0)
            new_X = k*(alpha+beta)-self.X
    
            idx_too_high = self.x_max < self.X
            idx_too_low = self.x_min > self.X
            
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
    
    def handle_bound(self):
        if self.strategy_bound==True:
            r5 = np.random.uniform()
            r6 = np.random.uniform()
            idx_too_high = self.x_max < self.X
            idx_too_low = self.x_min > self.X
            
            bound_max_map =  self.x_max + r5 * (self.x_max-self.X)/self.X * self.x_max
            bound_min_map =  self.x_min + r6 * np.abs( (self.x_min-self.X)/self.X * self.x_min )
            self.X[idx_too_high] = bound_max_map[idx_too_high].copy()
            self.X[idx_too_low] = bound_min_map[idx_too_low].copy()

        else:
            self.X = np.clip(self.X, a_min=self.x_min, a_max=self.x_max)
        
        