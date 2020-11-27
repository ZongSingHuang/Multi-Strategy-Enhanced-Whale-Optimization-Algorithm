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
import time

class MSEWOA():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500,
                 x_max=1, x_min=0, real_max=0, real_min=0, method='modif_s'):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.x_max = x_max
        self.x_min = x_min
        self.real_max = []
        self.real_min = []
        self.method = method
        self.final = None
        self.feature_size = self.num_dim - len(self.real_max)
        self.para_size = len(self.real_max)
        
        self._iter = 1
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.gBest_curve_error = np.zeros(self.max_iter)
        self.gBest_curve_feature = np.zeros(self.max_iter)
        self.super_hero_mode = True
        
        self.chaotic()
        
        score, score_error, score_feautre = self.obl()
        
        self.gBest_score = score.min().copy()
        self.gBest_X = self.X[score.argmin()].copy()
        self.gBest_curve[0] = self.gBest_score.copy()
        
        self.aaa = score_error[score.argmin()]
        self.bbb = score_feautre[score.argmin()]
        self.gBest_curve_error[0] = score_error[score.argmin()]
        self.gBest_curve_feature[0] = score_feautre[score.argmin()]
           
        
    def opt(self):
        self.bound_max = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_max[np.newaxis, :])
        self.bound_min = np.dot(np.ones(self.num_particle)[:, np.newaxis], self.x_min[np.newaxis, :])
        
        while(self._iter<self.max_iter):
            start = time.time()
            a2 = -1 - self._iter/self.max_iter
            
            for i in range(self.num_particle):
                p = np.random.uniform()
                R2 = np.random.uniform()
                C = 2*R2
                l = (a2-1)*np.random.uniform() + 1
                b = np.random.randint(low=0, high=500)

                if p>=0.5:
                    D = np.abs(self.gBest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(b*l)*np.cos(2*np.pi*l)+self.gBest_X
                else:
                    D = C*self.gBest_X - self.X[i, :]
                    self.X[i, :] = self.gBest_X - D*np.cos(2*np.pi*l)
            
            self.handle_bound()
            
            score, score_error, score_feautre = self.obl()
            
            if np.min(score) < self.gBest_score:
                self.gBest_X = self.X[score.argmin()].copy()
                self.gBest_score = score.min().copy()
                self.aaa = score_error[score.argmin()]
                self.bbb = score_feautre[score.argmin()]
            self.gBest_curve_error[self._iter] = self.aaa
            self.gBest_curve_feature[self._iter] = self.bbb
            self.gBest_curve[self._iter] = self.gBest_score.copy()
            self._iter = self._iter + 1
            
            print('iter '+'['+str(self._iter)+']'+'\t'+
                  'score '+'['+str(round(self.gBest_score, 3))+']'+'\t'+
                  'time '+'['+str(round(time.time()-start, 3))+']')

            feature = np.array(np.where(self.final[self.para_size:]==1)).ravel().tolist()
            para = np.round(self.gBest_X[:self.para_size], 5)
            print('para '+str(para)+'\t'+
                  'feature '+str(feature)+'\n')
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot_curve_plus(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss', )
        plt.plot(self.gBest_curve_error, label='point')
        plt.plot(self.gBest_curve_feature, label='feature')
        plt.grid()
        plt.legend()
        plt.show()

    def chaotic(self):
        init_X = np.random.uniform(low=0.0, high=1.0, size=[1, self.num_dim])
        self.X = np.zeros((self.num_particle, self.num_dim))
        for i in range(self.num_particle):
            self.X[i] = init_X
            bigger = init_X>=0.7
            smaller = init_X<0.7
            init_X[bigger] = (10/3)*(1-init_X[bigger])
            init_X[smaller] = init_X[smaller]/0.7
        self.X = self.X*(self.x_max-self.x_min) + self.x_min
        
    def obl(self):
        k = np.random.uniform()
        alpha = self.X.min(axis=0)
        beta = self.X.max(axis=0)

        new_X = k*(alpha+beta)-self.X
        
        idx_too_low = new_X < self.x_min
        idx_too_high = new_X > self.x_max
        
        rand_X = np.random.uniform(low=alpha, high=beta, size=[self.num_particle, self.num_dim])
        new_X[idx_too_high] = rand_X[idx_too_high].copy()
        new_X[idx_too_low] = rand_X[idx_too_low].copy()
        
        self.X = np.concatenate((new_X, self.X), axis=0)
        temp_X = self.feature_decode()
        
        while(self.super_hero_mode):
            check = np.where(np.sum(temp_X[:, self.para_size:], axis=1)==0)[0]
            if len(check)!=0:
                rand_X = np.random.uniform(low=alpha[self.para_size:], high=beta[self.para_size:], size=[len(check), self.feature_size])
                self.X[check, self.para_size:] = rand_X
                temp_X = self.feature_decode()
            else:
                break
        
        score, score_error, score_feautre = self.fit_func(temp_X)
        top_k = score.argsort()[:self.num_particle]
        score = score[top_k].copy()
        self.X = self.X[top_k].copy()
        self.final = temp_X[top_k[0]].copy()
        
        return score, score_error[top_k], score_feautre[top_k]
    
    def handle_bound(self):
        R5 = np.random.uniform()
        R6 = np.random.uniform()
        idx_too_high = self.x_max < self.X
        idx_too_low = self.x_min > self.X
        
        bound_max_map = self.bound_max[idx_too_high] + \
                        R5*self.bound_max[idx_too_high]*(self.bound_max[idx_too_high]-self.X[idx_too_high])/self.X[idx_too_high]
        bound_min_map = self.bound_min[idx_too_low] + \
                        R6*np.abs(self.bound_min[idx_too_low]*(self.bound_min[idx_too_low]-self.X[idx_too_low])/self.X[idx_too_low])

        self.X[idx_too_high] = bound_max_map.copy()
        self.X[idx_too_low] = bound_min_map.copy()

        idx_too_high = self.x_max < self.X
        idx_too_low = self.x_min > self.X

    def feature_decode(self):
        temp_X = self.X.copy()
        
        R = np.random.uniform()
        if self.method=='modif_s':
            aaa = np.exp( -10*(temp_X[:, self.para_size:]-0.5))
            y = 1/( 1 + aaa )
            temp_X[:, self.para_size:] = 1*(y<=R)
        elif self.method=='s':
            aaa = np.exp( -temp_X[:, self.para_size:] )
            y = 1/( 1 + aaa )
            temp_X[:, self.para_size:] = 1*(R<y)
        
        return temp_X