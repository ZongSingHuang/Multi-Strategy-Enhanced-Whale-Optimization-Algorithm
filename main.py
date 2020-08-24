# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:14:14 2020

@author: ZongSing_NB
"""

from MSEWOA import MSEWOA
import numpy as np
import time
import pandas as pd

def Sphere(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1)

def Schwefel_P222(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

def Quadric(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    outer = 0
    for i in range(x.shape[1]):
        inner = np.sum(x[:, :i+1], axis=1)**2
        outer = outer + inner
    
    return outer

def Rosenbrock(x):
    if x.ndim==1:
        x = x.reshape(1, -1) 
    
    left = x[:, :-1].copy()
    right = x[:, 1:].copy()
    
    return np.sum(100*(right - left**2)**2 + (left-1)**2, axis=1)

def Step(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(np.round((x+0.5), 0)**2, axis=1)

def Quadric_Noise(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    matrix = np.arange(x.shape[1])+1
     
    return np.sum((x**4)*matrix, axis=1)+np.random.rand(x.shape[0])

def Schwefel(x):
    if x.ndim==1:
        x = x.reshape(1, -1)        
     
    return -1*np.sum(x*np.sin(np.abs(x)**.5), axis=1)

def Rastrigin(x):
    if x.ndim==1:
        x = x.reshape(1, -1) 
    
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=1)

def Noncontinuous_Rastrigin(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    outlier = np.abs(x)>=0.5
    x[outlier] = np.round(2*x[outlier])/2
    
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=1)

def Ackley(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    left = 20*np.exp(-0.2*(np.sum(x**2, axis=1)/x.shape[1])**.5)
    right = np.exp(np.sum(np.cos(2*np.pi*x), axis=1)/x.shape[1])
    
    return -left - right + 20 + np.e

def Griewank(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    left = np.sum(x**2, axis=1)/4000
    right = np.prod( np.cos(x/((np.arange(x.shape[1])+1)**.5)), axis=1)
    return left - right + 1

def Generalized_Penalized01(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    y_head = 1 + (x[:, 0]+1)/4
    y_tail = 1 + (x[:, -1]+1)/4
    y_left = 1 + (x[:, :-1]+1)/4
    y_right = 1 + (x[:, 1:]+1)/4
    
    first = np.pi/x.shape[1]
    second = 10*np.sin(np.pi*y_head)**2
    third = np.sum( ((y_left-1)**2) * (1+10*np.sin(np.pi*y_right)**2), axis=1)
    fourth = (y_tail-1)**2
    five = np.sum(u_xakm(x, 10, 100, 4), axis=1)

    fitness = first*(second + third + fourth) + five
    
    return fitness

def Generalized_Penalized02(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    x_head = x[:, 0]
    x_tail = x[:, -1]
    x_left = x[:, :-1]
    x_right = x[:, 1:]
       
    first = 0.1
    second = np.sin(3*np.pi*x_head)**2 + (x_tail-1)**2
    third = np.sum( (x_left-1)**2 * (1+np.sin(3*np.pi*x_right)**2), axis=1)
    fourth = np.sum(u_xakm(x, 5, 100, 4), axis=1)

    fitness = first*(second + third) + fourth
    
    return fitness

def DE_JONG_N5(x):
    a1 = np.array([-32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32])
    a2 = np.array([-32, -32, -32, -32, -32,
                   -16, -16, -16, -16, -16,
                     0,   0,   0,   0,   0,
                    16,  16,  16,  16,  16,
                    32,  32,  32,  32,  32])
    
    first = 0.002
    second = np.sum(1/(np.arange(25)+1 + (x[:, 0]-a1)**6 + (x[:, 1]-a2)**6), axis=1)
    fitness = (first + second)**-1
    
    return fitness

def u_xakm(x, a, k, m):
    if x.ndim==1:
        x = x.reshape(1, -1)
    temp = x.copy()    
    
    case1 = x>a
    case3 = x<-a
    
    temp = np.zeros_like(x)
    temp[case1] = k*(x[case1]-a)**m         
    temp[case3] = k*(-1*x[case3]-a)**m
    
    return temp
    




d = 30
g = 3000
p = 20
times = 30
strategy_init = True
strategy_bound = True
table = np.zeros((5, 13))
table[2, :] = -np.ones(13)*np.inf
table[3, :] = np.ones(13)*np.inf
ALL = np.zeros((times, 13))
for i in range(times):
    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = MSEWOA(fit_func=Sphere, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 0]: table[2, 0] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 0]: table[3, 0] = optimizer.gBest_score
    table[0, 0] += optimizer.gBest_score
    table[1, 0] += end - start 
    ALL[i, 0] = optimizer.gBest_score


    x_max = 10*np.ones(d)
    x_min = -10*np.ones(d)
    optimizer = MSEWOA(fit_func=Schwefel_P222, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 1]: table[2, 1] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 1]: table[3, 1] = optimizer.gBest_score
    table[0, 1] += optimizer.gBest_score
    table[1, 1] += end - start  
    ALL[i, 1] = optimizer.gBest_score

    
    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = MSEWOA(fit_func=Quadric, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 2]: table[2, 2] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 2]: table[3, 2] = optimizer.gBest_score
    table[0, 2] += optimizer.gBest_score
    table[1, 2] += end - start
    ALL[i, 2] = optimizer.gBest_score
  
 
    x_max = 10*np.ones(d)
    x_min = -10*np.ones(d)
    optimizer = MSEWOA(fit_func=Rosenbrock, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 3]: table[2, 3] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 3]: table[3, 3] = optimizer.gBest_score  
    table[0, 3] += optimizer.gBest_score
    table[1, 3] += end - start  
    ALL[i, 3] = optimizer.gBest_score    

   
    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = MSEWOA(fit_func=Step, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 4]: table[2, 4] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 4]: table[3, 4] = optimizer.gBest_score  
    table[0, 4] += optimizer.gBest_score
    table[1, 4] += end - start
    ALL[i, 4] = optimizer.gBest_score
  
  
    x_max = 1.28*np.ones(d)
    x_min = -1.28*np.ones(d)
    optimizer = MSEWOA(fit_func=Quadric_Noise, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 5]: table[2, 5] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 5]: table[3, 5] = optimizer.gBest_score   
    table[0, 5] += optimizer.gBest_score
    table[1, 5] += end - start
    ALL[i, 5] = optimizer.gBest_score
 
 
    x_max = 500*np.ones(d)
    x_min = -500*np.ones(d)
    optimizer = MSEWOA(fit_func=Schwefel, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 6]: table[2, 6] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 6]: table[3, 6] = optimizer.gBest_score   
    table[0, 6] += optimizer.gBest_score
    table[1, 6] += end - start
    ALL[i, 6] = optimizer.gBest_score
  

    x_max = 5.12*np.ones(d)
    x_min = -5.12*np.ones(d)
    optimizer = MSEWOA(fit_func=Rastrigin, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 7]: table[2, 7] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 7]: table[3, 7] = optimizer.gBest_score   
    table[0, 7] += optimizer.gBest_score
    table[1, 7] += end - start  
    ALL[i, 7] = optimizer.gBest_score


    x_max = 5.12*np.ones(d)
    x_min = -5.12*np.ones(d)
    optimizer = MSEWOA(fit_func=Noncontinuous_Rastrigin, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 8]: table[2, 8] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 8]: table[3, 8] = optimizer.gBest_score  
    table[0, 8] += optimizer.gBest_score
    table[1, 8] += end - start
    ALL[i, 8] = optimizer.gBest_score
  
 
    x_max = 32*np.ones(d)
    x_min = -32*np.ones(d)
    optimizer = MSEWOA(fit_func=Ackley, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 9]: table[2, 9] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 9]: table[3, 9] = optimizer.gBest_score  
    table[0, 9] += optimizer.gBest_score
    table[1, 9] += end - start
    ALL[i, 9] = optimizer.gBest_score
   
 
    x_max = 600*np.ones(d)
    x_min = -600*np.ones(d)
    optimizer = MSEWOA(fit_func=Griewank, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 10]: table[2, 10] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 10]: table[3, 10] = optimizer.gBest_score  
    table[0, 10] += optimizer.gBest_score
    table[1, 10] += end - start  
    ALL[i, 10] = optimizer.gBest_score

    x_max = 50*np.ones(d)
    x_min = -50*np.ones(d)
    optimizer = MSEWOA(fit_func=Generalized_Penalized01, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 11]: table[2, 11] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 11]: table[3, 11] = optimizer.gBest_score  
    table[0, 11] += optimizer.gBest_score
    table[1, 11] += end - start  
    ALL[i, 11] = optimizer.gBest_score
    
    x_max = 50*np.ones(d)
    x_min = -50*np.ones(d)
    optimizer = MSEWOA(fit_func=Generalized_Penalized02, strategy_init=strategy_init, strategy_bound=strategy_bound,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 12]: table[2, 12] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 12]: table[3, 12] = optimizer.gBest_score  
    table[0, 12] += optimizer.gBest_score
    table[1, 12] += end - start  
    ALL[i, 12] = optimizer.gBest_score
    
    # x_max = 65.536*np.ones(2)
    # x_min = -65.536*np.ones(2)
    # optimizer = MSEWOA(fit_func=DE_JONG_N5, strategy_init=strategy_init, strategy_bound=strategy_bound,
    #                 num_dim=2, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    # start = time.time()
    # optimizer.opt()
    # end = time.time()
    # if optimizer.gBest_score>table[2, 13]: table[2, 13] = optimizer.gBest_score
    # if optimizer.gBest_score<table[3, 13]: table[3, 13] = optimizer.gBest_score  
    # table[0, 13] += optimizer.gBest_score
    # table[1, 13] += end - start  
    # ALL[i, 13] = optimizer.gBest_score
    
    print(i+1)
    
    
table[:2, :] = table[:2, :] / times
table[4, :] = np.std(ALL, axis=0)
table = pd.DataFrame(table)
table.columns=['Sphere', 'Schwefel_P222', 'Quadric', 'Rosenbrock', 'Step', 'Quadric_Noise', 'Schwefel', 
                'Rastrigin', 'Noncontinuous_Rastrigin', 'Ackley', 'Griewank', 'Generalized_Penalized01', 
                'Generalized_Penalized02', 'DE_JONG_N5']
table.index = ['avg', 'time', 'worst', 'best', 'std']