# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:14:14 2020

@author: ZongSing_NB
"""

from MSEWOA import MSEWOA
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

def F1(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1)

def F2(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

def F3(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    outer = 0
    for i in range(x.shape[1]):
        inner = np.sum(x[:, :i+1], axis=1)**2
        outer = outer + inner
    
    return outer

def F4(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    fitness = np.max(np.abs(x), 1)
    
    return fitness

def F5(x):
    if x.ndim==1:
        x = x.reshape(1, -1) 
    
    left = x[:, :-1].copy()
    right = x[:, 1:].copy()
    
    return np.sum(100*(right - left**2)**2 + (left-1)**2, axis=1)

# def F6(x):
#     if x.ndim==1:
#         x = x.reshape(1, -1)
#     return np.sum(np.round((x+0.5), 0)**2, axis=1)

def F6(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum((x+0.5)**2, axis=1)

def F7(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    matrix = np.arange(x.shape[1])+1
     
    return np.sum((x**4)*matrix, axis=1)+np.random.rand(x.shape[0])

def F8(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
     
    return -1*np.sum(x*np.sin(np.abs(x)**.5), axis=1)

def F9(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    return np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1) + 10*x.shape[1]

# def Noncontinuous_Rastrigin(x):
#     if x.ndim==1:
#         x = x.reshape(1, -1)
    
#     outlier = np.abs(x)>=0.5
#     x[outlier] = np.round(2*x[outlier])/2
    
#     return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=1)

def F10(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    left = 20*np.exp(-0.2*(np.sum(x**2, axis=1)/x.shape[1])**.5)
    right = np.exp(np.sum(np.cos(2*np.pi*x), axis=1)/x.shape[1])
    
    return -left - right + 20 + np.e

def F11(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    left = np.sum(x**2, axis=1)/4000
    right = np.prod( np.cos(x/((np.arange(x.shape[1])+1)**.5)), axis=1)
    return left - right + 1

def F12(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    y1 = 1 + (x[:, 0]+1)/4
    yi = 1 + (x[:, :-1]+1)/4
    yi_1 = 1 + (x[:, 1:]+1)/4
    yn = 1 + (x[:, -1]+1)/4
    
    return np.pi/x.shape[1] * \
                              ( 
                                10*np.sin(np.pi*y1)**2 + 
                                np.sum( (yi-1)**2 * (1+10*np.sin(np.pi*yi_1)**2), axis=1) +
                                (yn-1)**2
                              ) \
                            + u_xakm(x, 10, 100, 4)

def F13(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    return 0.1 * \
                 ( 
                   np.sin(3*np.pi*x[:, 0])**2 +
                   np.sum((x[:, :-1]-1)**2*(1+np.sin(3*np.pi*x[:, 1:])**2), axis=1) +
                   (x[:, -1]-1)**2*(1+np.sin(2*np.pi*x[:, -1])**2)
                  ) \
               + u_xakm(x, 5, 100, 4)

def F14(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
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
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        x1 = x[i, 0]
        x2 = x[i, 1]
        
        term1 = np.arange(25)+1
        term2 = (x1-a1)**6
        term3 = (x2-a2)**6
        term_left = np.sum(1/(term1 + term2 + term3))
        term_right = 1/500
        
        fitness[i] = 1/(term_right + term_left)
    
    return fitness

def vF15(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
    bK = 1/np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])

    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):    
        term1 = x[i, 0]*(bK**2+x[i, 1]*bK)
        term2 = bK**2+x[i, 2]*bK+x[i, 3]       
        fitness[i] = np.sum((aK - term1/term2)**2)
    
    return fitness

def vF16(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    return 4*(x[:, 0]**2)-2.1*(x[:, 0]**4)+(x[:, 0]**6)/3+x[:, 0]*x[:, 1]-4*(x[:, 1]**2)+4*(x[:, 1]**4)

def vF17(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    return (x[:, 1]-(x[:, 0]**2)*5.1/(4*(np.pi**2))+5/np.pi*x[:, 0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[:, 0])+10

def vF18(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    return (1+(x[:, 0]+x[:, 1]+1)**2*(19-14*x[:, 0]+3*(x[:, 0]**2)-14*x[:, 1]+6*x[:, 0]*x[:, 1]+3*x[:, 1]**2))* \
    (30+(2*x[:, 0]-3*x[:, 1])**2*(18-32*x[:, 0]+12*(x[:, 0]**2)+48*x[:, 1]-36*x[:, 0]*x[:, 1]+27*(x[:, 1]**2)))

def vF19(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aH = np.array([[3, 10, 30],
                   [.1, 10, 35],
                   [3, 10, 30],
                   [.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[.3689, .117, .2673],
                   [.4699, .4387, .747],
                   [.1091, .8732, .5547],
                   [.03815, .5743, .8828]])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        term1 = cH[0]*np.exp( -1*np.sum( aH[0, :]*(x[i, :]-pH[0, :])**2, axis=0 ) )
        term2 = cH[1]*np.exp( -1*np.sum( aH[1, :]*(x[i, :]-pH[1, :])**2, axis=0 ) )
        term3 = cH[2]*np.exp( -1*np.sum( aH[2, :]*(x[i, :]-pH[2, :])**2, axis=0 ) )
        term4 = cH[3]*np.exp( -1*np.sum( aH[3, :]*(x[i, :]-pH[3, :])**2, axis=0 ) )
        
        fitness[i] = -1*(term1 + term2 + term3 + term4)
    
    return fitness

def vF20(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [.05, 10, 17, .1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, .05, 10, .1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[.1312, .1696, .5569, .0124, .8283, .5886],
                   [.2329, .4135, .8307, .3736, .1004, .9991],
                   [.2348, .1415, .3522, .2883, .3047, .6650],
                   [.4047, .8828, .8732, .5743, .1091, .0381]])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        term1 = cH[0]*np.exp( -1*np.sum( aH[0, :]*(x[i, :]-pH[0, :])**2, axis=0 ) )
        term2 = cH[1]*np.exp( -1*np.sum( aH[1, :]*(x[i, :]-pH[1, :])**2, axis=0 ) )
        term3 = cH[2]*np.exp( -1*np.sum( aH[2, :]*(x[i, :]-pH[2, :])**2, axis=0 ) )
        term4 = cH[3]*np.exp( -1*np.sum( aH[3, :]*(x[i, :]-pH[3, :])**2, axis=0 ) )
        
        fitness[i] = -1*(term1 + term2 + term3 + term4)
    
    return fitness

def vF21(x):
    return Shekel(x, m=5)


def vF22(x):
    return Shekel(x, m=7)

def vF23(x):
    return Shekel(x, m=10)

def u_xakm(x, a, k, m):
    if x.ndim==1:
        x = x.reshape(1, -1)
    temp = x.copy()    
    
    case1 = x>a
    case3 = x<-a
    
    temp = np.zeros_like(x)
    temp[case1] = k*(x[case1]-a)**m         
    temp[case3] = k*(-1*x[case3]-a)**m
    
    return np.sum(temp, axis=1)

def Shekel(x, m=10):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aSH = np.array([[4, 4, 4, 4], 
                    [1, 1, 1, 1], 
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(m):
            fitness[i] = fitness[i] - 1/(np.sum((x[i, :]-aSH[j, :])**2)+cSH[j])
    
    return fitness
    


d2 = np.random.uniform(size=2)
d3 = np.random.uniform(size=3)
d4 = np.random.uniform(size=4)
d6 = np.random.uniform(size=6)
d30 = np.random.uniform(size=30)

result = np.zeros(23)

result[0] = F1(d30)
result[1] = F2(d30)
result[2] = F3(d30)
result[3] = F4(d30)
result[4] = F5(d30)
result[5] = F6(d30)
result[6] = F7(d30)
result[7] = F8(d30)
result[8] = F9(d30)
result[9] = F10(d30)
result[10] = F11(d30)
result[11] = F12(d30)
result[12] = F13(d30)
result[13] = F14(d2)
result[14] = F15(d4)
result[15] = F16(d2)
result[16] = F17(d2)
result[17] = F18(d2)
result[18] = F19(d3)
result[19] = F20(d6)
result[20] = F21(d4)
result[21] = F22(d4)
result[22] = F23(d4)
