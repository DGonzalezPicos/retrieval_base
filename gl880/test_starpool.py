import pathlib
import os
import numpy as np
from multiprocessing import Pool
import time


class PTGrid:
    
    def __init__(self, P, T, Nproc=1):
        
        self.P = P
        self.T = T
        self.Nproc = Nproc
        
    def set_func(self, func=None):
        
        if func is not None:
            self.func = func
        else:
            self.func = self.my_func
        return self
    
    def my_func(self, P, T):
        return P*T
        
    def __call__(self, func, args={}):
        # apply func to each P, T pair
        points = [(P, T) for P in self.P for T in self.T]
        if len(args) > 0:
            args_values = list(args.values())
            points = [(P, T, *args_values) for P, T in points]
            
        print(f' Number of points: {len(points)}')
            
        with Pool(self.Nproc) as pool:
            results = pool.starmap(func, points)
            
        return results
            

class CrossSections:
    
    def __init__(self, nu, gamma=1.0):
        
        self.nu = nu # vector shape (N,)
        self.gamma = gamma
        
    def __call__(self, P, T, alpha):
        
        # compute cross sections for P, T
        print(f'Computing cross sections for P={P}, T={T}')
        q = self.nu * P**self.gamma * T**self.gamma - alpha
        time.sleep(0.1)
        
        return q
    
if __name__=='__main__':
    
    P = np.linspace(1, 10, 10)
    T = np.linspace(100, 1000, 10)
    
    nu = np.linspace(1, 10, 10)
    gamma = 1.0
    cs = CrossSections(nu, gamma)
    
    Nproc = 16
    start = time.time()
    ptgrid = PTGrid(P, T, Nproc=Nproc)
    # ptgrid.set_func()
    q = ptgrid(cs, args={'alpha': 1.0})
    
    end = time.time()
    print(f' Elapsed time with {Nproc} processes: {end-start:.2f} s')
    
    # print(q)
    
        