#FastMGWR MPI Script
#Author: Ziqi Li
#Email: liziqi1992@gmail.com

import math
import numpy as np
from mpi4py import MPI
from scipy.spatial.distance import cdist,pdist
import argparse
from FastGWR import FastGWR
#Direct Example Call:
#mpiexec -np 2 python fastmgwr_mpi.py -data Zillow-test-dataset/zillow_1k.csv -out fast.csv


class FastMGWR(FastGWR):

    def __init__(self, comm, parser):
        FastGWR.__init__(self, comm, parser)
        
        
        
    def backfitting(self):
        if self.comm.rank ==0:
            print("MGWR Backfitting...")
        
        #Initalization
        betas,bw = self.fit(mgwr=True)
        if self.comm.rank ==0:
            print("Initialization Done...")
        XB = betas*self.X
        err = self.y.reshape(-1) - np.sum(XB,axis=1)
        bws = [None]*self.k

        for mgwr_iters in range(1,25):
            newXB = np.empty(XB.shape, dtype=np.float64)
            newbetas = np.empty(XB.shape, dtype=np.float64)
            
            for j in range(self.k):
                temp_y = (XB[:,j] + err).reshape(-1,1)
                temp_X = self.X[:,j].reshape(-1,1)
                
                betas,bw_j = self.fit(y=temp_y,X=temp_X,mgwr=True)
                XB_j = (betas*temp_X).reshape(-1)
                err = temp_y.reshape(-1) - XB_j
                newXB[:,j] = XB_j
                newbetas[:,j] = betas.reshape(-1)
                bws[j] = bw_j
        
                
            num = np.sum((newXB - XB)**2) / self.n
            den = np.sum(np.sum(newXB, axis=1)**2)
            score = (num / den)**0.5
            XB = newXB
            
            if self.comm.rank ==0:
                print("Iter:",mgwr_iters,"SOC:","{:.2e}".format(score))
                print("bws:",bws)
            
            if score < 1e-5:
                break
        
        
        RSS = np.sum(err**2)
        TSS = np.sum((self.y - np.mean(self.y))**2)
        R2 = 1- RSS/TSS
        print(R2)
            

if __name__ == "__main__":

    #Initializing MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    #Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-data")
    parser.add_argument("-out",default="fastgwr_rslt.csv")
    parser.add_argument("-bw")
    parser.add_argument("-minbw")
    parser.add_argument('-f','--fixed',action='store_true')
    parser.add_argument('-a','--adaptive',action='store_true')
    parser.add_argument('-c','--constant',action='store_true')

    #Timing starts
    t1 = MPI.Wtime()
    
    #Run function
    FastMGWR(comm,parser).backfitting()
    #Timing ends
    t_last = MPI.Wtime()
    
    wt = comm.gather(t_last-t1, root=0)
    if rank ==0:
        print("Total Time Elapsed:",np.round(max(wt),2),"seconds")
        print("-"*60)



