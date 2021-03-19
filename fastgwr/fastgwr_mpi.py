#FastGWR MPI Script
#Author: Ziqi Li
#Email: liziqi1992@gmail.com

import argparse
import numpy as np
from mpi4py import MPI
from FastGWR import FastGWR

#Direct Example Call:
#mpiexec -np 4 python fastgwr-mpi.py -data ../Zillow-test-dataset/zillow_5k.csv

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
    FastGWR(comm,parser).fit()
    #Timing ends
    t_last = MPI.Wtime()
    
    wt = comm.gather(t_last-t1, root=0)
    if rank ==0:
        print("Total Time Elapsed:",np.round(max(wt),2),"seconds")
        print("-"*60)



