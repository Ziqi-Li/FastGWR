#FastGWR MPI Script
#Author: Ziqi Li et al.
#Email: lziqi@asu.edu

import math
import numpy as np
from mpi4py import MPI
from scipy.spatial.distance import cdist,pdist
import argparse

#Example Call:
#mpiexec -np 2 python fastgwr-mpi.py -data Zillow-test-dataset/zillow_5k.csv -out fast.csv

def read(fname):
    #print("Reading",fname)
    if pysal:
        input = np.genfromtxt(fname, dtype=float, delimiter=',',skip_header=False)
    else:
        input = np.genfromtxt(fname, dtype=float, delimiter=',',skip_header=True)
    #input = np.load(fname)
    #Converting things into matrices
    y = input[:,2].reshape(-1,1)
    n = input.shape[0]

    if constant:
        X = np.hstack([np.ones((n,1)),input[:,3:]])
    else:
        X = input[:,3:]
    coords = input[:,:2]
    return n,X,y,coords

def local_fit(i,bw,fixed,final=False):

    #adaptive bisquare
    if not fixed:
        dist = cdist([coords[i]],coords).reshape(-1)
        maxd = np.partition(dist, int(bw)-1)[int(bw)-1]*1.0000001
        zs = dist/maxd
        zs[zs>=1] = 1
        wi = ((1-(zs)**2)**2).reshape(-1,1)
    #fixed gaussian
    else:
        zs = cdist([coords[i]],coords)/bw
        wi = np.exp(-0.5*(zs)**2).reshape(-1,1)


    if final:
        xT = (X * wi).T
        xtx_inv_xt = np.dot(np.linalg.inv(np.dot(xT, X)), xT)
        betas = np.dot(xtx_inv_xt, y).reshape(-1)
        ri = np.dot(X[i],xtx_inv_xt)
        
        predy = np.dot(X[i],betas)
        err = y[i][0] - predy
        CCT = np.diag(np.dot(xtx_inv_xt,xtx_inv_xt.T))
        
        #rss_ri = np.sum(ri**2)
        #return np.concatenate(([i,err,predy,ri[i],rss_ri],betas,CCT))
        
        return np.concatenate(([i,err,ri[i]],betas,CCT))
    else:
        X_new = X*np.sqrt(wi)
        Y_new = y*np.sqrt(wi)
        temp = np.dot(np.linalg.inv(np.dot(X_new.T,X_new)),X_new.T)
        hat = np.dot(X_new[i],temp[:,i])
        yhat = np.sum(np.dot(X_new,temp[:,i]).reshape(-1,1)*Y_new)
        err = Y_new[i][0]-yhat
        return err*err,hat


def golden_section(a, c, delta, function, tol=1.0e-6, max_iter=200, int_score=True):
    b = a + delta * np.abs(c-a)
    d = c - delta * np.abs(c-a)
    opt_bw = None
    score = None
    diff = 1.0e9
    iters  = 0
    dict = {}
    while np.abs(diff) > tol and iters < max_iter:
        iters += 1
        if not fixed:
            b = np.round(b)
            d = np.round(d)
        
        if b in dict:
            score_b = dict[b]
        else:
            score_b = function(b)
            dict[b] = score_b
        
        if d in dict:
            score_d = dict[d]
        else:
            score_d = function(d)
            dict[d] = score_d
        
        if rank == 0:
            if score_b <= score_d:
                opt_score = score_b
                opt_bw = b
                c = d
                d = b
                b = a + delta * np.abs(c-a)
            else:
                opt_score = score_d
                opt_bw = d
                a = b
                b = d
                d = c - delta * np.abs(c-a)
            
            diff = score_b - score_d
            score = opt_score
        
        b = comm.bcast(b,root=0)
        d = comm.bcast(d,root=0)
        opt_bw = comm.bcast(opt_bw,root=0)
        diff = comm.bcast(diff,root=0)
        score = comm.bcast(score,root=0)
    return opt_bw

def mpi_gwr_fit(bw,final=False,fout='./fastGWRResults.csv',fixed=False):
    #Need Betas
    if final:
        sub_Betas = np.empty((x_chunk.shape[0],2*k+3), dtype=np.float64)

        pos = 0
        for i in x_chunk:
            sub_Betas[pos] = local_fit(i,bw,fixed,final)
            pos+=1
        
        '''
        offset = rank*sub_Betas.nbytes
        
        fh.Write_at_all(offset, sub_Betas.reshape(-1))
        fh.Close()
        '''

        Betas_list = comm.gather(sub_Betas, root=0)
        
        if rank ==0:
            data = np.vstack(Betas_list)
            #data[:,-k:] = data[:,-k:]
            
            print("Fitting GWR Using Bandwidth:",bw)
            if pysal:
                np.savetxt(fout, data, delimiter=',',comments='')
            if not pysal:
                RSS = np.sum(data[:,1]**2)
                TSS = np.sum((y - np.mean(y))**2)
                R2 = 1- RSS/TSS
                trS = np.sum(data[:,2])
                #trSTS = np.sum(data[:,4])
                #sigma2_v1v2 = RSS/(n-2*trS+trSTS)
                sigma2_v1 = RSS/(n-trS)
                aicc = n*np.log(RSS/n) + n*np.log(2*np.pi) + n*(n+trS)/(n-trS-2.0)
                data[:,-k:] = np.sqrt(data[:,-k:]*sigma2_v1)
                print("Diagnostic Information:")
                print("AICc:",aicc)
                print("ENP:",trS)
                print("R2:",R2)
                header="index,residual,influ,"
                varNames = np.genfromtxt(fname, dtype=str, delimiter=',',names=True,max_rows=1).dtype.names[3:]
                if constant:
                    varNames = ['intercept'] + list(varNames)
                for x in varNames:
                    header += ("b_"+x+',')
                for x in varNames:
                    header += ("se_"+x+',')
                    np.savetxt(fout, data, delimiter=',',header=header[:-1],comments='')
        
            return
        return
    
    t3 = MPI.Wtime()
    sub_RSS = 0
    sub_trS = 0
    for i in x_chunk:
        err2,hat = local_fit(i,bw,fixed=fixed,final=False)
        sub_RSS += err2
        sub_trS += hat

    RSS_list = comm.gather(sub_RSS, root=0)
    trS_list = comm.gather(sub_trS, root=0)
    t4 = MPI.Wtime()
    #wt_43 = comm.gather(t4-t3, root=0)

    if rank == 0:
        tot_RSS = sum(RSS_list)
        tot_trS = sum(trS_list)
        tot_TSS = np.sum((y - np.mean(y))**2)
        llf = -np.log(tot_RSS)*n/2 - (1+np.log(np.pi/n*2))*n/2
        aicc = -2*llf + 2.0*n*(tot_trS + 1.0)/(n-tot_trS-2.0)
        R2 = 1- tot_RSS/tot_TSS
        print("BW, AICc",bw, aicc)
        return aicc
    return


if __name__ == "__main__":
    #Initializing
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data")
    parser.add_argument("-out")
    parser.add_argument("-bw")
    parser.add_argument("-minbw")
    parser.add_argument('-f','--fixed',action='store_true')
    parser.add_argument('-a','--adaptive',action='store_true')
    parser.add_argument('-p','--pysal',action='store_true')
    parser.add_argument('-c','--constant',action='store_true')

    fname = parser.parse_args().data
    fout  = parser.parse_args().out
    fixed = parser.parse_args().fixed
    pysal = parser.parse_args().pysal
    constant = parser.parse_args().constant

    bw = None
    if parser.parse_args().bw is not None:
        if fixed:
            bw = float(parser.parse_args().bw)
        else:
            bw = int(parser.parse_args().bw)



    if rank==0:
        print("Starting FastGWR with",size,"Processors")
        print("Data Input Path:",fname)
        print("Output Result Path:",fout)
        print("Constant:",constant)
        if fixed:
            print("Spatial Kernel: Fixed Gaussian")
        else:
            print("Spatial Kernel: Adaptive Bisquare")
    #Data Copying
    if rank ==0:
        n,X, y,coords = read(fname)
        k = X.shape[1]
        iter = np.arange(n)

    else:
        X = None
        y = None
        coords = None
        n = None
        k = None
        iter = None

    #t1
    t1 = MPI.Wtime()
    X = comm.bcast(X,root=0)
    y = comm.bcast(y,root=0)
    coords = comm.bcast(coords,root=0)
    iter = comm.bcast(iter,root=0)
    n = comm.bcast(n,root=0)
    k = comm.bcast(k,root=0)

    if not fixed:
        maxbw = n
        minbw = 40 + 2 * k
        if parser.parse_args().minbw is not None:
            minbw = int(parser.parse_args().minbw)
    else:
        minbw = float('Inf')
        maxbw = -100
        for i in range(n):
            dist = cdist([coords[i]],coords)
            tempmax = np.max(dist)
            tempmin = np.min(dist[np.nonzero(dist)])
            if tempmax > maxbw:
                maxbw = tempmax
            if tempmin < minbw:
                minbw = tempmin
        minbw = minbw / 2
        maxbw = maxbw * 2
        if parser.parse_args().minbw is not None:
            minbw = float(parser.parse_args().minbw)
    m = int(math.ceil(float(len(iter)) / size))
    x_chunk = iter[rank*m:(rank+1)*m]

    if bw is None:
        if rank ==0:
            print("Optimal Bandwidth Searching...")
        gwr_func = lambda bw: mpi_gwr_fit(bw,fixed=fixed)
        bw = golden_section(minbw, maxbw, 0.38197, gwr_func)
        if fixed:
            bw = round(bw,2)
        mpi_gwr_fit(bw,final=True,fout=fout,fixed=fixed)
    else:
        mpi_gwr_fit(bw,final=True,fout=fout,fixed=fixed)

    t_last = MPI.Wtime()
    wt = comm.gather(t_last-t1, root=0)
    if rank ==0:
        print("Total Time Elapsed:",np.round(max(wt),2),"seconds")



