import unittest
import os
import pandas as pd
from mgwr.gwr import GWR,GWRResults
from mgwr.sel_bw import Sel_BW

from fastgwr.FastGWR import FastGWR
#from fastgwr.FastMGWR import FastMGWR

class TestGWR(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")
        cmd1 = "mpiexec -np 4 python ../fastgwr_mpi.py -data ..//../Zillow-test-dataset/zillow_1k.csv -c -out test_gwr_adap.csv"
        os.system(cmd1)
        
        cmd2 = "mpiexec -np 4 python ../fastgwr_mpi.py -data ..//../Zillow-test-dataset/zillow_1k.csv -c -out test_gwr_fixed.csv"


    def gwr_test(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")
        cmd1 = "mpiexec -np 4 python ../fastgwr_mpi.py -data ..//../Zillow-test-dataset/zillow_1k.csv -c -out test_gwr_adap.csv"
        
        cmd2 = "mpiexec -np 4 python ../fastgwr_mpi.py -data ..//../Zillow-test-dataset/zillow_1k.csv -c -out test_gwr_fixed.csv"
        
        
        #subprocess.call(cmd1)
        #subprocess.call(cmd2)
        
        #os.system(cmd1)
        
        #os.system(cmd2)
        
        fastGWR_result_fixed = pd.read_csv("test_gwr_fixed.csv")
        fastGWR_result_adap = pd.read_csv("test_gwr_adap.csv")
        
        zillow = pd.read_csv("..//../Zillow-test-dataset/zillow_1k.csv",sep=',')
        
        #Converting things into matrices
        y = zillow.value.values.reshape(-1,1)
        X = zillow.iloc[:,3:].values
        k = zillow.shape[1]
        u = zillow.utmX
        v = zillow.utmY
        n = zillow.shape[0]
        coords = np.array(list(zip(u,v)))
        
        #bi-square
        opt_bw_adap = Sel_BW(coords,y,X).search(verbose=True)
        pysal_result_adap=GWR(coords,y,X,opt_bw_adap).fit()
        
        #fixed
        opt_bw_fixed = Sel_BW(coords,y,X,fixed=True,kernel="gaussian").search(verbose=True)
        pysal_result_fixed=GWR(coords,y,X,opt_bw_fixed,fixed=True,kernel="gaussian").fit()
        
        #Validate model residual
        self.assertEqual(np.allclose(fastGWR_result_fixed.residual,pysal_result_fixed.resid_response.reshape(-1)))
        self.assertEqual(np.allclose(fastGWR_result_adap.residual,pysal_result_adap.resid_response.reshape(-1)))
        
        #Validate parameter estimates
        self.assertEqual(np.allclose(np.array(fastGWR_result_fixed.iloc[:,3:8]),pysal_result_fixed.params))
        self.assertEqual(np.allclose(np.array(fastGWR_result_adap.iloc[:,3:8]),pysal_result_adap.params))

        #Validate parameter estimates standard errors
        self.assertEqual(np.allclose(np.array(fastGWR_result_adap.iloc[:,8:13]),pysal_result_adap.bse))
        self.assertEqual(np.allclose(np.array(fastGWR_result_fixed.iloc[:,8:13]),pysal_result_fixed.bse))
        

if __name__ == '__main__':
    unittest.main()
