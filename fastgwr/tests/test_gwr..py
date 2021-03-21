from unittest import TestCase

from FastGWR import FastGWR
from FastMGWR import FastMGWR

class TestGWR(TestCase):
    def test_is_string(self):
        cmd = "mpiexec -np 4 python fastgwr_mpi.py -data ../Zillow-test-dataset/zillow_1k.csv -c"
        
        
        self.assertTrue(isinstance(s, basestring))

if __name__ == '__main__':
    main()
