
# FastGWR
A command line tool for fast parallel computation of Geographically Weighted Regression models.
### New feature:
Multi-scale GWR model added!

### Installation:

1. Install any MPI implementation
    For Mac:

    OpenMPI: https://www.open-mpi.org

    or `conda install openmpi`

    For PC:

    Microsoft MPI: https://docs.microsoft.com/en-us/message-passing interface/microsoft-mpi

    or `conda install mpich`

  
    Make sure you have the mpi command on your path by running
    `mpiexec`

  

2. Install mpi4py

    `conda install mpi4py`

 
3. Install fastgwr

    `pip install fastgwr`
    

### Testing 
`fastgwr testgwr` or `fastgwr testmgwr`


### Example
Example call to FastGWR which can be called on desktop or HPC:
```
fastgwr run -np 4 -data input.csv -out results.csv -adaptive
```
Example call to an MGWR model
```
fastgwr run -np 4 -data input.csv -out results.csv -adaptive - mgwr
```
```
where:
-np 4: using 4 processors.
-data input.csv: input data matrix.
-out results.csv: output GWR results matrix including local parameter estimates, standard errors and local diagnostics.
-adaptive: Adaptive Bisquare kernel.
-fixed: Fixed Gaussian kernel.
-constant: Adding a constant column vector of 1 to the design matrix.
-bw 1000: Pre-defined bandwidth parameter. If missing, it will (golden-section) search for the optimal bandwidth and use that to fit GWR model.
-minbw 45: Lower bound in golden-section search.
-mgwr: fitting an MGWR model.
-chunks: #of chunks for MGWR computation (set a large number for reducing memory footprint).
```

The input needs to be prepared in this order:

|   | X-coord | y-coord | y    | X1  | X2  | X3  | Xk  |
|---|---------|---------|------|-----|-----|-----|-----|
|   | ...     | ...     | ...  | ... | ... | ... | ... |
|   | ...     | ...     | ...  | ... | ... | ... | ... |
|   |         |         |      |     |     |     |     |

```
where:
X-coord: X coordinate of the location point
Y-coord: Y coordinate of the location point
y: dependent variable
X1...Xk: independent variables
```



### Citation

[FastGWR](https://www.tandfonline.com/doi/full/10.1080/13658816.2018.1521523)

Li, Z., Fotheringham, A. S., Li, W., Oshan, T. (2019). Fast Geographically Weighted Regression (FastGWR): A Scalable Algorithm to Investigate Spatial Process Heterogeneity in Millions of Observations. International Journal of Geographic Information Science. doi: 10.1080/13658816.2018.1521523.
