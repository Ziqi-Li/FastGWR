![PyPI](https://img.shields.io/pypi/v/fastgwr)
![GitHub](https://img.shields.io/github/license/Ziqi-Li/fastgwr)

# FastGWR
A command line tool for fast parallel computation of Geographically Weighted Regression models.
### New feature:
Multi-scale GWR model added!

### Installation:

The `fastgwr` program is dependent on `mpi4py` package and a working MPI implementation. The easiest way to install both dependencies is to use `conda`:

```bash
$ conda install mpi4py
```

By installing `mpi4py`, `conda` will also install an MPI implementation based on your computer system (OpenMPI for Mac/Linux; MPICH for Windows). Users may want to check whether the MPI implementation is successfully installed and is on your path by running the `mpiexec` command. Then the `fastgwr` program can be installed from PyPi:

```bash
$ pip install fastgwr
```

After sucessful installation, users can test the functionalities from the command line by running:

```bash
# Using zillow sample data for testing MGWR model fitting.
$ fastgwr testgwr
```
or

```bash
# Using zillow sample data for testing MGWR model fitting.
$ fastgwr testmgwr
```


# Examples
Example call to the `fastgwr` to fit GWR model:

```bash
$ fastgwr run -np 4 -data input.csv -adaptive -constant
```

Example call to the `fastgwr` to fit MGWR model:

```bash
$ fastgwr run -np 4 -data input.csv -adaptive -constant -mgwr
```
where:

```bash
-np 4             Number of processors (e.g. 4).
-data input.csv   Input data matrix. (e.g. input.csv)
                  Can also be URL (e.g. https://raw.github.com/
                  Ziqi-Li/FastGWR/master/Zillow-test-dataset/zillow_1k.csv)
-out results.csv  Output GWR results matrix including local parameter 
                  estimates, standard errors and local diagnostics.
-adaptive         Adaptive Bisquare kernel.
-fixed            Fixed Gaussian kernel.
-constant         Adding a constant column vector of 1 to the design matrix.
-bw 1000          Pre-defined bandwidth parameter. If missing, it will
                  search (golden-section) for the optimal bandwidth and use
                  that to fit the GWR model.
-minbw 45         Lower bound in golden-section search. (e.g. 45)
-mgwr             Fitting an MGWR model.
-chunks           Number of chunks for MGWR computation (set to a larger 
                  number to reduce memory footprint).
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
See the example Zillow datasets in the repository.

### Results Validation

The results are validated against the [mgwr](https://github.com/pysal/mgwr), which can be seen in the [notebooks here](https://github.com/Ziqi-Li/FastGWR/tree/master/validation%20notebook).


### Citations

This program is developed based on these two papers:

[FastGWR](https://www.tandfonline.com/doi/full/10.1080/13658816.2018.1521523)

Li, Z., Fotheringham, A. S., Li, W., Oshan, T. (2019). Fast Geographically Weighted Regression (FastGWR): A Scalable Algorithm to Investigate Spatial Process Heterogeneity in Millions of Observations. International Journal of Geographic Information Science. doi: 10.1080/13658816.2018.1521523.

[FastMGWR](https://www.tandfonline.com/doi/abs/10.1080/13658816.2020.1720692)

Li, Z., & Fotheringham, A. S. (2020). Computational improvements to multi-scale geographically weighted regression. International Journal of Geographical Information Science, 34(7), 1378-1397.
