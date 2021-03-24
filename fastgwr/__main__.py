import os
import click
import fastgwr

@click.group()
@click.version_option("0.2.5")
def main():
    pass

@main.command()
@click.option("-np", default=4, required=True)
@click.option("-data", required=True)
@click.option("-out", default="fastgwr_rslt.csv", required=False)
@click.option("-adaptive/-fixed" ,default=True, required=True)
@click.option("-constant", required=False, is_flag=True)
@click.option("-bw", required=False)
@click.option("-minbw", required=False)
@click.option("-chunks", required=False)
@click.option("-mgwr", default=False, required=False, is_flag=True)
def run(np, data, out, adaptive, constant, bw, minbw, mgwr, chunks):
    """
    Fast(M)GWR
    
    -np: number of processors to use
    
    -data: input data matrix containing y and X
    
    -out: output GWR results (default: "fastgwr_rslt.csv")
    
    -adaptive: using adaptive bisquare kernel
    
    -fixed: using fixed gaussian kernel
    
    -constant: adding a constant vector to the X matrix
    
    -bw: using a pre-specified bandwidth to fit GWR
    
    -minbw: lower bound in the golden section search
    
    -mgwr: fitting an MGWR model
    
    -chunks: number of chunks for MGWR computation (default: 1).
             Increase the number if run out of memory but should keep it as low as possible.
    
    """
    
    mpi_path = os.path.dirname(fastgwr.__file__) + '/fastgwr_mpi.py'
    
    command = 'mpiexec ' + ' -np ' + str(np) + ' python ' + mpi_path + ' -data ' + data + ' -out ' + out
    if mgwr:
        command += ' -mgwr '
    
    if adaptive:
        command += ' -a '
    else:
        command += ' -f '
    if bw:
        command += (' -bw ' + bw)
    if minbw:
        command += (' -minbw ' + minbw)
    if constant:
        command += ' --constant '
    if chunks:
        command += (' -chunks ' + chunks)
        
    os.system(command)
    pass


@main.command()
def testgwr():
    """
    Testing GWR with zillow data
    """
    print("Testing GWR with zillow data:")
    mpi_path = os.path.dirname(fastgwr.__file__) + '/fastgwr_mpi.py'
    
    command = "mpiexec -np 2 python " + mpi_path + " -data https://raw.github.com/Ziqi-Li/FastGWR/master/Zillow-test-dataset/zillow_1k.csv -c"
    os.system(command)
    pass


@main.command()
def testmgwr():
    """
    Testing MGWR with zillow data
    """
    
    print("Testing MGWR with zillow data:")
    mpi_path = os.path.dirname(fastgwr.__file__) + '/fastgwr_mpi.py'
    
    command = "mpiexec -np 2 python " + mpi_path + " -data https://raw.github.com/Ziqi-Li/FastGWR/master/Zillow-test-dataset/zillow_1k.csv -c -mgwr"
    os.system(command)
    pass
    

if __name__ == '__main__':
    main()
