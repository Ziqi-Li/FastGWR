import os
import click

@click.group()
@click.version_option("0.1.1")
def main():
    pass

@main.command()
@click.option("-np", default=2, required=True)
@click.option("-data", required=True)
@click.option("-out", default="fastgwr_rslt.csv", required=False)
@click.option("-adaptive/-fixed" ,default=True,required=True)
@click.option("-constant", required=False, is_flag=True)
@click.option("-bw", required=False)
@click.option("-minbw", required=False)
def run(np, data, out, adaptive, constant, bw, minbw):
    """FastGWR"""
    
    command = 'mpiexec ' + ' -np ' + str(np) + ' python ' + ' fastgwr/fastgwr_mpi.py ' + ' -data ' + data + ' -out ' + out
    
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
        
    os.system(command)

if __name__ == '__main__':
    main()
