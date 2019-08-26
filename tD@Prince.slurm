#!/bin/bash
#SBATCH -p serial 
#SBATCH -n 1
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=5
#SBATCH -J formMap
#SBATCH -o log/fM_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2017a
ulimit -s unlimited

date
    matlab -nosplash -nodesktop -r "parameters;exit"
date
