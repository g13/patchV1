#!/bin/bash
#SBATCH --job-name=plotmap
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=5
#SBATCH -o log/plot_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2018a
ulimit -s unlimited

date
    matlab -nosplash -nodesktop -r "parameters;exit"
date
