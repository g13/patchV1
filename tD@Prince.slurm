#!/bin/bash
#SBATCH --job-name=formMap
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12GB
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=5
#SBATCH -o log/fM_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2017a
ulimit -s unlimited

date
    matlab -nosplash -nodesktop -r "parameters;exit"
date
