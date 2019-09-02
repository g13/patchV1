#!/bin/bash
#SBATCH -p serial 
#SBATCH -n 1
#SBATCH --mem-per-cpu=10GB
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=5
#SBATCH -J formMap
#SBATCH -o log/fM_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load all
module load matlab/R2017a
ulimit -s unlimited

fdr="t1"
if [ -d "$fdr" ]
then
	echo deleting contents in $fdr
	rm -rf $fdr/*
else
	mkdir $fdr	
fi
cp *.m $fdr
cd $fdr
date
    matlab -nosplash -nodesktop -r "parameters;exit"
date
cd ..
