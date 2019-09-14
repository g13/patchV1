#!/bin/bash
#SBATCH -p serial 
#SBATCH -n 1
#SBATCH --mem-per-cpu=60GB
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH -J formMap
#SBATCH -o log/fM_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load all
module load matlab/R2017a
ulimit -s unlimited

fdr="t"
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
