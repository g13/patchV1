#!/bin/bash
#SBATCH --job-name=formMap
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=13GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=5
#SBATCH -o log/fM_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load matlab/2017a
ulimit -s unlimited

fdr="qc"
if [ -d "$fdr" ]
then
	echo deleting contents in $fdr
	rm -rf $fdr/*
else
	mkdir $fdr	
fi
cp *.m $fdr
cp redo_or_plot.slurm $fdr
cd $fdr
mkdir log

date
    matlab -nosplash -nodesktop -r "parameters;exit"
date
