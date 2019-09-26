#!/bin/bash
#SBATCH -p serial 
#SBATCH --nodes=1
#SBATCH --mem=65GB
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=5
#SBATCH --job-name=formMap
#SBATCH -o log/fM_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
module load all
module load matlab/R2017a
ulimit -s unlimited

fdr="Large_sheet"
if [ -d "$fdr" ]
then
	echo deleting contents in $fdr
	rm -rf $fdr/*
else
	mkdir $fdr	
fi
cp *.m $fdr
cp redo_or_plot@Dalma.slurm $fdr
cd $fdr
mkdir log

date
    matlab -nosplash -nodesktop -r "parameters;exit"
date
cd ..
