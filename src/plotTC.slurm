#!/bin/bash
#SBATCH --job-name=plotTC
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH -o log/pTC_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END
set -e
module purge
# Greene modules
# Dalma modules
module list

date
echo python plotLGN_response.py $op $lgn $fdr $readNewSpike
python getTunfiningCurve.py $op $lgn $v1 $fdr $nOri
date
