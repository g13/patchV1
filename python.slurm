#!/bin/bash
#SBATCH -p serial 
#SBATCH --job-name=python
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH -o log/py_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge
conda activate general

python getLGNsurface.py
