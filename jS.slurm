#!/bin/bash
#SBATCH --job-name=jupyterServer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=06:00:00
#SBATCH -o log/jS_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END

module purge

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/wd554/local/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/wd554/local/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/wd554/local/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/wd554/local/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
date
conda activate general
# <<< conda initialize <<<

python jScriptR.py
date
