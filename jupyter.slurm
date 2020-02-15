#!/bin/bash
#SBATCH -p serial 
#SBATCH --job-name=jupyterServer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
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
conda activate general
# <<< conda initialize <<<

#port=$(shuf -i 6000-9999 -n 1)
port=6217
echo port: $port

ssh -N -f -R $port:localhost:$port login-0-1
ssh -N -f -R $port:localhost:$port login-0-2
ssh -N -f -R $port:localhost:$port login-0-3
ssh -N -f -R $port:localhost:$port login-0-4

#create ssh tunnel
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter lab --no-browser --port $port --notebook-dir=$(pwd)
