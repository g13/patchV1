#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plotV1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH -o log/pV1_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END
##SBATCH -p nvidia 
set -e
module purge
module load gcc/10.2.0
module load cuda/11.1.74
module load boost/intel/1.74.0
module list
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home/wd554/local/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/wd554/local/miniconda3/etc/profile.d/conda.sh" ]; then
#        . "/home/wd554/local/miniconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/wd554/local/miniconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup
#conda activate general
# <<< conda initialize <<<
TF=8
fdr=tune-008
op=evoked00-s15t8-const_n
op_cfg=evoked0.cfg
lgn=b_4_20-micro
v1=const_n

v1_cfg=connectome.cfg
lgn_cfg=LGN_V1.cfg

if [ -d "$fdr" ]
then
	echo overwrite contents in $fdr
else
	mkdir $fdr	
fi
cp $op_cfg $lgn_cfg $v1_cfg $fdr

echo using $op_cfg
echo outputs $op, $lgn and $v1
date
./patch -c $op_cfg
date
pid=""
#python plotV1_response.py $op $lgn $v1 $fdr $TF 0.25 True & 
python plotV1_response.py $op $lgn $v1 $fdr $TF 0.25 True True & 
pid+="${!} "
python plotLGN_response.py $op $lgn True &
pid+="${!} "
#python plotV1_response.py noDelay $lgn $v1 test 8 0.25 True &
#pid+="${!}"
wait $pid
date
