#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plotLearn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH -o log/pLearn_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END
##SBATCH -p nvidia 
set -e
module purge
module load gcc/10.2.0
module load cuda/11.1.74
module load boost/intel/1.74.0
module load matlab/2020b
module list

TF=13
fdr=lFF
op=lFF_rs3
#op=lFF_single2
op_cfg=lFF.cfg
lgn=lFF
v1=lFF

if [ -d "$fdr" ]
then
	echo overwrite contents in $fdr
else
	mkdir $fdr	
fi
cp $op_cfg inputLearnFF.m outputLearnFF.m $fdr

echo using $op_cfg
echo outputs to $fdr

date

matlab -r "inputLearnFF;exit;"
date
./patch -c $op_cfg
date
pid=""
python plotLGN_response.py $op $lgn $fdr True & 
pid+="${!} "
python plotV1_response_lFF.py $op $lgn $v1 $fdr $TF 0.25 True True & 
pid+="${!} "
matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', true, false);exit;" &
pid+="${!} "
wait $pid

#python patchOutput.py $op $fdr &
#pid+="${!} "

date
