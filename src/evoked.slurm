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
# Greene modules
module load gcc/10.2.0
module load cuda/11.1.74
module load boost/intel/1.74.0
module load matlab/2020b
# Dalma modules
#module load gcc/4.9.3
#module load cuda/9.2
#module load boost/gcc_4.9.3/openmpi_1.10.2/avx2/1.57.0
#module load matlab/R2017a
module list

TF=8
fdr=tune-008
op=evoked00-s15t8-test_CI
op_cfg=evoked0.cfg
lgn=b_4_20-micro
v1=test_CI

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
#./genCon -c $v1_cfg
./patch -c $op_cfg
date
pid=""
echo python plotLGN_response.py $op $lgn $fdr True
python plotLGN_response.py $op $lgn $fdr True &
pid+="${!} "
echo python plotV1_response.py $op $lgn $v1 $fdr $TF 0.25 True True
python plotV1_response.py $op $lgn $v1 $fdr $TF 0.25 True True & 
#python plotV1_response.py $op $lgn $v1 $fdr $TF 0.25 True False & 
pid+="${!} "
wait $pid
date
