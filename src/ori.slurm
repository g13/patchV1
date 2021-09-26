#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=plotV1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
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
####################### USED BY patchTC ############################

op_cfg=${op}-ori_${ori}.cfg

cp $op_cfg $fdr

date
if [ "$plotOnly" = False ]; then
	./patch -c $op_cfg
	date
	readNewSpike=True
	usePrefData=False
else
	readNewSpike=False
	usePrefData=True
	collectMeanDataOnly=False
fi

if [ "$fitTC" = True ]; then
	OPstatus=0
else
	OPstatus=1
fi

pid=""
echo python plotLGN_response.py ${op}_${ori} $lgn $fdr $readNewSpike
python plotLGN_response.py ${op}_${ori} $lgn $fdr $readNewSpike &
pid+="${!} "
echo python plotV1_response.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus
python plotV1_response.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus &
pid+="${!} "
if [ ${ori} -eq 1 ]; then
	echo python connections.py ${op}_${ori} $lgn $v1 $fdr
	python connections.py ${op}_${ori} $lgn $v1 $fdr & 
	pid+="${!} "
fi
wait $pid
date