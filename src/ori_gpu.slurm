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
module load python/intel/3.8.6
# Dalma modules
#module load gcc/4.9.3
#module load cuda/9.2
#module load boost/gcc_4.9.3/openmpi_1.10.2/avx2/1.57.0
#module load matlab/R2017a
module list

cd ${fdr0}
op_cfg=${op}-ori_${ori}.cfg

date
if [ "$plotOnly" = True ]; then
	readNewSpike=False
	usePrefData=True
	collectMeanDataOnly=False
else
	patch -c $op_cfg
	date
	cp $op_cfg ${op}-ori_${ori}_cfg.bak
	cp ${op}-ori_${ori}_cfg.bak $fig_fdr
	readNewSpike=True
	usePrefData=False
fi

if [ "$fitTC" = True ]; then
	OPstatus=0
else
	OPstatus=1
fi

pid=""
echo plotLGN_response.py ${op}_${ori} $lgn $res_fdr $data_fdr $fig_fdr $readNewSpike
plotLGN_response.py ${op}_${ori} $lgn $res_fdr $data_fdr $fig_fdr $readNewSpike &
pid+="${!} "

echo plotV1_response.py $op $lgn $v1 $res_fdr $data_fdr $fig_fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus
plotV1_response.py $op $lgn $v1 $res_fdr $data_fdr $fig_fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus &
pid+="${!} "
#if [ ${ori} -eq 1 ]; then
#	echo connections.py ${op}_${ori} $lgn $v1 $fdr
#	connections.py ${op}_${ori} $lgn $v1 $fdr & 
#	pid+="${!} "
#fi
wait $pid
date
