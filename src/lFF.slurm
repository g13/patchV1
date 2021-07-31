#!/bin/bash
##SBATCH --gres=gpu:1
#SBATCH --job-name=plotLearn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
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

readNewSpike=$1
if [ -z "$readNewSpike" ]; 
then
	readNewSpike=True
fi
plotOnly=true
TF=8
fdr=lFF_test_tau
#op=single_lFF_s
#op=lFF_4x3_small
#op=lFF1_1x12_1-3_-1-10_5
op=40x3_h
#op=lFF_1x12_108_equi0_1
#op_cfg=tau_test0.cfg
op_cfg=standing_test.cfg
lgn=lFF_40x3_full
v1=$lgn
usePrefData=False
collectMeanDataOnly=False
OPstatus=1
LGN_switch=false
seed=1924784
lgn0=lFF

ori=0
nOri=0

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

if [ "$plotOnly" = false ]
then
	echo matlab -r "inputLearnFF('$lgn', $seed, 0, 'lFF');exit;"
	matlab -r "inputLearnFF('$lgn', $seed);exit;"
	date
	./patch -c $op_cfg
	date
fi

pid=""
#echo python plotLGN_response.py $op $lgn $fdr $readNewSpike & 
#python plotLGN_response.py $op $lgn $fdr $readNewSpike & 
#pid+="${!} "
#echo python plotV1_response_lFF.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus & 
#python plotV1_response_lFF.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus & 
#pid+="${!} "
echo matlab -r "outputLearnFF('$lgn0', '$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &
matlab -r "outputLearnFF('$lgn0', '$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &
pid+="${!} "

#echo matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &
#matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &

#python patchOutput.py $op $fdr &
#pid+="${!} "

wait $pid

#isuffix=0
#echo python getReceptiveField_lFF.py collecting $op $lgn $fdr $isuffix 1
#python getReceptiveField_lFF.py collecting $op $lgn $fdr $isuffix 1
#echo python getReceptiveField_lFF.py plotting $isuffix $op $lgn $fdr $nOri $isuffix
#python getReceptiveField_lFF.py plotting $isuffix $op $lgn $fdr $nOri $isuffix
#
#isuffix=1
#echo python getReceptiveField_lFF.py collecting $op $lgn $fdr $isuffix 1
#python getReceptiveField_lFF.py collecting $op $lgn $fdr $isuffix 1
#echo python getReceptiveField_lFF.py plotting $isuffix $op $lgn $fdr $nOri $isuffix
#python getReceptiveField_lFF.py plotting $isuffix $op $lgn $fdr $nOri $isuffix

date
