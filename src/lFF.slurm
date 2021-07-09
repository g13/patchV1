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

readNewSpike=True
TF=8
fdr=mouse_lFF
#op=single_lFF_s
op=lFF_s3
op_cfg=lFF.cfg
lgn=lFF
v1=lFF
usePrefData=False
collectMeanDataOnly=False
OPstatus=1
LGN_switch=true
seed=1924784

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

matlab -r "inputLearnFF('$lgn', $seed);exit;"
date
./patch -c $op_cfg
date

pid=""
echo python plotLGN_response.py $op $lgn $fdr $readNewSpike & 
python plotLGN_response.py $op $lgn $fdr $readNewSpike & 
pid+="${!} "
echo plotV1_response_lFF.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus & 
python plotV1_response_lFF.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus & 
pid+="${!} "
echo matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &
matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &
pid+="${!} "

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
