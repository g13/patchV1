#!/bin/bash

# stage II wave goes here
waveStage=2 # stage number
fdr=wave_II # folder will be made to store figures and configuratoin files
op=40ii_test3 # theme string to be added to filenames of data and figures
op_cfg=waveII.cfg # the simulation config file to be used
lgn=lFF_40ii # the theme string to be used for LGN_V1 connections
lgn0=lFF_ii # the theme string to be used for V1, LGN positions etc.

# uncomment the following lines for stage III wave, DON'T FORGET to comment out the lines above
#waveStage=3
#fdr=wave_III
#op=40x3_iii1
#op_cfg=wave_III.cfg
#lgn=lFF_40x3
#lgn0=lFF_iii

seed=1924784 # RANDOM SEED for post-analysis sampling and input randomization 
# for outputLearnFF.m
st=2 # figures' output: 0 for temporal, 1 for spatial, 2 for both
examSingle=true # output the spatiotemporal figure, tLGN_V1_single if true


# NOT RELATED, leave alone
usePrefData=False
collectMeanDataOnly=False
OPstatus=1 
LGN_switch=false
TF=8
ori=0
nOri=0

v1=$lgn # cortical V1 connections used the same theme as LGN to V1 connections

echo $op # echo/print out the variable "op" in the job's output file

if [ -d "$fdr" ]
then
	echo overwrite contents in $fdr
else
	mkdir $fdr	
fi

echo using $op_cfg
echo outputs to $fdr

readNewSpike=$1
if [ -z "$readNewSpike" ]; 
then
	readNewSpike=True
fi
echo $readNewSpike

plotOnly=$2
if [ -z "$plotOnly" ]; 
then
	plotOnly=False
fi
echo $plotOnly

if [ "$plotOnly" = False ]
then
	gpu=1
	cp $op_cfg ${op}_cfg.bak
	cp inputLearnFF.m inputLearnFF_${op}_m.bak
	cp ${op}_cfg.bak inputLearnFF_${op}_m.bak $fdr
else
	gpu=0
fi

#isuffix=0
sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:$gpu
#SBATCH --job-name=plotLearn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH -o log/pLearn_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END
set -e
module purge
module load gcc/10.2.0
module load cuda/11.1.74
module load boost/intel/1.74.0
module load matlab/2020b
module list

date

if [ "$plotOnly" = False ]
then
	echo matlab -r "inputLearnFF('$lgn', $seed, 0, '$lgn0', $waveStage);exit;"
	matlab -r "inputLearnFF('$lgn', $seed, 0, '$lgn0', $waveStage);exit;"
	date
	./patch -c ${op}_cfg.bak
	date
fi


#echo python plotLGN_response.py $op $lgn $fdr $readNewSpike & 
#python plotLGN_response.py $op $lgn $fdr $readNewSpike & 

echo python plotV1_response_lFF.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus $waveStage & 
python plotV1_response_lFF.py $op $lgn $v1 $fdr $TF $ori $nOri $readNewSpike $usePrefData $collectMeanDataOnly $OPstatus $waveStage

echo matlab -r "outputLearnFF('$lgn0', '$lgn', '$op', '$fdr', $LGN_switch, false, $st, $examSingle);exit;" &
matlab -r "outputLearnFF('$lgn0', '$lgn', '$op', '$fdr', $LGN_switch, false, $st, $examSingle);exit;" &

cp outputLearnFF.m $fdr/outputLearnFF_m.bak

#echo matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &
#matlab -r "testLearnFF('$lgn', '$op', '$fdr');outputLearnFF('$lgn', '$op', '$fdr', $LGN_switch, false);exit;" &

#python patchOutput.py $op $fdr &

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

wait
date
EOT
