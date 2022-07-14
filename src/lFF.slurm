#!/bin/bash
cd ${HOME}/repos/patchV1/src

t=00:30:00
echo ${t}
#fdr0=/scratch/wd554/Jen
fdr0=/scratch/wd554/patchV1
res_fdr=${fdr0}/resource # resourceFolder in cfg stimulus, V1-static, LGN-static
setup_fdr=${fdr0}/setup # inputFolder in cfg LGN-V1 and V1 connectome files
data_fdr=${fdr0} # outputFolder in cfg


patch=patch_fast
waveStage=2 # set stage number

if [[ "${waveStage}" = "2" ]] ;then # stage II wave goes here
	#inputFn=32-sequential_10-30_stage_II 
	#inputFn=32-random3917563_10-30_stage_II 
	inputFn=32-opponent_10-10_stage_II
	fdr=wave_concat/wave_II # folder will be made to store figures and configuratoin files
	op=w2_iwi2 # theme string to be added to filenames of data and figures
	op_cfg=wave_II.cfg # the simulation config file to be used
	new_setup=True
	suffix=iwi2
	lgn=lFF_II_${suffix} # the theme string to be used for LGN_V1 connections
	res=lFF_II_${suffix} # the theme string to be used for V1, LGN positions etc.
	squareOrCircle=true # initial LGN-recruitment shape 
fi

relay=false
if [[ "${waveStage}" = "3" ]] ;then
	#inputFn=32x3-random3917563_5-15_stage_III_sym
	inputFn=32x3-opponent_5-10_stage_III_reverse
	#inputFn=32x3-sequential_5-15_stage_III_sym
	#inputFn=32x3-135_5-5_stage_III_sym
	fdr=wave_concat/wave_III
	op=w3r_opp_iwi2_on1off1_cap3_g7
	#op=single_test-135
	op_cfg=wave_III.cfg
	new_setup=True
	if [[ "${relay}" = false ]]; then
		suffix=g7
		lgn=lFF_III_${suffix}
		res=lFF_III_${suffix}
	else
		#fAsInput=${data_fdr}/sLGN_32_ii.bin
		fAsInput=
		if [ -z "${fAsInput}" ]; 
		then
			lgn=lFF_concat0
		else
			lgn=lFF_relay_concat
		fi
		res=lFF0
	fi
	squareOrCircle=false # initial LGN-recruitment shape 
fi

if [[ "${waveStage}" = "5" ]] ;then
	fdr=wave_concat
	op=II_l
	op_cfg=wave_concat.cfg
	new_setup=False
	lgn=lFF_l
	res=lFF_l # starts out from stage II
	squareOrCircle=true # initial LGN-recruitment shape 
fi

binary_thres=0.0
#binary_thres=0.5
std_ecc=0
#std_ecc=16
#std_ecc=12
#std_ecc=8
#std_ecc=6
#std_ecc=5
#std_ecc=4
#std_ecc=3
#std_ecc=2.66667
seed=1924784 # RANDOM SEED for post-analysis sampling and input randomization 
# for outputLearnFF.m
st=2 # figures' output: 0 for temporal, 1 for spatial, 2 for both, otherwise none
examSingle=true # output the spatiotemporal figure, tLGN_V1_single if true
use_local_max=0 # output the spatiotemporal figure, sLGN_V1 with strength normalized for each frame instead over all frames if set 1
collectMeanDataOnly=False # not ploting samples of V1 response variable traces

# NOT RELATED, leave alone
usePrefData=False
OPstatus=1 
LGN_switch=false
TF=8
ori=0
nOri=0

v1=${lgn} # cortical V1 connections used the same theme as LGN to V1 connections

echo ${op} # echo/print out the variable "op" in the job's output file
echo ${suffix}

fig_fdr=${fdr0}/${fdr}

if [ -d "${fig_fdr}" ]
then
	echo overwrite contents in ${fig_fdr}
else
	mkdir -p ${fig_fdr}
fi

echo using ${op_cfg}
echo outputs to ${fig_fdr}
cp lFF.slurm ${fig_fdr}/lFF_${op}.slurm

readNewSpike=$1
if [ -z "${readNewSpike}" ]; 
then
	readNewSpike=True
fi
echo readNewSpike=${readNewSpike}

plotOnly=$2
if [ -z "${plotOnly}" ]; 
then
	plotOnly=False
fi
echo plotOnly=${plotOnly}

cp outputLearnFF.m ${fig_fdr}/outputLearnFF_${op}.m
cp plotV1_response_lFF.py ${fig_fdr}/plotV1_response_lFF_${op}.py
if [ "${plotOnly}" = False ]
then
	gpu=1
	cp ${op_cfg} ${fig_fdr}/${op}.cfg
	cp inputLearnFF.m ${fig_fdr}/inputLearnFF_${op}.m
else
	gpu=0
fi

if ! [ -d "log" ]
then
	mkdir log
fi

sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:${gpu}
#SBATCH --job-name=plotLearn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=${t}
#SBATCH -o log/pLearn_%J.out
#SBATCH --mail-user=weidai@cims.nyu.edu
#SBATCH --mail-type=END
set -e
echo ${PATH}
module purge
module load gcc/10.2.0
module load cuda/11.1.74
module load boost/intel/1.74.0
module load matlab/2020b
module list

cd ${data_fdr}
date

if [ "${plotOnly}" = False ];
then
	if [ "${new_setup}" = True ];
	then
		echo matlab -nodisplay -nosplash -r "inputLearnFF('${inputFn}','${lgn}', ${seed}, ${std_ecc}, '${res}', ${waveStage}, '${res_fdr}', '${setup_fdr}', ${squareOrCircle}, '${fAsInput}', ${relay}, ${binary_thres});exit;"
		matlab -nodisplay -nosplash -r "inputLearnFF('${inputFn}', '${lgn}', ${seed}, ${std_ecc}, '${res}', ${waveStage}, '${res_fdr}', '${setup_fdr}', ${squareOrCircle}, '${fAsInput}', ${relay}, ${binary_thres});exit;"
	fi
	date
	${patch} -c ${fig_fdr}/${op}.cfg
	date
fi


date
echo plotLGN_response.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${readNewSpike} & 
plotLGN_response.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${readNewSpike} & 

date
echo plotV1_fr.py ${op} ${data_fdr} ${fig_fdr} ${nOri} ${readNewSpike} & 
plotV1_fr.py ${op} ${data_fdr} ${fig_fdr} ${nOri} ${readNewSpike} &

date
echo plotV1_response_lFF.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} & 
plotV1_response_lFF.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}

date
echo matlab -nodisplay -nosplash -r "outputLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', ${LGN_switch}, false, ${st}, ${examSingle}, ${use_local_max});exit;" &
matlab -nodisplay -nosplash -r "outputLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', ${LGN_switch}, false, ${st}, ${examSingle}, ${use_local_max});exit;" &

#date
#echo matlab -nodisplay -nosplash -r "testLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', 233, 5000);exit;" &
#matlab -nodisplay -nosplash -r "testLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', 233, 5000);exit;" &

#python patchOutput.py ${op} ${data_fdr} &

#isuffix=0
#echo python getReceptiveField_lFF.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#python getReceptiveField_lFF.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#echo python getReceptiveField_lFF.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}
#python getReceptiveField_lFF.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}
#
#isuffix=1
#echo python getReceptiveField_lFF.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#python getReceptiveField_lFF.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#echo python getReceptiveField_lFF.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}
#python getReceptiveField_lFF.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}

wait
date
EOT
