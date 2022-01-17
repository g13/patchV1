#!/bin/bash
cd ${HOME}/repos/patchV1/src

#fdr0=/scratch/wd554/Jen
fdr0=/scratch/wd554/patchV1
res_fdr=${fdr0}/resource # inputFolder in cfg stimulus, V1-static, LGN-static and connectome files
data_fdr=${fdr0} # outputFolder in cfg

patch=patch_fast
waveStage=3 # set stage number

if [[ "${waveStage}" = "2" ]] ;then # stage II wave goes here
	fdr=wave_concat/wave_II # folder will be made to store figures and configuratoin files
	op=32_ii # theme string to be added to filenames of data and figures
	op_cfg=wave_II.cfg # the simulation config file to be used
	new_setup=True
	lgn=lFF_ii # the theme string to be used for LGN_V1 connections
	res=lFF # the theme string to be used for V1, LGN positions etc.
fi

relay=false
if [[ "${waveStage}" = "3" ]] ;then
	fdr=wave_concat/wave_III
	op=wave_III_scrambled
	op_cfg=wave_III.cfg
	smallRange=true
	new_setup=False
	if [[ "${relay}" = false ]]; then
		lgn=lFF_III
		res=lFF_III
	else
		#fAsInput=${fdr0}/sLGN_32_ii.bin
		fAsInput=
		if [ -z "${fAsInput}" ]; 
		then
			lgn=lFF_concat0
		else
			lgn=lFF_relay_concat
		fi
		res=lFF0
	fi
fi

if [[ "${waveStage}" = "5" ]] ;then
	fdr=wave_concat
	op=32_32x3_concat
	op_cfg=wave_concat.cfg
	new_setup=True
	lgn=lFF_concat
	res=lFF # starts out from stage II
fi

std_ecc=0
seed=1924784 # RANDOM SEED for post-analysis sampling and input randomization 
# for outputLearnFF.m
st=2 # figures' output: 0 for temporal, 1 for spatial, 2 for both
examSingle=true # output the spatiotemporal figure, tLGN_V1_single if true
squareOrCircle=false # initial LGN-recruitment shape 
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

#isuffix=0
sbatch <<EOT
#!/bin/bash
#SBATCH --gres=gpu:${gpu}
#SBATCH --job-name=plotLearn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=00:40:00
#SBATCH -o log/pLearn_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END
#SBATCH -p nvidia
set -e
echo ${PATH}
module purge
#module load gcc/10.2.0
#module load cuda/11.1.74
#module load boost/intel/1.74.0
#module load matlab/2020b
# Dalma modules
module load all
module load gcc/4.9.3
module load cuda/9.2
module load boost/gcc_4.9.3/openmpi_1.10.2/avx2/1.57.0
module load matlab/R2017a
#module list

cd ${fdr0}
date

if [ "${plotOnly}" = False ];
then
	if [ "${new_setup}" = True ];
	then
		echo matlab -nodisplay -nosplash -r "inputLearnFF('${lgn}', ${seed}, ${std_ecc}, '${res}', ${waveStage}, '${res_fdr}', ${squareOrCircle}, '${fAsInput}', ${relay}, ${smallRange});exit;"
		matlab -nodisplay -nosplash -r "inputLearnFF('${lgn}', ${seed}, ${std_ecc}, '${res}', ${waveStage}, '${res_fdr}', ${squareOrCircle}, '${fAsInput}', ${relay}, ${smallRange});exit;"
	fi
	date
	${patch} -c ${fig_fdr}/${op}.cfg
	date
fi


#echo plotLGN_response.py ${op} ${lgn} ${res_fdr} ${data_fdr} ${fig_fdr} ${readNewSpike} & 
#plotLGN_response.py ${op} ${lgn} ${res_fdr} ${data_fdr} ${fig_fdr} ${readNewSpike} & 

echo plotV1_response_lFF.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} & 
pwd
#python $HOME/repos/patchV1/src/plotV1_response_lFF.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}

echo matlab -nodisplay -nosplash -r "outputLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${data_fdr}', '${fig_fdr}', ${LGN_switch}, false, ${st}, ${examSingle}, ${use_local_max});exit;" &
matlab -nodisplay -nosplash -r "outputLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${data_fdr}', '${fig_fdr}', ${LGN_switch}, false, ${st}, ${examSingle}, ${use_local_max});exit;" &

echo matlab -nodisplay -nosplash -r "testLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${data_fdr}', '${fig_fdr}', 233, 5000);exit;" &
matlab -nodisplay -nosplash -r "testLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${data_fdr}', '${fig_fdr}', 233, 5000);exit;" &

#python patchOutput.py ${op} ${data_fdr} &

#echo python getReceptiveField_lFF.py collecting ${op} ${lgn} ${data_fdr} ${isuffix} 1
#python getReceptiveField_lFF.py collecting ${op} ${lgn} ${data_fdr} ${isuffix} 1
#echo python getReceptiveField_lFF.py plotting ${isuffix} ${op} ${res} ${lgn} ${data_fdr} ${nOri} ${isuffix}
#python getReceptiveField_lFF.py plotting ${isuffix} ${op} ${res} ${lgn} ${data_fdr} ${nOri} ${isuffix}
#
#isuffix=1
#echo python getReceptiveField_lFF.py collecting ${op} ${lgn} ${data_fdr} ${isuffix} 1
#python getReceptiveField_lFF.py collecting ${op} ${lgn} ${data_fdr} ${isuffix} 1
#echo python getReceptiveField_lFF.py plotting ${isuffix} ${op} ${res} ${lgn} ${data_fdr} ${nOri} ${isuffix}
#python getReceptiveField_lFF.py plotting ${isuffix} ${op} ${res} ${lgn} ${data_fdr} $nOri $isuffix

wait
date
EOT
