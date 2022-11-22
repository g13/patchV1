#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate general
set -e

cd ${data_fdr}

if [ "${plotOnly}" = False ];
then
	if [ "${new_setup}" = True ];
	then
		#echo matlab -nodisplay -nosplash -r "inputLearnFF('${inputFn}','${lgn}', ${seed}, ${std_ecc}, '${res}', ${waveStage}, '${res_fdr}', '${setup_fdr}', ${squareOrCircle}, '${fAsInput}', ${relay}, ${binary_thres});exit;"
		#matlab -nodisplay -nosplash -r "inputLearnFF('${inputFn}', '${lgn}', ${seed}, ${std_ecc}, '${res}', ${waveStage}, '${res_fdr}', '${setup_fdr}', ${squareOrCircle}, '${fAsInput}', ${relay}, ${binary_thres});exit;"
		echo python ${fig_fdr}/inputLearnFF_${op}.py ${inputFn} ${lgn} ${seed} ${std_ecc} ${res} ${waveStage} ${res_fdr} ${setup_fdr} ${squareOrCircle} ${relay} ${binary_thres} ${fAsInput}
		python ${fig_fdr}/inputLearnFF_${op}.py ${inputFn} ${lgn} ${seed} ${std_ecc} ${res} ${waveStage} ${res_fdr} ${setup_fdr} ${squareOrCircle} ${relay} ${binary_thres} ${fAsInput}
	fi
	date
	echo ${patch} -c ${fig_fdr}/${op}.cfg
	${patch} -c ${fig_fdr}/${op}.cfg
	date
fi

#echo python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${readNewSpike}
#python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${readNewSpike} & 

date
echo python ${fig_fdr}/plotV1_fr_${op}.py ${op} ${data_fdr} ${fig_fdr} ${nOri} ${readNewSpike}
python ${fig_fdr}/plotV1_fr_${op}.py ${op} ${data_fdr} ${fig_fdr} ${nOri} ${readNewSpike} &

#date
#echo python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
#python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}

date
#echo matlab -nodisplay -nosplash -r "outputLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', ${LGN_switch}, false, ${st}, ${examSingle}, ${use_local_max});exit;" &
#matlab -nodisplay -nosplash -r "outputLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', ${LGN_switch}, false, ${st}, ${examSingle}, ${use_local_max});exit;" &
echo python ${fig_fdr}/outputLearnFF_${op}.py ${seed} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${LGN_switch} false ${st} ${examSingle} ${use_local_max}
python ${fig_fdr}/outputLearnFF_${op}.py ${seed} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${LGN_switch} false ${st} ${examSingle} ${use_local_max} &

#date
#echo matlab -nodisplay -nosplash -r "testLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', 233, 5000);exit;" &
#matlab -nodisplay -nosplash -r "testLearnFF('${res}', '${lgn}', '${op}', '${res_fdr}', '${setup_fdr}', '${data_fdr}', '${fig_fdr}', 233, 5000);exit;" &

#python frameOutput.py ${op} ${data_fdr} &

#### not yet available for virtual_LGN
#isuffix=0
#echo python ${fig_fdr}/getReceptiveField_lFF_${op}.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#python ${fig_fdr}/getReceptiveField_lFF_${op}.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#echo python ${fig_fdr}/getReceptiveField_lFF_${op}.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}
#python ${fig_fdr}/getReceptiveField_lFF_${op}.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}
#
#isuffix=1
#echo python ${fig_fdr}/getReceptiveField_lFF_${op}.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#python ${fig_fdr}/getReceptiveField_lFF_${op}.py collecting ${op} ${lgn} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${isuffix} ${seed} 1
#echo python ${fig_fdr}/getReceptiveField_lFF_${op}.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}
#python ${fig_fdr}/getReceptiveField_lFF_${op}.py plotting ${isuffix} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${isuffix}

wait
date
