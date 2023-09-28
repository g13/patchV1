#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.zsh hook)"
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

jobID=""
date
echo python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${readNewSpike}
python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${readNewSpike} & 
jobID+="${!} "

#echo python ${fig_fdr}/plotV1_fr_${op}.py ${op} ${res_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${nOri} ${readNewSpike} ${ns}
#python ${fig_fdr}/plotV1_fr_${op}.py ${op} ${res_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${nOri} ${readNewSpike} ${ns} &
#jobID+="${!} "

echo python ${fig_fdr}/outputLearnFF_${op}.py ${seed} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${LGN_switch} false ${st} ${examSingle} ${use_local_max} ${waveStage} ${ns} ${examLTD} ${find_peak}
python ${fig_fdr}/outputLearnFF_${op}.py ${seed} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${LGN_switch} false ${st} ${examSingle} ${use_local_max} ${waveStage} ${ns} ${examLTD} ${find_peak} &
jobID+="${!} "

#echo python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
#python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}

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

wait ${jobID}
date

if [[ "${cleanData}" = "1" ]]
then
    date
    echo delete data
    python ${fig_fdr}/clean_data_${op}.py ${data_fdr} ${op}
    rm ${data_fdr}/snapShot_*-${op}.bin
fi
