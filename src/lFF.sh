#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.zsh hook)"
conda activate general

cd ${data_fdr}

if [ "${plotOnly}" = False ];
then
	if [ "${new_setup}" = True ];
	then
		echo python ${fig_fdr}/inputLearnFF_${op}.py ${inputFn} ${lgn} ${seed} ${std_ecc} ${res} ${waveStage} ${res_fdr} ${setup_fdr} ${squareOrCircle} ${relay} ${binary_thres} ${fAsInput} ${retino_cross} ${retinotopy}
		python ${fig_fdr}/inputLearnFF_${op}.py ${inputFn} ${lgn} ${seed} ${std_ecc} ${res} ${waveStage} ${res_fdr} ${setup_fdr} ${squareOrCircle} ${relay} ${binary_thres} ${fAsInput} ${retino_cross} ${retinotopy}
        if [ "$?" -ne 0 ];
        then
            mail -s "lFF ${op} FAIL" gueux13@gmail.com <<< "inputLearnFF.py FAIL"
            exit
        fi

	fi
	date
	echo ${patch} -c ${fig_fdr}/${op}.cfg
	${patch} -c ${fig_fdr}/${op}.cfg
    if [ "$?" -ne 0 ];
    then
        mail -s "lFF ${op} FAIL" gueux13@gmail.com <<< "patchfast EXIT_FAILURE"
        exit
    fi
	date
fi

jobID=""
date
#echo python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${setup_fdr} ${readNewSpike}
#python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} ${setup_fdr} ${readNewSpike} & 
#jobID+="${!} "

date
#echo python ${fig_fdr}/plotV1_fr_${op}.py ${op} ${res_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${nOri} ${readNewSpike} ${ns}
#python ${fig_fdr}/plotV1_fr_${op}.py ${op} ${res_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${nOri} ${readNewSpike} ${ns} &
#jobID+="${!} "

date
echo python ${fig_fdr}/outputLearnFF_${op}.py ${seed} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${LGN_switch} false ${st} ${examSingle} ${use_local_max} ${waveStage} ${ns} ${examLTD} ${find_peak} ${retino_cross} ${retinotopy}
python ${fig_fdr}/outputLearnFF_${op}.py ${seed} ${res} ${lgn} ${op} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${inputFn} ${LGN_switch} false ${st} ${examSingle} ${use_local_max} ${waveStage} ${ns} ${examLTD} ${find_peak} ${retino_cross} ${retinotopy} &
jobID+="${!} "

if [ "$?" -ne 0 ];
then
    mail -s "lFF ${op} FAIL" gueux13@gmail.com <<< "outputLearnFF.py FAIL"
    exit
fi

#echo python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
#python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
jobID+="${!} "
#
if [ "$?" -ne 0 ];
then
    mail -s "lFF ${op} FAIL" gueux13@gmail.com <<< "plotV1_response_lFF.py FAIL"
    exit
fi

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

mail -s "lFF ${op} SUCCESSFUL" gueux13@gmail.com <<< "SUCCESSFUL"
