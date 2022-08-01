#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate general
set -e

date
pid=""
echo python ${fig_fdr}/getTuningCurve_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${fitTC} ${fitDataReady}
python ${fig_fdr}/getTuningCurve_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${fitTC} ${fitDataReady} &
pid+="${!} "

if [ "${generate_V1_connection}" = True ]; then
	echo python ${fig_fdr}/connections_${V1_connectome_suffix}.py ${trial_suffix}_1 ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}
	python ${fig_fdr}/connections_${V1_connectome_suffix}.py ${trial_suffix}_1 ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} &
	pid+="${!} "
fi

wait $pid

if [ "$collectMeanDataOnly" = True ]; then
	OPstatus=1
else
	OPstatus=2
fi

if [ "${usePrefData}" = True ]; then

	pid=""
	for ori in $( seq 1 $nOri )
	do
		echo python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} False ${usePrefData} False ${OPstatus}
		python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} False ${usePrefData} False ${OPstatus} &
		pid+="${!} "
	done
	wait $pid
fi
date
conda activate base
