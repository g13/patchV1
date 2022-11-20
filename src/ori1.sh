#!/bin/bash
# source /root/miniconda3/etc/profile.d/conda.sh
# eval "$(conda shell.bash hook)"
# conda activate general
# set -e

cd ${data_fdr}
pwd
patch_cfg=${fig_fdr}/${trial_suffix}-ori_${ori}.cfg
date
# check 1000
# a=`check 1200`
# echo $a
# check 1200
# RETURN=$?
# echo $RETURN
# date
if [ "$plotOnly" = False ]; then
	echo ${patch} -c $patch_cfg
	${patch} -c $patch_cfg
	# a=`check 1200`
	# echo $a
	RETURN=$?
	echo $RETURN
	date
fi

# echo ${a}
# date

# usePrefData=False
# collectMeanDataOnly=False

# if [ "$fitTC" = True ]; then
# 	OPstatus=0
# else
# 	OPstatus=1
# fi

# echo python

# pid=""
# echo python ${fig_fdr}/plotLGN_response_${trial_suffix}.py ${trial_suffix}_${ori} ${LGN_V1_suffix} ${data_fdr} ${fig_fdr}
# python ${fig_fdr}/plotLGN_response_${trial_suffix}.py ${trial_suffix}_${ori} ${LGN_V1_suffix} ${data_fdr} ${fig_fdr} &
# pid+="${!} "

# echo python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
# python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} &
# pid+="${!} "

# echo python ${fig_fdr}/plotFrameOutput_${trial_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}
# python ${fig_fdr}/plotFrameOutput_${trial_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}

# if [ "${singleOri}" = True ]; then
# 	if [ "${generate_V1_connection}" = True ]; then
# 		echo python ${fig_fdr}/connections_${V1_connectome_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}
# 		python ${fig_fdr}/connections_${V1_connectome_suffix}.py ${trial_suffix}_${ori} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr}
# 		pid+="${!} "
# 	fi
# fi

# wait $pid
# date
# conda activate base
