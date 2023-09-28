#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.zsh hook)"
conda activate general
set -e

cd ${data_fdr}

if [ "${plotOnly}" = False ];
then
    if [ "${newSetup}" = True ];
    then
        python ${fig_fdr}/assignLearnedWeights_${op}.py ${original_suffix} ${setup_fdr} ${data_fdr} ${new_sLGN_fn}
    fi
	date
	echo ${patch} -c ${fig_fdr}/${op}.cfg
	${patch} -c ${fig_fdr}/${op}.cfg
	date
fi

readNewSpike=True
nOri=0
ns=10
lgn=${original_suffix}

python ${fig_fdr}/plotSTA_sample_${op}.py ${op} ${data_fdr} ${fig_fdr} &

python ${fig_fdr}/plotLGN_response_${op}.py ${op} ${lgn} ${data_fdr} ${fig_fdr} & 

collectMeanDataOnly=False
usePrefData=False
LGN_swithc=false
OPstatus=1
TF=8
ori=0
v1=${lgn}
res=${lgn}

python ${fig_fdr}/plotV1_response_lFF_${op}.py ${op} ${res} ${lgn} ${v1} ${res_fdr} ${setup_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} &
if [[ "${cleanData}" = "1" ]]
then
    date
    echo delete data
    python ${fig_fdr}/clean_data_${op}.py ${data_fdr} ${op}
    rm ${data_fdr}/snapShot_*-${op}.bin
fi
