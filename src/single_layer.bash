#!/bin/bash
repo=$HOME/repos/patchV1
cfg_fdr=${repo}/src
fdr0=/scratch/wd554/patchV1/
res_fdr=${fdr0}/resource
fig_fdr=${fdr0}/test


trial_suffix=single_test
patch_cfg=ori.cfg

generate_V1_connection=True
V1_connectome_suffix=single_test
genCon_cfg=connectome.cfg

generate_LGN_V1_connection=True
LGN_V1_suffix=single_test
retino_cfg=LGN_V1.cfg

TF=8 # input frequency if applicable
ori=0 # input orientation index if applicable
nOri=0 # total # orientaton if applicable
OPstatus=1 
usePrefData=False # use fitted preference from plotTC
collectMeanDataOnly=False # no plots, just collect average data

readNewSpike=$1
if [ -z "${readNewSpike}" ]; 
then
	readNewSpike=True
fi

plotOnly=$2
if [ -z "${plotOnly}" ]; 
then
	plotOnly=False
fi

if [ -d "${fig_fdr}" ]
then
	echo overwrite contents in ${fig_fdr}
else
	mkdir -p ${fig_fdr}
fi

# copy files
if [ "${plotOnly}" = False ]
then

	if [ "${generate_LGN_V1_connection}" = True ]
	then
		cp ${cfg_fdr}/${retino_cfg} ${fig_fdr}/retino_${LGN_V1_suffix}.cfg
	fi

	if [ "${generate_V1_connection}" = True ]
	then
		cp ${cfg_fdr}/${genCon_cfg} ${fig_fdr}/genCon_${V1_connectome_suffix}.cfg
	fi

	cp ${cfg_fdr}/${patch_cfg} ${fig_fdr}/patch_${trial_suffix}.cfg
fi
cp ${repo}/src/plotV1_response.py ${fig_fdr}/plotV1_response_${trial_suffix}.cfg

# run
cd ${fdr0}
if [ "${plotOnly}" = False ]
then
	if [ "${generate_LGN_V1_connection}" = True ]
	then
		retino -c ${fig_fdr}/retino_${LGN_V1_suffix}.cfg
	fi

	if [ "${generate_V1_connection}" = True ]
	then
		genCon -c ${fig_fdr}/genCon_${V1_connectome_suffix}.cfg
	fi

	patch_fast -c {fig_fdr}/patch_${trial_suffix}.cfg
fi

data_fdr=${fdr0}

python plotV1_response.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
