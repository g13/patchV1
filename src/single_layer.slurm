#!/bin/bash
#SBATCH --job-name=plotV1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH -o log/pV1_%J.out
#SBATCH --mail-user=weidai@cims.nyu.edu
#SBATCH --mail-type=END
#SBATCH --gres=gpu:1
##SBATCH -p nvidia 
set -e
repo=$HOME/repos/patchV1
cfg_fdr=${repo}/src
fdr0=/scratch/wd554/patchV1
res_fdr=${fdr0}/resource
fig_fdr=${fdr0}/test


res_suffix=square-single_layer
trial_suffix=single_test
patch_cfg=ori.cfg

generate_V1_connection=False
V1_connectome_suffix=single_test
genCon_cfg=connectome.cfg

generate_LGN_V1_connection=True
LGN_V1_suffix=single_test
retino_cfg=LGN_V1.cfg

TF=8 # input frequency if applicable
ori=2 # input orientation index if applicable
nOri=6 # total # orientaton if applicable
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
if [ "${plotOnly}" = False ]
then
	if [ "${generate_LGN_V1_connection}" = True ]
	then
		cd ${res_fdr}
		retino -c ${fig_fdr}/retino_${LGN_V1_suffix}.cfg
	fi

	if [ "${generate_V1_connection}" = True ]
	then
		cd ${res_fdr}
		genCon -c ${fig_fdr}/genCon_${V1_connectome_suffix}.cfg
	fi
	
	date
	cd ${fdr0}
	echo patch_fast -c ${fig_fdr}/patch_${trial_suffix}.cfg
	#nsys profile -o ori_profile-small.qdstrm patch_fast -c ${fig_fdr}/patch_${trial_suffix}.cfg
	#patch_fast -c ${fig_fdr}/patch_${trial_suffix}.cfg
fi

data_fdr=${fdr0}

date
echo plotV1_response.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus}
plotV1_response.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} ${readNewSpike} ${usePrefData} ${collectMeanDataOnly} ${OPstatus} &

date
echo connections.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr}
connections.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} &

wait
date
