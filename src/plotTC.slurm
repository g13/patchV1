#!/bin/bash
#SBATCH --job-name=plotTC
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH -o log/pTC_%J.out
#SBATCH --mail-user=wd554@nyu.edu
#SBATCH --mail-type=END
set -e
module purge
# Greene modules
module load python/intel/3.8.6
# Dalma modules
module list
default=$1 # if run plotTC.slurm alone
if [ "$default" = True ]; then
	nOri=6
	fdr=tc-test
	op=tc-s17t8-no_CI
	lgn=b_4_20-micro
	v1=no_CI
fi

date
echo python ${fig_fdr}/getTuningCurve_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${fitTC} ${fitDataReady}
python ${fig_fdr}/getTuningCurve_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${nOri} ${fitTC} ${fitDataReady}

if [ "$collectMeanDataOnly" = True ]; then
	OPstatus=1
else
	OPstatus=2
fi

if [ "${usePrefData}" = True ]; then

	pid=""
	for ori in $( seq 1 $nOri )
	do
		echo python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} False ${usePrefData} False ${OPstatus}
		python ${fig_fdr}/plotV1_response_${trial_suffix}.py ${trial_suffix} ${res_suffix} ${LGN_V1_suffix} ${V1_connectome_suffix} ${res_fdr} ${data_fdr} ${fig_fdr} ${TF} ${ori} ${nOri} False ${usePrefData} False ${OPstatus} &
		pid+="${!} "
	done
	wait $pid
fi
date
