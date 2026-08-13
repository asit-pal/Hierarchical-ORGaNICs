#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=SDE_val_arr
#SBATCH --account=torch_pr_239_general
#
# One array task per contrast in SDE_validation.c_vals. Submitted by
# Job_Scripts/submit_sde_validation.sh, which supplies --array and --output and
# exports BASE_DIR / NEW_CONFIG / RESULTS_DIR.
#
# Serial over 9 contrasts would be ~6.5 h against an 8 h wall limit, with every
# completed contrast lost if the last one overran. As an array it is ~75 min and a
# single failed contrast is a single failed task.

module purge

IDX=${SLURM_ARRAY_TASK_ID}
PREFIX="${SDE_OUT_PREFIX:-sde_validation}"
EXTRA="${SDE_EXTRA_ARGS:-}"

# The config carries `paired: True` so the c = 0 background keeps the noise-matched
# linear shadow: it is the same system on the same Wiener path with no rectification
# clipping, and so the only measurement of how much clipping distorts the background
# that every other contrast's normalised curve divides by. No other contrast needs it.
# (The CLI has --no-paired but no --paired, so this is the direction that works.)
if [ "${IDX}" != "0" ]; then
    EXTRA="${EXTRA} --no-paired"
fi

OUT_NAME="${PREFIX}_c$(printf %02d ${IDX})"

echo "SDE validation: c_index=${IDX} -> Data/${OUT_NAME}.npy"
echo "  extra args: ${EXTRA}"

singularity exec \
    --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
source /ext3/env.sh
cd "${BASE_DIR}"
python -u ${BASE_DIR}/Analysis/SDE_validation.py "${NEW_CONFIG}" \
    --c-index ${IDX} --out-name ${OUT_NAME} \
    --n-jobs ${SLURM_CPUS_PER_TASK} ${EXTRA}
EOF
