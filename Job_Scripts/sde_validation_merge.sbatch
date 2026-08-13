#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=SDE_val_merge
#SBATCH --account=torch_pr_239_general
#
# Collect the per-contrast shards, report the comparison, and draw the figures.
# Submitted with --dependency=afterok on the array, so a failed contrast blocks this
# rather than producing a figure whose background is silently the wrong condition.

module purge

PREFIX="${SDE_OUT_PREFIX:-sde_validation}"
NBOOT="${SDE_N_BOOT:-1000}"

singularity exec \
    --overlay /scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro \
    /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash << EOF
source /ext3/env.sh
cd "${BASE_DIR}"

echo "=== Merging shards and reporting the contrast-index comparison ==="
python -u ${BASE_DIR}/Analysis/merge_sde_validation.py "${RESULTS_DIR}" \
    --prefix ${PREFIX} --n-boot ${NBOOT}

echo "=== Per-condition figures (absolute PSD, coherence, traces) ==="
python -u ${BASE_DIR}/Plotting/Plot_SDE_validation.py "${RESULTS_DIR}"

echo "=== Published-normalisation comparison across contrasts ==="
python -u ${BASE_DIR}/Plotting/Plot_SDE_validation_contrast.py "${RESULTS_DIR}" \
    --n-boot ${NBOOT}
EOF
