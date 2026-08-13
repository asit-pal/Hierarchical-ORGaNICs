#!/bin/bash
# Submit the SDE validation as a SLURM array -- one task per contrast in
# SDE_validation.c_vals -- followed by a merge/plot job that depends on all of them.
#
# Usage:
#   bash Job_Scripts/submit_sde_validation.sh <config_suffix>
#
#   SDE_EXTRA_ARGS   extra flags passed to every array task (used for cheap preflights)
#   SDE_OUT_PREFIX   shard basename prefix (default: sde_validation)
#   SDE_N_BOOT       paired bootstrap draws for the index error band (default: 1000)
#
# Examples:
#   bash Job_Scripts/submit_sde_validation.sh 79
#   SDE_EXTRA_ARGS="--n-trials 4 --T 1.0 --burn-in 0.2" \
#       bash Job_Scripts/submit_sde_validation.sh 79_preflight

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_suffix>   (e.g. 79, or 79_preflight)"
    exit 1
fi

CONFIG_NUM=$1
BASE_DIR="/home/ap6603/Hierarchical-ORGaNICs"
SOURCE_CONFIG="${BASE_DIR}/configs/config.yaml"
RESULTS_DIR="${BASE_DIR}/Results_5/config_${CONFIG_NUM}"
NEW_CONFIG="${RESULTS_DIR}/config_${CONFIG_NUM}.yaml"
JOB_SCRIPTS_DIR="${BASE_DIR}/Job_Scripts"
SIF=/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif
OVERLAY=/scratch/ap6603/Lightning/overlay-50G-10M.ext3:ro

mkdir -p "${RESULTS_DIR}/Data" "${RESULTS_DIR}/Plots"
if [ ! -f "${NEW_CONFIG}" ]; then
    cp "${SOURCE_CONFIG}" "${NEW_CONFIG}"
    echo "Copied ${SOURCE_CONFIG} -> ${NEW_CONFIG}"
fi

# Read the array size straight from the config so the contrast literals live in exactly
# one place. The tasks select their contrast by *index* into this same list, and the
# merge step checks the shards it finds against it.
N_C=$(singularity exec --overlay "${OVERLAY}" "${SIF}" /bin/bash -c \
    "source /ext3/env.sh; python -c \"import yaml,sys; print(len(yaml.safe_load(open(sys.argv[1]))['SDE_validation']['c_vals']))\" '${NEW_CONFIG}'")

if ! [[ "${N_C}" =~ ^[0-9]+$ ]] || [ "${N_C}" -lt 1 ]; then
    echo "Could not read SDE_validation.c_vals from ${NEW_CONFIG} (got '${N_C}')"
    exit 1
fi

export BASE_DIR CONFIG_NUM NEW_CONFIG RESULTS_DIR
export SDE_EXTRA_ARGS="${SDE_EXTRA_ARGS:-}"
export SDE_OUT_PREFIX="${SDE_OUT_PREFIX:-sde_validation}"
export SDE_N_BOOT="${SDE_N_BOOT:-1000}"

echo "Config      : ${NEW_CONFIG}"
echo "Contrasts   : ${N_C} (array 0-$((N_C - 1)))"
echo "Extra args  : ${SDE_EXTRA_ARGS:-<none>}"

ARRAY_JOB=$(sbatch --parsable \
    --array=0-$((N_C - 1)) \
    --output="${RESULTS_DIR}/sde.%A_%a.out" \
    "${JOB_SCRIPTS_DIR}/sde_validation_array.sbatch")
echo "Submitted contrast array : ${ARRAY_JOB}"

# afterok, not afterany: a missing contrast would make the merge pick the wrong
# background and rescale every normalised curve without complaining.
MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_JOB}" \
    --output="${RESULTS_DIR}/sde_merge.%j.out" \
    "${JOB_SCRIPTS_DIR}/sde_validation_merge.sbatch")
echo "Submitted merge + plots  : ${MERGE_JOB}  (after ${ARRAY_JOB})"
echo
echo "Watch:   squeue -u \$USER"
echo "Logs:    ${RESULTS_DIR}/sde.${ARRAY_JOB}_*.out"
echo "Report:  ${RESULTS_DIR}/sde_merge.${MERGE_JOB}.out"
