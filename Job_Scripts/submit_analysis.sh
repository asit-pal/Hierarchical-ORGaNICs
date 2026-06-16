#!/bin/bash
# NOTE: Update BASE_DIR below to point to your local clone of this repository.
# The sbatch scripts also contain hardcoded paths that must be updated for your
# HPC environment (singularity container, conda environment, etc.).

# Check if config number is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_number>"
    exit 1
fi

# Set config number and paths
CONFIG_NUM=$1
BASE_DIR="/home/ap6603/Hierarchical-ORGaNICs"
SOURCE_CONFIG="${BASE_DIR}/configs/config.yaml"
RESULTS_DIR="${BASE_DIR}/Results_5/config_${CONFIG_NUM}"
NEW_CONFIG="${RESULTS_DIR}/config_${CONFIG_NUM}.yaml"
JOB_SCRIPTS_DIR="${BASE_DIR}/Job_Scripts"

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}/Data"
mkdir -p "${RESULTS_DIR}/Plots"

# Copy config file if it doesn't exist
if [ ! -f "${NEW_CONFIG}" ]; then
    cp "${SOURCE_CONFIG}" "${NEW_CONFIG}"
fi

# Export variables for use in job scripts
export BASE_DIR
export CONFIG_NUM
export NEW_CONFIG
export RESULTS_DIR

echo "Submitting analysis jobs for config ${CONFIG_NUM}"

# Submit power spectra analysis
power_analysis_job=$(sbatch ${JOB_SCRIPTS_DIR}/power_analysis_job.sbatch)
echo "Submitted power spectra analysis job: ${power_analysis_job}"

# Submit coherence analysis
# coherence_job=$(sbatch ${JOB_SCRIPTS_DIR}/coherence_analysis_job.sbatch)
# echo "Submitted coherence analysis job: ${coherence_job}"

# Submit communication analysis
# comm_job=$(sbatch ${JOB_SCRIPTS_DIR}/communication_analysis_job.sbatch)
# echo "Submitted communication analysis job: ${comm_job}"
