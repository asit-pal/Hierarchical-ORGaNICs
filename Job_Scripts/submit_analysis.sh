#!/bin/bash

# Check if config number is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_number>"
    exit 1
fi

# Set config number and paths
CONFIG_NUM=$1
BASE_DIR="/home/ap6603/Script_analysis"
SOURCE_CONFIG="${BASE_DIR}/configs/config.yaml"
RESULTS_DIR="${BASE_DIR}/Results_4/config_${CONFIG_NUM}"
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

# # # Submit power spectra analysis (use full path)
power_analysis_job=$(sbatch ${JOB_SCRIPTS_DIR}/power_analysis_job.sbatch)
echo "Submitted power spectra analysis job: ${power_analysis_job}"

# # # # # # # Submit coherence analysis
# coherence_job=$(sbatch ${JOB_SCRIPTS_DIR}/coherence_analysis_job.sbatch)
# echo "Submitted coherence analysis job: ${coherence_job}"

# # # # # Submit communication analysis
# comm_job=$(sbatch ${JOB_SCRIPTS_DIR}/communication_analysis_job.sbatch)
# echo "Submitted communication analysis job: ${comm_job}"

# # Submit gain modulation analysis
# gain_mod_job=$(sbatch ${JOB_SCRIPTS_DIR}/gain_modulation_job.sbatch)
# echo "Submitted gain modulation analysis job: ${gain_mod_job}"

# # Submit three area communication analysis
# comm_3_job=$(sbatch ${JOB_SCRIPTS_DIR}/communication_analysis_3_job.sbatch)
# echo "Submitted communication analysis job: ${comm_3_job}"

# # # Submit frequency wise decomposition analysis
# freq_job=$(sbatch ${JOB_SCRIPTS_DIR}/dimension_vs_freq.sbatch)
# echo "Submitted dimension vs freq analysis job: ${freq_job}"

# freq_job=$(sbatch ${JOB_SCRIPTS_DIR}/prediction_perf_vs_freq.sbatch)
# echo "Submitted prediction perf vs freq analysis job: ${freq_job}"

# Submit stability analysis
# stability_job=$(sbatch ${JOB_SCRIPTS_DIR}/stability_analysis_job.sbatch)
# echo "Submitted stability analysis job: ${stability_job}"

# # Submit weight matrices plotting
# weight_mat_job=$(sbatch ${JOB_SCRIPTS_DIR}/plot_weight_matrices.sbatch)
# echo "Submitted weight matrices plotting job: ${weight_mat_job}"