#!/bin/bash

# Train a 2D nnU-Net baseline for TotalSpineSeg X-ray experiments.

set -e
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

DATASET=${1:-201}
FOLD=${2:-0}

TOTALSPINESEG_DATA="$(realpath "${TOTALSPINESEG_DATA:-data}")"

CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}
MEMGB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
JOBS=${TOTALSPINESEG_JOBS:-$CORES}
JOBSNN=$(( JOBS < $((MEMGB / 8)) ? JOBS : $((MEMGB / 8)) ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))
JOBSNN=${TOTALSPINESEG_JOBSNN:-$JOBSNN}
DEVICE=${TOTALSPINESEG_DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}

export nnUNet_def_n_proc=$JOBSNN
export nnUNet_n_proc_DA=$JOBSNN
export nnUNet_raw="$TOTALSPINESEG_DATA"/nnUNet/raw
export nnUNet_preprocessed="$TOTALSPINESEG_DATA"/nnUNet/preprocessed
export nnUNet_results="$TOTALSPINESEG_DATA"/nnUNet/results

nnUNetTrainer=${3:-nnUNetTrainer}
nnUNetPlanner=${4:-ExperimentPlanner}
nnUNetPlans=${5:-nnUNetPlans}
gpu_memory_target=${6:-${TOTALSPINESEG_NNUNET_GPU_MEMORY_TARGET:-}}
configuration=2d

d_name=$(basename "$(ls -d "$nnUNet_raw"/Dataset${DATASET}_*)")

echo ""
echo "Running X-ray training with the following parameters:"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNetTrainer=${nnUNetTrainer}"
echo "nnUNetPlanner=${nnUNetPlanner}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "gpu_memory_target=${gpu_memory_target}"
echo "configuration=${configuration}"
echo "JOBSNN=${JOBSNN}"
echo "DEVICE=${DEVICE}"
echo "DATASET=${DATASET}"
echo "FOLD=${FOLD}"
echo ""

if [ ! -f "$nnUNet_preprocessed"/$d_name/dataset_fingerprint.json ]; then
    echo "Extracting fingerprint dataset $d_name"
    nnUNetv2_extract_fingerprint -d $DATASET -np $JOBSNN --verify_dataset_integrity
fi

if [ ! -f "$nnUNet_preprocessed"/$d_name/${nnUNetPlans}.json ]; then
    echo "Planning dataset $d_name"
    if [ -n "$gpu_memory_target" ]; then
        nnUNetv2_plan_experiment -d $DATASET -pl $nnUNetPlanner -gpu_memory_target $gpu_memory_target -overwrite_plans_name $nnUNetPlans
    else
        nnUNetv2_plan_experiment -d $DATASET -pl $nnUNetPlanner -overwrite_plans_name $nnUNetPlans
    fi
fi

data_identifier=$(python3 - <<PY
import json
from pathlib import Path

plans_path = Path(r"$nnUNet_preprocessed") / r"$d_name" / r"${nnUNetPlans}.json"
with plans_path.open("r", encoding="utf-8") as handle:
    plans = json.load(handle)
print(plans["configurations"]["$configuration"]["data_identifier"])
PY
)
echo "data_identifier=${data_identifier}"

if [[ ! -d "$nnUNet_preprocessed"/$d_name/$data_identifier ]]; then
    echo "Preprocessing dataset $d_name"
    nnUNetv2_preprocess -d $DATASET -plans_name $nnUNetPlans -c $configuration -np $JOBSNN
fi

echo "Training dataset $d_name fold $FOLD"
nnUNetv2_train $DATASET $configuration $FOLD -tr $nnUNetTrainer -p $nnUNetPlans --c -device $DEVICE

if [[ -d "$nnUNet_raw"/$d_name/imagesTs && -d "$nnUNet_raw"/$d_name/labelsTs ]]; then
    echo "Running test-set prediction for dataset $d_name"
    test_dir="$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/fold_${FOLD}/test
    mkdir -p "$test_dir"
    nnUNetv2_predict -d $DATASET -i "$nnUNet_raw"/$d_name/imagesTs -o "$test_dir" -f $FOLD -c $configuration -tr $nnUNetTrainer -p $nnUNetPlans -npp $JOBSNN -nps $JOBSNN
    nnUNetv2_evaluate_folder "$nnUNet_raw"/$d_name/labelsTs "$test_dir" -djfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/dataset.json -pfile "$nnUNet_results"/$d_name/${nnUNetTrainer}__${nnUNetPlans}__${configuration}/plans.json -np $JOBSNN
fi
