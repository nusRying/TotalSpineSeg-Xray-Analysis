param(
    [int]$Dataset = 201,
    [int]$Fold = 0,
    [string]$nnUNetTrainer = "nnUNetTrainer",
    [string]$nnUNetPlanner = "ExperimentPlanner",
    [string]$nnUNetPlans = "nnUNetPlans",
    [double]$GpuMemoryTargetGb = 0,
    [switch]$StopAfterPreprocess
)

$ErrorActionPreference = "Stop"

function Get-EnvOrDefault([string]$Name, [string]$DefaultValue) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return $value
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultDataRoot = Join-Path $repoRoot "data"
$dataRoot = Get-EnvOrDefault "TOTALSPINESEG_DATA" $defaultDataRoot
$dataRoot = (Resolve-Path $dataRoot).Path

$cores = [Environment]::ProcessorCount
$memoryBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$memoryGb = [Math]::Max([int]($memoryBytes / 1GB), 1)
$jobs = [int](Get-EnvOrDefault "TOTALSPINESEG_JOBS" "$cores")
$jobsNn = [Math]::Min($jobs, [Math]::Max([int]($memoryGb / 8), 1))
$jobsNn = [int](Get-EnvOrDefault "TOTALSPINESEG_JOBSNN" "$jobsNn")

$pythonExecutable = "C:\Users\umair\miniconda3\envs\totalspineseg-xray\python.exe"
$device = Get-EnvOrDefault "TOTALSPINESEG_DEVICE" (
    & $pythonExecutable -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
)

$env:nnUNet_def_n_proc = "$jobsNn"
$env:nnUNet_n_proc_DA = "$jobsNn"
$env:nnUNet_raw = Join-Path $dataRoot "nnUNet\\raw"
$env:nnUNet_preprocessed = Join-Path $dataRoot "nnUNet\\preprocessed"
$env:nnUNet_results = Join-Path $dataRoot "nnUNet\\results"

$datasetDirs = Get-ChildItem $env:nnUNet_raw -Directory | Where-Object { $_.Name -like ("Dataset{0:D3}_*" -f $Dataset) }
if ($datasetDirs.Count -ne 1) {
    throw "Expected exactly one dataset directory for id $Dataset under $($env:nnUNet_raw)."
}
$datasetDir = $datasetDirs[0]
$datasetName = $datasetDir.Name
$configuration = "2d"

Write-Host ""
Write-Host "Running X-ray training with the following parameters:"
Write-Host "nnUNet_raw=$($env:nnUNet_raw)"
Write-Host "nnUNet_preprocessed=$($env:nnUNet_preprocessed)"
Write-Host "nnUNet_results=$($env:nnUNet_results)"
Write-Host "nnUNetTrainer=$nnUNetTrainer"
Write-Host "nnUNetPlanner=$nnUNetPlanner"
Write-Host "nnUNetPlans=$nnUNetPlans"
Write-Host "GpuMemoryTargetGb=$GpuMemoryTargetGb"
Write-Host "configuration=$configuration"
Write-Host "JOBSNN=$jobsNn"
Write-Host "DEVICE=$device"
Write-Host "DATASET=$Dataset"
Write-Host "FOLD=$Fold"
Write-Host ""

$pythonScriptsDir = "C:\Users\umair\miniconda3\envs\totalspineseg-xray\Scripts"

$fingerprintPath = Join-Path $env:nnUNet_preprocessed "$datasetName\\dataset_fingerprint.json"
if (-not (Test-Path $fingerprintPath)) {
    Write-Host "Extracting fingerprint for dataset $datasetName"
    Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_extract_fingerprint.exe") -d $Dataset -np $jobsNn --verify_dataset_integrity } -FailureMessage "nnUNet fingerprint extraction failed."
}

$plansPath = Join-Path $env:nnUNet_preprocessed "$datasetName\\$nnUNetPlans.json"
if (-not (Test-Path $plansPath)) {
    Write-Host "Planning dataset $datasetName"
    if ($GpuMemoryTargetGb -gt 0) {
        Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_plan_experiment.exe") -d $Dataset -pl $nnUNetPlanner -gpu_memory_target $GpuMemoryTargetGb -overwrite_plans_name $nnUNetPlans } -FailureMessage "nnUNet experiment planning failed."
    } else {
        Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_plan_experiment.exe") -d $Dataset -pl $nnUNetPlanner -overwrite_plans_name $nnUNetPlans } -FailureMessage "nnUNet experiment planning failed."
    }
}

$plansJson = Get-Content $plansPath -Raw | ConvertFrom-Json
$dataIdentifier = $plansJson.configurations.$configuration.data_identifier
Write-Host "data_identifier=$dataIdentifier"

$preprocessedPath = Join-Path $env:nnUNet_preprocessed "$datasetName\\$dataIdentifier"
if (-not (Test-Path $preprocessedPath)) {
    Write-Host "Preprocessing directory missing. Starting preprocessing..."
    Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_preprocess.exe") -d $Dataset -plans_name $nnUNetPlans -c $configuration -np $jobsNn } -FailureMessage "nnUNet preprocessing failed."
}

# --- NEW: Robustness Check ---
$datasetJson = Get-Content (Join-Path $env:nnUNet_raw "$datasetName\\dataset.json") -Raw | ConvertFrom-Json
$expectedCount = $datasetJson.numTraining
$actualCount = (Get-ChildItem $preprocessedPath -Filter "*_seg.b2nd").Count

if ($actualCount -lt $expectedCount) {
    Write-Host "WARNING: Partial preprocessing detected ($actualCount/$expectedCount). Resuming..."
    Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_preprocess.exe") -d $Dataset -plans_name $nnUNetPlans -c $configuration -np $jobsNn } -FailureMessage "nnUNet preprocessing resume failed."
}
# -----------------------------

$freeSpaceGb = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
if ($freeSpaceGb -lt 5) {
    Write-Host "CRITICAL: Only $freeSpaceGb GB free. Training will likely fail due to disk space."
    # We don't exit, but we warn loudly
}

if ($StopAfterPreprocess) {
    Write-Host "Stopped after preprocessing as requested."
    exit 0
}

Write-Host "Training dataset $datasetName fold $Fold"
Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_train.exe") $Dataset $configuration $Fold -tr $nnUNetTrainer -p $nnUNetPlans --c -device $device } -FailureMessage "nnUNet training failed."

$imagesTs = Join-Path $datasetDir.FullName "imagesTs"
$labelsTs = Join-Path $datasetDir.FullName "labelsTs"
if ((Test-Path $imagesTs) -and (Test-Path $labelsTs)) {
    Write-Host "Running test-set prediction for dataset $datasetName"
    $testDir = Join-Path $env:nnUNet_results "$datasetName\\${nnUNetTrainer}__${nnUNetPlans}__${configuration}\\fold_${Fold}\\test"
    New-Item -ItemType Directory -Force -Path $testDir | Out-Null
    Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_predict.exe") -d $Dataset -i $imagesTs -o $testDir -f $Fold -c $configuration -tr $nnUNetTrainer -p $nnUNetPlans -npp $jobsNn -nps $jobsNn } -FailureMessage "nnUNet test prediction failed."
    Invoke-CheckedCommand -Command { & (Join-Path $pythonScriptsDir "nnUNetv2_evaluate_folder.exe") $labelsTs $testDir -djfile (Join-Path $env:nnUNet_results "$datasetName\\${nnUNetTrainer}__${nnUNetPlans}__${configuration}\\dataset.json") -pfile (Join-Path $env:nnUNet_results "$datasetName\\${nnUNetTrainer}__${nnUNetPlans}__${configuration}\\plans.json") -np $jobsNn } -FailureMessage "nnUNet test evaluation failed."
}
