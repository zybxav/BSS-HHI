param(
    [Parameter(Mandatory = $true)][string]$MatlabExe,
    [Parameter(Mandatory = $true)][string]$RunnerDir,
    [Parameter(Mandatory = $true)][string]$ConfigFile,
    [Parameter(Mandatory = $true)][ValidateRange(1, 2147483647)][int]$DatasetCount,
    [switch]$SmokeTest
)

$ErrorActionPreference = 'Stop'
$logDir = Join-Path $RunnerDir 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$matlabRunnerDir = $RunnerDir.Replace("'", "''")
$matlabConfigFile = $ConfigFile.Replace("'", "''")
$jobs = @()

for ($i = 0; $i -lt $DatasetCount; $i++) {
    $name = "dataset_$($i + 1)"
    if ($SmokeTest) {
        $batch = "addpath('$matlabRunnerDir'); run_dataset_experiments($($i + 1),'$matlabConfigFile',true)"
    } else {
        $batch = "addpath('$matlabRunnerDir'); run_dataset_experiments($($i + 1),'$matlabConfigFile')"
    }
    $stdout = Join-Path $logDir "$name.stdout.log"
    $stderr = Join-Path $logDir "$name.stderr.log"
    $process = Start-Process -FilePath $MatlabExe -ArgumentList "-batch `"$batch`"" `
        -WorkingDirectory $RunnerDir -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $jobs += [pscustomobject]@{ Name = $name; Process = $process; Stdout = $stdout; Stderr = $stderr }
    Write-Host "Started $name (PID $($process.Id))"
}

$failed = $false
foreach ($job in $jobs) {
    $job.Process.WaitForExit()
    $exitCode = $job.Process.ExitCode
    Write-Host "$($job.Name): exit=$exitCode; log=$($job.Stdout)"
    if ($exitCode -ne 0) {
        $failed = $true
        Write-Host "Error log: $($job.Stderr)"
    }
}
if ($failed) { exit 1 }
