$ErrorActionPreference = 'Stop'

$projectRoot = 'C:\Users\BLESSING MAMBWE\Pictures\Projects\Latent Probing For Toxicity'
$pythonExe = 'C:\Python312\python.exe'
$submitScript = Join-Path $projectRoot 'codabench_submit.py'
$verifyScript = Join-Path $projectRoot 'verify_codabench_submission.py'
$submissionZip = Join-Path $projectRoot 'artifacts\kaggle\select\submissions\best_single.zip'
$submissionLog = Join-Path $projectRoot 'results\submission_log.json'
$runLog = Join-Path $projectRoot 'results\scheduled_testing_submit.log'
$verificationJson = Join-Path $projectRoot 'artifacts\testing_submission_result.json'
$expectedHash = '0bd11ac11c0a09a8b38d834e06c558f0e17236dae680c215a5e33af8c746a601'
$label = 'final_testing_pooled_std_lda06'
$phaseStart = [DateTimeOffset]::Parse('2026-09-10T00:00:00Z')
$phaseEnd = [DateTimeOffset]::Parse('2026-09-12T00:00:00Z')

function Write-RunLog([string]$message) {
    $stamp = [DateTimeOffset]::UtcNow.ToString('o')
    Add-Content -LiteralPath $runLog -Value "$stamp $message"
}

Set-Location -LiteralPath $projectRoot
Write-RunLog 'scheduled Testing Phase submission started'

$actualHash = (Get-FileHash -LiteralPath $submissionZip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    Write-RunLog "ABORT hash mismatch: $actualHash"
    throw 'Submission ZIP hash mismatch'
}

$now = [DateTimeOffset]::UtcNow
if ($now -lt $phaseStart -or $now -ge $phaseEnd) {
    Write-RunLog "ABORT outside Testing Phase window: $($now.ToString('o'))"
    throw 'Outside Testing Phase window'
}
Write-RunLog "guards passed; exact hash $actualHash"

& $pythonExe $submitScript submit $submissionZip --yes --phase 'Testing Phase' --label $label 2>&1 |
    Tee-Object -FilePath $runLog -Append
$uploadExit = $LASTEXITCODE
Write-RunLog "upload command exit status $uploadExit"
if ($uploadExit -ne 0) {
    throw "Upload command failed with status $uploadExit; no retry will be attempted"
}

$rows = Get-Content -LiteralPath $submissionLog -Raw | ConvertFrom-Json
$match = $rows | Where-Object { $_.label -eq $label -and $_.phase -eq 'Testing Phase' } |
    Select-Object -Last 1
if (-not $match -or -not $match.submission_id) {
    Write-RunLog 'ABORT upload returned no recorded submission ID; no retry will be attempted'
    throw 'No unambiguous Testing submission ID was recorded'
}
Write-RunLog "verifying submission ID $($match.submission_id) against phase 29479"

& $pythonExe $verifyScript $match.submission_id --expected-phase 29479 --output $verificationJson --max-wait 720 2>&1 |
    Tee-Object -FilePath $runLog -Append
$verifyExit = $LASTEXITCODE
Write-RunLog "verification exit status $verifyExit"
exit $verifyExit
