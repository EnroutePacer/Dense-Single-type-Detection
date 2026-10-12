[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [string]$RepositoryRoot,

    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path

foreach ($required in @('engine', 'configs', 'tools', 'train.py', 'requirements.txt', 'LICENSE')) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $required))) {
        throw "Source is not a DEIM root; missing: $required"
    }
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$importRoot = Join-Path $repo "_local\imports\CODE-$timestamp"
$changes = [System.Collections.Generic.List[object]]::new()

$sourceFiles = @()
foreach ($dir in @('engine', 'configs', 'tools')) {
    $sourceFiles += Get-ChildItem -LiteralPath (Join-Path $source $dir) -Recurse -File |
        Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -ne '.pyc' }
}
$sourceFiles += Get-Item -LiteralPath (Join-Path $source 'train.py'), (Join-Path $source 'requirements.txt'), (Join-Path $source 'LICENSE')

foreach ($file in $sourceFiles | Sort-Object FullName) {
    $relative = $file.FullName.Substring($source.Length + 1)
    $destination = Join-Path $repo $relative
    $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    $destinationHash = if (Test-Path -LiteralPath $destination) {
        (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    } else {
        $null
    }

    if ($sourceHash -ne $destinationHash) {
        $changes.Add([pscustomobject]@{
            relative_path = $relative.Replace('\', '/')
            source_sha256 = $sourceHash
            previous_sha256 = $destinationHash
            action = if ($destinationHash) { 'update' } else { 'add' }
        })

        if ($Apply -and $PSCmdlet.ShouldProcess($destination, 'Import active source file')) {
            if (Test-Path -LiteralPath $destination) {
                $backup = Join-Path (Join-Path $importRoot 'before') $relative
                New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
                Copy-Item -LiteralPath $destination -Destination $backup -Force
            }
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
        }
    }
}

$changes | Format-Table -AutoSize
Write-Host "Changed files: $($changes.Count)"

if ($Apply) {
    New-Item -ItemType Directory -Path $importRoot -Force | Out-Null
    $changes | Export-Csv -LiteralPath (Join-Path $importRoot 'changes.csv') -NoTypeInformation -Encoding utf8
    [pscustomobject]@{
        imported_at = (Get-Date).ToString('o')
        source_root = $source
        repository_root = $repo
        changed_files = $changes.Count
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $importRoot 'import.json') -Encoding utf8
    Write-Host "Import evidence: $importRoot"
} else {
    Write-Host 'Audit only. Re-run with -Apply after reviewing the change list.'
}
