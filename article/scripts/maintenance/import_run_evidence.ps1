[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^EXP-[A-Z]+-[0-9]{3}$')]
    [string]$ExperimentId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^RUN-[0-9]{8}-[0-9]{6}-s[0-9]{2}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [string]$SourceRunDirectory,

    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$source = (Resolve-Path -LiteralPath $SourceRunDirectory).Path
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$destination = Join-Path $repo "_local\raw_evidence\$ExperimentId\$RunId"

if (Test-Path -LiteralPath $destination) {
    throw "Destination already exists; run evidence is append-only: $destination"
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $source '*') -Destination $destination -Recurse -Force

$manifest = Get-ChildItem -LiteralPath $destination -Recurse -File | Sort-Object FullName | ForEach-Object {
    [pscustomobject]@{
        relative_path = $_.FullName.Substring($destination.Length + 1).Replace('\', '/')
        size_bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
}
$manifest | Export-Csv -LiteralPath (Join-Path $destination 'private_inventory.csv') -NoTypeInformation -Encoding utf8
Write-Host "Imported $($manifest.Count) files to $destination"
Write-Host 'Next: create/update the public experiment summary and register ART IDs. Do not commit _local.'
