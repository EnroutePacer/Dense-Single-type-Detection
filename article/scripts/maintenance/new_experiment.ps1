[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^EXP-[A-Z]+-[0-9]{3}$')]
    [string]$ExperimentId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9-]+$')]
    [string]$Slug,

    [Parameter(Mandatory = $true)]
    [string]$Title,

    [ValidateSet('train', 'eval', 'diagnostic', 'benchmark', 'ablation', 'audit')]
    [string]$Type = 'diagnostic',

    [string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$template = Join-Path $RepositoryRoot 'experiments\_template'
$destination = Join-Path $RepositoryRoot "experiments\${ExperimentId}_$Slug"
if (Test-Path -LiteralPath $destination) {
    throw "Experiment already exists: $destination"
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $template 'README.md'), (Join-Path $template 'experiment.yaml'), (Join-Path $template 'reproduce.sh') -Destination $destination

$replacements = @{
    'EXP-TYPE-000' = $ExperimentId
    'short_ascii_slug' = $Slug
    '实验标题' = $Title
    'type: diagnostic' = "type: $Type"
}
foreach ($file in @('README.md', 'experiment.yaml', 'reproduce.sh')) {
    $path = Join-Path $destination $file
    $content = Get-Content -LiteralPath $path -Raw
    foreach ($key in $replacements.Keys) {
        $content = $content.Replace($key, $replacements[$key])
    }
    Set-Content -LiteralPath $path -Value $content -Encoding utf8
}

New-Item -ItemType Directory -Path (Join-Path $destination 'configs') -Force | Out-Null
Write-Host "Created: $destination"
Write-Host 'Register it in experiments/registry.csv before running.'
