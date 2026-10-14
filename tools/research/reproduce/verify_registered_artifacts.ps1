[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [switch]$RequireAll
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path }
$ArticleRoot = Join-Path $RepositoryRoot 'article'
$registry = Join-Path $ArticleRoot 'docs\evidence\artifact_registry.csv'
$rows = Import-Csv -LiteralPath $registry
$failed = 0
$verified = 0
$skipped = 0

foreach ($row in $rows) {
    if ($row.local_locator -notlike 'legacy://*') {
        Write-Warning "$($row.artifact_id): unsupported locator $($row.local_locator)"
        continue
    }
    $relative = $row.local_locator.Substring('legacy://'.Length).Replace('/', '\')
    $path = Join-Path $ArticleRoot $relative
    if (-not (Test-Path -LiteralPath $path)) {
        if ($RequireAll) {
            Write-Warning "$($row.artifact_id): missing local artifact"
            $failed++
        } else {
            Write-Verbose "$($row.artifact_id): private artifact is not present in this checkout"
            $skipped++
        }
        continue
    }
    $item = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $ok = ($item.Length -eq [int64]$row.size_bytes) -and ($hash -eq $row.sha256)
    [pscustomobject]@{
        artifact_id = $row.artifact_id
        bytes_ok = $item.Length -eq [int64]$row.size_bytes
        hash_ok = $hash -eq $row.sha256
        status = if ($ok) { 'OK' } else { 'FAILED' }
    }
    if ($ok) { $verified++ } else { $failed++ }
}

if ($failed -gt 0) {
    throw "$failed artifact verification(s) failed"
}
Write-Host "Verified $verified available registered artifact(s); skipped $skipped unavailable private artifact(s)."
