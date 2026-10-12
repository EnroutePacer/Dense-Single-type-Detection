[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [string]$ManifestPath = 'docs\reproducibility\source_snapshot.sha256',
    [switch]$CanonicalLF
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$manifest = Join-Path $RepositoryRoot $ManifestPath
$failed = 0
$count = 0

foreach ($line in Get-Content -LiteralPath $manifest) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw "Malformed manifest line: $line" }
    $expected = $Matches[1]
    $relative = $Matches[2].Replace('/', '\')
    $path = Join-Path $RepositoryRoot $relative
    $count++
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Warning "Missing: $relative"
        $failed++
        continue
    }
    if ($CanonicalLF) {
        # New text snapshots use UTF-8 / LF independent of Git checkout settings.
        # Keep the historical byte-for-byte verification as the default.
        $bytes = [System.IO.File]::ReadAllBytes($path)
        $content = [System.Text.Encoding]::UTF8.GetString($bytes).Replace("`r`n", "`n")
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $digest = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($content))
            $actual = ([System.BitConverter]::ToString($digest)).Replace('-', '')
        } finally { $sha.Dispose() }
    } else {
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    }
    if ($actual -ne $expected) {
        Write-Warning "Hash mismatch: $relative"
        $failed++
    }
}

if ($failed -gt 0) { throw "$failed source hash verification(s) failed" }
Write-Host "Verified $count source snapshot files."
