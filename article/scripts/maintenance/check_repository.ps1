[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [switch]$ReleaseStrict
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$excludedTop = @('_local', '00_overview', '01_sources', '02_experiments', '03_evidence', '04_analysis', '05_paper', '99_archive', '.git')
$files = Get-ChildItem -LiteralPath $repo -Recurse -Force -File | Where-Object {
    $relative = $_.FullName.Substring($repo.Length + 1)
    $top = ($relative -split '[\\/]')[0]
    $excludedTop -notcontains $top -and $relative -notmatch '(^|[\\/])__pycache__([\\/]|$)'
}

$errors = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()

$forbidden = $files | Where-Object {
    $_.FullName -match '[\\/](output|outputs|__pycache__|checkpoints)[\\/]' -or
    $_.Name -match '^(events\.out|.*\.(pyc|pth|pt|onnx|pdiparams|pdmodel|tar\.gz|log))$'
}
foreach ($item in $forbidden) { $errors.Add("Forbidden public artifact: $($item.FullName)") }

foreach ($item in $files | Where-Object { $_.Length -gt 10MB }) {
    $errors.Add("Public file exceeds 10 MiB: $($item.FullName)")
}

$pathPattern = '(\b[A-Z]:\\[A-Za-z0-9_. ()-]{2,}|/aicamera-mlp/|(?:root|admin|ubuntu)@)'
foreach ($item in $files | Where-Object { $_.FullName -ne $PSCommandPath -and ($_.Extension -match '^\.(md|txt|yaml|yml|json|csv|py|ps1|sh)$' -or $_.Name -in @('README', 'NOTICE')) }) {
    $matches = Select-String -LiteralPath $item.FullName -Pattern $pathPattern -AllMatches -ErrorAction SilentlyContinue
    foreach ($match in $matches) {
        $message = "Machine-specific path: $($item.FullName):$($match.LineNumber)"
        if ($ReleaseStrict) { $errors.Add($message) } else { $warnings.Add($message) }
    }
}

foreach ($warning in $warnings | Sort-Object -Unique) { Write-Warning $warning }
foreach ($entryError in $errors | Sort-Object -Unique) { Write-Warning "ERROR: $entryError" }

if ($errors.Count -gt 0) { throw "Repository check failed with $($errors.Count) error(s)." }
Write-Host "Repository structure check passed: $($files.Count) public-tree files; $($warnings.Count) release warning(s)."
