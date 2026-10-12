[CmdletBinding()]
param(
    [string]$RepositoryRoot,
    [switch]$ReleaseStrict
)

$ErrorActionPreference = 'Stop'
$RepositoryRoot = if ($RepositoryRoot) { $RepositoryRoot } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path }
$repo = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$excludedTop = @('_local', '00_overview', '01_sources', '02_experiments', '03_evidence', '04_analysis', '05_paper', '99_archive', '.git')
$files = Get-ChildItem -LiteralPath $repo -Recurse -Force -File | Where-Object {
    $relative = $_.FullName.Substring($repo.Length + 1)
    $top = ($relative -split '[\\/]')[0]
    $excludedTop -notcontains $top -and $relative -notmatch '(^|[\\/])__pycache__([\\/]|$)'
}

$errors = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()

foreach ($item in $files) {
    $relative = $item.FullName.Substring($repo.Length + 1)
    $gitPath = $relative.Replace('\', '/')
    $inCanonicalOutput = $relative -match '^(output|outputs)[\\/]'
    $requiresLfs = $item.Name -match '(^events\.out\.tfevents\.|.*\.(pth|pt|onnx|pdiparams|pdmodel|engine|tar|tar\.gz|tgz|zip|7z)$)'
    $filterLine = (& git -C $repo check-attr filter -- $gitPath 2>$null) -join ''
    $usesLfs = $filterLine -match ': filter: lfs$'

    if ($item.FullName -match '[\\/](__pycache__|checkpoints)[\\/]' -or $item.Extension -eq '.pyc') {
        $errors.Add("Forbidden public artifact: $($item.FullName)")
    }
    elseif ($requiresLfs -and (-not $inCanonicalOutput -or -not $usesLfs)) {
        $errors.Add("Large/binary artifact is outside canonical LFS output policy: $($item.FullName)")
    }
    elseif ($item.Extension -eq '.log' -and -not $inCanonicalOutput) {
        $errors.Add("Raw log is outside output/ or outputs/: $($item.FullName)")
    }

    if ($item.Length -gt 10MB -and -not $usesLfs) {
        $errors.Add("Non-LFS public file exceeds 10 MiB: $($item.FullName)")
    }
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
