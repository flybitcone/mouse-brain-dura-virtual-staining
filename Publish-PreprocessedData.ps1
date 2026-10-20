$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')

function Assert-WorkspacePath([string]$Path) {
    $resolved = [System.IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($workspace + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside this workspace: $resolved"
    }
    # Refuse junctions/symlinks anywhere along an existing path.
    $cursor = $resolved
    while ($cursor.Length -gt $workspace.Length) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse-point path: $cursor"
            }
        }
        $cursor = [System.IO.Path]::GetDirectoryName($cursor)
    }
    return $resolved
}

$stage = Assert-WorkspacePath (Join-Path $workspace 'preprocessing_build')
$previous = Assert-WorkspacePath (Join-Path $workspace 'preprocessing_previous')
$marker = Join-Path $stage 'SUCCESS.json'
if (-not (Test-Path -LiteralPath $marker)) { throw 'Validated build marker is missing.' }
$status = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
if ($status.status -ne 'validated') { throw 'Build is not validated.' }
$inventory = Join-Path $workspace 'preprocessing_reports/raw_inventory.csv'
if ((Get-FileHash -LiteralPath $inventory -Algorithm SHA256).Hash.ToLower() -ne $status.source_inventory_sha256) {
    throw 'Source inventory changed since validation.'
}
if (Test-Path -LiteralPath $previous) { throw 'Previous replacement needs review before continuing.' }
$moves = @(
    @{ Source = (Join-Path $stage 'dataset_split'); Destination = (Join-Path $workspace 'dataset_split'); Name = 'dataset_split' },
    @{ Source = (Join-Path $stage 'dataset_patches'); Destination = (Join-Path $workspace 'deep learning/dataset_patches'); Name = 'dataset_patches' }
)
foreach ($move in $moves) {
    $move.Source = Assert-WorkspacePath $move.Source
    $move.Destination = Assert-WorkspacePath $move.Destination
    if (-not (Test-Path -LiteralPath $move.Source)) { throw "Missing staged directory: $($move.Source)" }
}
New-Item -ItemType Directory -Path $previous | Out-Null
$installed = @()
$backedUp = @()
try {
    foreach ($move in $moves) {
        if (Test-Path -LiteralPath $move.Destination) {
            Move-Item -LiteralPath $move.Destination -Destination (Join-Path $previous $move.Name)
            $backedUp += $move
        }
        Move-Item -LiteralPath $move.Source -Destination $move.Destination
        $installed += $move
    }
} catch {
    foreach ($move in $installed) {
        Move-Item -LiteralPath $move.Destination -Destination $move.Source
    }
    foreach ($move in $backedUp) {
        Move-Item -LiteralPath (Join-Path $previous $move.Name) -Destination $move.Destination
    }
    throw
}
Copy-Item -LiteralPath $marker -Destination (Join-Path $workspace 'preprocessing_reports/SUCCESS.json') -Force
# Only derived directories backed up by this invocation are removed.
$previous = Assert-WorkspacePath $previous
Get-ChildItem -LiteralPath $previous -Recurse -Force | ForEach-Object {
    if ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Reparse point in backup: $($_.FullName)" }
}
Remove-Item -LiteralPath $previous -Recurse -Force
Remove-Item -LiteralPath $marker
# Non-recursive removal refuses unexpected staging contents.
Remove-Item -LiteralPath $stage
Write-Output 'Installed verified dataset_split and deep learning/dataset_patches. raw_data was preserved.'
