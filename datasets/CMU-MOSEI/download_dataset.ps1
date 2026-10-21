<#
.SYNOPSIS
    CMU-MOSEI Dataset Verification, SDK Helper & Safe Extraction Utility for Windows PowerShell
.DESCRIPTION
    Provides automated environment validation, setup instructions for CMU-MultimodalSDK,
    and a safe archive extraction utility for CMU-MOSEI (Carnegie Mellon University Multimodal
    Opinion Sentiment and Emotion Intensity) dataset archives with path traversal prevention.

    CMU-MOSEI data is distributed by the CMU MultiComp Lab through CMU-MultimodalSDK
    (Computational Sequence Datasets / .csd) and academic research download repositories.

.PARAMETER TargetDir
    The base directory for CMU-MOSEI data. Defaults to the local directory (datasets/CMU-MOSEI).
.PARAMETER ExtractArchives
    Switch to safely extract any .tar.gz, .tar, or .zip archives placed in raw/ or features/.
.PARAMETER ForceOverwrite
    Switch to allow overwriting previously unpacked split media or feature directories.
.PARAMETER CheckEnvironment
    Switch to check local Python environment, mmdatasdk, and required scientific packages.
.PARAMETER ShowSDKUsage
    Display the exact Python commands to download CMU-MOSEI features via CMU-MultimodalSDK.

.EXAMPLE
    .\download_dataset.ps1
.EXAMPLE
    .\download_dataset.ps1 -CheckEnvironment
.EXAMPLE
    .\download_dataset.ps1 -ExtractArchives
#>

[CmdletBinding()]
param (
    [string]$TargetDir = "$PSScriptRoot",
    [switch]$ExtractArchives,
    [switch]$ForceOverwrite,
    [switch]$CheckEnvironment,
    [switch]$ShowSDKUsage
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "CMU-MOSEI Dataset Setup, Verification & Extraction Utility" -ForegroundColor Cyan
Write-Host "Multimodal Emotion Recognition Research Project" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Directory Initialization
$TargetDir = [System.IO.Path]::GetFullPath($TargetDir)
$AnnotationsDir = Join-Path $TargetDir "annotations"
$RawDir = Join-Path $TargetDir "raw"
$FeaturesDir = Join-Path $TargetDir "features"
$ManifestsDir = Join-Path $TargetDir "manifests"
$ReportsDir = Join-Path $TargetDir "reports"

foreach ($dir in @($AnnotationsDir, $RawDir, $FeaturesDir, $ManifestsDir, $ReportsDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[+] Initialized directory: $dir" -ForegroundColor Green
    }
}

# 2. Check Environment & Prerequisites
if ($CheckEnvironment -or $PSCmdlet.ParameterSetName -eq "") {
    Write-Host "`n[*] Step 1: Checking Python environment and multimodal tools..." -ForegroundColor Yellow
    $pyCmd = Get-Command "py" -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        $pyCmd = Get-Command "python" -ErrorAction SilentlyContinue
    }

    if ($pyCmd) {
        $pyVersion = & $pyCmd.Source --version 2>&1
        Write-Host "  [OK] Python detected: $pyVersion ($($pyCmd.Source))" -ForegroundColor Green

        # Check for mmdatasdk or h5py
        $sdkCheck = & $pyCmd.Source -c "
try:
    import mmdatasdk
    print('mmdatasdk: available')
except ImportError:
    print('mmdatasdk: not installed (optional, used for direct SDK download)')
try:
    import h5py
    print('h5py: available')
except ImportError:
    print('h5py: not installed')
" 2>&1
        foreach ($line in $sdkCheck) {
            Write-Host "    $line" -ForegroundColor Gray
        }
    } else {
        Write-Warning "Python launcher ('py' or 'python') was not detected in PATH."
    }
}

# 3. Safe Extraction Helper
function Expand-SafeArchive {
    param (
        [string]$ArchivePath,
        [string]$DestinationDir,
        [bool]$AllowOverwrite
    )

    if (-not (Test-Path $ArchivePath)) {
        return @{ Success = $false; Reason = "Archive file not found." }
    }

    $archiveItem = Get-Item $ArchivePath
    if ($archiveItem.Length -lt 1048576) {
        return @{ Success = $false; Reason = "Archive size is abnormally small ($($archiveItem.Length) bytes); extraction aborted." }
    }

    # Path traversal protection
    $canonicalDest = [System.IO.Path]::GetFullPath($DestinationDir)
    $canonicalRoot = [System.IO.Path]::GetFullPath($TargetDir)
    if (-not $canonicalDest.StartsWith($canonicalRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return @{ Success = $false; Reason = "Security violation: destination '$canonicalDest' escapes target root '$canonicalRoot'." }
    }

    Write-Host "  [>] Safely unpacking $($archiveItem.Name) into $DestinationDir..." -ForegroundColor Cyan
    try {
        if ($archiveItem.Extension -match "gz|tar") {
            $tarOutput = tar -xf "$ArchivePath" -C "$DestinationDir" 2>&1
            if ($LASTEXITCODE -ne 0) {
                return @{ Success = $false; Reason = "tar exited with code $LASTEXITCODE : $tarOutput" }
            }
        } elseif ($archiveItem.Extension -eq ".zip") {
            Expand-Archive -Path $ArchivePath -DestinationPath $DestinationDir -Force:$AllowOverwrite
        }
        return @{ Success = $true; Skipped = $false }
    } catch {
        return @{ Success = $false; Reason = $_.Exception.Message }
    }
}

# 4. Check Local Data / Archives
Write-Host "`n[*] Step 2: Checking local CMU-MOSEI dataset files and archives..." -ForegroundColor Yellow

$annoFile = Join-Path $AnnotationsDir "mosei_annotations.csv"
if (Test-Path $annoFile) {
    $annoSize = (Get-Item $annoFile).Length
    Write-Host "  [OK] Found local annotation file: mosei_annotations.csv ($annoSize bytes)" -ForegroundColor Green
} else {
    Write-Host "  [-] No local mosei_annotations.csv detected in $AnnotationsDir" -ForegroundColor Gray
}

$archives = Get-ChildItem -Path @($RawDir, $FeaturesDir) -Include "*.tar.gz", "*.tar", "*.zip" -Recurse -ErrorAction SilentlyContinue
if ($archives) {
    Write-Host "  [*] Detected $(($archives | Measure-Object).Count) archive(s) locally:" -ForegroundColor Yellow
    foreach ($arc in $archives) {
        $szMB = [math]::Round($arc.Length / 1MB, 2)
        Write-Host "      - $($arc.Name) ($szMB MB in $($arc.DirectoryName))" -ForegroundColor Gray
        if ($ExtractArchives) {
            $res = Expand-SafeArchive -ArchivePath $arc.FullName -DestinationDir $arc.DirectoryName -AllowOverwrite $ForceOverwrite
            if ($res.Success) {
                Write-Host "        [+] Successfully unpacked $($arc.Name)" -ForegroundColor Green
            } else {
                Write-Warning "Failed to unpack $($arc.Name): $($res.Reason)"
            }
        }
    }
} else {
    Write-Host "  [-] No compressed archives (.tar.gz, .zip) found in raw/ or features/." -ForegroundColor Gray
}

# 5. SDK Download Instructions
Write-Host "`n[*] Step 3: CMU-MultimodalSDK Data Acquisition Reference" -ForegroundColor Yellow
Write-Host "    Official Dataset Page : https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/" -ForegroundColor Gray
Write-Host "    Official SDK Repo     : https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK" -ForegroundColor Gray
Write-Host ""
Write-Host "To download CMU-MOSEI features via the official CMU-MultimodalSDK in Python:" -ForegroundColor Cyan
Write-Host @"
--------------------------------------------------------------------------------
# Install official SDK:
pip install mmdatasdk

# Download labels and aligned modalities:
python -c "
import mmdatasdk
# Download sentiment and emotion labels
mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels, 'datasets/CMU-MOSEI/features/')
# Download acoustic features (COVAREP)
mosei.add_computational_sequences(mmdatasdk.cmu_mosei.covarep, 'datasets/CMU-MOSEI/features/')
# Download visual features (Facet 4.2)
mosei.add_computational_sequences(mmdatasdk.cmu_mosei.facet, 'datasets/CMU-MOSEI/features/')
# Download text features (Glove embeddings)
mosei.add_computational_sequences(mmdatasdk.cmu_mosei.glove_vectors, 'datasets/CMU-MOSEI/features/')
print('CMU-MOSEI download complete.')
"
--------------------------------------------------------------------------------
"@ -ForegroundColor DarkGray

Write-Host "`n[+] Verification and setup checks complete.`n" -ForegroundColor Green
