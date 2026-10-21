<#
.SYNOPSIS
    MELD Dataset Download, Verification, and Safe Extraction Script for Windows PowerShell
.DESCRIPTION
    Automates the acquisition and verification of the MELD (Multimodal EmotionLines Dataset)
    for team members. Downloads official annotation CSVs directly from upstream Declare Lab,
    verifies local raw media folders or archives, and safely unpacks official .tar.gz archives
    if present locally, with path traversal prevention and overwrite safeguards.
.PARAMETER TargetDir
    The base directory where MELD annotations and raw data should be placed.
    Defaults to the local dataset directory (datasets/MELD).
.PARAMETER ExtractArchives
    Switch to safely extract any .tar.gz archives found in the raw directory.
.PARAMETER ForceOverwrite
    Switch to allow overwriting already extracted split media directories.
.PARAMETER SkipMedia
    If specified, checks and downloads only annotations and skips media verification.
.EXAMPLE
    .\download_dataset.ps1
.EXAMPLE
    .\download_dataset.ps1 -ExtractArchives
.EXAMPLE
    .\download_dataset.ps1 -TargetDir ".\datasets\MELD" -ExtractArchives
#>

[CmdletBinding()]
param (
    [string]$TargetDir = "$PSScriptRoot",
    [switch]$ExtractArchives,
    [switch]$ForceOverwrite,
    [switch]$SkipMedia
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "MELD Dataset Setup, Verification & Extraction Utility" -ForegroundColor Cyan
Write-Host "Multimodal Emotion Recognition Project" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Define and sanitize base directories
$TargetDir = [System.IO.Path]::GetFullPath($TargetDir)
$AnnotationsDir = Join-Path $TargetDir "annotations"
$RawDir = Join-Path $TargetDir "raw"
$ManifestsDir = Join-Path $TargetDir "manifests"
$ReportsDir = Join-Path $TargetDir "reports"

foreach ($dir in @($AnnotationsDir, $RawDir, $ManifestsDir, $ReportsDir)) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[+] Initialized directory: $dir" -ForegroundColor Green
    }
}

# 1. Official annotation CSV URLs (Declare Lab upstream repository)
$AnnotationBaseUrl = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
$AnnotationFiles = @(
    "train_sent_emo.csv",
    "dev_sent_emo.csv",
    "test_sent_emo.csv"
)

Write-Host "`n[*] Step 1: Checking and fetching official annotation CSV files..." -ForegroundColor Yellow
Write-Host "    Source: $AnnotationBaseUrl" -ForegroundColor Gray
foreach ($file in $AnnotationFiles) {
    $dest = Join-Path $AnnotationsDir $file
    if (Test-Path $dest) {
        $size = (Get-Item $dest).Length
        Write-Host "  [OK] $file already present locally ($size bytes)." -ForegroundColor Gray
    } else {
        $url = "$AnnotationBaseUrl/$file"
        Write-Host "  [↓] Downloading $file from upstream repository..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
            $size = (Get-Item $dest).Length
            Write-Host "  [+] Successfully downloaded $file ($size bytes)." -ForegroundColor Green
        } catch {
            Write-Warning "Failed to download $file from $url : $_"
        }
    }
}

# 2. Check for local raw media folders or archives
Write-Host "`n[*] Step 2: Checking MELD raw media archives and split folders..." -ForegroundColor Yellow

$ArchiveSpecs = @(
    @{
        Split = "train"
        Archive = "train.tar.gz"
        TargetDirName = "train_splits"
        ApproxSize = "~7.8 GB"
    },
    @{
        Split = "dev"
        Archive = "dev.tar.gz"
        TargetDirName = "dev_splits_complete"
        ApproxSize = "~776 MB"
    },
    @{
        Split = "test"
        Archive = "test.tar.gz"
        TargetDirName = "output_repeated_splits_test"
        ApproxSize = "~2.2 GB"
    }
)

# Helper function for safe archive extraction
function Expand-MeldArchive {
    param (
        [string]$ArchivePath,
        [string]$DestinationDir,
        [string]$ExpectedFolder,
        [bool]$AllowOverwrite
    )

    if (-not (Test-Path $ArchivePath)) {
        return @{ Success = $false; Reason = "Archive file not found." }
    }

    $archiveItem = Get-Item $ArchivePath
    if ($archiveItem.Length -lt 1048576) { # Less than 1 MB is likely corrupted/incomplete
        return @{ Success = $false; Reason = "Archive size is abnormally small ($($archiveItem.Length) bytes); extraction aborted." }
    }

    # Path traversal safeguard: verify destination canonical path stays within RawDir
    $canonicalDest = [System.IO.Path]::GetFullPath($DestinationDir)
    $canonicalRaw = [System.IO.Path]::GetFullPath($RawDir)
    if (-not $canonicalDest.StartsWith($canonicalRaw, [System.StringComparison]::OrdinalIgnoreCase)) {
        return @{ Success = $false; Reason = "Security violation: destination '$canonicalDest' escapes raw root '$canonicalRaw'." }
    }

    $finalFolderPath = Join-Path $DestinationDir $ExpectedFolder
    if ((Test-Path $finalFolderPath) -and -not $AllowOverwrite) {
        $existingCount = (Get-ChildItem -Path $finalFolderPath -Filter "*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($existingCount -gt 0) {
            return @{ Success = $true; Skipped = $true; Reason = "Target directory already contains $existingCount MP4 files; overwrite skipped." }
        }
    }

    Write-Host "  [>] Safely extracting $($archiveItem.Name) into $DestinationDir..." -ForegroundColor Cyan
    try {
        # Utilize Windows native tar tool
        $tarOutput = tar -xzf "$ArchivePath" -C "$DestinationDir" 2>&1
        if ($LASTEXITCODE -ne 0) {
            return @{ Success = $false; Reason = "tar exited with code $LASTEXITCODE : $tarOutput" }
        }
        $extractedCount = (Get-ChildItem -Path $finalFolderPath -Filter "*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count
        return @{ Success = $true; Skipped = $false; ExtractedCount = $extractedCount }
    } catch {
        return @{ Success = $false; Reason = $_.Exception.Message }
    }
}

$allSplitsReady = $true

foreach ($spec in $ArchiveSpecs) {
    $split = $spec.Split
    $targetFolderName = $spec.TargetDirName
    $targetPath = Join-Path $RawDir $targetFolderName
    $archivePath = Join-Path $RawDir $spec.Archive

    $mediaFound = $false
    if (Test-Path $targetPath) {
        $count = (Get-ChildItem -Path $targetPath -Filter "*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($count -gt 0) {
            Write-Host "  [OK] Found $split media folder: $targetFolderName ($count MP4 clips present)" -ForegroundColor Green
            $mediaFound = $true
        }
    }

    # If media folder is missing but archive is present locally
    if (-not $mediaFound -and (Test-Path $archivePath)) {
        $arcSize = [math]::Round((Get-Item $archivePath).Length / 1MB, 2)
        Write-Host "  [*] Detected local archive: $($spec.Archive) ($arcSize MB)" -ForegroundColor Yellow
        if ($ExtractArchives) {
            $extractResult = Expand-MeldArchive -ArchivePath $archivePath -DestinationDir $RawDir -ExpectedFolder $targetFolderName -AllowOverwrite $ForceOverwrite
            if ($extractResult.Success) {
                if ($extractResult.Skipped) {
                    Write-Host "      [!] $($extractResult.Reason)" -ForegroundColor Yellow
                } else {
                    Write-Host "      [+] Successfully extracted $($extractResult.ExtractedCount) MP4 clips into $targetFolderName." -ForegroundColor Green
                }
                $mediaFound = $true
            } else {
                Write-Host "      [X] Extraction failed: $($extractResult.Reason)" -ForegroundColor Red
            }
        } else {
            Write-Host "      Run script with -ExtractArchives to unpack this archive." -ForegroundColor Gray
        }
    }

    if (-not $mediaFound) {
        $allSplitsReady = $false
        Write-Host "  [!] Missing $split media. Expected folder: '$targetFolderName' or archive '$($spec.Archive)'." -ForegroundColor DarkYellow
    }
}

if (-not $allSplitsReady -and -not $SkipMedia) {
    Write-Host "`n[!] Notice on MELD Media Download:" -ForegroundColor Yellow
    Write-Host "    Because redistribution permission is not established for public Git hosting,"
    Write-Host "    media archives are not hosted directly in this repository."
    Write-Host "`n    Teammates can download the official raw media archives from:"
    Write-Host "    - Hugging Face: https://huggingface.co/datasets/declare-lab/MELD"
    Write-Host "    - Official Site: https://affective-meld.github.io/"
    Write-Host "    - Archives: train.tar.gz (~7.8 GB), dev.tar.gz (~776 MB), test.tar.gz (~2.2 GB)"
    Write-Host "`n    Place the downloaded archives in '$RawDir' and execute:"
    Write-Host "    .\datasets\MELD\download_dataset.ps1 -ExtractArchives" -ForegroundColor Cyan
}

# 3. Next Steps instructions
Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. To run preprocessing on local raw media in datasets/MELD/raw:" -ForegroundColor White
Write-Host "   python datasets/MELD/prepare_dataset.py --meld-raw-dir `"$RawDir`"" -ForegroundColor Yellow
Write-Host "2. Or to run against custom workspace storage (e.g. EmotionProject):" -ForegroundColor White
Write-Host "   python datasets/MELD/prepare_dataset.py --meld-raw-dir `"C:\path\to\MELD.Raw`" --audio-dir `"C:\path\to\audio`"" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
