# Reproduce the whole study (Windows PowerShell). Needs: JDK 17+, Maven, Docker Desktop (>= 6 GB for its VM), gcc + the Assignment 3
# build (only for the native reference), uv (or Python 3.11+ with numpy pandas matplotlib), Chrome or Edge (report PDF).
# Inputs (~360 MB) are written OUTSIDE the project folder by default so that cloud-sync tools do not touch them while timing.
param(
  [string]$DataDir = "C:\bench_a4\data",
  [string]$Python = "uv run --with numpy --with pandas --with matplotlib python"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = $Python.Split(" ")
function Py { & $py[0] $py[1..($py.Length-1)] @args; if ($LASTEXITCODE -ne 0) { throw "python step failed" } }

.\build.ps1
.\scripts\collect_environment.ps1
Py scripts/generate_inputs.py --outdir $DataDir `
   --check "..\..\assigment-3\Assignment03_AlejandroSantana_Materials\data\inputs\manifest.json" `
   --check "..\..\Assignment01_AlejandroSantana_Materials\data\inputs\manifest.json"
Py tests/validate.py --data-dir $DataDir
if (Test-Path results\raw) { throw "results\raw already contains measurements: move or delete it before a full re-run" }
Py scripts/run_benchmarks.py --data-dir $DataDir        # D1..D7 (Spark) and L1 (local baselines), about 1.5 h
Py scripts/run_reference_a3.py --data-dir $DataDir      # optional: native C/OpenMP of Assignment 3 on this machine
Py scripts/analyze.py
Py report/build_report.py
