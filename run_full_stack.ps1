param(
    [ValidateSet("setup", "api", "streamlit", "benchmark", "triton", "full")]
    [string]$Target = "full",
    [switch]$SkipInstall,
    [switch]$NoVenv,
    [int]$ApiPort = 8000,
    [int]$MaxFrames = 1520,
    [string]$TritonModelRepo = "Basketball Dribble Analysis/Deployment/TRITON/model_repository",
    [string]$BenchmarkOut = "Basketball Dribble Analysis/benchmark_results.json"
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    return Split-Path -Parent $MyInvocation.PSCommandPath
}

function Get-PythonExe {
    param(
        [string]$Root
    )
    if ($NoVenv) {
        return "python"
    }

    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    Write-Host "Creating virtual environment at .venv ..."
    & python -m venv (Join-Path $Root ".venv")
    return $venvPython
}

function Install-Dependencies {
    param(
        [string]$PythonExe,
        [string]$Root
    )
    if ($SkipInstall) {
        Write-Host "Skipping dependency install."
        return
    }
    Write-Host "Installing dependencies from requirements.txt ..."
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r (Join-Path $Root "requirements.txt")
}

function Run-Api {
    param(
        [string]$PythonExe,
        [string]$Root
    )
    $codeDir = Join-Path $Root "Basketball Dribble Analysis\Code"
    & $PythonExe -m uvicorn api:app --app-dir $codeDir --host 127.0.0.1 --port $ApiPort --reload
}

function Run-Streamlit {
    param(
        [string]$PythonExe,
        [string]$Root
    )
    $appPath = Join-Path $Root "Basketball Dribble Analysis\Code\streamlit_app.py"
    & $PythonExe -m streamlit run $appPath
}

function Run-Benchmark {
    param(
        [string]$PythonExe,
        [string]$Root
    )
    $benchPath = Join-Path $Root "Basketball Dribble Analysis\Code\benchmark_models.py"
    $outPath = Join-Path $Root $BenchmarkOut
    & $PythonExe $benchPath --max-frames $MaxFrames --output $outPath
}

function Run-Triton {
    param(
        [string]$Root
    )
    $repoPath = Join-Path $Root $TritonModelRepo
    if (-not (Get-Command tritonserver -ErrorAction SilentlyContinue)) {
        throw "tritonserver command not found. Install Triton server or run this target on Triton host."
    }
    & tritonserver --model-repository $repoPath
}

$root = Resolve-ProjectRoot
Set-Location $root
$pythonExe = Get-PythonExe -Root $root

switch ($Target) {
    "setup" {
        Install-Dependencies -PythonExe $pythonExe -Root $root
        Write-Host "Setup complete."
    }
    "api" {
        Install-Dependencies -PythonExe $pythonExe -Root $root
        Run-Api -PythonExe $pythonExe -Root $root
    }
    "streamlit" {
        Install-Dependencies -PythonExe $pythonExe -Root $root
        Run-Streamlit -PythonExe $pythonExe -Root $root
    }
    "benchmark" {
        Install-Dependencies -PythonExe $pythonExe -Root $root
        Run-Benchmark -PythonExe $pythonExe -Root $root
    }
    "triton" {
        Run-Triton -Root $root
    }
    "full" {
        Install-Dependencies -PythonExe $pythonExe -Root $root
        Write-Host "Starting API and Streamlit in separate PowerShell windows ..."
        $apiCmd = "& `"$pythonExe`" -m uvicorn api:app --app-dir `"$($root)\Basketball Dribble Analysis\Code`" --host 127.0.0.1 --port $ApiPort --reload"
        $stCmd = "& `"$pythonExe`" -m streamlit run `"$($root)\Basketball Dribble Analysis\Code\streamlit_app.py`""
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd | Out-Null
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $stCmd | Out-Null
        Write-Host "API: http://127.0.0.1:$ApiPort"
        Write-Host "Streamlit: http://127.0.0.1:8501"
    }
}
