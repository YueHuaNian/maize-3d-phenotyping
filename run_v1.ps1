[CmdletBinding(PositionalBinding=$false)]
param(
    [Parameter(Mandatory=$true)][string]$InputRoot,
    [string]$PythonPath,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ModelArgs
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    $PythonPath = $env:MAIZE_PYTHON
}
if (-not $PythonPath) {
    # Host configuration recorded 2026-11-18; recheck on another machine.
    $PythonPath = 'D:/YHN/03-工作与事务/本机工具/Nature-Skills/.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Set -PythonPath or MAIZE_PYTHON to a Python with requirements.txt installed.' }
$env:PYTHONUTF8 = '1'
Push-Location $PSScriptRoot
try {
    & $PythonPath (Join-Path $PSScriptRoot 'scripts/run_leaf_modeling.py') --input-root $InputRoot @ModelArgs
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode
