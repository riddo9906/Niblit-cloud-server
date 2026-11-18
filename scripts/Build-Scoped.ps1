param([string]$BuildRoot='D:\NiblitAIOS',[string]$BuildId='',[string]$CoreSourceRoot=$env:NIBLIT_CORE_SOURCE_ROOT,[switch]$Execute)
$ErrorActionPreference='Stop'
if (!$CoreSourceRoot) { $CoreSourceRoot=Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'Niblit' }
$entry=Join-Path $CoreSourceRoot 'scripts\Invoke-ScopedBuild.ps1'
if (!(Test-Path -LiteralPath $entry -PathType Leaf)) { throw 'Set NIBLIT_CORE_SOURCE_ROOT to the approved Core source checkout' }
& $entry -BuildRoot $BuildRoot -BuildId $BuildId -Component cloud -Execute:$Execute
exit $LASTEXITCODE
