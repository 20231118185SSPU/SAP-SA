# Sa-core test verifier
# Refresh PATH and run full test suite for sa-core, then summarize results.

# Refresh PATH from system and user scopes
$env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
Write-Host "PATH refreshed"

"""Run full test suite for sa-core and capture output"""
$testOutput = cargo test -p sa-core 2>&1

"""Show last 15 lines of test output"""
$testLast15 = $testOutput -split "`r?`n" | Select-Object -Last 15
Write-Host "`n--- TEST OUTPUT (last 15 lines) ---" -ForegroundColor Green
$testLast15 | ForEach-Object { Write-Host $_ }
Write-Host "--- END TEST OUTPUT ---`n" -ForegroundColor Green

"""Parse summary from the full test output"""
$status = ''
if ($testOutput -match 'test result:\s*(\w+)\.') { $status = $matches[1] }
$passed = 0; $failed = 0; $ignored = 0
if ($testOutput -match '(\d+)\s+passed') { $passed = [int]$matches[1] }
if ($testOutput -match '(\d+)\s+failed') { $failed = [int]$matches[1] }
if ($testOutput -match '(\d+)\s+ignored') { $ignored = [int]$matches[1] }
$total = $passed + $failed + $ignored

Write-Host "Test Summary: Total=$total Passed=$passed Failed=$failed Ignored=$ignored Status=$status"
if ($status -eq 'ok') { Write-Host "Test result: ok" } else { Write-Host "Test result: $status" }
