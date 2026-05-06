$MachinePath = [Environment]::GetEnvironmentVariable('Path','Machine')
$UserPath = [Environment]::GetEnvironmentVariable('Path','User')
$env:Path = if ([string]::IsNullOrEmpty($MachinePath)) { $UserPath } else { $MachinePath + ';' + $UserPath }
$Output = cargo check -p sa-core 2>&1
$Output
$Output | Select-String -Pattern '^error' -Context 15,10
