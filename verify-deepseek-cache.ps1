# DeepSeek V4 缓存优化 - 验证脚本
# 使用方法：在 PowerShell 中运行此脚本

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "DeepSeek V4 缓存优化 - 验证脚本" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 切换到 sa 目录
Set-Location "E:\SA\SAP-6.2\sa"

Write-Host "[1/3] 检查 Rust 工具链..." -ForegroundColor Yellow
$rustVersion = rustc --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 错误：未找到 Rust 工具链" -ForegroundColor Red
    Write-Host "请安装 Rust：https://rustup.rs/" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Rust 版本：$rustVersion" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] 执行编译检查 (cargo check)..." -ForegroundColor Yellow
Write-Host "--------------------------------------" -ForegroundColor Gray
cargo check -p sa-core 2>&1 | Tee-Object -Variable checkOutput
Write-Host "--------------------------------------" -ForegroundColor Gray

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 编译检查失败" -ForegroundColor Red
    Write-Host "请检查上述错误信息" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 编译检查通过" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] 执行测试 (cargo test)..." -ForegroundColor Yellow
Write-Host "--------------------------------------" -ForegroundColor Gray
cargo test -p sa-core 2>&1 | Tee-Object -Variable testOutput
Write-Host "--------------------------------------" -ForegroundColor Gray

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 测试失败" -ForegroundColor Red
    Write-Host "请检查上述错误信息" -ForegroundColor Red
    exit 1
}
Write-Host "✅ 所有测试通过" -ForegroundColor Green
Write-Host ""

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✅ 验证完成！" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "1. 复制配置文件：cp sa.deepseek.toml sa.toml" -ForegroundColor White
Write-Host "2. 启动服务：cargo run -p sa --release" -ForegroundColor White
Write-Host ""
