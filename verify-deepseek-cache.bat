@echo off
chcp 65001 >nul
echo ======================================
echo DeepSeek V4 缓存优化 - 验证脚本
echo ======================================
echo.

cd /d E:\SA\SAP-6.2\sa

echo [1/3] 检查 Rust 工具链...
rustc --version
if errorlevel 1 (
    echo ❌ 错误：未找到 Rust 工具链
    echo 请安装 Rust：https://rustup.rs/
    pause
    exit /b 1
)
echo ✅ Rust 工具链已安装
echo.

echo [2/3] 执行编译检查 (cargo check)...
echo --------------------------------------
cargo check -p sa-core
echo --------------------------------------
if errorlevel 1 (
    echo ❌ 编译检查失败
    echo 请检查上述错误信息
    pause
    exit /b 1
)
echo ✅ 编译检查通过
echo.

echo [3/3] 执行测试 (cargo test)...
echo --------------------------------------
cargo test -p sa-core
echo --------------------------------------
if errorlevel 1 (
    echo ❌ 测试失败
    echo 请检查上述错误信息
    pause
    exit /b 1
)
echo ✅ 所有测试通过
echo.

echo ======================================
echo ✅ 验证完成！
echo ======================================
echo.
echo 下一步：
echo 1. 复制配置文件：copy sa.deepseek.toml sa.toml
echo 2. 启动服务：cargo run -p sa --release
echo.
pause
