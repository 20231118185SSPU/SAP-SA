// scripts/generate-types.js
// Generates TypeScript types from Rust protocol definitions.
//
// Usage: node scripts/generate-types.js
// Prerequisites: Rust toolchain installed
//
// This script:
// 1. Builds the export_types binary
// 2. Runs it to generate .ts files in sa/bindings/
// 3. Copies them to WebUI/src/lib/sa-protocol/generated/
// 4. Creates an index.ts that re-exports everything

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const SA_ROOT = path.resolve(__dirname, '..');
const BINDINGS_DIR = path.join(SA_ROOT, 'bindings');
const WEBUI_DIR = path.join(SA_ROOT, '..', 'WebUI', 'src', 'lib', 'sa-protocol', 'generated');

function run(cmd, opts = {}) {
  console.log(`  $ ${cmd}`);
  execSync(cmd, { cwd: SA_ROOT, stdio: 'inherit', ...opts });
}

console.log('🔧 Building export_types binary...');
run('cargo build -p sa --bin export_types --quiet');

console.log('📦 Running type export...');
run('cargo run -p sa --bin export_types --quiet 2>/dev/null');

console.log('📁 Copying to WebUI...');
fs.mkdirSync(WEBUI_DIR, { recursive: true });

const files = fs.readdirSync(BINDINGS_DIR).filter(f => f.endsWith('.ts'));
for (const file of files) {
  fs.copyFileSync(path.join(BINDINGS_DIR, file), path.join(WEBUI_DIR, file));
}

// Generate index.ts
const exports = files
  .map(f => `export * from './${f.replace('.ts', '')}';`)
  .join('\n');
fs.writeFileSync(path.join(WEBUI_DIR, 'index.ts'), `// Auto-generated — do not edit\n${exports}\n`);

console.log(`✅ Generated ${files.length} type files in sa-protocol/generated/`);
