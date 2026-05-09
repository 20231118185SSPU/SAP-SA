<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-09 | Updated: 2026-05-09 -->

# sa/workflows/

## Purpose

YAML 格式的工作流定义文件。定义 Agent 的多步骤执行流程。

## Key Files

| File | Description |
|------|-------------|
| `idea-to-pr.yaml` | 从创意到 PR 的完整流程 |
| `fix-issue.yaml` | 修复 issue 的工作流 |
| `code-review.yaml` | 代码审查工作流 |
| `architect.yaml` | 架构设计工作流 |
| `plan-to-pr.yaml` | 从计划到 PR |
| `refactor-safely.yaml` | 安全重构工作流 |
| `piv-loop.yaml` | PIV 循环工作流 |
| `validate.yaml` | 验证工作流 |

## For AI Agents

### Working In This Directory
- 工作流为 YAML 格式，定义步骤序列
- 被 sa-core 的 workflow_engine.rs 加载执行

<!-- MANUAL: -->
