<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-09 | Updated: 2026-05-09 -->

# sa/bindings/

## Purpose

WebSocket 协议的 TypeScript 类型绑定。由 `sa/scripts/generate-types.js` 从 Rust 类型定义自动生成，供 WebUI 前端使用。

## Key Files

| File | Description |
|------|-------------|
| `ServerMessage.ts` | 服务端消息类型（最大的类型文件） |
| `ClientMessage.ts` | 客户端消息类型 |
| `ClientHello.ts` | 客户端握手消息 |
| `ServerHello.ts` | 服务端握手响应 |
| `InitRequired.ts` | 初始化必需参数 |
| `InitMethodOption.ts` | 初始化方法选项 |
| `AgentIdentity.ts` | Agent 身份信息 |
| `AuthStyle.ts` | 认证风格 |
| `Event.ts` | 事件类型 |
| `MemoryFact.ts` | 记忆事实类型 |
| `UserQuestion.ts` | 用户提问类型 |
| `UserQuestionAnswer.ts` | 用户回答类型 |
| `WireApi.ts` | 线路 API 类型 |
| `HelloReject.ts` | 握手拒绝 |
| `InitCompleted.ts` | 初始化完成 |
| `InitFailed.ts` | 初始化失败 |
| `InitMethod.ts` | 初始化方法 |
| `QuestionMode.ts` | 提问模式 |
| `QuestionOption.ts` | 提问选项 |

## For AI Agents

### Working In This Directory
- 这些文件由脚本自动生成，不要手动修改
- 修改协议类型请改 Rust 端，然后运行 `sa/scripts/generate-types.js` 重新生成
- WebUI 端对应的解析代码在 `WebUI/src/lib/sa-protocol/`

<!-- MANUAL: -->
