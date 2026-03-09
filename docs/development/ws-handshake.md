# SA WebSocket 双向身份握手协议

## 目的

SA 的 WebSocket 连接在进入正常消息流之前，必须先完成一次“前端先证明、后端再证明”的双向握手。

这个握手的目标不是做互联网级别的强认证，而是回答一个更实际的问题：

- 当前连接到的端口，是否大概率就是本机上预期的 SA 后端。
- 当前连接发起方，是否大概率就是预期的 SA 前端。
- 即使端口被其他程序占用，前后端也能通过协议细节快速识别出“对不上”。

## 适用边界

这个机制适合本机前后端联调和误连检测，不能视为对抗恶意本地进程的强安全认证。

原因：

- 机器指纹来自本机可获得的信息（主机名、MAC 地址）。
- 协议材料和算法都在本地程序中。
- 因此前提是“用于探测是不是正确的 SA 端口/进程”，不是“抵抗拥有本机权限的攻击者”。

## 握手顺序

握手顺序是强制固定的：

1. 前端建立 WebSocket 连接。
2. 前端必须在 5 秒内发送第一条文本帧，并且该帧必须是 `client_hello`。
3. 后端验证 `client_hello`。
4. 只有验证通过后，后端才会返回 `server_hello`。
5. 前端验证 `server_hello`。
6. 只有双方都验证通过后，才进入正常的 `submit`、`event`、`ask`、`show` 等消息流。

如果前端首包超时、不是文本 JSON、不是 `client_hello`、或者证明不匹配，后端会返回 `hello_reject`。

## 协议常量

当前实现中的固定常量如下：

- `protocol`: `sa-ws/v1`
- `hash_algo`: `sha256`
- `time_step_secs`: `5`
- `allowed_skew_buckets`: `1`
- `client_name`: `sa-cli`
- `server_name`: `sa`
- 握手超时: `5` 秒

## 时间桶规则

握手不是按秒直接签名，而是按 5 秒一个时间桶进行计算。

公式：

```text
time_bucket = floor(unix_utc_seconds / 5)
```

校验规则：

- 接收方计算当前 `now_bucket`
- 只接受 `received_bucket` 落在 `now_bucket - 1` 到 `now_bucket + 1` 的范围内

也就是允许前后各 1 个桶的误差，总共容忍最多约 15 秒量级的时钟偏差窗口。

## 机器指纹

前后端都使用同一套本机指纹派生方式。

原始材料：

- 主机名（转小写）
- 一个 MAC 地址；若读取不到则使用 `no-mac`

拼接格式：

```text
sa-machine-fingerprint/v1|host=<hostname>|mac=<mac>
```

随后做 SHA-256，得到：

- `machine_fingerprint`: 完整十六进制哈希，只在本地参与证明计算，不直接上传
- `machine_hint`: `machine_fingerprint` 的前 12 个十六进制字符，用于协议中做可观测调试

## client_hello

前端首包格式：

```json
{
  "type": "client_hello",
  "hello": {
    "protocol": "sa-ws/v1",
    "hash_algo": "sha256",
    "time_step_secs": 5,
    "allowed_skew_buckets": 1,
    "client_name": "sa-cli",
    "client_version": "0.1.0",
    "time_bucket": 352000000,
    "machine_hint": "0123abcd4567",
    "client_nonce": "8a4c0d4a-2e42-4ef3-aec9-c8c2d2b2e8d1",
    "proof": "<sha256-hex>"
  }
}
```

字段说明：

- `client_nonce` 由前端为本次连接随机生成
- `proof` 是前端证明，绑定了时间桶、程序版本、机器指纹和本次 nonce

前端证明公式：

```text
client_proof = sha256_hex(
  "sa-cli-proof/v1"
  + "|sa-ws/v1"
  + "|sa-cli"
  + "|<client_version>"
  + "|<time_bucket>"
  + "|<machine_fingerprint>"
  + "|<client_nonce>"
)
```

后端会校验：

- 协议号、哈希算法、时间桶步长、允许偏差是否全部匹配
- `client_name` 是否为 `sa-cli`
- `machine_hint` 是否与本机一致
- `time_bucket` 是否落在允许窗口内
- `proof` 是否能由本机指纹和消息内容重算得到

## server_hello

只有在 `client_hello` 校验通过后，后端才会返回：

```json
{
  "type": "server_hello",
  "hello": {
    "protocol": "sa-ws/v1",
    "hash_algo": "sha256",
    "time_step_secs": 5,
    "allowed_skew_buckets": 1,
    "server_name": "sa",
    "server_version": "0.1.0",
    "time_bucket": 352000000,
    "machine_hint": "0123abcd4567",
    "client_nonce": "8a4c0d4a-2e42-4ef3-aec9-c8c2d2b2e8d1",
    "server_nonce": "6f9f8f52-6db9-4af0-82ea-f1a9e2a1d4c4",
    "proof": "<sha256-hex>"
  }
}
```

后端证明公式：

```text
server_proof = sha256_hex(
  "sa-server-proof/v1"
  + "|sa-ws/v1"
  + "|sa"
  + "|<server_version>"
  + "|<time_bucket>"
  + "|<machine_fingerprint>"
  + "|<client_nonce>"
  + "|<server_nonce>"
)
```

前端应当校验：

- `server_name` 是否为 `sa`
- `machine_hint` 是否与本机一致
- `client_nonce` 是否与自己发出的首包一致
- `time_bucket` 是否在允许窗口内
- `proof` 是否能被本机重算

## hello_reject

若握手失败，后端会发送：

```json
{
  "type": "hello_reject",
  "reject": {
    "protocol": "sa-ws/v1",
    "server_name": "sa",
    "server_version": "0.1.0",
    "reason": "Client hello verification failed: ..."
  }
}
```

典型失败原因：

- 首包超过 5 秒未发送
- 首包不是 UTF-8 文本 JSON
- 首包不是 `client_hello`
- 时间桶超出允许偏差范围
- `machine_hint` 与本机不一致
- `client_name`、`server_name`、`protocol`、`hash_algo` 不匹配
- 重算后的 proof 不一致

## 进入正常消息流后的限制

握手完成后，连接才会开始发送：

- `pending_questions`
- `recent_shows`
- `accepted`
- `history`
- `event`
- `question`
- `question_resolved`
- `show`
- `error`

如果客户端在握手完成后再次发送 `client_hello`，后端会返回普通 `error`，说明该消息类型只允许出现在连接首包。

## 设计取舍

这个协议刻意选择了以下取舍：

- 不让后端先说话，避免“任意端口先回个像样 JSON”就被前端误判。
- 先验客户端，再回服务端证明，保证服务端身份只对通过初验的前端暴露。
- 使用可调试的 `machine_hint`，方便排查连接到了哪一台机器、哪一套环境。
- 使用 5 秒时间桶和前后 1 桶容错，降低轻微时钟偏差带来的误判。
- 使用方向不同的 proof label，避免前后端证明被直接互相复用。

## 实现位置

后端实现位于：

- `crates/sa-core/src/ws_protocol.rs`
- `crates/sa-core/src/ws_identity.rs`
- `crates/sa/src/main.rs`