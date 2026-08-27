# Changelog

所有重要变更记录于此。版本号与 [metadata.yaml](metadata.yaml) 保持一致。

## [0.7.0] - 2026-08-27

### 新增

- **Kimi（Moonshot）视频理解链路**：视频经 `/v1/files`（`purpose=video`）上传后以 `ms://<file-id>` 引用（官方唯一视频接入方式）。
  - 仅 Moonshot 官方接口（`api.moonshot.cn` / `api.moonshot.ai`）且模型名含 `kimi` 时触发；中转站的 kimi 模型走原内嵌路径，不受影响。
  - 单文件上限 100MB；仅 `kimi-k3`、`kimi-k2.6`、`kimi-k2.7-code`、`kimi-k2.7-code-highspeed` 支持视频理解。
- **通义千问（阿里云百炼）大视频链路**：小视频 base64 内嵌（按官方「编码后 < 10MB」限制留余量，约 7.4MB 原始视频）；大视频自动走百炼免费临时 URL 上传（`oss://` 前缀，单文件 ≤ 1GB，48 小时有效，文件与调用模型绑定），调用时自动携带官方要求的 `X-DashScope-OssResourceResolve: enable` 请求头。
- **配置面板模板**：`_conf_schema.json` 新增 `qwen`（通义千问·阿里云百炼）与 `kimi`（Kimi·Moonshot）接入方模板；qwen 模板支持抽帧频率 `fps`（[0.1, 10]，默认 2.0）。
- 新增独立上传模块 [`kimi_uploader.py`](kimi_uploader.py) / [`qwen_uploader.py`](qwen_uploader.py)：文件以流式上传，避免大文件整体读入内存。

### 修复

- 修正 qwen base64 内嵌上限：7.5MB 原始视频编码后恰好等于官方 10MB 上限（严格应小于），调整为 7.4MB 并留出 data URL 前缀余量。
- Kimi 上传链路增加官方域名门卫（`_is_moonshot`），避免名称含 `kimi` 的中转站模型被误路由到 `/v1/files` 上传（此前会破坏其原有内嵌路径）。
- 上传接口对非 JSON 对象响应给出明确错误提示，不再抛出 `AttributeError` 绕过统一错误处理。

### 测试

- 新增 `tests/test_kimi_uploader.py`、`tests/test_qwen_uploader.py`，并扩充 `tests/test_gemini_client.py`（Kimi/qwen 分析分支、域名门卫、fps、base64 边界与异常路径），共 196 个用例。
