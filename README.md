# astrbot_plugin_video_sense

让 Bot 看懂视频——分析群里分享的视频文件的画面、内容与氛围，以角色口吻把结果告诉你。

## 功能

- 分析视频文件的画面内容、场景、人物动作和整体氛围
- 消息中的视频自动记录（支持 `File`/`Video` 组件，如 QQ/Telegram 视频），缓存带接收时间，LLM 可分辨新旧
- **视频感知提示**：收到视频后自动向 LLM 注入提示，引导它主动调用分析工具
- **后台异步分析**：LLM 工具触发立即返回，分析在后台执行（绕过框架 120 秒工具超时）
- **AI 唤醒发送**：分析结果由主 Agent 以角色口吻发送，失败原因也能说得像人话
- **双协议支持**：gemini 模型走 Gemini 协议；qwen-vl、gpt 等模型自动走 OpenAI 兼容协议
- **双传输模式**：小视频内嵌传输；大视频走官方 Files API（免费层 2GB）或 ffmpeg 自动压缩
- 指令触发和 LLM 工具双通道，工具可携带具体追问
- 分析结果缓存，重复分析不消耗 API

## 使用方式

### 指令触发

唤醒 Bot（群聊 @ 机器人或唤醒词，私聊默认直接触发）后发送 `/分析视频`，附带或引用视频：

```
/分析视频
/分析视频 这个视频在干嘛？
```

没有附带文件时，Bot 会列出对话中缓存的视频（带时间），用序号指定：

```
/分析视频 1
/分析视频 1 这个视频适合发到朋友圈吗？
```

> 若唤醒词不含 `/`（如 `!`），请发送 `分析视频`（不带斜杠）。

### LLM 工具自动触发（推荐）

开启「启用 LLM 工具调用」后，Bot 提供三个工具，LLM 按场景自动选择：

| 工具 | 使用场景 |
|------|----------|
| `analyze_current_video` | **本条消息直接携带**的视频（用户刚发的、或引用消息中的） |
| `list_video_files` | 用户提到**之前发过**的视频；列表带接收时间（刚刚/N分钟前/N小时前/昨天），LLM 据此分辨新旧 |
| `analyze_video_by_number` | 按序号分析历史视频（先 `list_video_files` 拿序号） |

分析链路：

```
用户发视频 → 自动缓存（去重 + 记录时间）→ 注入"视频感知"提示
  → LLM 判断：
     本条消息带视频 → analyze_current_video → ⏳ 立即返回
     提到历史视频   → list_video_files → analyze_video_by_number N
     误调 analyze_current_video 但消息无视频 → 返回带时间的列表引导 LLM 选序号
  → 后台分析完成 → 唤醒主 Agent → 以角色口吻发送结果（失败原因同样口吻化）
```

视频分析耗时较长（数十秒），工具调用**立即返回"分析中"**，结果稍后主动送达，无需等待。

## 支持的视频格式

mp4、mov、webm、avi、mpeg、mpg、flv、wmv、3gpp（官方支持的视频格式）

## 配置说明

### 添加 API 接入方

**Gemini 官方接口**

| 字段 | 说明 |
|------|------|
| 供应商名称 | 自定义标识，如 `gemini-official` |
| API Key | 在 [Google AI Studio](https://aistudio.google.com/apikey) 获取 |
| 模型名称 | 推荐 `gemini-2.0-flash` 或 `gemini-2.5-flash` |
| 超时时间 | 建议不低于 180 秒 |

**OpenAI 兼容中转站**

| 字段 | 说明 |
|------|------|
| 供应商名称 | 自定义标识 |
| API Key | 中转站提供的访问密钥 |
| Base URL | 中转站地址，如 `https://xxx.com/v1` |
| 模型名称 | 按中转站实际支持的模型填写 |
| 协议（可选） | 留空自动判断：模型名含 `gemini` 走 Gemini 协议，其他模型走 OpenAI 兼容协议；可强制 `gemini` / `openai` |
| 超时时间 | 建议不低于 180 秒 |

> **协议自动选择**：模型名包含 `gemini` 时用 Gemini 协议（`generateContent` 端点）；其他模型（`qwen-vl-max`、`gpt-4o` 等）自动用 OpenAI 兼容协议（`/v1/chat/completions`，视频通过 `video_url` data URL 内嵌）。中转站需支持对应协议与视频输入。

### 分析设置

**分析模型**

格式为 `供应商名称/模型名称`，供应商名称需与上方配置完全一致（区分大小写）。留空自动使用第一个接入方。例如：`gemini-official/gemini-2.0-flash`

**其他配置项**

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| 分析系统提示词 | 分析指令，指令触发和 LLM 工具触发时使用 | 内置提示词 |
| 指令触发使用不同提示词 | 开启后可单独配置 `/分析视频` 的提示词 | 关闭 |
| 指令触发提示词 | 仅 `/分析视频` 指令触发时使用 | 同分析系统提示词 |
| 启用 LLM 工具调用 | 开启后 Bot 可主动分析 | 开启 |
| 最大文件大小 | 文件总大小上限，超过直接拒绝 | 100 MB |
| 内嵌传输上限 | ≤ 此大小内嵌传输（base64 约膨胀 33%，请求体上限 20MB），超过走 Files API 或自动压缩 | 15 MB |
| 启用 Files API | 超过内嵌上限时官方接口走 Files API（免费层 2GB）；**中转站自动跳过此开关**，无需手动关闭 | 开启 |
| 每会话缓存视频数上限 | 超出自动丢弃最旧条目 | 50 |
| 支持的视频格式 | 仅列表内格式会被缓存 | mp4/mov/webm/avi/mpeg/mpg/flv/wmv/3gpp |
| 调试模式 | 输出详细日志，方便排查 | 关闭 |

### 自动压缩（可选）

中转站接入且视频超过「内嵌传输上限」时，开启「自动压缩超限视频」后，插件用 ffmpeg 压缩到上限以内再分析（目标码率阶梯降级：720p → 480p，长视频先截取前 N 秒）。

**插件不会自动安装 ffmpeg**，二选一：

1. **WebUI「平台日志」页面 → 「安装 pip 库」按钮 → 输入 `imageio-ffmpeg` 并安装**（⚠ 依赖较大，约 80MB，内含 ffmpeg 二进制）
2. 系统安装 ffmpeg（如 `winget install ffmpeg` / `apt install ffmpeg`）

插件启动时检测 ffmpeg 并在平台日志输出状态与安装建议；未检测到时压缩功能自动禁用。安装后重启 AstrBot 生效。

| 压缩配置项 | 说明 | 默认值 |
|------------|------|--------|
| 自动压缩超限视频 | 超限且无法 Files API 时自动压缩 | 关闭 |
| 压缩前截取时长上限 | 超过先无损截取前 N 秒 | 120 秒 |
| 压缩分辨率 | 视频高度压缩到该值（宽度等比） | 720 |
| 压缩质量（CRF） | H.264 质量参数，越大体积越小 | 28 |

## 常见问题

**分析失败提示"网关返回 HTML 错误页（502）"？**
网关层限制（nginx 默认请求体上限 1MB）或上游超时。缓解：把「内嵌传输上限」调低（如 5MB）并开启「自动压缩」，减小请求体；或改用 Gemini 官方接口（无此限制）。

**LLM 不主动分析视频？**
确认插件已更新到 v0.5.1+（视频感知提示注入生效），查看平台日志中 `hook(OnLLMRequestEvent) -> astrbot_plugin_video_sense - _on_llm_request` 是否出现；工具描述已引导 LLM 区分"本条消息视频"与"历史视频"。

**视频分析很慢？**
视频理解本身耗时（上传 + 模型处理）。指令 `/分析视频` 同步等待，LLM 工具异步后台执行——两者都不受 120 秒工具超时影响。

## 开发

```bash
pip install pytest pytest-asyncio aiohttp ruff
ruff check .          # lint（pyproject.toml 配置，规则与 AstrBot 官方一致）
ruff format --check . # 格式检查
python -m pytest tests -v
```

> 压缩功能真实联调需要 ffmpeg：`pip install imageio-ffmpeg`（仅测试环境安装，插件不声明该依赖）。

---

![Moe Counter](https://count.getloli.com/get/@lingyun14beta-video_sense?theme=miku)
