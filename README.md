# DeepSeek Vision MCP Server

将 DeepSeek 网页版 (chat.deepseek.com) 的原生多模态识图能力包装为 MCP Server。

**不是 OCR！是真·多模态视觉理解模型。** 支持识别人物、物体、场景、颜色、文字等。

## 前置条件

1. Python 3.11+
2. DeepSeek 账号 (chat.deepseek.com)

## 获取 Token

1. 在 Chrome/Edge 中打开 chat.deepseek.com 并登录
2. F12 → Application → Local Storage → chat.deepseek.com
3. 找到 `userToken` 键，复制值 `JSON.parse(value).value`

## 安装

```bash
# 克隆
git clone https://github.com/lmtttt/deepseek-vision-mcp.git
cd deepseek-vision-mcp

# 安装依赖
pip install curl-cffi==0.8.1b9 httpx mcp Pillow wasmtime numpy anyio

# 或者用 uv
uv sync
```

## 配置

设置环境变量启动：

```bash
export DEEPSEEK_USER_TOKEN="your_token_here"
export DEEPSEEK_SMIDV2="smidV2_cookie_value"  # 可选，提升稳定性

python -m deepseek_vision_mcp
```

### 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_USER_TOKEN` | ✅ | 认证 Token |
| `DEEPSEEK_SMIDV2` | ❌ | smidV2 cookie，提升请求稳定性 |
| `DEEPSEEK_BASE_URL` | ❌ | API 地址 (默认: https://chat.deepseek.com) |
| `DEEPSEEK_POLL_TIMEOUT` | ❌ | 文件上传轮询超时(秒) (默认: 60) |
| `DEEPSEEK_CHAT_TIMEOUT` | ❌ | 聊天超时(秒) (默认: 120) |

## OpenClaw 集成

```bash
openclaw mcp set deepseek-vision '{
  "command": "python3",
  "args": ["-m", "deepseek_vision_mcp"],
  "env": {
    "DEEPSEEK_USER_TOKEN": "your_token_here",
    "PYTHONPATH": "/path/to/deepseek-vision-mcp/src"
  },
  "cwd": "/path/to/deepseek-vision-mcp"
}'
```

## Claude Desktop 集成

```json
{
  "mcpServers": {
    "deepseek-vision": {
      "command": "python",
      "args": ["-m", "deepseek_vision_mcp"],
      "env": {
        "DEEPSEEK_USER_TOKEN": "your_token_here"
      }
    }
  }
}
```

## 使用

启动后，MCP 客户端可调用以下工具：

### `deepseek_vision`

分析图片内容。

**参数：**
- `image` (必填): 本地图片路径
- `prompt` (可选): 提问内容 (默认: "请详细描述这张图片中的内容")
- `thinking` (可选): 启用 DeepThink 推理 (默认: false)

**示例：**
```
deepseek_vision(image="/path/to/photo.jpg")
deepseek_vision(image="/path/to/photo.jpg", prompt="这张图片里有什么人？什么物体？")
```

### `deepseek_vision_status`

检查认证和运行状态。

## 项目结构

```
deepseek-vision-mcp/
├── pyproject.toml
├── README.md
└── src/
    └── deepseek_vision_mcp/
        ├── __init__.py     # 版本号
        ├── __main__.py     # python -m 入口
        ├── server.py       # MCP Server 主逻辑 + Vision 流水线
        ├── config.py       # 配置管理 (从环境变量加载)
        ├── auth.py         # Token 认证管理
        ├── hif_auth.py     # HIF 签名 token 获取
        ├── pow.py          # PoW 挑战求解器 (WASM)
        ├── upload.py       # 文件上传 + 状态轮询
        ├── models.py       # 数据模型
        └── wasm/           # PoW WASM 模块
```

## 技术实现

完整的 Vision 调用流程：

```
1. POST /api/v0/file/upload_file         → 上传图片，获取 file_id (OCR处理)
2. POST /api/v0/file/fork_file_task      → Fork 到 vision 模型 (真正的图像处理)
3. GET  hif-leim.deepseek.com/query       → 获取 leim 签名 token
4. GET  hif-dliq.deepseek.com/query       → 获取 dliq 签名 token
5. POST /api/v0/chat/completion           → Vision 模型分析 (需要 curl_cffi TLS指纹)
```

## License

MIT
