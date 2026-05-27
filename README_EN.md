# DeepSeek Vision MCP Server

Wrap DeepSeek web chat's (chat.deepseek.com) native multimodal vision capability as an MCP Server.

**Not OCR — a real multimodal vision understanding model.** Supports recognizing people, objects, scenes, colors, text, and more.

## Prerequisites

1. Python 3.11+
2. A DeepSeek account (chat.deepseek.com)

## Get Your Token

1. Open chat.deepseek.com in Chrome/Edge and log in
2. F12 → Application → Local Storage → chat.deepseek.com
3. Find the `userToken` key, copy `JSON.parse(value).value`

## Installation

```bash
# Clone
git clone https://github.com/lmtttt/deepseek-vision-mcp.git
cd deepseek-vision-mcp

# Install dependencies
pip install curl-cffi==0.8.1b9 httpx mcp Pillow wasmtime numpy anyio

# Or use uv
uv sync
```

## Configuration

Set environment variables and start:

```bash
export DEEPSEEK_USER_TOKEN="your_token_here"
export DEEPSEEK_SMIDV2="smidV2_cookie_value"  # optional, improves reliability

python -m deepseek_vision_mcp
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_USER_TOKEN` | Yes | Auth token |
| `DEEPSEEK_SMIDV2` | No | smidV2 cookie for improved reliability |
| `DEEPSEEK_BASE_URL` | No | API base URL (default: https://chat.deepseek.com) |
| `DEEPSEEK_POLL_TIMEOUT` | No | File upload poll timeout in seconds (default: 60) |
| `DEEPSEEK_CHAT_TIMEOUT` | No | Chat timeout in seconds (default: 120) |

## OpenClaw Integration

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

## Claude Desktop Integration

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

## Usage

Once started, the MCP client can invoke the following tools:

### `deepseek_vision`

Analyze image content.

**Parameters:**
- `image` (required): Local file path to the image (jpg, png, etc.)
- `prompt` (optional): Question or prompt about the image (default: "Please describe the content of this image in detail")
- `thinking` (optional): Enable DeepThink reasoning (default: false)
- `continue_conversation` (optional): Continue the previous conversation instead of creating a new one (default: false)
- `session_id` (optional): Explicit chat_session_id to reuse

**Examples:**
```
deepseek_vision(image="/path/to/photo.jpg")
deepseek_vision(image="/path/to/photo.jpg", prompt="What people and objects are in this picture?")
```

### `deepseek_vision_status`

Check authentication and service health status.

## Project Structure

```
deepseek-vision-mcp/
├── pyproject.toml
├── README.md
├── README_CN.md
└── src/
    └── deepseek_vision_mcp/
        ├── __init__.py     # Version
        ├── __main__.py     # python -m entry point
        ├── server.py       # MCP Server + Vision pipeline
        ├── config.py       # Config management (from env vars)
        ├── auth.py         # Token auth management
        ├── hif_auth.py     # HIF signature token fetching
        ├── pow.py          # PoW challenge solver (WASM)
        ├── upload.py       # File upload + status polling
        ├── models.py       # Data models
        └── wasm/           # PoW WASM module
```

## Technical Implementation

Full vision call flow:

```
1. POST /api/v0/file/upload_file         → Upload image, get file_id (OCR processing)
2. POST /api/v0/file/fork_file_task      → Fork to vision model (real image understanding)
3. GET  hif-leim.deepseek.com/query       → Get leim signature token
4. GET  hif-dliq.deepseek.com/query       → Get dliq signature token
5. POST /api/v0/chat/completion           → Vision model analysis (requires curl_cffi TLS fingerprinting)
```

## License

MIT
