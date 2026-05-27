"""MCP Server for DeepSeek Vision (real multimodal model).

Provides:
- deepseek_vision: Upload an image and analyze with DeepSeek's vision model
- deepseek_vision_status: Check authentication and service health

The vision flow requires:
1. Upload image → get OCR file_id
2. Fork file to vision model via fork_file_task → get vision file_id
3. Get HIF signature tokens (hif-leim, hif-dliq)
4. Call chat/completion with model_type=vision and the forked file_id

Environment variables:
- DEEPSEEK_USER_TOKEN (required): Auth token from chat.deepseek.com
- DEEPSEEK_SMIDV2 (optional): smidV2 cookie for reliability
- DEEPSEEK_BASE_URL (optional): API base URL
"""

import asyncio
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import httpx
from curl_cffi import requests as curl_requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import __version__
from .auth import AuthManager
from .config import Config
from .hif_auth import HIFAuth
from .pow import PoWSolver
from .upload import FileUploader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session persistence — allows continuing a conversation across tool calls
# ---------------------------------------------------------------------------

_last_session_state: dict | None = None


def _get_session_file() -> Path:
    return Path.home() / ".deepseek-vision" / "session.json"


def _load_session_state() -> dict | None:
    """Load the last persisted session state from disk."""
    global _last_session_state
    if _last_session_state is not None:
        return _last_session_state
    try:
        sf = _get_session_file()
        if sf.exists():
            _last_session_state = json.loads(sf.read_text())
            return _last_session_state
    except Exception:
        pass
    return None


def _save_session_state(state: dict) -> None:
    """Persist session state to disk so it survives server restarts."""
    global _last_session_state
    _last_session_state = state
    try:
        config_dir = Path.home() / ".deepseek-vision"
        config_dir.mkdir(parents=True, exist_ok=True)
        _get_session_file().write_text(json.dumps(state))
    except Exception:
        pass


def create_app() -> Server:
    """Create and configure the MCP server."""
    config = Config.from_env()
    auth = AuthManager(config)
    hif = HIFAuth(config)

    app = Server("deepseek-vision")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="deepseek_vision",
                description="Upload an image and analyze it using DeepSeek's vision model. "
                            "Supports photos, screenshots, documents with images.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Local file path to the image (jpg, png, etc.)",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Question or prompt about the image (default: 请详细描述这张图片中的内容)",
                            "default": "请详细描述这张图片中的内容",
                        },
                        "thinking": {
                            "type": "boolean",
                            "description": "Enable DeepThink reasoning",
                            "default": False,
                        },
                        "continue_conversation": {
                            "type": "boolean",
                            "description": "Continue the previous conversation instead of creating a new one. "
                                         "When True, reuses the last session_id and chains messages so the "
                                         "model can compare with previously uploaded images.",
                            "default": False,
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Explicit chat_session_id to reuse. Overrides continue_conversation "
                                         "when both are provided. Use this to switch between multiple "
                                         "conversation threads.",
                        },
                    },
                    "required": ["image"],
                },
            ),
            Tool(
                name="deepseek_vision_status",
                description="Check authentication and service health status",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == "deepseek_vision":
                return await handle_vision(arguments, config, auth, hif)
            elif name == "deepseek_vision_status":
                return await handle_status(config, auth)
        except Exception as e:
            logger.exception("Tool call failed")
            return [TextContent(type="text", text=f"Error: {e}")]
        raise ValueError(f"Unknown tool: {name}")

    return app


async def handle_vision(
    arguments: dict,
    config: Config,
    auth: AuthManager,
    hif: HIFAuth,
) -> list[TextContent]:
    """Handle the deepseek_vision tool call."""
    if not auth.validate():
        return [TextContent(
            type="text",
            text="DeepSeek token not configured.\n\n"
                 "Set DEEPSEEK_USER_TOKEN environment variable.\n\n"
                 "To get your token:\n"
                 "1. Open chat.deepseek.com in Chrome\n"
                 "2. DevTools → Application → Local Storage\n"
                 "3. Key: userToken → Copy JSON.parse(value).value"
        )]

    # Read image
    image_path: str = arguments["image"]
    prompt: str = arguments.get("prompt", "请详细描述这张图片中的内容")
    thinking: bool = arguments.get("thinking", False)
    continue_conversation: bool = arguments.get("continue_conversation", False)
    explicit_session_id: str | None = arguments.get("session_id")

    # Resolve session continuity
    reuse_session_id: str | None = None
    reuse_parent_message_id: str | None = None

    if explicit_session_id:
        reuse_session_id = explicit_session_id
        # When explicit session_id is given, try to load its last message_id
        saved = _load_session_state()
        if saved and saved.get("session_id") == explicit_session_id:
            reuse_parent_message_id = saved.get("parent_message_id")
    elif continue_conversation:
        saved = _load_session_state()
        if saved:
            reuse_session_id = saved.get("session_id")
            reuse_parent_message_id = saved.get("parent_message_id")

    try:
        if image_path.startswith("data:") or _is_base64(image_path):
            image_data = _decode_base64_image(image_path)
        else:
            with open(image_path, "rb") as f:
                image_data = f.read()
    except FileNotFoundError:
        return [TextContent(type="text", text=f"Image not found: {image_path}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Failed to read image: {e}")]

    try:
        result_text, new_session_id, new_parent_message_id = await _run_vision_pipeline(
            config, image_data, prompt, thinking, hif,
            session_id=reuse_session_id,
            parent_message_id=reuse_parent_message_id,
        )

        # Persist session state for future continuation
        _save_session_state({
            "session_id": new_session_id,
            "parent_message_id": new_parent_message_id,
        })

        # Build response
        lines = [result_text]
        if continue_conversation or explicit_session_id:
            lines.append(f"\n---\n[会话继续中] session_id: {new_session_id}")
        else:
            lines.append(f"\n---\n[session_id: {new_session_id}] (可用 continue_conversation=true 继续此对话)")
        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.exception("Vision pipeline failed")
        return [TextContent(type="text", text=f"Vision analysis failed: {e}")]


async def _run_vision_pipeline(
    config: Config,
    image_data: bytes,
    prompt: str,
    thinking: bool,
    hif: HIFAuth,
    session_id: str | None = None,
    parent_message_id: str | None = None,
) -> tuple[str, str, str | None]:
    """Complete vision pipeline: upload → fork → (optional session) → vision completion.

    Returns (response_text, session_id, new_parent_message_id).
    """
    async with httpx.AsyncClient() as cli:
        # Step 1: Upload image (OCR processing)
        logger.info("Step 1: Uploading image...")
        uploader = FileUploader(AuthManager(config), config, cli)
        file_info = await uploader.upload(image_data)
        file_info = await uploader.wait_for_success(file_info.id)
        logger.info("  Uploaded: %s (OCR)", file_info.id)

        # Step 2: Fork to vision model
        logger.info("Step 2: Forking to vision model...")
        vision_file_id = await _fork_to_vision(config, file_info.id)
        logger.info("  Vision file: %s", vision_file_id)

        # Step 3: Create or reuse chat session
        if session_id:
            logger.info("Step 3: Reusing session: %s", session_id)
        else:
            logger.info("Step 3: Creating session...")
            session_id = await _create_session(config)
            logger.info("  Session: %s", session_id)

        # Step 4: Get HIF tokens
        logger.info("Step 4: Getting HIF tokens...")
        hif_headers = await hif.get_headers()

        # Step 5: Vision completion
        logger.info("Step 5: Vision completion...")
        solver = PoWSolver()

        # Get PoW for completion via httpx
        r = await cli.post(
            f"{config.base_url}/api/v0/chat/create_pow_challenge",
            headers={"Authorization": f"Bearer {config.user_token}",
                     "content-type": "application/json"},
            json={"target_path": "/api/v0/chat/completion"},
        )
        pow_header = solver.solve_challenge(
            r.json()["data"]["biz_data"]["challenge"]
        )

    # Vision completion via curl_cffi (for TLS fingerprint + proper headers)
    result_text, new_parent_message_id = await _vision_completion(
        config, pow_header, hif_headers, session_id, vision_file_id,
        prompt, thinking, parent_message_id,
    )
    return result_text, session_id, new_parent_message_id


async def _fork_to_vision(config: Config, file_id: str) -> str:
    """Fork uploaded file to vision model.

    The uploaded file only has OCR text extraction.
    Forking creates a new file with full image understanding.
    """
    async with httpx.AsyncClient() as cli:
        r = await cli.post(
            f"{config.base_url}/api/v0/file/fork_file_task",
            headers={"Authorization": f"Bearer {config.user_token}",
                     "content-type": "application/json"},
            json={"file_id": file_id, "to_model_type": "vision"},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Fork failed: {r.text[:200]}")

        data = r.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Fork failed: {data.get('msg', 'unknown')}")

        vision_file_id = data["data"]["biz_data"]["id"]

        # Wait for vision processing
        for i in range(30):
            await asyncio.sleep(1)
            r = await cli.get(
                f"{config.base_url}/api/v0/file/fetch_files?file_ids={vision_file_id}",
                headers={"Authorization": f"Bearer {config.user_token}"},
            )
            files = r.json().get("data", {}).get("biz_data", {}).get("files", [])
            if files:
                status = files[0].get("status", "").upper()
                if status == "SUCCESS":
                    return vision_file_id
                if status in ("FAILED", "ERROR"):
                    raise RuntimeError(f"Vision file processing failed: {status}")

        raise TimeoutError("Vision file did not become ready within 30s")


async def _create_session(config: Config) -> str:
    """Create a chat session."""
    async with httpx.AsyncClient() as cli:
        r = await cli.post(
            f"{config.base_url}/api/v0/chat_session/create",
            headers={"Authorization": f"Bearer {config.user_token}",
                     "content-type": "application/json"},
            json={"agent": "chat"},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Session creation failed: {r.text[:200]}")
        return r.json()["data"]["biz_data"]["id"]


async def _vision_completion(
    config: Config,
    pow_header: str,
    hif_headers: dict,
    session_id: str,
    vision_file_id: str,
    prompt: str,
    thinking: bool,
    parent_message_id: str | None = None,
) -> tuple[str, str | None]:
    """Send vision request using curl_cffi for proper TLS fingerprint.

    Returns (response_text, new_parent_message_id).
    The new_parent_message_id should be used as parent_message_id for the
    next message in the same conversation.
    """
    session = curl_requests.Session()
    session.impersonate = "chrome131"

    # Add user cookies if configured
    if config.cookies:
        session.cookies.update(config.cookies)

    headers = {
        "accept": "*/*",
        "accept-language": f"{config.client_locale},{config.client_locale};q=0.9,en;q=0.8",
        "authorization": f"Bearer {config.user_token}",
        "content-type": "application/json",
        "priority": "u=1, i",
        "origin": config.base_url,
        "referer": f"{config.base_url}/a/chat/s/{session_id}",
        "x-app-version": config.app_version,
        "x-client-locale": config.client_locale,
        "x-client-platform": "web",
        "x-client-timezone-offset": "28800",
        "x-client-version": config.app_version,
        "x-ds-pow-response": pow_header,
        **hif_headers,
    }

    body = {
        "chat_session_id": session_id,
        "parent_message_id": parent_message_id,
        "model_type": "vision",
        "prompt": prompt,
        "ref_file_ids": [vision_file_id],
        "thinking_enabled": thinking,
        "search_enabled": False,
        "action": None,
        "preempt": False,
    }

    r = session.post(
        f"{config.base_url}/api/v0/chat/completion",
        headers=headers,
        json=body,
        stream=True,
    )

    if r.status_code != 200:
        raise RuntimeError(
            f"Vision API returned HTTP {r.status_code}: {r.text[:300]}"
        )

    text_parts = []
    new_parent_message_id: str | None = None
    for line in r.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if payload == "[DONE]":
            break

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "error":
            raise RuntimeError(
                f"Vision error: {event.get('content', 'unknown error')}"
            )

        # Capture message_id for conversation continuity.
        # DeepSeek returns response_message_id at top level in the first
        # stream event, and also inside v.response.message_id.
        msg_id = event.get("response_message_id")
        if not msg_id:
            v = event.get("v", {})
            if isinstance(v, dict):
                resp = v.get("response", {})
                if isinstance(resp, dict):
                    msg_id = resp.get("message_id")
        if not msg_id:
            msg_id = event.get("message_id") or event.get("msg_id")
        if msg_id:
            new_parent_message_id = msg_id

        # Text tokens
        if "v" in event and isinstance(event.get("v"), str):
            text_parts.append(event["v"])

        # Alternative text format
        if event.get("type") == "text":
            txt = event.get("text", "") or event.get("content", "")
            if isinstance(txt, list):
                txt = "".join(
                    x.get("text", "") if isinstance(x, dict) else str(x)
                    for x in txt
                )
            if txt:
                text_parts.append(str(txt))

    return "".join(text_parts), new_parent_message_id


async def handle_status(
    config: Config,
    auth: AuthManager,
) -> list[TextContent]:
    """Handle the deepseek_vision_status tool call."""
    token_valid = auth.validate()
    status_lines = [
        f"DeepSeek Vision MCP Server v{__version__}",
        "",
        f"- Authenticated: {'✅' if token_valid else '❌'}",
        f"- Token configured: {'Yes' if config.user_token else 'No'}",
        f"- smidV2 cookie: {'✅' if config.smid_v2 else '❌ (optional)'}",
        f"- Base URL: {config.base_url}",
    ]
    if not token_valid:
        status_lines.extend([
            "",
            "Setup:",
            "  export DEEPSEEK_USER_TOKEN='your_token_here'",
        ])

    return [TextContent(type="text", text="\n".join(status_lines))]


def _is_base64(s: str) -> bool:
    """Check if a string looks like base64-encoded image data."""
    import base64
    try:
        if len(s) > 100:
            base64.b64decode(s, validate=True)
            return True
        return False
    except Exception:
        return False


def _decode_base64_image(data: str) -> bytes:
    """Decode a base64 image (with or without data URI prefix)."""
    import base64
    if data.startswith("data:"):
        _, encoded = data.split(",", 1)
    else:
        encoded = data
    return base64.b64decode(encoded)


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting DeepSeek Vision MCP Server v%s", __version__)

    app = create_app()

    import anyio
    anyio.run(_run_stdio, app)


async def _run_stdio(app: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    main()
