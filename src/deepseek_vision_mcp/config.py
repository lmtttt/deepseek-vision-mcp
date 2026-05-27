"""Configuration management for DeepSeek Vision MCP Server."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    user_token: str = ""
    smid_v2: str = ""
    cf_clearance: str = ""
    base_url: str = "https://chat.deepseek.com"
    poll_timeout: int = 60
    poll_interval: float = 1.0
    chat_timeout: int = 120
    upload_timeout: int = 60
    max_retries: int = 3
    app_version: str = "2.0.0"
    client_locale: str = "zh_CN"

    @classmethod
    def from_env(cls) -> "Config":
        """Create Config from environment variables.

        Required:
            DEEPSEEK_USER_TOKEN - Auth token from chat.deepseek.com localStorage

        Optional:
            DEEPSEEK_SMIDV2 - smidV2 cookie value for better reliability
            DEEPSEEK_BASE_URL - API base URL (default: https://chat.deepseek.com)
            DEEPSEEK_POLL_TIMEOUT - File poll timeout in seconds (default: 60)
            DEEPSEEK_CHAT_TIMEOUT - Chat timeout in seconds (default: 120)
        """
        token = os.environ.get("DEEPSEEK_USER_TOKEN", "")
        if not token:
            token = _read_token_from_file()

        return cls(
            user_token=token,
            smid_v2=os.environ.get("DEEPSEEK_SMIDV2", ""),
            cf_clearance=os.environ.get("DEEPSEEK_CF_CLEARANCE", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://chat.deepseek.com"),
            poll_timeout=int(os.environ.get("DEEPSEEK_POLL_TIMEOUT", "60")),
            poll_interval=float(os.environ.get("DEEPSEEK_POLL_INTERVAL", "1.0")),
            chat_timeout=int(os.environ.get("DEEPSEEK_CHAT_TIMEOUT", "120")),
        )

    @property
    def is_authenticated(self) -> bool:
        return bool(self.user_token)

    @property
    def cookies(self) -> dict[str, str]:
        """Browser cookies needed for vision API calls."""
        cookies = {}
        if self.smid_v2:
            cookies["smidV2"] = self.smid_v2
        return cookies


def _read_token_from_file() -> str:
    """Try to read token from ~/.deepseek-vision/config.json."""
    try:
        config_dir = Path.home() / ".deepseek-vision"
        config_file = config_dir / "config.json"
        if config_file.exists():
            import json
            data = json.loads(config_file.read_text())
            return data.get("user_token", "")
    except Exception:
        pass
    return ""
