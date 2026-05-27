"""Authentication management for DeepSeek API."""

from .config import Config


class AuthManager:
    """Manages authentication headers and cookies for DeepSeek API."""

    def __init__(self, config: Config):
        self.config = config
        self._token_valid: bool | None = None

    def get_headers(self) -> dict[str, str]:
        """Build headers for DeepSeek API requests."""
        headers = {
            "Authorization": f"Bearer {self.config.user_token}",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/event-stream, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "X-App-Version": self.config.app_version,
            "X-Client-Version": "1.0.0-always",
            "X-Client-Locale": "zh-CN",
            "X-Client-Platform": "web",
        }
        return headers

    def get_cookies(self) -> dict[str, str]:
        """Build cookies for DeepSeek API requests."""
        cookies: dict[str, str] = {}
        if self.config.cf_clearance:
            cookies["cf_clearance"] = self.config.cf_clearance
        return cookies

    def validate(self) -> bool:
        """Check if token is configured (non-empty).

        Full validation requires an actual API call, but we can at
        least check that a token was provided.
        """
        if self._token_valid is not None:
            return self._token_valid
        valid = self.config.is_authenticated
        self._token_valid = valid
        return valid

    @property
    def token_source(self) -> str:
        """Describe where the token was sourced from."""
        import os
        if os.environ.get("DEEPSEEK_USER_TOKEN"):
            return "env"
        return "file"
