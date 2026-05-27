"""HIF (High-Integrity Framework) token manager.

Fetches x-hif-leim and x-hif-dliq tokens from DeepSeek's internal
signing service. These tokens are required for vision model requests.
Tokens expire every 600 seconds (10 minutes).

Uses httpx for token fetching (more reliable than curl_cffi for these endpoints).
"""

import asyncio
import logging
import time

import httpx

from .config import Config

logger = logging.getLogger(__name__)


class HIFAuth:
    """Manages HIF signature tokens for DeepSeek vision requests."""

    LEIM_URL = "https://hif-leim.deepseek.com/query"
    DLIQ_URL = "https://hif-dliq.deepseek.com/query"
    TTL = 600  # tokens expire after 10 minutes

    def __init__(self, config: Config):
        self.config = config
        self._leim: str | None = None
        self._dliq: str | None = None
        self._expires_at: float = 0

    async def get_headers(self) -> dict[str, str]:
        """Get HIF auth headers, refreshing if needed."""
        if time.time() >= self._expires_at or not self._leim or not self._dliq:
            await self._refresh()
        return {
            "x-hif-leim": self._leim or "",
            "x-hif-dliq": self._dliq or "",
        }

    async def _refresh(self):
        """Fetch fresh HIF tokens from DeepSeek signing service."""
        common = {
            "Authorization": f"Bearer {self.config.user_token}",
            "Origin": "https://chat.deepseek.com",
            "Referer": "https://chat.deepseek.com/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        }

        # hif-dliq.deepseek.com only has AAAA records → fails on WSL2 without IPv6 routing.
        # Patch anyio.connect_tcp to redirect to hif-leim's IPv4 (same CloudFront distribution).
        import anyio
        _orig_connect_tcp = anyio.connect_tcp
        async def _patched_connect_tcp(remote_host, remote_port, *, local_host=None,
                                       tls=False, ssl_context=None,
                                       tls_standard_compatible=True, tls_hostname=None,
                                       happy_eyeballs_delay=0.25):
            if isinstance(remote_host, str) and remote_host == 'hif-dliq.deepseek.com':
                remote_host = 'hif-leim.deepseek.com'
            return await _orig_connect_tcp(
                remote_host, remote_port, local_host=local_host,
                tls=tls, ssl_context=ssl_context,
                tls_standard_compatible=tls_standard_compatible,
                tls_hostname=tls_hostname,
                happy_eyeballs_delay=happy_eyeballs_delay,
            )
        anyio.connect_tcp = _patched_connect_tcp

        try:
            async with httpx.AsyncClient(timeout=15.0) as cli:
                leim_resp = await cli.get(self.LEIM_URL, headers=common)
                dliq_resp = await cli.get(self.DLIQ_URL, headers=common)
        finally:
            anyio.connect_tcp = _orig_connect_tcp

        leim_resp.raise_for_status()
        dliq_resp.raise_for_status()

        self._leim = leim_resp.json()["data"]["biz_data"]["value"]
        self._dliq = dliq_resp.json()["data"]["biz_data"]["value"]
        self._expires_at = time.time() + self.TTL * 0.8

        logger.info(
            "HIF tokens refreshed (leim=%s..., dliq=%s...)",
            self._leim[:20],
            self._dliq[:20],
        )
