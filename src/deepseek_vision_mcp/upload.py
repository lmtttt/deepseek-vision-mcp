"""File upload and status polling for DeepSeek Vision API."""

import asyncio
import io
import logging
from pathlib import Path

import httpx
from PIL import Image

from .auth import AuthManager
from .config import Config
from .models import FileInfo
from .pow import PoWSolver

logger = logging.getLogger(__name__)


class FileUploader:
    """Handles image upload and status polling."""

    def __init__(self, auth: AuthManager, config: Config, http_client: httpx.AsyncClient):
        self.auth = auth
        self.config = config
        self.http = http_client
        self._pow_solver = PoWSolver()

    async def _get_pow_challenge(self, target_path: str) -> dict:
        """Obtain and solve a PoW challenge for the given target path."""
        url = f"{self.config.base_url}/api/v0/chat/create_pow_challenge"
        headers = self.auth.get_headers()
        headers["Accept"] = "application/json"
        headers["Content-Type"] = "application/json"

        payload = {"target_path": target_path}

        response = await self.http.post(
            url,
            headers=headers,
            cookies=self.auth.get_cookies(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("biz_code", 0) != 0:
            raise RuntimeError(
                f"PoW challenge failed: biz_code={data.get('biz_code')}, "
                f"msg={data.get('msg', 'unknown error')}"
            )

        challenge = data.get("data", {}).get("biz_data", {}).get("challenge", {})
        return challenge

    async def upload(self, image_data: bytes) -> FileInfo:
        """Upload image bytes to DeepSeek server.

        Returns FileInfo with the uploaded file's metadata.
        """
        # Verify image dimensions and optionally resize
        image_data = self._maybe_compress(image_data)

        # Solve PoW challenge before upload
        pow_config = await self._get_pow_challenge("/api/v0/file/upload_file")
        pow_header = self._pow_solver.solve_challenge(pow_config)

        url = f"{self.config.base_url}/api/v0/file/upload_file"
        headers = self.auth.get_headers()
        headers["x-ds-pow-response"] = pow_header
        headers.pop("Accept", None)
        headers["Accept"] = "application/json"

        files = {"file": ("image.png", image_data, "image/png")}

        response = await self.http.post(
            url,
            headers=headers,
            cookies=self.auth.get_cookies(),
            files=files,
            timeout=self.config.upload_timeout,
        )

        if response.status_code == 403:
            body = response.text
            raise RuntimeError(
                f"Upload failed with 403 after PoW: {body}"
            )

        response.raise_for_status()
        data = response.json()

        if data.get("biz_code", 0) != 0:
            raise RuntimeError(
                f"Upload failed: biz_code={data.get('biz_code')}, "
                f"msg={data.get('msg', 'unknown error')}"
            )

        file_data = data.get("data", {}).get("biz_data", {})
        return FileInfo(
            id=file_data["id"],
            file_name=file_data.get("file_name", "image.png"),
            file_size=file_data.get("file_size", 0),
            status=file_data.get("status", "pending"),
            is_image=file_data.get("is_image", False),
            width=file_data.get("width"),
            height=file_data.get("height"),
            signed_path=file_data.get("signed_path"),
            audit_result=file_data.get("audit_result"),
        )

    async def wait_for_success(
        self,
        file_id: str,
        timeout: int | None = None,
        interval: float | None = None,
    ) -> FileInfo:
        """Poll file status until SUCCESS or timeout.

        Raises TimeoutError if file processing doesn't complete within timeout.
        Raises RuntimeError if file processing fails.
        """
        timeout = timeout or self.config.poll_timeout
        interval = interval or self.config.poll_interval
        deadline = asyncio.get_event_loop().time() + timeout

        terminal_failures = {
            "FAILED",
            "CONTENT_EMPTY",
            "CONTENT_FILTER",
            "CONTENT_TOO_LONG",
            "CANCELLED",
        }

        file_info = FileInfo(id=file_id, status="PENDING")

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"File {file_id} did not reach SUCCESS within {timeout}s "
                    f"(last status: {file_info.status})"
                )

            file_info = await self._fetch_file(file_id)

            if file_info.status == "SUCCESS":
                logger.info("File %s processing completed", file_id)
                return file_info

            if file_info.status in terminal_failures:
                raise RuntimeError(
                    f"File {file_id} processing failed: status={file_info.status}"
                )

            await asyncio.sleep(min(interval, remaining))

    def _maybe_compress(self, image_data: bytes, max_dim: int = 2048) -> bytes:
        """Compress image if it exceeds max_dim or is too large."""
        try:
            img = Image.open(io.BytesIO(image_data))
            # Check if resizing is needed
            w, h = img.size
            if w <= max_dim and h <= max_dim and len(image_data) <= 20 * 1024 * 1024:
                return image_data

            # Resize while maintaining aspect ratio
            if w > max_dim or h > max_dim:
                ratio = min(max_dim / w, max_dim / h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            # Convert RGBA to RGB if needed
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception as e:
            logger.warning("Image compression failed, using original: %s", e)
            return image_data

    async def _fetch_file(self, file_id: str) -> FileInfo:
        """Fetch file status from the API."""
        url = f"{self.config.base_url}/api/v0/file/fetch_files"
        headers = self.auth.get_headers()
        headers["Accept"] = "application/json"

        params = {"file_ids": file_id}

        response = await self.http.get(
            url,
            headers=headers,
            cookies=self.auth.get_cookies(),
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        biz_data = data.get("data", {}).get("biz_data", {})
        files = biz_data.get("files", [])
        if not files:
            return FileInfo(id=file_id, status="PENDING")

        f = files[0]
        return FileInfo(
            id=f.get("id", file_id),
            file_name=f.get("file_name", ""),
            file_size=f.get("file_size", 0),
            status=f.get("status", "PENDING").upper(),
            is_image=f.get("is_image", False),
            width=f.get("width"),
            height=f.get("height"),
            signed_path=f.get("signed_path"),
            audit_result=f.get("audit_result"),
        )
