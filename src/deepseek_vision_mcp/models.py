"""Pydantic data models for DeepSeek Vision MCP Server."""

from pydantic import BaseModel


class FileInfo(BaseModel):
    """Information about an uploaded file."""
    id: str
    file_name: str = ""
    file_size: int = 0
    status: str = "pending"
    is_image: bool = False
    width: int | None = None
    height: int | None = None
    signed_path: str | None = None
    audit_result: str | None = None


class SSEFragment(BaseModel):
    """A fragment from the SSE chat stream."""
    type: str  # "text" | "thinking" | "error" | "done"
    content: str = ""
    message_id: str | None = None


class VisionResult(BaseModel):
    """Result from a vision query."""
    text: str
    thinking: str | None = None
    file_id: str
    session_id: str
    message_id: str | None = None
