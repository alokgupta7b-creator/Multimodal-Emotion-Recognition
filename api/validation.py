"""Upload validation and safe temporary-file handling."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, Request, UploadFile

from api.config import ApiSettings

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_upload_label(filename: str | None) -> str:
    """Return a path-free, ASCII-safe display label (never joined into filesystem paths)."""
    if not filename:
        return "upload"
    base = Path(filename.replace("\\", "/")).name.strip()
    cleaned = _SAFE_LABEL_RE.sub("_", base).strip("._")
    return cleaned[:64] or "upload"


class StreamTooLargeError(Exception):
    """Raised when an incoming request body stream exceeds the allowed byte limit."""
    pass


def validate_content_length(request: Request, settings: ApiSettings, extra_bytes: int = 65_536) -> None:
    """Validate Content-Length header upfront to prevent unbounded parsing/spooling."""
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            length = int(raw_length)
            if length > settings.max_upload_bytes + extra_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request payload exceeds maximum allowed size ({settings.max_upload_bytes} bytes).",
                )
        except ValueError:
            pass


def _detect_video_suffix(data: bytes) -> str | None:
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".mp4"
    if data.startswith(b"\x1aE\xdf\xa3") and b"webm" in data[:64].lower():
        return ".webm"
    return None


def _signature(data: bytes, modality: str) -> bool:
    if modality == "audio":
        return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE"
    if modality == "video":
        return _detect_video_suffix(data) is not None
    return False


def _open_validated_temp(initial_bytes: bytes, modality: str) -> tuple[BinaryIO, Path]:
    header = initial_bytes[:64]
    if not initial_bytes or not _signature(header, modality):
        raise HTTPException(status_code=415, detail=f"Uploaded content is not a valid {modality} file.")
    suffix = ".wav" if modality == "audio" else (_detect_video_suffix(header) or ".mp4")
    handle = tempfile.NamedTemporaryFile(prefix="emotion_", suffix=suffix, delete=False)
    temporary = Path(handle.name)
    handle.write(initial_bytes)
    return handle, temporary


async def store_upload(upload: UploadFile, modality: str, settings: ApiSettings) -> Path:
    if modality not in {"audio", "video"}:
        raise ValueError(f"Unsupported upload modality: {modality}")
    total = 0
    prefix = bytearray()
    handle: BinaryIO | None = None
    temporary: Path | None = None
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit.")
            if handle is None:
                prefix.extend(chunk)
                if len(prefix) >= 64:
                    handle, temporary = _open_validated_temp(bytes(prefix), modality)
                    prefix.clear()
            else:
                handle.write(chunk)
        if handle is None:
            handle, temporary = _open_validated_temp(bytes(prefix), modality)
        assert temporary is not None
        return temporary
    except HTTPException:
        if handle is not None:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if handle is not None:
            handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file could not be stored.") from exc
    finally:
        if handle is not None:
            handle.close()
        await upload.close()


def cleanup_file(path: Path) -> None:
    path.unlink(missing_ok=True)
