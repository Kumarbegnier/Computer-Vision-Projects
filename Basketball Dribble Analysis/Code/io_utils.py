from __future__ import annotations

import ipaddress
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_TIMEOUT_SECONDS = 20


def _is_public_ip(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for info in infos:
        ip = info[4][0]
        ip_obj = ipaddress.ip_address(ip)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            return False
    return True


def validate_remote_video_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        raise ValueError("Video URL is empty.")

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("Only http/https URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("URL hostname is missing.")

    if not _is_public_ip(parsed.hostname):
        raise ValueError("URL host is not public/reachable.")

    return url


def download_video_to_temp_file(url: str) -> Path:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp.name)

    total = 0
    try:
        with temp:
            with urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Remote file is too large (>{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB)."
                        )
                    temp.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if total == 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Downloaded file is empty.")

    return temp_path
