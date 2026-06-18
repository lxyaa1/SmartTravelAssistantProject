from __future__ import annotations

import os
import logging
from pathlib import Path
from urllib.parse import quote

from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAILIAN_AMAP_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"
OFFICIAL_AMAP_MCP_BASE_URL = "https://mcp.amap.com/mcp"
AMAP_MCP_URL = BAILIAN_AMAP_MCP_URL
AMAP_MCP_TRANSPORT = "http"
AMAP_KEY_ENV_NAMES = ("Amap_Key", "AMAP_KEY", "AMAP_MAPS_API_KEY")


def build_mock_mcp_client() -> MultiServerMCPClient:
    """Create a local stdio MCP client config for future tool-backed nodes."""
    server_path = PROJECT_ROOT / "mcp_servers" / "mock_travel_server" / "server.py"
    return MultiServerMCPClient(
        {
            "mock_travel": {
                "command": "python",
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )


async def load_mock_tools():
    client = build_mock_mcp_client()
    return await client.get_tools()


def build_amap_mcp_client(
    api_key: str | None = None,
    provider: str | None = None,
) -> MultiServerMCPClient:
    """Create a Streamable HTTP MCP client for Amap Maps.

    provider:
      - official: use Amap's own MCP endpoint with an Amap Web service key.
      - bailian: use Bailian's MCP endpoint with DASHSCOPE_API_KEY.
      - auto/None: prefer official when an Amap key is available.
    """
    _quiet_mcp_http_logging()
    selected_provider = _select_amap_provider(provider, has_explicit_api_key=api_key is not None)
    if selected_provider == "official":
        amap_key = api_key or get_amap_api_key()
        if not amap_key:
            raise ValueError(f"Amap key is required in one of: {', '.join(AMAP_KEY_ENV_NAMES)}")
        endpoint = f"{OFFICIAL_AMAP_MCP_BASE_URL}?key={quote(amap_key, safe='')}"
        return MultiServerMCPClient(
            {
                "amap-official": {
                    "url": endpoint,
                    "transport": AMAP_MCP_TRANSPORT,
                }
            }
        )

    dashscope_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        raise ValueError("DASHSCOPE_API_KEY is required for Bailian Amap MCP")
    return MultiServerMCPClient(
        {
            "amap-server": {
                "url": BAILIAN_AMAP_MCP_URL,
                "headers": {"Authorization": f"Bearer {dashscope_key}"},
                "transport": AMAP_MCP_TRANSPORT,
            }
        }
    )


async def load_amap_tools(api_key: str | None = None, provider: str | None = None):
    client = build_amap_mcp_client(api_key=api_key, provider=provider)
    return await client.get_tools()


def get_amap_api_key() -> str:
    return _get_first_env(AMAP_KEY_ENV_NAMES)


def has_amap_api_key() -> bool:
    return bool(get_amap_api_key())


def get_bailian_api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY", "")


def _select_amap_provider(provider: str | None, has_explicit_api_key: bool = False) -> str:
    if provider is None and has_explicit_api_key:
        return "bailian"

    normalized = (provider or os.getenv("TRAVEL_AGENT_AMAP_PROVIDER") or "auto").strip().lower()
    if normalized in {"official", "amap_official"}:
        return "official"
    if normalized in {"bailian", "dashscope", "amap_bailian"}:
        return "bailian"
    if normalized == "auto":
        return "official" if has_amap_api_key() else "bailian"
    raise ValueError("TRAVEL_AGENT_AMAP_PROVIDER must be one of: auto, official, bailian")


def _get_first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value

    if os.name == "nt":
        for name in names:
            value = _get_windows_env(name)
            if value:
                return value
    return ""


def _get_windows_env(name: str) -> str:
    try:
        import winreg
    except ImportError:
        return ""

    for root, subkey in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    ):
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return str(value)
    return ""


def _quiet_mcp_http_logging() -> None:
    for logger_name in ("httpx", "httpcore", "mcp", "mcp.client", "mcp.client.streamable_http"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
