from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import ToolException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.mcp.client import (  # noqa: E402
    AMAP_KEY_ENV_NAMES,
    AMAP_MCP_TRANSPORT,
    BAILIAN_AMAP_MCP_URL,
    OFFICIAL_AMAP_MCP_BASE_URL,
    get_amap_api_key,
    get_bailian_api_key,
    load_amap_tools,
)


async def main() -> int:
    _configure_stdout()

    parser = argparse.ArgumentParser(description="Minimal Amap MCP connectivity check.")
    parser.add_argument(
        "--provider",
        choices=("auto", "official", "bailian"),
        default="auto",
        help="MCP provider. auto prefers official Amap MCP when an Amap key is available.",
    )
    parser.add_argument("--call-weather", action="store_true", help="Also call maps_weather after listing tools.")
    parser.add_argument("--city", default="\u676d\u5dde", help="City name or adcode for maps_weather.")
    args = parser.parse_args()

    amap_key = get_amap_api_key()
    dashscope_key = get_bailian_api_key()
    provider = args.provider
    if provider == "auto":
        provider = "official" if amap_key else "bailian"

    print(f"Provider: {provider}")
    print(f"Transport: {AMAP_MCP_TRANSPORT!r}")
    print(f"Amap key set: {bool(amap_key)}")
    print(f"DASHSCOPE_API_KEY set: {bool(dashscope_key)}")

    if provider == "official":
        if not amap_key:
            print(f"ERROR: no Amap key found in {', '.join(AMAP_KEY_ENV_NAMES)}.")
            return 2
        print(f"Endpoint: {OFFICIAL_AMAP_MCP_BASE_URL}?key=***")
        load_tools = lambda: load_amap_tools(api_key=amap_key, provider="official")
    else:
        if not dashscope_key:
            print("ERROR: DASHSCOPE_API_KEY is not set.")
            return 2
        print(f"Endpoint: {BAILIAN_AMAP_MCP_URL}")
        load_tools = lambda: load_amap_tools(api_key=dashscope_key, provider="bailian")

    try:
        print("Loading Amap MCP tools...")
        tools = await load_tools()
    except Exception as exc:
        print("ERROR: failed to connect/list Amap MCP tools.")
        print(f"{type(exc).__name__}: {exc}")
        return 3

    tool_map = {tool.name: tool for tool in tools}
    print(f"Tool count: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {_tool_args(tool)}")

    if not args.call_weather:
        print("Tool discovery succeeded. Add --call-weather to test an actual tool call.")
        return 0

    tool = tool_map.get("maps_weather")
    if not tool:
        print("ERROR: maps_weather was not discovered.")
        return 4

    try:
        print(f"Calling maps_weather(city={args.city!r})...")
        result = await tool.ainvoke({"city": args.city})
    except ToolException as exc:
        print("ERROR: maps_weather returned a tool error.")
        print(str(exc))
        if "INVALID_USER_KEY" in str(exc):
            print("Diagnosis: MCP connection works, but the Amap tool rejected this key.")
            print("Check Bailian Amap MCP activation/authorization or whether an extra Amap service key is required.")
        return 5
    except Exception as exc:
        print("ERROR: maps_weather call failed.")
        print(f"{type(exc).__name__}: {exc}")
        return 6

    print("maps_weather result:")
    print(result)
    return 0


def _tool_args(tool: Any) -> str:
    args = getattr(tool, "args", None)
    if not args:
        return "{}"
    return ", ".join(str(key) for key in args)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
