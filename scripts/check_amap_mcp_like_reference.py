from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient


AMAP_MCP_URL = "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp"


async def main() -> int:
    _configure_stdout()
    _load_dotenv_if_available()

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    print(f"Endpoint: {AMAP_MCP_URL}")
    print('Transport: "http"')
    print(f"DASHSCOPE_API_KEY set: {bool(api_key)}")

    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY is not set.")
        return 2

    client = MultiServerMCPClient(
        {
            "amap-server": {
                "transport": "http",
                "url": AMAP_MCP_URL,
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
        }
    )

    try:
        tools = await client.get_tools()
    except Exception as exc:
        print("ERROR: failed to connect/list Amap MCP tools.")
        print(f"{type(exc).__name__}: {exc}")
        return 3

    tool_map = {tool.name: tool for tool in tools}
    print(f"Tool count: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {_tool_args(tool)}")

    weather_tool = tool_map.get("maps_weather")
    if weather_tool is None:
        print("ERROR: maps_weather was not discovered.")
        return 4

    try:
        print("Calling maps_weather(city='杭州')...")
        result = await weather_tool.ainvoke({"city": "杭州"})
    except ToolException as exc:
        print("ERROR: maps_weather returned a tool error.")
        print(str(exc))
        if "INVALID_USER_KEY" in str(exc):
            print("Diagnosis: MCP connection works, but Amap rejected the key.")
        return 5
    except Exception as exc:
        print("ERROR: maps_weather call failed.")
        print(f"{type(exc).__name__}: {exc}")
        return 6

    print("maps_weather result:")
    print(result)
    return 0


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv()


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _tool_args(tool: Any) -> str:
    args = getattr(tool, "args", None)
    if not args:
        return "{}"
    return ", ".join(str(key) for key in args)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
