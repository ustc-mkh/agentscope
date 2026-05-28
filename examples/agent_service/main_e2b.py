# -*- coding: utf-8 -*-
"""Start the agent service with an E2B cloud sandbox workspace backend.

Requires:
    - A valid E2B API key (set via the ``E2B_API_KEY`` env var or passed
      directly).
    - A running Redis server on localhost:6379.

Environment variables:
    - E2B_API_KEY (required): Your E2B API key.
    - E2B_TEMPLATE (optional): E2B template id. Defaults to "base".
    - E2B_DOMAIN (optional): Custom E2B domain for self-hosted setups.
    - E2B_TIMEOUT (optional): Sandbox keep-alive timeout in seconds.
      Defaults to 300.
    - GATEWAY_PORT (optional): TCP port the in-sandbox gateway listens on.
      Defaults to 9140.
"""

import os

import uvicorn
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from agentscope.app import (
    RedisStorage,
    create_app,
)
from agentscope.app._manager import E2BWorkspaceManager

app = create_app(
    RedisStorage(
        host="localhost",
        port=6379,
    ),
    workspace_manager=E2BWorkspaceManager(
        # E2B template — "base" is a stock Ubuntu image with python3 + curl
        template=os.getenv("E2B_TEMPLATE", "base"),
        # E2B API key — falls back to E2B_API_KEY env var if empty
        api_key=os.getenv("E2B_API_KEY", ""),
        # Optional custom E2B domain
        domain=os.getenv("E2B_DOMAIN", ""),
        # Sandbox keep-alive timeout (seconds)
        timeout_seconds=int(os.getenv("E2B_TIMEOUT", "300")),
        # In-sandbox gateway port
        gateway_port=int(os.getenv("GATEWAY_PORT", "9140")),
        # Environment variables baked into the sandbox
        env={},
        # Extra pip packages for the gateway venv
        extra_pip=[],
        # Default MCP servers seeded into new workspaces
        default_mcps=[],
        # TTL for idle workspace eviction (seconds)
        ttl=3600.0,
    ),
    extra_middlewares=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ],
)


if __name__ == "__main__":
    uvicorn.run(
        "main_e2b:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
