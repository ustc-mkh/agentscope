# -*- coding: utf-8 -*-
"""Start the agent service with a Docker workspace backend.

Requires:
    - A running Docker daemon reachable by the current user.
    - A running Redis server on localhost:6379.

Environment variables (optional):
    - NODE_VERSION: Major Node.js version to bake into the workspace image
      (e.g. "20"). Defaults to None (no Node).
    - GATEWAY_PORT: TCP port the in-container MCP gateway listens on.
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
from agentscope.app._manager import DockerWorkspaceManager

app = create_app(
    RedisStorage(
        host="localhost",
        port=6379,
    ),
    workspace_manager=DockerWorkspaceManager(
        basedir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "workspaces_docker",
        ),
        # Base Docker image — must have python3 in $PATH
        base_image=os.getenv("DOCKER_BASE_IMAGE", "python:3.11-slim"),
        # Optional Node.js version to install in the image
        node_version=os.getenv("NODE_VERSION"),
        # Extra pip packages to install in the gateway venv
        extra_pip=[],
        # In-container gateway port
        gateway_port=int(os.getenv("GATEWAY_PORT", "9140")),
        # Environment variables passed into every container
        env={},
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
        "main_docker:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
