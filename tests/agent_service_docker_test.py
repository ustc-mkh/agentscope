# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Integration tests for the agent service with a Docker workspace backend.

Tests the full HTTP API surface — agent CRUD, session CRUD, workspace MCP
and skill management — with a real Docker backend.  Storage uses a real
Redis server (connection configurable via ``REDIS_HOST``, ``REDIS_PORT``,
``REDIS_DB``, ``REDIS_PASSWORD`` env vars).  The Docker daemon **and**
a Redis server are required for all tests.

The whole module is skipped when no Docker daemon is reachable.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
import uuid
from unittest.async_case import IsolatedAsyncioTestCase

from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from agentscope.app import create_app
from agentscope.app._manager import DockerWorkspaceManager
from agentscope.app.storage import RedisKeyConfig, RedisStorage

# ── docker daemon detection ────────────────────────────────────────


def _docker_available() -> bool:
    """Return ``True`` iff the Docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


_DOCKER_OK = _docker_available()
_SKIP_REASON = "Docker daemon not available"

# Default user header for all requests
_USER_HEADERS = {"X-User-ID": "test-user"}


# ── helpers ────────────────────────────────────────────────────────


def _make_storage() -> RedisStorage:
    """Create a RedisStorage connected to a real Redis server.

    Connection parameters are read from environment variables:
    ``REDIS_HOST`` (default ``localhost``), ``REDIS_PORT`` (default ``6379``),
    ``REDIS_PASSWORD`` (default ``None``), ``REDIS_DB`` (default ``0``).
    """
    return RedisStorage(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        password=os.environ.get("REDIS_PASSWORD", None) or None,
        key_ttl=None,
        key_config=RedisKeyConfig(),
    )


# ── test base class ────────────────────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class DockerAgentServiceTestBase(IsolatedAsyncioTestCase):
    """Base class that sets up a FastAPI test client with Docker backend."""

    async def asyncSetUp(self) -> None:
        # pylint: disable=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = _make_storage()

        self.workspace_manager = DockerWorkspaceManager(
            basedir=self.temp_dir.name,
        )

        self.app = create_app(
            self.storage,
            workspace_manager=self.workspace_manager,
            extra_middlewares=[
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                ),
            ],
        )

        # Manually trigger lifespan startup
        from contextlib import AsyncExitStack

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        await self._exit_stack.enter_async_context(self.storage)
        await self._exit_stack.enter_async_context(self.workspace_manager)

        # Flush Redis to ensure test isolation
        await self.storage.get_client().flushdb()

        # Attach lifespan-like state
        from agentscope.app._manager import (
            BackgroundTaskManager,
            SchedulerManager,
            SessionManager,
        )

        self.app.state.storage = self.storage
        self.app.state.workspace_manager = self.workspace_manager
        self.app.state.session_manager = SessionManager()
        self.app.state.background_task_manager = BackgroundTaskManager()
        scheduler = SchedulerManager(
            storage=self.storage,
            session_manager=self.app.state.session_manager,
            background_task_manager=self.app.state.background_task_manager,
            workspace_manager=self.workspace_manager,
        )
        self.app.state.scheduler_manager = scheduler
        await scheduler.start()

        transport = ASGITransport(app=self.app)
        self.client = AsyncClient(
            transport=transport,
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        try:
            self.app.state.session_manager.cancel()
            self.app.state.background_task_manager.cancel()
            await self.app.state.scheduler_manager.shutdown()
        except Exception:
            pass
        await self._exit_stack.__aexit__(None, None, None)
        self.temp_dir.cleanup()


# ── Agent CRUD tests ──────────────────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerAgentCRUD(DockerAgentServiceTestBase):
    """Test agent CRUD endpoints with Docker backend."""

    async def test_create_agent(self) -> None:
        """POST /agent/ creates a new agent and returns its id."""
        resp = await self.client.post(
            "/agent/",
            json={"name": "docker-agent", "system_prompt": "You help."},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("agent_id", data)
        self.assertTrue(len(data["agent_id"]) > 0)

    async def test_list_agents_empty(self) -> None:
        """GET /agent/ returns empty list when no agents exist."""
        resp = await self.client.get("/agent/", headers=_USER_HEADERS)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["agents"], [])
        self.assertEqual(data["total"], 0)

    async def test_list_agents_after_create(self) -> None:
        """GET /agent/ returns agents after creation."""
        await self.client.post(
            "/agent/",
            json={"name": "agent-1"},
            headers=_USER_HEADERS,
        )
        await self.client.post(
            "/agent/",
            json={"name": "agent-2"},
            headers=_USER_HEADERS,
        )
        resp = await self.client.get("/agent/", headers=_USER_HEADERS)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 2)
        names = {a["data"]["name"] for a in data["agents"]}
        self.assertEqual(names, {"agent-1", "agent-2"})

    async def test_update_agent(self) -> None:
        """PATCH /agent/{id} updates agent fields."""
        create_resp = await self.client.post(
            "/agent/",
            json={"name": "old-name", "system_prompt": "old prompt"},
            headers=_USER_HEADERS,
        )
        agent_id = create_resp.json()["agent_id"]

        patch_resp = await self.client.patch(
            f"/agent/{agent_id}",
            json={"name": "new-name", "system_prompt": "new prompt"},
            headers=_USER_HEADERS,
        )
        self.assertEqual(patch_resp.status_code, 200)
        updated = patch_resp.json()
        self.assertEqual(updated["data"]["name"], "new-name")
        self.assertEqual(updated["data"]["system_prompt"], "new prompt")

    async def test_update_agent_not_found(self) -> None:
        """PATCH /agent/{id} returns 404 for non-existent agent."""
        resp = await self.client.patch(
            "/agent/nonexistent",
            json={"name": "x"},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)

    async def test_delete_agent(self) -> None:
        """DELETE /agent/{id} removes the agent."""
        create_resp = await self.client.post(
            "/agent/",
            json={"name": "to-delete"},
            headers=_USER_HEADERS,
        )
        agent_id = create_resp.json()["agent_id"]

        del_resp = await self.client.delete(
            f"/agent/{agent_id}",
            headers=_USER_HEADERS,
        )
        self.assertEqual(del_resp.status_code, 204)

        # Confirm it's gone
        list_resp = await self.client.get("/agent/", headers=_USER_HEADERS)
        self.assertEqual(list_resp.json()["total"], 0)

    async def test_delete_agent_not_found(self) -> None:
        """DELETE /agent/{id} returns 404 for non-existent agent."""
        resp = await self.client.delete(
            "/agent/nonexistent",
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)


# ── Session CRUD tests ────────────────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerSessionCRUD(DockerAgentServiceTestBase):
    """Test session CRUD endpoints with Docker backend."""

    async def _create_agent(self, name: str = "test-agent") -> str:
        """Helper: create an agent and return its id."""
        resp = await self.client.post(
            "/agent/",
            json={"name": name},
            headers=_USER_HEADERS,
        )
        return resp.json()["agent_id"]

    async def test_create_session(self) -> None:
        """POST /sessions/ creates a new session."""
        agent_id = await self._create_agent()
        resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("session_id", data)
        self.assertTrue(len(data["session_id"]) > 0)

    async def test_create_session_with_workspace_id(self) -> None:
        """POST /sessions/ with explicit workspace_id."""
        agent_id = await self._create_agent()
        ws_id = uuid.uuid4().hex
        resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id, "workspace_id": ws_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 201)

    async def test_create_session_agent_not_found(self) -> None:
        """POST /sessions/ returns 404 if agent doesn't exist."""
        resp = await self.client.post(
            "/sessions/",
            json={"agent_id": "nonexistent"},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)

    async def test_list_sessions(self) -> None:
        """GET /sessions/?agent_id=... lists sessions for an agent."""
        agent_id = await self._create_agent()

        # Create two sessions with different workspace ids
        await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id, "workspace_id": "ws-1"},
            headers=_USER_HEADERS,
        )
        await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id, "workspace_id": "ws-2"},
            headers=_USER_HEADERS,
        )

        resp = await self.client.get(
            "/sessions/",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 2)

    async def test_list_sessions_agent_not_found(self) -> None:
        """GET /sessions/?agent_id=... returns 404 for unknown agent."""
        resp = await self.client.get(
            "/sessions/",
            params={"agent_id": "nonexistent"},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)

    async def test_delete_session(self) -> None:
        """DELETE /sessions/{id} removes the session."""
        agent_id = await self._create_agent()
        create_resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        session_id = create_resp.json()["session_id"]

        del_resp = await self.client.delete(
            f"/sessions/{session_id}",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(del_resp.status_code, 204)

    async def test_delete_session_not_found(self) -> None:
        """DELETE /sessions/{id} returns 404 for non-existent session."""
        agent_id = await self._create_agent()
        resp = await self.client.delete(
            "/sessions/nonexistent",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)

    async def test_list_messages_empty(self) -> None:
        """GET /sessions/{id}/messages returns empty list for new session."""
        agent_id = await self._create_agent()
        create_resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        session_id = create_resp.json()["session_id"]

        resp = await self.client.get(
            f"/sessions/{session_id}/messages",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["messages"], [])
        self.assertFalse(data["is_running"])

    async def test_list_messages_session_not_found(self) -> None:
        """GET /sessions/{id}/messages returns 404 for unknown session."""
        agent_id = await self._create_agent()
        resp = await self.client.get(
            "/sessions/nonexistent/messages",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)


# ── Workspace (MCP + skill) tests ─────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerWorkspaceEndpoints(DockerAgentServiceTestBase):
    """Test workspace MCP and skill endpoints with a Docker backend.

    These tests require an actual Docker daemon because they exercise the
    full workspace lifecycle: container creation, MCP gateway startup, and
    skill upload/listing via ``docker exec`` + ``put_archive``.
    """

    async def _setup_session(self) -> tuple[str, str]:
        """Helper: create an agent + session and return (agent_id, session_id)."""
        agent_resp = await self.client.post(
            "/agent/",
            json={"name": f"ws-agent-{uuid.uuid4().hex[:6]}"},
            headers=_USER_HEADERS,
        )
        agent_id = agent_resp.json()["agent_id"]

        session_resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        session_id = session_resp.json()["session_id"]
        return agent_id, session_id

    async def test_list_mcps_empty(self) -> None:
        """GET /workspace/mcp returns empty list for a new Docker workspace."""
        agent_id, session_id = await self._setup_session()
        resp = await self.client.get(
            "/workspace/mcp",
            params={"agent_id": agent_id, "session_id": session_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    async def test_list_skills_empty(self) -> None:
        """GET /workspace/skill returns empty list for a new workspace."""
        agent_id, session_id = await self._setup_session()
        resp = await self.client.get(
            "/workspace/skill",
            params={"agent_id": agent_id, "session_id": session_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    async def test_add_and_list_skill(self) -> None:
        """POST /workspace/skill uploads a skill; GET /workspace/skill lists it."""
        agent_id, session_id = await self._setup_session()

        # Create a skill directory on the host
        # pylint: disable=consider-using-with
        skill_dir = tempfile.TemporaryDirectory()
        try:
            skill_name = "docker_test_skill"
            skill_md_content = (
                f"---\nname: {skill_name}\n"
                f"description: A test skill for Docker\n---\n\n"
                f"# {skill_name}\n\nTest skill content.\n"
            )
            with open(
                os.path.join(skill_dir.name, "SKILL.md"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(skill_md_content)

            # Add the skill
            add_resp = await self.client.post(
                "/workspace/skill",
                json={"skill_path": skill_dir.name},
                params={"agent_id": agent_id, "session_id": session_id},
                headers=_USER_HEADERS,
            )
            self.assertEqual(add_resp.status_code, 201)

            # List and verify
            list_resp = await self.client.get(
                "/workspace/skill",
                params={"agent_id": agent_id, "session_id": session_id},
                headers=_USER_HEADERS,
            )
            self.assertEqual(list_resp.status_code, 200)
            skills = list_resp.json()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0]["name"], skill_name)
            self.assertEqual(
                skills[0]["description"],
                "A test skill for Docker",
            )
        finally:
            skill_dir.cleanup()

    async def test_remove_skill(self) -> None:
        """DELETE /workspace/skill/{name} removes a previously added skill."""
        agent_id, session_id = await self._setup_session()

        # Create and add a skill
        # pylint: disable=consider-using-with
        skill_dir = tempfile.TemporaryDirectory()
        try:
            skill_name = "removable_skill"
            with open(
                os.path.join(skill_dir.name, "SKILL.md"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"---\nname: {skill_name}\n"
                    f"description: To be removed\n---\n",
                )
            await self.client.post(
                "/workspace/skill",
                json={"skill_path": skill_dir.name},
                params={"agent_id": agent_id, "session_id": session_id},
                headers=_USER_HEADERS,
            )
        finally:
            skill_dir.cleanup()

        # Remove
        del_resp = await self.client.delete(
            f"/workspace/skill/{skill_name}",
            params={"agent_id": agent_id, "session_id": session_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(del_resp.status_code, 204)

        # Confirm removed
        list_resp = await self.client.get(
            "/workspace/skill",
            params={"agent_id": agent_id, "session_id": session_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(list_resp.json(), [])

    async def test_workspace_session_not_found(self) -> None:
        """Workspace endpoints return 404 when session doesn't exist."""
        agent_resp = await self.client.post(
            "/agent/",
            json={"name": "orphan-agent"},
            headers=_USER_HEADERS,
        )
        agent_id = agent_resp.json()["agent_id"]

        resp = await self.client.get(
            "/workspace/mcp",
            params={"agent_id": agent_id, "session_id": "nonexistent"},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 404)


# ── Workspace lifecycle tests ─────────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerWorkspaceLifecycle(DockerAgentServiceTestBase):
    """Test workspace lifecycle with Docker backend via the HTTP API.

    Validates that the Docker workspace manager correctly creates, caches,
    and cleans up workspaces through the service API.
    """

    async def test_workspace_created_on_first_access(self) -> None:
        """Accessing workspace endpoints for a session triggers container
        creation and the workspace is cached in the manager."""
        agent_resp = await self.client.post(
            "/agent/",
            json={"name": "lifecycle-agent"},
            headers=_USER_HEADERS,
        )
        agent_id = agent_resp.json()["agent_id"]
        session_resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        session_id = session_resp.json()["session_id"]

        # Before accessing workspace, the cache should be empty
        self.assertEqual(len(self.workspace_manager._cache), 0)

        # Access workspace endpoint — triggers container creation
        resp = await self.client.get(
            "/workspace/mcp",
            params={"agent_id": agent_id, "session_id": session_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(resp.status_code, 200)

        # Now the workspace manager should have cached the workspace
        self.assertEqual(len(self.workspace_manager._cache), 1)

    async def test_multiple_sessions_same_agent_share_workspace(self) -> None:
        """Two sessions for the same agent share the same workspace (via
        the workspace_id upsert mechanism)."""
        agent_resp = await self.client.post(
            "/agent/",
            json={"name": "shared-agent"},
            headers=_USER_HEADERS,
        )
        agent_id = agent_resp.json()["agent_id"]

        # Create two sessions with the same workspace id
        ws_id = uuid.uuid4().hex
        s1 = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id, "workspace_id": ws_id},
            headers=_USER_HEADERS,
        )
        s2 = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id, "workspace_id": ws_id},
            headers=_USER_HEADERS,
        )
        session_id_1 = s1.json()["session_id"]
        session_id_2 = s2.json()["session_id"]

        # Access workspace for first session
        await self.client.get(
            "/workspace/mcp",
            params={"agent_id": agent_id, "session_id": session_id_1},
            headers=_USER_HEADERS,
        )
        # The second session with the same workspace_id should hit the cache
        await self.client.get(
            "/workspace/mcp",
            params={"agent_id": agent_id, "session_id": session_id_2},
            headers=_USER_HEADERS,
        )

        # Only one workspace should be cached (same workspace_id)
        self.assertEqual(len(self.workspace_manager._cache), 1)


# ── Cross-cutting tests ──────────────────────────────────────────


@unittest.skipUnless(_DOCKER_OK, _SKIP_REASON)
class TestDockerCrossCutting(DockerAgentServiceTestBase):
    """Cross-cutting tests for the agent service with Docker backend."""

    async def test_user_isolation(self) -> None:
        """Agents created by one user are not visible to another."""
        user1_headers = {"X-User-ID": "user-1"}
        user2_headers = {"X-User-ID": "user-2"}

        await self.client.post(
            "/agent/",
            json={"name": "user1-agent"},
            headers=user1_headers,
        )
        await self.client.post(
            "/agent/",
            json={"name": "user2-agent"},
            headers=user2_headers,
        )

        resp1 = await self.client.get("/agent/", headers=user1_headers)
        resp2 = await self.client.get("/agent/", headers=user2_headers)

        self.assertEqual(resp1.json()["total"], 1)
        self.assertEqual(
            resp1.json()["agents"][0]["data"]["name"], "user1-agent"
        )
        self.assertEqual(resp2.json()["total"], 1)
        self.assertEqual(
            resp2.json()["agents"][0]["data"]["name"], "user2-agent"
        )

    async def test_agent_full_lifecycle(self) -> None:
        """Full agent lifecycle: create -> update -> list -> delete -> list."""
        # Create
        create_resp = await self.client.post(
            "/agent/",
            json={"name": "lifecycle", "system_prompt": "v1"},
            headers=_USER_HEADERS,
        )
        agent_id = create_resp.json()["agent_id"]

        # Update
        await self.client.patch(
            f"/agent/{agent_id}",
            json={"system_prompt": "v2"},
            headers=_USER_HEADERS,
        )

        # List — verify updated
        list_resp = await self.client.get("/agent/", headers=_USER_HEADERS)
        agent = list_resp.json()["agents"][0]
        self.assertEqual(agent["data"]["system_prompt"], "v2")

        # Delete
        await self.client.delete(f"/agent/{agent_id}", headers=_USER_HEADERS)

        # List — should be empty
        final = await self.client.get("/agent/", headers=_USER_HEADERS)
        self.assertEqual(final.json()["total"], 0)

    async def test_session_full_lifecycle(self) -> None:
        """Full session lifecycle: create agent -> create session ->
        list messages -> delete session -> delete agent."""
        # Create agent
        agent_resp = await self.client.post(
            "/agent/",
            json={"name": "session-lifecycle-agent"},
            headers=_USER_HEADERS,
        )
        agent_id = agent_resp.json()["agent_id"]

        # Create session
        sess_resp = await self.client.post(
            "/sessions/",
            json={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        session_id = sess_resp.json()["session_id"]

        # List messages — empty
        msg_resp = await self.client.get(
            f"/sessions/{session_id}/messages",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(msg_resp.status_code, 200)
        self.assertEqual(msg_resp.json()["messages"], [])

        # Delete session
        del_sess = await self.client.delete(
            f"/sessions/{session_id}",
            params={"agent_id": agent_id},
            headers=_USER_HEADERS,
        )
        self.assertEqual(del_sess.status_code, 204)

        # Delete agent
        del_agent = await self.client.delete(
            f"/agent/{agent_id}",
            headers=_USER_HEADERS,
        )
        self.assertEqual(del_agent.status_code, 204)


if __name__ == "__main__":
    unittest.main()
