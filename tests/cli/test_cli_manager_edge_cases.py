import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_register_real_session_id_moves_pending_to_active_and_maps():
    from cli.manager import CLISessionManager

    with patch("cli.manager.CLISession") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.is_busy = False
        mock_session.stop = AsyncMock(return_value=True)
        mock_session_cls.return_value = mock_session

        manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")
        session, temp_id, is_new = await manager.get_or_create_session()
        assert session is mock_session
        assert is_new is True

        ok = await manager.register_real_session_id(temp_id, "real_1")
        assert ok is True

        real = await manager.get_real_session_id(temp_id)
        assert real == "real_1"

        # Lookup via temp id should resolve to the real session id.
        s2, sid2, is_new2 = await manager.get_or_create_session(session_id=temp_id)
        assert s2 is mock_session
        assert sid2 == "real_1"
        assert is_new2 is False


@pytest.mark.asyncio
async def test_register_real_session_id_missing_temp_id_returns_false():
    from cli.manager import CLISessionManager

    manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")
    ok = await manager.register_real_session_id("missing", "real_1")
    assert ok is False


@pytest.mark.asyncio
async def test_remove_session_pending_stops_and_returns_true():
    from cli.manager import CLISessionManager

    with patch("cli.manager.CLISession") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.is_busy = False
        mock_session.stop = AsyncMock(return_value=True)
        mock_session_cls.return_value = mock_session

        manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")
        _, temp_id, _ = await manager.get_or_create_session()

        removed = await manager.remove_session(temp_id)
        assert removed is True
        mock_session.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_session_active_removes_temp_mapping():
    from cli.manager import CLISessionManager

    with patch("cli.manager.CLISession") as mock_session_cls:
        mock_session = MagicMock()
        mock_session.is_busy = False
        mock_session.stop = AsyncMock(return_value=True)
        mock_session_cls.return_value = mock_session

        manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")
        _, temp_id, _ = await manager.get_or_create_session()
        await manager.register_real_session_id(temp_id, "real_1")

        removed = await manager.remove_session("real_1")
        assert removed is True

        real = await manager.get_real_session_id(temp_id)
        assert real is None


@pytest.mark.asyncio
async def test_get_or_create_session_cleans_up_idle_sessions_when_at_limit():
    from cli.manager import CLISessionManager

    manager = CLISessionManager(
        workspace_path="/tmp", api_url="http://x/v1", max_sessions=1
    )

    idle_session = MagicMock()
    idle_session.is_busy = False
    idle_session.stop = AsyncMock(return_value=True)
    manager._sessions["s1"] = idle_session

    with patch("cli.manager.CLISession") as mock_session_cls:
        new_session = MagicMock()
        new_session.is_busy = False
        new_session.stop = AsyncMock(return_value=True)
        mock_session_cls.return_value = new_session

        s2, sid2, is_new2 = await manager.get_or_create_session()
        assert s2 is new_session
        assert sid2.startswith("pending_")
        assert is_new2 is True

    idle_session.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_or_create_session_raises_when_max_sessions_reached_with_busy_sessions():
    from cli.manager import CLISessionManager

    manager = CLISessionManager(
        workspace_path="/tmp", api_url="http://x/v1", max_sessions=1
    )

    busy_session = MagicMock()
    busy_session.is_busy = True
    busy_session.stop = AsyncMock(return_value=True)
    manager._sessions["s1"] = busy_session

    with pytest.raises(RuntimeError):
        await manager.get_or_create_session()


@pytest.mark.asyncio
async def test_stop_all_handles_stop_exceptions():
    from cli.manager import CLISessionManager

    manager = CLISessionManager(workspace_path="/tmp", api_url="http://x/v1")

    s1 = MagicMock()
    s1.stop = AsyncMock(side_effect=RuntimeError("boom"))
    s1.is_busy = False

    s2 = MagicMock()
    s2.stop = AsyncMock(return_value=True)
    s2.is_busy = False

    manager._sessions["a"] = s1
    manager._pending_sessions["b"] = s2

    await manager.stop_all()
    s1.stop.assert_awaited_once()
    s2.stop.assert_awaited_once()
    assert manager.get_stats()["active_sessions"] == 0
    assert manager.get_stats()["pending_sessions"] == 0
