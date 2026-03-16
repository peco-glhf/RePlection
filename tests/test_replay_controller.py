"""ReplayStateController のユニットテスト.

Replay API への実接続は行わない。
- requests.Session をモックして HTTP リクエストの内容を検証する
- desired_state の更新・reconcile ロジックを確認する
- 接続不可時の縮退動作（crash しない）を確認する
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from sidecar.replay_controller import (
    REPLAY_API_BASE,
    SPEED_MAX,
    SPEED_MIN,
    TIME_CORRECTION_THRESHOLD,
    ReplayStateController,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def make_mock_session(
    get_response: dict | None = None,
    post_response: dict | None = None,
    connection_error: bool = False,
) -> MagicMock:
    """HTTP セッションのモックを生成する.

    Args:
        get_response: GET /replay/playback の返却 JSON。None の場合デフォルト状態を返す。
        post_response: POST /replay/playback の返却 JSON。None の場合 {} を返す。
        connection_error: True の場合 ConnectionError を発生させる。
    """
    session = MagicMock(spec=requests.Session)

    if connection_error:
        session.get.side_effect = requests.exceptions.ConnectionError("接続不可")
        session.post.side_effect = requests.exceptions.ConnectionError("接続不可")
        return session

    # GET レスポンス
    get_resp = MagicMock()
    get_resp.json.return_value = get_response or {
        "time": 0.0,
        "paused": False,
        "speed": 1.0,
        "length": 1200.0,
        "seeking": False,
    }
    get_resp.raise_for_status.return_value = None
    session.get.return_value = get_resp

    # POST レスポンス
    post_resp = MagicMock()
    post_resp.json.return_value = post_response or {}
    post_resp.raise_for_status.return_value = None
    session.post.return_value = post_resp

    return session


def make_controller(
    get_response: dict | None = None,
    post_response: dict | None = None,
    connection_error: bool = False,
) -> tuple[ReplayStateController, MagicMock]:
    """テスト用 ReplayStateController とモックセッションを生成する."""
    mock_session = make_mock_session(get_response, post_response, connection_error)
    controller = ReplayStateController(http_session=mock_session)
    return controller, mock_session


@pytest.fixture(autouse=True)
def run_to_thread_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """ユニットテストでは to_thread を同期実行に置き換える."""

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)


# ---------------------------------------------------------------------------
# 定数テスト
# ---------------------------------------------------------------------------


class TestConstants:
    def test_replay_api_base(self) -> None:
        assert "localhost" in REPLAY_API_BASE or "127.0.0.1" in REPLAY_API_BASE

    def test_time_threshold_is_half_second(self) -> None:
        assert TIME_CORRECTION_THRESHOLD == 0.5

    def test_speed_bounds(self) -> None:
        assert SPEED_MIN < 1.0 < SPEED_MAX


# ---------------------------------------------------------------------------
# update_desired テスト
# ---------------------------------------------------------------------------


class TestUpdateDesired:
    def test_update_time_seconds(self) -> None:
        controller, _ = make_controller()
        controller.update_desired(time_seconds=941.0)
        assert controller._desired.time_seconds == 941.0

    def test_update_paused_true(self) -> None:
        controller, _ = make_controller()
        controller.update_desired(paused=True)
        assert controller._desired.paused is True

    def test_update_paused_false(self) -> None:
        controller, _ = make_controller()
        controller.update_desired(paused=False)
        assert controller._desired.paused is False

    def test_update_speed(self) -> None:
        controller, _ = make_controller()
        controller.update_desired(speed=0.5)
        assert controller._desired.speed == 0.5

    def test_update_multiple_fields(self) -> None:
        controller, _ = make_controller()
        controller.update_desired(time_seconds=100.0, paused=True, speed=0.25)
        assert controller._desired.time_seconds == 100.0
        assert controller._desired.paused is True
        assert controller._desired.speed == 0.25

    def test_update_unknown_field_does_not_raise(self) -> None:
        controller, _ = make_controller()
        controller.update_desired(nonexistent_field=999)  # 例外なし


# ---------------------------------------------------------------------------
# _build_payload テスト
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_seek_replay(self) -> None:
        payload = ReplayStateController._build_payload(
            "seek_replay", {"time_seconds": 941.0}
        )
        assert payload == {"time": 941.0}

    def test_seek_replay_default_time(self) -> None:
        payload = ReplayStateController._build_payload("seek_replay", {})
        assert payload == {"time": 0.0}

    def test_pause_replay(self) -> None:
        payload = ReplayStateController._build_payload("pause_replay", {})
        assert payload == {"paused": True}

    def test_resume_replay(self) -> None:
        payload = ReplayStateController._build_payload("resume_replay", {})
        assert payload == {"paused": False}

    def test_slow_motion(self) -> None:
        payload = ReplayStateController._build_payload("slow_motion", {"speed": 0.5})
        assert payload == {"speed": 0.5}

    def test_slow_motion_clamps_to_min(self) -> None:
        payload = ReplayStateController._build_payload(
            "slow_motion", {"speed": 0.001}
        )
        assert payload is not None
        assert payload["speed"] == SPEED_MIN

    def test_slow_motion_clamps_to_max(self) -> None:
        payload = ReplayStateController._build_payload(
            "slow_motion", {"speed": 100.0}
        )
        assert payload is not None
        assert payload["speed"] == SPEED_MAX

    def test_unknown_tool_returns_none(self) -> None:
        payload = ReplayStateController._build_payload("unknown_tool", {})
        assert payload is None


# ---------------------------------------------------------------------------
# _call_replay_api テスト（HTTP リクエスト検証）
# ---------------------------------------------------------------------------


class TestCallReplayApi:
    def test_seek_sends_correct_post(self) -> None:
        """seek_replay が正しい URL・payload で POST することを確認."""
        controller, mock_session = make_controller()
        controller._call_replay_api("seek_replay", {"time_seconds": 941.0})

        mock_session.post.assert_called_once_with(
            f"{REPLAY_API_BASE}/replay/playback",
            json={"time": 941.0},
            timeout=(0.5, 1.0),
        )

    def test_pause_sends_correct_post(self) -> None:
        controller, mock_session = make_controller()
        controller._call_replay_api("pause_replay", {})
        mock_session.post.assert_called_once_with(
            f"{REPLAY_API_BASE}/replay/playback",
            json={"paused": True},
            timeout=(0.5, 1.0),
        )

    def test_resume_sends_correct_post(self) -> None:
        controller, mock_session = make_controller()
        controller._call_replay_api("resume_replay", {})
        mock_session.post.assert_called_once_with(
            f"{REPLAY_API_BASE}/replay/playback",
            json={"paused": False},
            timeout=(0.5, 1.0),
        )

    def test_slow_motion_sends_correct_post(self) -> None:
        controller, mock_session = make_controller()
        controller._call_replay_api("slow_motion", {"speed": 0.25})
        mock_session.post.assert_called_once_with(
            f"{REPLAY_API_BASE}/replay/playback",
            json={"speed": 0.25},
            timeout=(0.5, 1.0),
        )

    def test_connection_error_raises_runtime_error(self) -> None:
        """接続不可時に呼び出し元へ失敗を伝播することを確認."""
        controller, _ = make_controller(connection_error=True)
        with pytest.raises(RuntimeError, match="Replay API 接続不可"):
            controller._call_replay_api("pause_replay", {})
        assert not controller.available

    def test_timeout_raises_runtime_error_and_marks_unavailable(self) -> None:
        """タイムアウトを接続不可として扱うことを確認."""
        controller, mock_session = make_controller()
        mock_session.post.side_effect = requests.exceptions.Timeout("timeout")

        with pytest.raises(RuntimeError, match="Replay API 接続不可"):
            controller._call_replay_api("pause_replay", {})

        assert not controller.available

    def test_unknown_tool_does_not_send(self) -> None:
        controller, mock_session = make_controller()
        controller._call_replay_api("invalid_tool", {})
        mock_session.post.assert_not_called()


# ---------------------------------------------------------------------------
# _get_actual_state テスト
# ---------------------------------------------------------------------------


class TestGetActualState:
    def test_returns_state_dict(self) -> None:
        state = {"time": 123.45, "paused": False, "speed": 1.0, "length": 1800.0}
        controller, _ = make_controller(get_response=state)
        result = controller._get_actual_state()
        assert result["time"] == 123.45
        assert result["paused"] is False

    def test_sends_get_to_correct_url(self) -> None:
        controller, mock_session = make_controller()
        controller._get_actual_state()
        mock_session.get.assert_called_once_with(
            f"{REPLAY_API_BASE}/replay/playback", timeout=(0.5, 1.0)
        )

    def test_connection_error_returns_empty_dict(self) -> None:
        controller, _ = make_controller(connection_error=True)
        result = controller._get_actual_state()
        assert result == {}
        assert not controller.available

    def test_connection_error_does_not_raise(self) -> None:
        controller, _ = make_controller(connection_error=True)
        controller._get_actual_state()  # 例外なし

    def test_timeout_returns_empty_dict_and_marks_unavailable(self) -> None:
        """actual_state 取得タイムアウト時は縮退動作に入ることを確認."""
        controller, mock_session = make_controller()
        mock_session.get.side_effect = requests.exceptions.Timeout("timeout")

        result = controller._get_actual_state()

        assert result == {}
        assert not controller.available


class TestApplyCorrections:
    def test_sends_post_with_short_timeout(self) -> None:
        """補正 POST も短い timeout 設定を使うことを確認."""
        controller, mock_session = make_controller()

        controller._apply_corrections({"paused": True})

        mock_session.post.assert_called_once_with(
            f"{REPLAY_API_BASE}/replay/playback",
            json={"paused": True},
            timeout=(0.5, 1.0),
        )


# ---------------------------------------------------------------------------
# apply_and_reconcile テスト
# ---------------------------------------------------------------------------


class TestApplyAndReconcile:
    @pytest.mark.asyncio
    async def test_seek_sends_post(self) -> None:
        """seek_replay が POST を送信することを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 941.0, "paused": False, "speed": 1.0}
        )
        await controller.apply_and_reconcile("seek_replay", {"time_seconds": 941.0})
        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["json"] == {"time": 941.0}

    @pytest.mark.asyncio
    async def test_connection_error_raises_runtime_error(self) -> None:
        """Replay API 接続不可時に apply_and_reconcile が失敗を伝播することを確認."""
        controller, _ = make_controller(connection_error=True)
        with pytest.raises(RuntimeError, match="Replay API 接続不可"):
            await controller.apply_and_reconcile("pause_replay", {})


# ---------------------------------------------------------------------------
# reconcile_now テスト
# ---------------------------------------------------------------------------


class TestReconcileNow:
    @pytest.mark.asyncio
    async def test_reconcile_corrects_paused_mismatch(self) -> None:
        """desired.paused=True に対して actual.paused=False の場合、補正 POST を送信することを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 0.0, "paused": False, "speed": 1.0}
        )
        controller.update_desired(paused=True)
        await controller.reconcile_now()

        # 補正 POST が送信されること
        assert mock_session.post.called
        correction = mock_session.post.call_args[1]["json"]
        assert correction.get("paused") is True

    @pytest.mark.asyncio
    async def test_reconcile_corrects_time_mismatch(self) -> None:
        """desired.time_seconds=941.0 に対して actual.time=0.0 の場合（閾値超過）、補正 POST を送信することを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 0.0, "paused": False, "speed": 1.0}
        )
        controller.update_desired(time_seconds=941.0)
        await controller.reconcile_now()

        assert mock_session.post.called
        correction = mock_session.post.call_args[1]["json"]
        assert correction.get("time") == 941.0

    @pytest.mark.asyncio
    async def test_reconcile_no_correction_within_threshold(self) -> None:
        """時刻のズレが閾値以内の場合、補正 POST を送信しないことを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 941.2, "paused": False, "speed": 1.0}
        )
        # ズレ 0.2 秒（閾値 0.5 秒未満）
        controller.update_desired(time_seconds=941.0)
        await controller.reconcile_now()

        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_no_correction_when_state_matches(self) -> None:
        """desired_state と actual_state が一致している場合、補正 POST を送信しないことを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 941.0, "paused": True, "speed": 1.0}
        )
        controller.update_desired(time_seconds=941.0, paused=True)
        await controller.reconcile_now()

        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_connection_error_does_not_raise(self) -> None:
        """Replay API 接続不可時に reconcile_now が例外を送出しないことを確認."""
        controller, _ = make_controller(connection_error=True)
        await controller.reconcile_now()  # 例外なし


# ---------------------------------------------------------------------------
# DoD-3 統合シナリオ: AI が「15:41 を見て」→ リプレイが 15:41 にシーク
# ---------------------------------------------------------------------------


class TestDod3Scenario:
    """DoD-3: AI が「15:41 を見て」→ リプレイが 15:41（941秒）にシークする."""

    @pytest.mark.asyncio
    async def test_seek_to_941_seconds(self) -> None:
        """seek_replay(941) が POST /replay/playback {"time": 941} を送信することを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 941.0, "paused": False, "speed": 1.0}
        )
        # desired_state を更新してから apply_and_reconcile
        controller.update_desired(time_seconds=941.0)
        await controller.apply_and_reconcile("seek_replay", {"time_seconds": 941.0})

        # POST が送信されていること
        mock_session.post.assert_called()
        first_call = mock_session.post.call_args_list[0]
        assert first_call[1]["json"] == {"time": 941.0}
        assert first_call[0][0] == f"{REPLAY_API_BASE}/replay/playback"

    @pytest.mark.asyncio
    async def test_seek_and_pause_workflow(self) -> None:
        """シーク後に一時停止する典型的なワークフローを確認."""
        controller, mock_session = make_controller(
            get_response={"time": 941.0, "paused": True, "speed": 1.0}
        )
        # 1. シーク
        controller.update_desired(time_seconds=941.0)
        await controller.apply_and_reconcile("seek_replay", {"time_seconds": 941.0})
        # 2. 一時停止
        controller.update_desired(paused=True)
        await controller.apply_and_reconcile("pause_replay", {})

        # POST が 2 回以上呼ばれること
        assert mock_session.post.call_count >= 2
