"""RealtimeEngine のユニットテスト.

Gemini Live API への実接続は行わない。
- sounddevice 縮退モード（WSL 環境など）での動作
- AudioPlayer / AudioCapture の単体動作
- 状態遷移ロジック（モック session を使用）
- barge-in 時のバッファ flush と状態遷移

Windows 実機でのみ検証が必要な項目は test_windows_* または末尾にコメントで明記する。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from google.genai import types

from sidecar.models import DialogueState, SubtitleEvent
from sidecar.realtime_engine import (
    CHUNK_FRAMES,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    AudioCapture,
    AudioPlayer,
    RealtimeEngine,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def make_engine(
    system_instruction: str = "テストコーチ",
    on_subtitle: Callable[[SubtitleEvent], None] | None = None,
    on_state_change: Callable[[DialogueState], None] | None = None,
    replay_controller: Any | None = None,
) -> RealtimeEngine:
    """テスト用 RealtimeEngine を生成する."""
    return RealtimeEngine(
        system_instruction=system_instruction,
        on_subtitle=on_subtitle or (lambda _: None),
        on_state_change=on_state_change or (lambda _: None),
        replay_controller=replay_controller,
    )


# ---------------------------------------------------------------------------
# 定数テスト
# ---------------------------------------------------------------------------


class TestConstants:
    """定数値が仕様通りか確認する."""

    def test_input_sample_rate(self) -> None:
        assert INPUT_SAMPLE_RATE == 16_000

    def test_output_sample_rate(self) -> None:
        assert OUTPUT_SAMPLE_RATE == 24_000

    def test_chunk_frames_is_40ms(self) -> None:
        # 40ms @ 16kHz = 640 サンプル
        assert CHUNK_FRAMES == int(INPUT_SAMPLE_RATE * 0.04)


# ---------------------------------------------------------------------------
# AudioPlayer テスト（縮退モード）
# ---------------------------------------------------------------------------


class TestAudioPlayerDegraded:
    """sounddevice なし環境（縮退モード）での AudioPlayer テスト."""

    def setup_method(self) -> None:
        # sounddevice を利用不可として AudioPlayer をテスト
        self.player = AudioPlayer()

    def test_start_does_not_raise_without_sounddevice(self) -> None:
        with patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", False):
            player = AudioPlayer()
            player.start()  # 例外なし
            assert not player.active

    def test_write_does_not_raise_without_sounddevice(self) -> None:
        with patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", False):
            player = AudioPlayer()
            player.write(b"\x00" * 48)  # 例外なし

    def test_clear_and_stop_does_not_raise_without_sounddevice(self) -> None:
        with patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", False):
            player = AudioPlayer()
            player.clear_and_stop()  # 例外なし

    def test_close_does_not_raise_without_sounddevice(self) -> None:
        with patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", False):
            player = AudioPlayer()
            player.close()  # 例外なし

    def test_active_false_when_stream_not_started(self) -> None:
        player = AudioPlayer()
        # start() 未呼び出し
        assert not player.active


class TestAudioPlayerRecovery:
    """sounddevice 利用時の再生ストリーム復旧を確認する."""

    def test_clear_and_stop_detaches_stream_after_abort(self) -> None:
        stream = MagicMock()
        sounddevice = SimpleNamespace(RawOutputStream=MagicMock(return_value=stream))

        with (
            patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", True),
            patch("sidecar.realtime_engine._sd", sounddevice),
        ):
            player = AudioPlayer()
            player.start()
            stream.abort.reset_mock()
            stream.start.reset_mock()

            player.clear_and_stop()

            stream.abort.assert_called_once()
            stream.stop.assert_not_called()
            stream.close.assert_called_once()
            stream.start.assert_not_called()
            assert player._stream is None
            assert not player.active

    def test_write_recreates_stream_after_write_error(self) -> None:
        first_stream = MagicMock()
        second_stream = MagicMock()
        first_stream.write.side_effect = RuntimeError("Stream is stopped")
        sounddevice = SimpleNamespace(
            RawOutputStream=MagicMock(side_effect=[first_stream, second_stream])
        )

        with (
            patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", True),
            patch("sidecar.realtime_engine._sd", sounddevice),
        ):
            player = AudioPlayer()
            player.start()

            player.write(b"\x00\x01")

            assert sounddevice.RawOutputStream.call_count == 2
            first_stream.write.assert_called_once_with(b"\x00\x01")
            second_stream.start.assert_called_once()
            second_stream.write.assert_called_once_with(b"\x00\x01")

    def test_write_after_clear_and_stop_creates_new_stream(self) -> None:
        first_stream = MagicMock()
        second_stream = MagicMock()
        sounddevice = SimpleNamespace(
            RawOutputStream=MagicMock(side_effect=[first_stream, second_stream])
        )

        with (
            patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", True),
            patch("sidecar.realtime_engine._sd", sounddevice),
        ):
            player = AudioPlayer()
            player.start()

            player.clear_and_stop()
            player.write(b"\x00\x01")

            assert sounddevice.RawOutputStream.call_count == 2
            first_stream.write.assert_not_called()
            second_stream.start.assert_called_once()
            second_stream.write.assert_called_once_with(b"\x00\x01")

    def test_write_does_not_retry_interrupted_chunk(self) -> None:
        stream = MagicMock()
        sounddevice = SimpleNamespace(RawOutputStream=MagicMock(return_value=stream))

        with (
            patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", True),
            patch("sidecar.realtime_engine._sd", sounddevice),
        ):
            player = AudioPlayer()
            player.start()

            def interrupted_write(_pcm_data: bytes) -> None:
                player.clear_and_stop()
                raise RuntimeError("割り込みで停止")

            stream.write.side_effect = interrupted_write

            player.write(b"\x00\x01")

            # 割り込みで落ちたバッチは再試行せず、そのまま捨てる。
            assert sounddevice.RawOutputStream.call_count == 1

    def test_clear_and_stop_does_not_touch_stream_while_write_is_running(self) -> None:
        """別スレッドの write 中に abort/close が並行実行されないことを確認."""

        class BlockingStream:
            def __init__(self) -> None:
                self.write_started = threading.Event()
                self.allow_write_finish = threading.Event()
                self.concurrent_abort = threading.Event()
                self.concurrent_close = threading.Event()
                self._write_active = False
                self._guard = threading.Lock()

            def start(self) -> None:
                return None

            def write(self, _pcm_data: bytes) -> None:
                with self._guard:
                    self._write_active = True
                self.write_started.set()
                self.allow_write_finish.wait(timeout=1.0)
                with self._guard:
                    self._write_active = False

            def abort(self) -> None:
                with self._guard:
                    if self._write_active:
                        self.concurrent_abort.set()

            def close(self) -> None:
                with self._guard:
                    if self._write_active:
                        self.concurrent_close.set()

        stream = BlockingStream()
        sounddevice = SimpleNamespace(RawOutputStream=MagicMock(return_value=stream))

        with (
            patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", True),
            patch("sidecar.realtime_engine._sd", sounddevice),
        ):
            player = AudioPlayer()
            player.start()

            write_thread = threading.Thread(
                target=player.write,
                args=(b"\x00\x01",),
            )
            stop_thread = threading.Thread(target=player.clear_and_stop)

            write_thread.start()
            assert stream.write_started.wait(timeout=0.5) is True
            stop_thread.start()
            time.sleep(0.05)
            stream.allow_write_finish.set()

            write_thread.join(timeout=1.0)
            stop_thread.join(timeout=1.0)

            assert stream.concurrent_abort.is_set() is False
            assert stream.concurrent_close.is_set() is False


# ---------------------------------------------------------------------------
# AudioCapture テスト（縮退モード）
# ---------------------------------------------------------------------------


class TestAudioCaptureDegraded:
    """sounddevice なし環境（縮退モード）での AudioCapture テスト."""

    def test_start_does_not_raise_without_sounddevice(self) -> None:
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        with patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", False):
            cap = AudioCapture(queue)
            cap.start()  # 例外なし
            assert not cap.active

    def test_stop_does_not_raise_without_sounddevice(self) -> None:
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        with patch("sidecar.realtime_engine._SOUNDDEVICE_AVAILABLE", False):
            cap = AudioCapture(queue)
            cap.stop()  # 例外なし

    def test_callback_puts_bytes_to_queue(self) -> None:
        """コールバックが put_nowait でキューにデータを追加することを確認."""
        loop = asyncio.new_event_loop()
        try:
            queue: asyncio.Queue[bytes] = asyncio.Queue()
            cap = AudioCapture(queue)
            dummy_data = b"\x01\x02" * 640  # 640 frames × 2 bytes
            # コールバックを直接呼び出す（await なし）
            cap._callback(dummy_data, CHUNK_FRAMES, None, None)
            assert queue.qsize() == 1
            chunk = queue.get_nowait()
            assert chunk == dummy_data
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# RealtimeEngine 初期状態テスト
# ---------------------------------------------------------------------------


class TestRealtimeEngineInit:
    """RealtimeEngine の初期化テスト."""

    def test_initial_state_is_idle(self) -> None:
        engine = make_engine()
        assert engine.state == DialogueState.IDLE

    def test_system_instruction_stored(self) -> None:
        engine = make_engine(system_instruction="カスタム instruction")
        assert engine._system_instruction == "カスタム instruction"

    def test_callbacks_stored(self) -> None:
        on_sub = MagicMock()
        on_state = MagicMock()
        engine = RealtimeEngine(
            system_instruction="test",
            on_subtitle=on_sub,
            on_state_change=on_state,
        )
        assert engine._on_subtitle is on_sub
        assert engine._on_state_change is on_state

    def test_playback_queue_is_unbounded(self) -> None:
        engine = make_engine()
        assert engine._playback_queue.maxsize == 0


# ---------------------------------------------------------------------------
# 状態遷移テスト
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """_set_state の状態遷移とコールバック呼び出しを確認する."""

    def test_set_state_updates_state(self) -> None:
        engine = make_engine()
        engine._set_state(DialogueState.LISTENING)
        assert engine.state == DialogueState.LISTENING

    def test_set_state_calls_callback(self) -> None:
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        engine._set_state(DialogueState.LISTENING)
        assert DialogueState.LISTENING in states

    def test_set_same_state_does_not_call_callback(self) -> None:
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        # IDLE → IDLE（変化なし）
        engine._set_state(DialogueState.IDLE)
        assert len(states) == 0

    def test_full_transition_sequence(self) -> None:
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        engine._set_state(DialogueState.LISTENING)
        engine._set_state(DialogueState.USER_SPEAKING)
        engine._set_state(DialogueState.SPEAKING)
        engine._set_state(DialogueState.INTERRUPTED)
        engine._set_state(DialogueState.LISTENING)
        assert states == [
            DialogueState.LISTENING,
            DialogueState.USER_SPEAKING,
            DialogueState.SPEAKING,
            DialogueState.INTERRUPTED,
            DialogueState.LISTENING,
        ]


# ---------------------------------------------------------------------------
# barge-in (flush) テスト
# ---------------------------------------------------------------------------


class TestFlushPlaybackQueue:
    """再生キュー flush 動作を確認する."""

    def test_flush_playback_queue_clears_all_items(self) -> None:
        engine = make_engine()
        for _ in range(5):
            engine._playback_queue.put_nowait(b"\x00" * 1280)
        assert engine._playback_queue.qsize() == 5
        engine._flush_playback_queue()
        assert engine._playback_queue.qsize() == 0

    def test_flush_empty_playback_queue_does_not_raise(self) -> None:
        engine = make_engine()
        engine._flush_playback_queue()  # 例外なし


class TestPlaybackQueue:
    """再生キューの制御を確認する."""

    def test_enqueue_playback_chunk_keeps_all_items_when_queue_is_unbounded(self) -> None:
        engine = make_engine()
        engine._enqueue_playback_chunk(b"\x01")
        engine._enqueue_playback_chunk(b"\x02")
        engine._enqueue_playback_chunk(b"\xff")

        items = list(engine._playback_queue._queue)
        assert items == [b"\x01", b"\x02", b"\xff"]

    @pytest.mark.asyncio
    async def test_collect_playback_batch_joins_ready_chunks(self) -> None:
        engine = make_engine()
        engine._playback_queue.put_nowait(b"\x02")
        engine._playback_queue.put_nowait(b"\x03")

        batch = await engine._collect_playback_batch(b"\x01")

        assert batch == b"\x01\x02\x03"


# ---------------------------------------------------------------------------
# barge-in 状態遷移テスト（モック session）
# ---------------------------------------------------------------------------


class TestBargeinHandling:
    """barge-in 受信時の状態遷移を確認する（受信ループ単体テスト）."""

    @staticmethod
    def _make_interrupted_response() -> MagicMock:
        """interrupted=True の server_content を持つモックレスポンスを生成する."""
        interrupted_content = MagicMock()
        interrupted_content.interrupted = True
        interrupted_content.model_turn = None
        interrupted_content.output_transcription = None
        interrupted_content.input_transcription = None
        interrupted_content.turn_complete = False

        mock_response = MagicMock()
        mock_response.server_content = interrupted_content
        mock_response.go_away = None
        mock_response.session_resumption_update = None
        return mock_response

    @staticmethod
    def _make_audio_response(*chunks: bytes) -> MagicMock:
        content = MagicMock()
        content.interrupted = False
        content.output_transcription = None
        content.input_transcription = None
        content.turn_complete = False

        model_turn = MagicMock()
        model_turn.parts = []
        for chunk in chunks:
            inline_data = MagicMock()
            inline_data.data = chunk

            part = MagicMock()
            part.inline_data = inline_data
            model_turn.parts.append(part)
        content.model_turn = model_turn

        response = MagicMock()
        response.server_content = content
        response.go_away = None
        response.session_resumption_update = None
        return response

    @pytest.mark.asyncio
    async def test_interrupted_transitions_to_listening(self) -> None:
        """interrupted フラグ付きのレスポンスで INTERRUPTED → LISTENING に遷移することを確認."""
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        engine._set_state(DialogueState.SPEAKING)
        states.clear()  # SPEAKING の記録をリセット

        mock_response = self._make_interrupted_response()

        # 1件だけ yield するジェネレータ（stop_event なしで自然終了）
        async def mock_receive() -> Any:
            yield mock_response
            engine._stop_event.set()

        mock_session = MagicMock()
        mock_session.receive = mock_receive

        await engine._recv_loop(mock_session)

        # barge-in 後は LISTENING に戻ること
        assert DialogueState.INTERRUPTED in states
        assert states[-1] == DialogueState.LISTENING

    @pytest.mark.asyncio
    async def test_interrupted_flushes_playback_queue(self) -> None:
        """barge-in 時に再生キューが空になることを確認."""
        engine = make_engine()
        for _ in range(3):
            engine._playback_queue.put_nowait(b"\x00" * 1280)

        mock_response = self._make_interrupted_response()

        async def mock_receive() -> Any:
            yield mock_response
            engine._stop_event.set()

        mock_session = MagicMock()
        mock_session.receive = mock_receive

        await engine._recv_loop(mock_session)

        assert engine._playback_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_model_audio_is_enqueued_without_inline_playback(self) -> None:
        """受信ループは直接再生せず、再生キューへ積むだけにする."""
        engine = make_engine()
        engine._player.write = MagicMock()
        mock_response = self._make_audio_response(b"\x01\x02", b"\x03\x04")

        async def mock_receive() -> Any:
            yield mock_response
            engine._stop_event.set()

        mock_session = MagicMock()
        mock_session.receive = mock_receive

        await engine._recv_loop(mock_session)

        engine._player.write.assert_not_called()
        assert engine._playback_queue.qsize() == 1
        assert engine._playback_queue.get_nowait() == b"\x01\x02\x03\x04"


# ---------------------------------------------------------------------------
# Function Calling テスト
# ---------------------------------------------------------------------------


class TestToolCalling:
    """Function Calling と Replay controller 連携を確認する."""

    @staticmethod
    def _make_function_call(
        name: str,
        args: dict[str, Any] | None = None,
        call_id: str = "call-1",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=call_id,
            name=name,
            args=args or {},
        )

    @pytest.mark.asyncio
    async def test_handle_tool_call_updates_controller_and_sends_ack(self) -> None:
        controller = MagicMock()
        controller.apply_and_reconcile = AsyncMock()
        session = MagicMock()
        session.send_tool_response = AsyncMock()
        engine = make_engine(replay_controller=controller)
        function_call = self._make_function_call(
            "seek_replay",
            {"time_seconds": 941.0},
        )

        await engine._handle_tool_call(session, function_call)

        controller.update_desired.assert_called_once_with(time_seconds=941.0)
        controller.apply_and_reconcile.assert_awaited_once_with(
            "seek_replay",
            {"time_seconds": 941.0},
        )
        responses = session.send_tool_response.await_args.kwargs["function_responses"]
        assert len(responses) == 1
        assert responses[0].id == "call-1"
        assert responses[0].name == "seek_replay"
        assert responses[0].response == {"output": {"ok": True}}

    @pytest.mark.asyncio
    async def test_handle_tool_call_returns_error_when_controller_fails(self) -> None:
        controller = MagicMock()
        controller.apply_and_reconcile = AsyncMock(side_effect=RuntimeError("boom"))
        session = MagicMock()
        session.send_tool_response = AsyncMock()
        engine = make_engine(replay_controller=controller)
        function_call = self._make_function_call("pause_replay")

        await engine._handle_tool_call(session, function_call)

        responses = session.send_tool_response.await_args.kwargs["function_responses"]
        assert responses[0].response == {"error": {"message": "boom"}}

    @pytest.mark.asyncio
    async def test_handle_tool_call_returns_error_when_controller_raises_connection_error(
        self,
    ) -> None:
        controller = MagicMock()
        controller.apply_and_reconcile = AsyncMock(
            side_effect=requests.exceptions.ConnectionError(
                "Replay API 接続不可（LoL リプレイ起動中か確認してください）"
            )
        )
        session = MagicMock()
        session.send_tool_response = AsyncMock()
        engine = make_engine(replay_controller=controller)
        function_call = self._make_function_call("pause_replay")

        await engine._handle_tool_call(session, function_call)

        responses = session.send_tool_response.await_args.kwargs["function_responses"]
        assert responses[0].response == {
            "error": {
                "message": "Replay API 接続不可（LoL リプレイ起動中か確認してください）"
            }
        }

    @pytest.mark.asyncio
    async def test_handle_tool_call_cancellation_reconciles_actual_state(self) -> None:
        controller = MagicMock()
        controller._get_actual_state.return_value = {
            "time": 941.0,
            "paused": True,
            "speed": 0.5,
        }
        controller.reconcile_now = AsyncMock()
        engine = make_engine(replay_controller=controller)
        cancellation = SimpleNamespace(ids=["call-1"])

        async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        with patch("sidecar.realtime_engine.asyncio.to_thread", new=fake_to_thread):
            await engine._handle_tool_call_cancellation(cancellation)

        controller.update_desired.assert_called_once_with(
            time_seconds=941.0,
            paused=True,
            speed=0.5,
        )
        controller.reconcile_now.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_interruption_flushes_playback_and_reconciles(self) -> None:
        states: list[DialogueState] = []
        engine = make_engine(
            on_state_change=states.append,
            replay_controller=MagicMock(),
        )
        engine._player.clear_and_stop = MagicMock()
        engine._sync_replay_state_from_actual = AsyncMock()
        scheduled: list[asyncio.Task[None]] = []
        for _ in range(3):
            engine._playback_queue.put_nowait(b"\x00" * 1280)

        original_create_task = asyncio.create_task

        def fake_create_task(coro: Any) -> asyncio.Task[None]:
            task = original_create_task(coro)
            scheduled.append(task)
            return task

        with patch("sidecar.realtime_engine.asyncio.create_task", side_effect=fake_create_task):
            await engine._handle_interruption()
            assert engine._sync_replay_state_from_actual.await_count == 0
            await asyncio.gather(*scheduled)

        assert engine._playback_queue.qsize() == 0
        engine._player.clear_and_stop.assert_called_once()
        engine._sync_replay_state_from_actual.assert_awaited_once()
        assert len(scheduled) == 1
        assert DialogueState.INTERRUPTED in states
        assert states[-1] == DialogueState.LISTENING


# ---------------------------------------------------------------------------
# Push-to-talk テスト
# ---------------------------------------------------------------------------


class TestPushToTalk:
    """Push-to-talk 開始/終了時の制御を確認する."""

    @pytest.mark.asyncio
    async def test_start_push_to_talk_interrupts_playback_and_sends_activity_start(
        self,
    ) -> None:
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        engine._session = MagicMock()
        engine._session.send_realtime_input = AsyncMock()
        engine._player.clear_and_stop = MagicMock()
        engine._set_state(DialogueState.SPEAKING)
        engine._playback_queue.put_nowait(b"\x00" * 1280)

        started = await engine.start_push_to_talk()

        assert started is True
        assert engine._playback_queue.qsize() == 0
        engine._player.clear_and_stop.assert_called_once()
        engine._session.send_realtime_input.assert_awaited_once()
        kwargs = engine._session.send_realtime_input.await_args.kwargs
        assert isinstance(kwargs["activity_start"], types.ActivityStart)
        assert states[-1] == DialogueState.USER_SPEAKING

    @pytest.mark.asyncio
    async def test_start_push_to_talk_from_listening_does_not_drop_next_turn_audio(
        self,
    ) -> None:
        engine = make_engine()
        engine._session = MagicMock()
        engine._session.send_realtime_input = AsyncMock()
        engine._player.clear_and_stop = MagicMock()
        engine._set_state(DialogueState.LISTENING)

        started = await engine.start_push_to_talk()

        assert started is True
        assert engine._drop_playback_until_turn_complete is False
        engine._player.clear_and_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_end_push_to_talk_sends_activity_end_and_moves_to_processing(
        self,
    ) -> None:
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        engine._session = MagicMock()
        engine._session.send_realtime_input = AsyncMock()
        engine._push_to_talk_active = True

        ended = await engine.end_push_to_talk()

        assert ended is True
        engine._session.send_realtime_input.assert_awaited_once()
        kwargs = engine._session.send_realtime_input.await_args.kwargs
        assert isinstance(kwargs["activity_end"], types.ActivityEnd)
        assert states[-1] == DialogueState.PROCESSING

    @pytest.mark.asyncio
    async def test_start_push_to_talk_while_speaking_drops_late_audio_until_turn_complete(
        self,
    ) -> None:
        engine = make_engine()
        engine._session = MagicMock()
        engine._session.send_realtime_input = AsyncMock()
        engine._player.clear_and_stop = MagicMock()
        engine._set_state(DialogueState.SPEAKING)
        engine._playback_queue.put_nowait(b"\x00" * 1280)

        started = await engine.start_push_to_talk()

        assert started is True
        assert engine._drop_playback_until_turn_complete is True
        assert engine._playback_queue.qsize() == 0
        engine._player.clear_and_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_turn_complete_waits_until_playback_finishes(self) -> None:
        states: list[DialogueState] = []
        engine = make_engine(on_state_change=states.append)
        engine._set_state(DialogueState.SPEAKING)
        states.clear()
        engine._playback_in_progress = True

        engine._mark_turn_complete()

        assert engine.state == DialogueState.SPEAKING
        assert engine._pending_turn_completion is True
        assert states == []

        engine._playback_in_progress = False
        engine._complete_turn_if_ready()

        assert engine.state == DialogueState.LISTENING
        assert states == [DialogueState.LISTENING]


# ---------------------------------------------------------------------------
# 字幕イベントテスト
# ---------------------------------------------------------------------------


class TestSubtitleEmission:
    """字幕イベントの発行を確認する."""

    def test_emit_subtitle_ai(self) -> None:
        subtitles: list[SubtitleEvent] = []
        engine = make_engine(on_subtitle=subtitles.append)
        engine._emit_subtitle("テストテキスト", is_user=False)
        assert len(subtitles) == 1
        assert subtitles[0].text == "テストテキスト"
        assert not subtitles[0].is_user

    def test_emit_subtitle_user(self) -> None:
        subtitles: list[SubtitleEvent] = []
        engine = make_engine(on_subtitle=subtitles.append)
        engine._emit_subtitle("ユーザー発話", is_user=True)
        assert len(subtitles) == 1
        assert subtitles[0].is_user

    def test_emit_subtitle_has_timestamp(self) -> None:
        subtitles: list[SubtitleEvent] = []
        engine = make_engine(on_subtitle=subtitles.append)
        engine._emit_subtitle("時刻テスト", is_user=False)
        assert subtitles[0].timestamp > 0


# ---------------------------------------------------------------------------
# build_config テスト
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """_build_config が正しい LiveConnectConfig を生成することを確認する."""

    def test_config_has_audio_modality(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert "AUDIO" in config.response_modalities

    def test_config_has_output_transcription(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.output_audio_transcription is not None

    def test_config_has_input_transcription(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.input_audio_transcription is not None

    def test_config_has_realtime_input_config(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.realtime_input_config is not None

    def test_config_defaults_to_no_interruption(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.realtime_input_config is not None
        assert (
            config.realtime_input_config.activity_handling
            == types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
        )
        assert config.realtime_input_config.automatic_activity_detection is not None
        assert config.realtime_input_config.automatic_activity_detection.disabled is True

    def test_config_can_disable_push_to_talk_via_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"GEMINI_LIVE_PUSH_TO_TALK": "false"},
            clear=False,
        ):
            engine = make_engine()
            config = engine._build_config()

        assert config.realtime_input_config is not None
        assert (
            config.realtime_input_config.activity_handling
            == types.ActivityHandling.NO_INTERRUPTION
        )
        assert config.realtime_input_config.automatic_activity_detection is None

    def test_config_can_enable_interruptions_via_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GEMINI_LIVE_PUSH_TO_TALK": "false",
                "GEMINI_LIVE_ACTIVITY_HANDLING": "interrupt",
            },
            clear=False,
        ):
            engine = make_engine()
            config = engine._build_config()

        assert config.realtime_input_config is not None
        assert (
            config.realtime_input_config.activity_handling
            == types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
        )

    def test_config_can_disable_automatic_activity_detection_via_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "GEMINI_LIVE_PUSH_TO_TALK": "false",
                "GEMINI_LIVE_DISABLE_AUTOMATIC_ACTIVITY_DETECTION": "true",
            },
            clear=False,
        ):
            engine = make_engine()
            config = engine._build_config()

        assert config.realtime_input_config is not None
        assert config.realtime_input_config.automatic_activity_detection is not None
        assert config.realtime_input_config.automatic_activity_detection.disabled is True

    def test_config_has_context_window_compression(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.context_window_compression is not None

    def test_config_has_session_resumption(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.session_resumption is not None

    def test_config_system_instruction_contains_text(self) -> None:
        engine = make_engine(system_instruction="コーチ指示テスト")
        config = engine._build_config()
        parts = config.system_instruction.parts
        assert any("コーチ指示テスト" in p.text for p in parts)

    def test_config_has_replay_tools(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        assert config.tools is not None
        declarations = config.tools[0].function_declarations
        assert declarations is not None
        names = [declaration.name for declaration in declarations]
        assert names == [
            "seek_replay",
            "pause_replay",
            "resume_replay",
            "slow_motion",
        ]

    def test_replay_tools_are_non_blocking(self) -> None:
        engine = make_engine()
        config = engine._build_config()
        declarations = config.tools[0].function_declarations
        assert declarations is not None
        assert all(declaration.behavior == "NON_BLOCKING" for declaration in declarations)


# ---------------------------------------------------------------------------
# stop() テスト
# ---------------------------------------------------------------------------


class TestStop:
    """stop() が stop_event をセットすることを確認する."""

    @pytest.mark.asyncio
    async def test_stop_sets_stop_event(self) -> None:
        engine = make_engine()
        assert not engine._stop_event.is_set()
        await engine.stop()
        assert engine._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_stop_closes_active_session(self) -> None:
        engine = make_engine()
        engine._session = MagicMock()
        engine._session.close = AsyncMock()

        await engine.stop()

        engine._session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recv_loop_returns_when_stop_event_is_set_and_receive_fails(self) -> None:
        """stop 後の close 起因例外を正常終了として扱うことを確認."""
        engine = make_engine()

        async def mock_receive() -> Any:
            engine._stop_event.set()
            raise RuntimeError("session closed")
            yield  # pragma: no cover

        mock_session = MagicMock()
        mock_session.receive = mock_receive

        await engine._recv_loop(mock_session)


# ---------------------------------------------------------------------------
# start() テスト（API キーなし → 即終了）
# ---------------------------------------------------------------------------


class TestStartWithoutApiKey:
    """GEMINI_API_KEY が未設定の場合、start() が安全に終了することを確認する."""

    @pytest.mark.asyncio
    async def test_start_exits_without_api_key(self) -> None:
        engine = make_engine()
        with patch.dict("os.environ", {}, clear=True):
            # 例外なし、かつ状態が IDLE に戻ること
            await engine.start()
        assert engine.state == DialogueState.IDLE
