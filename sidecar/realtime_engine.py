"""Gemini Live API との双方向音声対話エンジン（最小構成）.

sounddevice（WASAPI）でマイク入力を取得し、Gemini Live API に送信する。
AI の音声応答を sounddevice で再生する。
barge-in（server_content.interrupted）時に再生を即停止する。

縮退モード（sounddevice 利用不可の場合）:
  - マイク入力: なし（WSL 環境など）
  - 音声再生: なし
  - 字幕・状態遷移は正常動作
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import types

from sidecar.models import DialogueState, SubtitleEvent
from sidecar.replay_controller import ReplayStateController

logger = logging.getLogger(__name__)

# 定数
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
INPUT_SAMPLE_RATE = 16_000  # Hz（マイク入力）
OUTPUT_SAMPLE_RATE = 24_000  # Hz（AI 音声出力）
CHUNK_FRAMES = 640  # 40ms @ 16kHz
INPUT_QUEUE_MAXSIZE = 8
# AI 音声は途中欠落より連続性を優先し、interrupt 時だけ明示的に flush する。
PLAYBACK_QUEUE_MAXSIZE = 0
PLAYBACK_BATCH_MAX_CHUNKS = 4
PLAYBACK_BATCH_WAIT_SECONDS = 0.02
PTT_SUFFIX_PADDING_SECONDS = 0.15
CONTEXT_WINDOW_TRIGGER_TOKENS = 104_857
CONTEXT_WINDOW_TARGET_TOKENS = 52_428
TOOL_ACK_RESPONSE = {"output": {"ok": True}}
DEFAULT_ACTIVITY_HANDLING = types.ActivityHandling.NO_INTERRUPTION
INTERRUPTING_ACTIVITY_HANDLING = types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS

# sounddevice のオプショナルインポート（WSL 等で利用不可な場合は縮退）
try:
    import sounddevice as _sd  # type: ignore[import-untyped]

    _SOUNDDEVICE_AVAILABLE = True
except Exception as _e:
    _sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice が利用不可: %s — 縮退モードで動作します", _e)


def _is_truthy_env(value: str | None) -> bool:
    """環境変数の真偽値表現を正規化する."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _push_to_talk_enabled() -> bool:
    """Push-to-talk を有効にするかを返す.

    ハッカソン提出版では、manual VAD で明示的に話者交代する方が
    デモの安定性と説明しやすさの両立に向くため既定で有効化する。
    """
    raw_value = os.getenv("GEMINI_LIVE_PUSH_TO_TALK")
    if raw_value is None:
        return True
    return _is_truthy_env(raw_value)


def _resolve_activity_handling() -> types.ActivityHandling:
    """割り込み方針を環境変数から解決する.

    既定値は NO_INTERRUPTION とし、Live API の自動 VAD が
    再生中の AI 音声を自己割り込みしにくい構成にする。
    """
    raw_value = os.getenv("GEMINI_LIVE_ACTIVITY_HANDLING", "").strip().lower()
    if raw_value in {"", "no_interruption", "no-interruption", "none"}:
        return DEFAULT_ACTIVITY_HANDLING
    if raw_value in {"interrupt", "start_of_activity_interrupts"}:
        return INTERRUPTING_ACTIVITY_HANDLING

    logger.warning(
        "未知の GEMINI_LIVE_ACTIVITY_HANDLING=%r のため %s を使用します",
        raw_value,
        DEFAULT_ACTIVITY_HANDLING.value,
    )
    return DEFAULT_ACTIVITY_HANDLING


def _build_realtime_input_config() -> types.RealtimeInputConfig:
    """Live API の音声入力制御設定を構築する."""
    if _push_to_talk_enabled():
        return types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            ),
            activity_handling=INTERRUPTING_ACTIVITY_HANDLING,
        )

    automatic_activity_detection: types.AutomaticActivityDetection | None = None
    if _is_truthy_env(os.getenv("GEMINI_LIVE_DISABLE_AUTOMATIC_ACTIVITY_DETECTION")):
        automatic_activity_detection = types.AutomaticActivityDetection(disabled=True)

    return types.RealtimeInputConfig(
        automatic_activity_detection=automatic_activity_detection,
        activity_handling=_resolve_activity_handling(),
    )


REPLAY_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="seek_replay",
            description="リプレイを指定した時刻（秒）にシークする",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "time_seconds": types.Schema(
                        type="NUMBER",
                        description="シーク先の時刻（秒）",
                    ),
                },
                required=["time_seconds"],
            ),
            behavior="NON_BLOCKING",
        ),
        types.FunctionDeclaration(
            name="pause_replay",
            description="リプレイを一時停止する",
            parameters=types.Schema(type="OBJECT", properties={}),
            behavior="NON_BLOCKING",
        ),
        types.FunctionDeclaration(
            name="resume_replay",
            description="リプレイの再生を再開する",
            parameters=types.Schema(type="OBJECT", properties={}),
            behavior="NON_BLOCKING",
        ),
        types.FunctionDeclaration(
            name="slow_motion",
            description="リプレイの再生速度を変更する",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "speed": types.Schema(
                        type="NUMBER",
                        description="再生速度（1.0 = 等速）",
                    ),
                },
                required=["speed"],
            ),
            behavior="NON_BLOCKING",
        ),
    ]
)


class AudioPlayer:
    """sounddevice RawOutputStream による音声再生."""

    def __init__(self) -> None:
        self._stream: Any = None
        self._active = False
        self._stream_generation = 0
        # stream 参照の切り替えとネイティブ操作を同じロックで直列化する。
        # Windows では write と abort/close の並行実行で不安定化しやすいため、
        # barge-in 時も必ず現在の write が抜けてから停止処理に入る。
        self._stream_lock = threading.RLock()

    def _create_stream(self) -> Any:
        """再生ストリームを生成して開始する."""
        stream = _sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            latency=0.1,  # 100ms バッファ（"low" だと初回再生の頭切れが起きる）
        )
        stream.start()
        return stream

    def _snapshot(self) -> tuple[Any | None, int]:
        """現在の stream と世代番号を取得する."""
        with self._stream_lock:
            return self._stream, self._stream_generation

    def _generation_changed(self, generation: int) -> bool:
        """割り込みで stream 世代が切り替わったかを返す."""
        with self._stream_lock:
            return self._stream_generation != generation

    def _detach_stream(self, *, bump_generation: bool = False) -> Any | None:
        """現在の stream を切り離し、以後の書き込み対象から外す."""
        with self._stream_lock:
            if bump_generation:
                self._stream_generation += 1
            stream = self._stream
            self._stream = None
            self._active = False
            return stream

    def _install_stream(self, stream: Any) -> Any | None:
        """新しい stream を現在の再生先として登録する."""
        with self._stream_lock:
            previous_stream = self._stream
            self._stream = stream
            self._active = True
            return previous_stream

    def _finalize_stream(self, stream: Any | None) -> None:
        """不要になった stream を best-effort で解放する."""
        if stream is None:
            return
        with self._stream_lock:
            for method_name in ("stop", "close"):
                try:
                    getattr(stream, method_name)()
                except Exception:
                    pass

    def _close_stream(self, stream: Any | None) -> None:
        """abort 済み stream を close のみに絞って解放する."""
        if stream is None:
            return
        with self._stream_lock:
            try:
                stream.close()
            except Exception:
                pass

    def _replace_stream(
        self,
        *,
        success_message: str,
        failure_message: str,
    ) -> bool:
        """再生 stream を新しく作り直す."""
        if not _SOUNDDEVICE_AVAILABLE:
            return False

        try:
            with self._stream_lock:
                stream = self._create_stream()
        except Exception as e:
            previous_stream = self._detach_stream()
            self._finalize_stream(previous_stream)
            logger.warning(failure_message, e)
            return False

        previous_stream = self._install_stream(stream)
        self._finalize_stream(previous_stream)
        logger.info(success_message, OUTPUT_SAMPLE_RATE)
        return True

    def _recover_stream(self) -> bool:
        """停止した再生 stream を再作成する."""
        return self._replace_stream(
            success_message="音声再生ストリーム再作成（%d Hz）",
            failure_message="音声再生ストリーム再作成失敗: %s",
        )

    def start(self) -> None:
        """再生ストリームを開始する."""
        if not _SOUNDDEVICE_AVAILABLE:
            return
        self._replace_stream(
            success_message="音声再生ストリーム開始（%d Hz）",
            failure_message="音声再生ストリーム開始失敗: %s — 縮退モード",
        )

    def write(self, pcm_data: bytes) -> None:
        """PCM データを再生バッファに書き込む."""
        stream, generation = self._snapshot()
        if stream is None:
            if not self._recover_stream():
                return
            stream, generation = self._snapshot()
            if stream is None:
                return

        try:
            with self._stream_lock:
                if self._generation_changed(generation):
                    logger.debug("割り込みで音声書き込みを中断")
                    return
                stream.write(pcm_data)
        except Exception as e:
            if self._generation_changed(generation):
                logger.debug("割り込みで音声書き込みを中断")
                return
            logger.warning("音声書き込みエラー: %s", e)
            if not self._recover_stream():
                return

            retry_stream, retry_generation = self._snapshot()
            if retry_stream is None:
                return
            try:
                with self._stream_lock:
                    if self._generation_changed(retry_generation):
                        logger.debug("割り込みで音声書き込みを中断")
                        return
                    retry_stream.write(pcm_data)
            except Exception as retry_error:
                if self._generation_changed(retry_generation):
                    logger.debug("割り込みで音声書き込みを中断")
                    return
                logger.warning("音声書き込みエラー: %s", retry_error)

    def clear_and_stop(self) -> None:
        """再生バッファを即クリアし停止する（barge-in 用）."""
        stream = self._detach_stream(bump_generation=True)
        if stream is None:
            return
        try:
            with self._stream_lock:
                stream.abort()
                logger.debug("再生バッファ即停止（barge-in）")
        except Exception as e:
            logger.warning("再生停止エラー: %s", e)
        finally:
            # Windows MME では abort 後に stop まで触ると不安定なことがある。
            self._close_stream(stream)

    def close(self) -> None:
        """ストリームをクローズする."""
        stream = self._detach_stream()
        self._finalize_stream(stream)

    @property
    def active(self) -> bool:
        """音声再生が有効かどうか."""
        return self._active


class AudioCapture:
    """sounddevice RawInputStream によるマイク入力キャプチャ."""

    def __init__(self, audio_queue: asyncio.Queue[bytes]) -> None:
        self._queue = audio_queue
        self._stream: Any = None
        self._active = False

    def start(self) -> None:
        """マイク入力ストリームを開始する."""
        if not _SOUNDDEVICE_AVAILABLE:
            return
        try:
            self._stream = _sd.RawInputStream(
                samplerate=INPUT_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_FRAMES,
                callback=self._callback,
            )
            self._stream.start()
            self._active = True
            logger.info(
                "マイク入力ストリーム開始（%d Hz, %d frames/chunk）",
                INPUT_SAMPLE_RATE,
                CHUNK_FRAMES,
            )
        except Exception as e:
            logger.warning("マイク入力ストリーム開始失敗: %s — 縮退モード", e)
            self._active = False

    def _callback(self, indata: Any, frames: int, _time: Any, status: Any) -> None:
        if status:
            logger.debug("sounddevice input status: %s", status)
        chunk = bytes(indata)
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # 最新音声を優先し、古い未送信チャンクを捨てる。
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(chunk)

    def stop(self) -> None:
        """マイク入力ストリームを停止・クローズする."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._active = False

    @property
    def active(self) -> bool:
        """マイク入力が有効かどうか."""
        return self._active


def _append_transcript_fragment(buffer: str, fragment: str) -> str:
    """transcript バッファに断片を追記する（ASCII英数字同士はスペース補完）."""
    if not buffer:
        return fragment
    prev = buffer[-1]
    curr = fragment[0]
    if prev.isascii() and prev.isalnum() and curr.isascii() and curr.isalnum():
        return buffer + " " + fragment
    return buffer + fragment


class RealtimeEngine:
    """Gemini Live API との双方向音声対話エンジン（最小構成）."""

    def __init__(
        self,
        system_instruction: str,
        on_subtitle: Callable[[SubtitleEvent], None],
        on_state_change: Callable[[DialogueState], None],
        replay_controller: ReplayStateController | None = None,
    ) -> None:
        self._system_instruction = system_instruction
        self._on_subtitle = on_subtitle
        self._on_state_change = on_state_change
        self._replay_controller = replay_controller

        self._state = DialogueState.IDLE
        self._stop_event = asyncio.Event()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=INPUT_QUEUE_MAXSIZE
        )
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=PLAYBACK_QUEUE_MAXSIZE
        )
        self._resumption_handle: str | None = None  # session_resumption_update で更新
        self._session: genai.live.AsyncSession | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._tool_response_lock = asyncio.Lock()
        self._realtime_input_lock = asyncio.Lock()
        self._push_to_talk_active = False
        self._pending_turn_completion = False
        self._playback_in_progress = False
        self._drop_playback_until_turn_complete = False

        self._player = AudioPlayer()
        self._capture = AudioCapture(self._audio_queue)

    @property
    def state(self) -> DialogueState:
        """現在の対話状態."""
        return self._state

    def _set_state(self, new_state: DialogueState) -> None:
        if self._state == new_state:
            return
        logger.info("DialogueState: %s → %s", self._state.value, new_state.value)
        self._state = new_state
        self._on_state_change(new_state)

    def _build_config(self) -> types.LiveConnectConfig:
        """LiveConnectConfig を構築する."""
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            realtime_input_config=_build_realtime_input_config(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=CONTEXT_WINDOW_TRIGGER_TOKENS,
                sliding_window=types.SlidingWindow(
                    target_tokens=CONTEXT_WINDOW_TARGET_TOKENS
                ),
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=self._system_instruction)]
            ),
            tools=[REPLAY_TOOLS],
            # 接続断時にトークンで再開できるよう有効化（再接続時は handle を渡す）
            session_resumption=types.SessionResumptionConfig(
                handle=self._resumption_handle,
            ),
        )

    def _flush_audio_queue(self) -> None:
        """未送信のマイク音声キューを空にする."""
        self._flush_queue(self._audio_queue)

    def _flush_playback_queue(self) -> None:
        """未再生の AI 音声キューを空にする."""
        self._flush_queue(self._playback_queue)

    @staticmethod
    def _flush_queue(queue: asyncio.Queue[bytes]) -> None:
        """Queue を安全に空にする."""
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _enqueue_playback_chunk(self, chunk: bytes) -> None:
        """再生キューへ音声チャンクを積む."""
        self._playback_queue.put_nowait(chunk)

    async def _collect_playback_batch(self, first_chunk: bytes) -> bytes:
        """近接して届いた音声チャンクをまとめて再生する."""
        chunks = [first_chunk]
        while len(chunks) < PLAYBACK_BATCH_MAX_CHUNKS:
            try:
                chunks.append(
                    await asyncio.wait_for(
                        self._playback_queue.get(),
                        timeout=PLAYBACK_BATCH_WAIT_SECONDS,
                    )
                )
            except asyncio.TimeoutError:
                break
        return b"".join(chunks)

    def _collect_model_audio(self, model_turn: Any) -> bytes | None:
        """model_turn 内の音声パーツを 1 つの PCM チャンクへまとめる."""
        audio_buffer = bytearray()
        for part in getattr(model_turn, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None or not inline_data.data:
                continue
            if self._drop_playback_until_turn_complete:
                logger.debug("barge-in 後の late chunk を破棄")
                return None
            audio_buffer.extend(inline_data.data)
        return bytes(audio_buffer) if audio_buffer else None

    def _mark_turn_complete(self) -> None:
        """ターン完了を記録し、再生完了後に LISTENING へ戻す."""
        self._pending_turn_completion = True
        self._drop_playback_until_turn_complete = False
        if self._complete_turn_if_ready():
            logger.info("ターン完了 → 接続維持・次ターン待機")
        else:
            logger.info("ターン完了待機中: 再生キュー排出を待つ")

    def _complete_turn_if_ready(self) -> bool:
        """再生完了済みならターンを閉じて待機状態へ戻す."""
        if not self._pending_turn_completion:
            return False
        if self._playback_in_progress or not self._playback_queue.empty():
            return False

        self._pending_turn_completion = False
        if self._push_to_talk_active:
            self._set_state(DialogueState.USER_SPEAKING)
        else:
            self._set_state(DialogueState.LISTENING)
        return True

    def _should_interrupt_current_turn_for_push_to_talk(self) -> bool:
        """現在の PTT 開始が既存ターンへの割り込みかどうかを返す."""
        if self._state in {
            DialogueState.SPEAKING,
            DialogueState.PROCESSING,
            DialogueState.INTERRUPTED,
        }:
            return True
        return (
            self._playback_in_progress
            or not self._playback_queue.empty()
            or self._pending_turn_completion
        )

    async def _send_realtime_signal(
        self,
        *,
        activity_start: types.ActivityStart | None = None,
        activity_end: types.ActivityEnd | None = None,
    ) -> bool:
        """現在のセッションへ manual VAD シグナルを送る."""
        session = self._session
        if session is None:
            return False

        try:
            async with self._realtime_input_lock:
                await session.send_realtime_input(
                    activity_start=activity_start,
                    activity_end=activity_end,
                )
        except Exception as e:
            logger.warning("manual VAD シグナル送信失敗: %s [%s]", e, type(e).__name__)
            return False
        return True

    async def start_push_to_talk(self) -> bool:
        """ユーザーの発話ターンを明示的に開始する."""
        if self._push_to_talk_active:
            return False

        is_interrupting_current_turn = self._should_interrupt_current_turn_for_push_to_talk()
        self._flush_audio_queue()
        if is_interrupting_current_turn:
            self._flush_playback_queue()
            self._player.clear_and_stop()
        self._drop_playback_until_turn_complete = is_interrupting_current_turn
        # start シグナル送信中に先頭音声を取りこぼさないよう先に有効化する。
        self._push_to_talk_active = True

        if not await self._send_realtime_signal(activity_start=types.ActivityStart()):
            self._push_to_talk_active = False
            self._drop_playback_until_turn_complete = False
            return False

        self._pending_turn_completion = False
        self._set_state(DialogueState.USER_SPEAKING)
        logger.info("Push-to-talk 開始")
        return True

    async def end_push_to_talk(self) -> bool:
        """ユーザーの発話ターンを終了し、AI 応答待ちへ戻す."""
        if not self._push_to_talk_active:
            return False

        # 語尾の取りこぼしを減らすため、短い余白を持たせてから終了通知する。
        await asyncio.sleep(PTT_SUFFIX_PADDING_SECONDS)
        if not await self._send_realtime_signal(activity_end=types.ActivityEnd()):
            self._push_to_talk_active = False
            self._set_state(DialogueState.LISTENING)
            return False

        self._push_to_talk_active = False
        self._set_state(DialogueState.PROCESSING)
        logger.info("Push-to-talk 終了")
        return True

    def _track_background_task(self, task: asyncio.Task[None]) -> None:
        """終了待ちが必要な補助 task を追跡する."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _schedule_replay_sync_after_interruption(self) -> None:
        """barge-in 後の Replay 実状態同期を hot path の外で実行する."""
        if self._replay_controller is None:
            return
        self._track_background_task(
            asyncio.create_task(self._sync_replay_state_from_actual())
        )

    @staticmethod
    def _get_tool_args(function_call: Any) -> dict[str, Any]:
        """SDK の args を通常の辞書に正規化する."""
        raw_args = getattr(function_call, "args", {}) or {}
        return raw_args if isinstance(raw_args, dict) else dict(raw_args)

    def _update_desired_state_for_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """ツールごとに desired_state を更新する."""
        if self._replay_controller is None:
            return

        if tool_name == "seek_replay":
            self._replay_controller.update_desired(
                time_seconds=float(args.get("time_seconds", 0.0))
            )
        elif tool_name == "pause_replay":
            self._replay_controller.update_desired(paused=True)
        elif tool_name == "resume_replay":
            self._replay_controller.update_desired(paused=False)
        elif tool_name == "slow_motion":
            self._replay_controller.update_desired(speed=float(args.get("speed", 0.5)))

    async def _sync_replay_state_from_actual(self) -> None:
        """Replay の実状態を desired_state に反映して再補正する."""
        if self._replay_controller is None:
            return

        actual = await asyncio.to_thread(self._replay_controller._get_actual_state)
        if not actual:
            return

        self._replay_controller.update_desired(
            time_seconds=float(actual["time"]) if "time" in actual else None,
            paused=actual.get("paused"),
            speed=float(actual.get("speed", 1.0)),
        )
        await self._replay_controller.reconcile_now()

    async def _handle_interruption(self) -> None:
        """barge-in 発生時の状態補正を行う."""
        self._flush_playback_queue()
        self._player.clear_and_stop()
        self._set_state(DialogueState.INTERRUPTED)
        self._schedule_replay_sync_after_interruption()
        if self._push_to_talk_active:
            self._set_state(DialogueState.USER_SPEAKING)
        else:
            self._set_state(DialogueState.LISTENING)
        logger.info("barge-in 検知: 再生停止")

    @staticmethod
    def _build_tool_response(
        function_call: Any,
        *,
        response: dict[str, Any],
    ) -> types.FunctionResponse:
        """FunctionResponse を構築する."""
        return types.FunctionResponse(
            id=getattr(function_call, "id", None),
            name=getattr(function_call, "name", None),
            response=response,
        )

    async def _send_tool_response(
        self,
        session: genai.live.AsyncSession,
        function_response: types.FunctionResponse,
    ) -> None:
        """Live API セッションへ tool response を送信する."""
        tool_name = getattr(function_response, "name", None)
        call_id = getattr(function_response, "id", None)
        try:
            async with self._tool_response_lock:
                await session.send_tool_response(
                    function_responses=[function_response]
                )
            logger.info("tool_response 送信: %s (id=%s)", tool_name, call_id)
        except Exception as e:
            logger.warning("tool_response 送信失敗: %s [%s]", e, type(e).__name__)

    async def _handle_tool_call(
        self,
        session: genai.live.AsyncSession,
        function_call: Any,
    ) -> None:
        """1件の function call を処理し、最小 ACK を返す."""
        tool_name = getattr(function_call, "name", "")
        args = self._get_tool_args(function_call)
        logger.info(
            "tool_call 受信: %s id=%s args=%s",
            tool_name,
            getattr(function_call, "id", None),
            args,
        )

        try:
            if self._replay_controller is None:
                raise RuntimeError("ReplayStateController が未設定です")

            self._update_desired_state_for_tool(tool_name, args)
            await self._replay_controller.apply_and_reconcile(tool_name, args)
            function_response = self._build_tool_response(
                function_call,
                response=TOOL_ACK_RESPONSE,
            )
        except Exception as e:
            logger.exception("tool_call 処理エラー: %s", tool_name)
            function_response = self._build_tool_response(
                function_call,
                response={"error": {"message": str(e)}},
            )

        await self._send_tool_response(session, function_response)

    async def _handle_tool_call_cancellation(self, cancellation: Any) -> None:
        """tool_call_cancellation に合わせて desired_state を再同期する."""
        ids = list(getattr(cancellation, "ids", []) or [])
        logger.info("tool_call_cancellation 受信: ids=%s", ids)
        await self._sync_replay_state_from_actual()
        self._flush_playback_queue()
        self._player.clear_and_stop()
        self._set_state(DialogueState.LISTENING)

    async def _playback_loop(self) -> None:
        """再生キューを順に消費してスピーカーへ書き出す."""
        while not self._stop_event.is_set():
            try:
                first_chunk = await asyncio.wait_for(
                    self._playback_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue

            batch = await self._collect_playback_batch(first_chunk)
            self._playback_in_progress = True
            await asyncio.to_thread(self._player.write, batch)
            self._playback_in_progress = False
            self._complete_turn_if_ready()

    async def start(self) -> None:
        """対話セッションを開始する.

        同一接続で複数ターンを継続する（turn_complete で再接続しない）。
        go_away / 接続エラー時のみ再接続する。stop() で終了する。
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY が未設定")
            return

        client = genai.Client(api_key=api_key)
        self._player.start()
        self._capture.start()
        self._set_state(DialogueState.LISTENING)
        playback_task = asyncio.create_task(self._playback_loop())

        logger.info(
            "RealtimeEngine 開始（sounddevice: mic=%s, speaker=%s）",
            self._capture.active,
            self._player.active,
        )

        try:
            while not self._stop_event.is_set():
                config = self._build_config()
                try:
                    realtime_input_config = config.realtime_input_config
                    logger.info(
                        "Live 設定: push_to_talk=%s, activity_handling=%s, auto_vad_disabled=%s",
                        _push_to_talk_enabled(),
                        getattr(realtime_input_config, "activity_handling", None),
                        getattr(
                            getattr(
                                realtime_input_config,
                                "automatic_activity_detection",
                                None,
                            ),
                            "disabled",
                            False,
                        ),
                    )
                    async with client.aio.live.connect(
                        model=MODEL_ID, config=config
                    ) as session:
                        self._session = session
                        logger.info("Gemini Live API セッション開始（モデル: %s）", MODEL_ID)

                        send_task = asyncio.create_task(self._send_loop(session))
                        recv_task = asyncio.create_task(self._recv_loop(session))

                        # recv_task が終わるまで待つ（send_task はサブタスク）
                        # recv_task が例外を持つ場合はここで再スロー → 外側 except へ
                        try:
                            await recv_task
                        finally:
                            # セッションが閉じる前に送信系 task を確実に停止する。
                            if not send_task.done():
                                send_task.cancel()
                            session_background_tasks = list(self._background_tasks)
                            for task in session_background_tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(
                                *session_background_tasks,
                                send_task,
                                return_exceptions=True,
                            )
                        self._session = None

                    # go_away または予期せぬ接続終了 → stop でなければ再接続
                    if not self._stop_event.is_set():
                        self._flush_playback_queue()
                        logger.info("接続終了（go_away / 切断）→ 0.5秒後に再接続")
                        self._set_state(DialogueState.LISTENING)
                        await asyncio.sleep(0.5)

                except Exception:
                    if self._stop_event.is_set():
                        break
                    self._flush_playback_queue()
                    logger.exception("セッションエラー → 2秒後に再接続試行")
                    await asyncio.sleep(2.0)

        finally:
            self._session = None
            self._push_to_talk_active = False
            self._pending_turn_completion = False
            self._playback_in_progress = False
            self._drop_playback_until_turn_complete = False
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            playback_task.cancel()
            await asyncio.gather(playback_task, return_exceptions=True)
            self._flush_audio_queue()
            self._flush_playback_queue()
            self._capture.stop()
            self._player.close()
            self._set_state(DialogueState.IDLE)

    async def stop(self) -> None:
        """対話セッションを停止する."""
        logger.info("RealtimeEngine 停止要求")
        self._stop_event.set()
        session = self._session
        if session is not None:
            try:
                await session.close()
            except Exception:
                # 停止要求時は close 失敗でも後続のキャンセルへ進める。
                pass

    async def _send_loop(self, session: genai.live.AsyncSession) -> None:
        """マイク音声を Gemini Live API に送信するループ.

        send_realtime_input が失敗しても break せず継続する。
        セッション終了は recv_loop の終了に委ねる。
        """
        while not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if _push_to_talk_enabled() and not self._push_to_talk_active:
                continue

            try:
                async with self._realtime_input_lock:
                    await session.send_realtime_input(
                        audio=types.Blob(
                            mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                            data=chunk,
                        )
                    )
            except Exception as e:
                # セッション終了の判断は recv_loop に委ねるため break しない
                logger.warning(
                    "send_realtime_input エラー（継続）: %s [%s]",
                    e,
                    type(e).__name__,
                )

    async def _recv_loop(self, session: genai.live.AsyncSession) -> None:
        """Gemini Live API から応答を受信するループ.

        session.receive() は 1 ターン分のメッセージ列を返し turn_complete で終わる
        async iterator（SDK 仕様）。ターン完了後も接続を維持して次のターンを待つ。
        go_away / stop_event を受けたら return して start() に再接続を委ねる。
        """
        while not self._stop_event.is_set():
            broke_early = False  # 我々側で break したか（True = return が必要）
            break_reason = "unknown"

            ai_transcript_buffer = ""

            try:
                async for response in session.receive():
                    if self._stop_event.is_set():
                        broke_early = True
                        break_reason = "stop_event"
                        break

                    # go_away: サーバーが接続終了を予告 → 再接続準備
                    go_away = getattr(response, "go_away", None)
                    if go_away:
                        logger.warning("go_away 受信 → 再接続準備: %s", go_away)
                        broke_early = True
                        break_reason = "go_away"
                        break

                    # session_resumption_update: resumable な handle を保存
                    sru = getattr(response, "session_resumption_update", None)
                    if sru:
                        resumable = getattr(sru, "resumable", False)
                        new_handle = getattr(sru, "new_handle", None)
                        if resumable and new_handle:
                            self._resumption_handle = new_handle
                            logger.info("session_resumption_update: handle 更新")
                        else:
                            logger.debug(
                                "session_resumption_update: resumable=%s", resumable
                            )

                    tool_call = getattr(response, "tool_call", None)
                    if tool_call is not None:
                        function_calls = getattr(tool_call, "function_calls", []) or []
                        for function_call in function_calls:
                            self._track_background_task(
                                asyncio.create_task(
                                    self._handle_tool_call(session, function_call)
                                )
                            )

                    tool_call_cancellation = getattr(
                        response,
                        "tool_call_cancellation",
                        None,
                    )
                    if tool_call_cancellation is not None:
                        await self._handle_tool_call_cancellation(
                            tool_call_cancellation
                        )

                    server = getattr(response, "server_content", None)
                    if server is None:
                        continue

                    # barge-in: 再生バッファを即クリア（transcription は先に拾う）
                    if server.interrupted:
                        if server.output_transcription and server.output_transcription.text:
                            frag = server.output_transcription.text.strip()
                            if frag:
                                ai_transcript_buffer = _append_transcript_fragment(
                                    ai_transcript_buffer, frag
                                )
                            self._emit_subtitle(ai_transcript_buffer, is_user=False)
                        if server.input_transcription and server.input_transcription.text:
                            self._emit_subtitle(server.input_transcription.text, is_user=True)
                        logger.warning("server.interrupted 受信: 音声再生を中断します")
                        self._drop_playback_until_turn_complete = True
                        await self._handle_interruption()
                        continue

                    # AI 音声チャンク受信 → 再生キューへ積む
                    if server.model_turn:
                        audio_chunk = self._collect_model_audio(server.model_turn)
                        if audio_chunk:
                            self._set_state(DialogueState.SPEAKING)
                            self._pending_turn_completion = False
                            self._enqueue_playback_chunk(audio_chunk)

                    # AI 音声の文字起こし（デルタを蓄積して累積テキストを emit）
                    if server.output_transcription and server.output_transcription.text:
                        fragment = server.output_transcription.text.strip()
                        if fragment:
                            ai_transcript_buffer = _append_transcript_fragment(
                                ai_transcript_buffer, fragment
                            )
                        self._emit_subtitle(ai_transcript_buffer, is_user=False)

                    # ユーザー音声の文字起こし
                    if server.input_transcription and server.input_transcription.text:
                        self._emit_subtitle(
                            server.input_transcription.text, is_user=True
                        )

                    # デバッグ用
                    if getattr(server, "generation_complete", None):
                        logger.debug("generation_complete 受信")
                    if getattr(server, "waiting_for_input", None):
                        logger.debug("waiting_for_input 受信")

                    # ターン完了: SDK がこのメッセージを yield した後に break する
                    # → ここで接続は切らない。while ループで次のターンへ進む
                    if server.turn_complete:
                        if ai_transcript_buffer:
                            self._emit_subtitle(
                                ai_transcript_buffer, is_user=False, finished=True
                            )
                        ai_transcript_buffer = ""
                        self._mark_turn_complete()

            except asyncio.CancelledError:
                logger.info("recv_loop: キャンセル")
                raise
            except Exception as e:
                if self._stop_event.is_set():
                    logger.info(
                        "recv_loop 終了（stop 済みで受信終了）: %s [%s]",
                        e,
                        type(e).__name__,
                    )
                    return
                logger.error(
                    "recv_loop ターン例外: %s [%s]", e, type(e).__name__, exc_info=True
                )
                raise

            if broke_early:
                logger.info("recv_loop 終了（理由: %s）", break_reason)
                return

            # SDK の async iterator が turn_complete で自然終了 → 次ターンへ
            logger.debug("1ターン完了 → 次ターン receive() 開始")

    def _emit_subtitle(
        self, text: str, *, is_user: bool, finished: bool = False
    ) -> None:
        """字幕イベントを発行する."""
        event = SubtitleEvent(
            text=text,
            timestamp=time.time(),
            is_user=is_user,
            finished=finished,
        )
        self._on_subtitle(event)
        # 逐次字幕は断片数が多いため、通常運用では INFO ログを汚さない。
        logger.debug("%s字幕: %s", "ユーザー" if is_user else "AI", text)
