"""FastAPI sidecar エントリポイント.

Tauri フロントエンドからポーリングされる REST API を提供する。
"""

from __future__ import annotations

import asyncio
import logging
import logging.config
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sidecar.models import CoachState, CoachingContext, DialogueState, SubtitleEvent
from sidecar.pipeline import context_to_system_instruction, run_analyze
from sidecar.realtime_engine import RealtimeEngine
from sidecar.replay_controller import ReplayStateController

# .env ファイルを読み込む（GEMINI_API_KEY 等）
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUBTITLE_MERGE_WINDOW_SECONDS = 1.2
QUIET_ACCESS_LOG_PATHS = frozenset({"/health", "/state", "/subtitles"})


class PollingAccessLogFilter(logging.Filter):
    """高頻度ポーリングの access log を抑止する."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = getattr(record, "args", ())
        if (
            record.name == "uvicorn.access"
            and isinstance(args, tuple)
            and len(args) >= 3
            and args[2] in QUIET_ACCESS_LOG_PATHS
        ):
            return False
        return True


def _configure_access_log_filter(
    access_logger: logging.Logger | None = None,
) -> None:
    """uvicorn.access へポーリング抑止フィルタを一度だけ登録する."""
    target_logger = access_logger or logging.getLogger("uvicorn.access")
    if any(
        isinstance(log_filter, PollingAccessLogFilter)
        for log_filter in target_logger.filters
    ):
        return
    target_logger.addFilter(PollingAccessLogFilter())


_configure_access_log_filter()

# --- アプリ状態（シングルトン） ---

_coach_state: CoachState = CoachState.IDLE
_dialogue_state: DialogueState = DialogueState.IDLE
_latest_subtitle: SubtitleEvent | None = None
_latest_ai_subtitle: SubtitleEvent | None = None
_latest_user_subtitle: SubtitleEvent | None = None
_analyze_task: asyncio.Task | None = None  # type: ignore[type-arg]
_session_task: asyncio.Task | None = None  # type: ignore[type-arg]
_engine: RealtimeEngine | None = None
_context_path: str | None = None


class AnalyzeRequest(BaseModel):
    """POST /analyze の request body."""

    video_path: str
    match_id: str = ""  # 空なら .env RIOT_MATCH_ID を参照


class StartSessionRequest(BaseModel):
    """POST /start の request body."""

    context_path: str


# --- FastAPI アプリ ---

app = FastAPI(title="RePlection sidecar", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tauri WebView からのアクセスを許可
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- エンドポイント ---


@app.get("/health")
async def health() -> dict[str, str]:
    """ヘルスチェック.

    Returns:
        {{"status": "ok", "state": "<CoachState>"}}
    """
    return {"status": "ok", "state": _coach_state.value}


@app.get("/state")
async def get_state() -> dict[str, object]:
    """現在の状態を返す.

    Returns:
        CoachState と DialogueState を含む辞書。READY 時は context_path も含む。
    """
    result: dict[str, object] = {
        "coach_state": _coach_state.value,
        "dialogue_state": _dialogue_state.value,
    }
    if _coach_state == CoachState.READY and _context_path is not None:
        result["context_path"] = _context_path
    return result


@app.get("/subtitles")
async def get_subtitles() -> dict[str, object]:
    """最新の字幕イベントを返す（Tauri ポーリング用）.

    Returns:
        字幕テキスト・タイムスタンプ・話者フラグを含む辞書。
        字幕がない場合は text が空文字列。
    """
    ai = _latest_ai_subtitle
    user = _latest_user_subtitle
    return {
        "text": (ai.text if ai else ""),
        "timestamp": (ai.timestamp if ai else 0.0),
        "is_user": False,
        "ai_text": (ai.text if ai else ""),
        "ai_timestamp": (ai.timestamp if ai else 0.0),
        "ai_finished": (ai.finished if ai else False),
        "user_text": (user.text if user else ""),
        "user_timestamp": (user.timestamp if user else 0.0),
        "user_finished": (user.finished if user else False),
    }


@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest) -> dict[str, object]:
    """動画を分析して中間生成物を生成する.

    バックグラウンドで Stage 1-4 パイプラインを実行し、
    完了後に context_path を保存して READY 状態に遷移する。

    Returns:
        {{"status": "analyzing"}} または {{"status": "busy"}} または {{"status": "ready", "context_path": "..."}}
    """
    global _coach_state, _analyze_task, _context_path

    if _coach_state != CoachState.IDLE:
        return {"status": "busy", "coach_state": _coach_state.value}

    _coach_state = CoachState.ANALYZING
    _context_path = None
    _analyze_task = asyncio.create_task(
        _run_analyze_session(request.video_path, request.match_id)
    )
    return {"status": "analyzing"}


@app.post("/start")
async def start_session(request: StartSessionRequest) -> dict[str, str]:
    """コーチングセッションを開始する（context_path 必須）.

    事前に POST /analyze で生成した中間生成物 JSON のパスを受け取り、
    即座に RealtimeEngine を起動する。

    Returns:
        {{"status": "started"}} または {{"status": "not_ready"}}
    """
    global _coach_state, _dialogue_state, _latest_subtitle, _latest_ai_subtitle, _latest_user_subtitle, _session_task

    if _coach_state != CoachState.READY:
        return {"status": "not_ready"}

    _coach_state = CoachState.DIALOGUE
    _dialogue_state = DialogueState.IDLE
    _latest_subtitle = None
    _latest_ai_subtitle = None
    _latest_user_subtitle = None
    _session_task = asyncio.create_task(_run_live_session(request.context_path))
    return {"status": "started"}


@app.post("/stop")
async def stop_session() -> dict[str, str]:
    """コーチングセッションを停止する.

    Returns:
        {{"status": "stopped"}} または {{"status": "not_running"}}
    """
    global _coach_state, _dialogue_state, _analyze_task, _session_task, _engine, _context_path, _latest_subtitle, _latest_ai_subtitle, _latest_user_subtitle

    if _coach_state == CoachState.IDLE and (
        _analyze_task is None or _analyze_task.done()
    ) and (
        _session_task is None or _session_task.done()
    ):
        return {"status": "not_running"}

    # 分析タスクのキャンセル
    if _analyze_task is not None and not _analyze_task.done():
        _analyze_task.cancel()
        await asyncio.gather(_analyze_task, return_exceptions=True)
    _analyze_task = None

    # RealtimeEngine を停止
    if _engine is not None:
        await _engine.stop()

    session_task = _session_task
    if session_task is not None and not session_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(session_task), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("セッション停止待機がタイムアウトしたためキャンセルします")
            if not session_task.done():
                session_task.cancel()
            await asyncio.gather(session_task, return_exceptions=True)

    _engine = None
    _session_task = None
    _context_path = None
    _latest_subtitle = None
    _latest_ai_subtitle = None
    _latest_user_subtitle = None

    _coach_state = CoachState.IDLE
    _dialogue_state = DialogueState.IDLE
    logger.info("セッション停止")
    return {"status": "stopped"}


@app.post("/ptt/start")
async def start_push_to_talk() -> dict[str, str]:
    """Push-to-talk を開始する.

    Returns:
        {{"status": "started"}} / {{"status": "already_active"}} / {{"status": "not_running"}}
    """
    if _coach_state != CoachState.DIALOGUE or _engine is None:
        return {"status": "not_running"}

    started = await _engine.start_push_to_talk()
    return {"status": "started" if started else "already_active"}


@app.post("/ptt/end")
async def end_push_to_talk() -> dict[str, str]:
    """Push-to-talk を終了する.

    Returns:
        {{"status": "ended"}} / {{"status": "not_active"}} / {{"status": "not_running"}}
    """
    if _coach_state != CoachState.DIALOGUE or _engine is None:
        return {"status": "not_running"}

    ended = await _engine.end_push_to_talk()
    return {"status": "ended" if ended else "not_active"}


# --- バックグラウンドタスク ---


async def _run_analyze_session(video_path: str, match_id: str = "") -> None:
    """動画分析セッション（バックグラウンド）: ANALYZING → READY or ERROR."""
    global _coach_state, _context_path

    logger.info("分析開始: %s (match_id=%s)", video_path, match_id or "(env)")
    try:
        ctx, saved_path = await asyncio.to_thread(run_analyze, video_path, match_id)
        _context_path = str(saved_path)
        _coach_state = CoachState.READY
        logger.info("分析完了: context_path=%s", _context_path)
    except asyncio.CancelledError:
        logger.info("分析キャンセル")
        _coach_state = CoachState.IDLE
        raise
    except Exception:
        logger.exception("分析エラー")
        _coach_state = CoachState.ERROR


async def _run_live_session(context_path: str) -> None:
    """Live 会話セッション（バックグラウンド）: DIALOGUE → IDLE."""
    global _coach_state, _dialogue_state, _engine, _session_task

    logger.info("Live セッション開始")
    try:
        # パストラバーサル対策: data/sessions/ 配下の .json のみ許可
        sessions_dir = Path("data/sessions").resolve()
        target = Path(context_path).resolve(strict=True)
        target.relative_to(sessions_dir)  # 配下でなければ ValueError
        if not target.is_file() or target.suffix != ".json":
            raise ValueError("不正なファイル指定")

        json_str = await asyncio.to_thread(
            target.read_text, encoding="utf-8"
        )
        ctx = CoachingContext.from_json(json_str)
        system_instruction = context_to_system_instruction(ctx)
        logger.info("system_instruction 復元: %d 文字", len(system_instruction))

        replay_ctrl = ReplayStateController()

        # カメラを自分視点に固定
        player_champion = ctx.match_info.player_champion
        if player_champion:
            await asyncio.to_thread(replay_ctrl.set_camera_to_player, player_champion)

        _engine = RealtimeEngine(
            system_instruction=system_instruction,
            on_subtitle=_on_subtitle,
            on_state_change=_on_state_change,
            replay_controller=replay_ctrl,
        )
        logger.info("RealtimeEngine 起動")
        await _engine.start()  # stop() が呼ばれるまでブロック

    except asyncio.CancelledError:
        logger.info("Live セッションキャンセル")
        raise
    except Exception:
        logger.exception("Live セッションエラー")
        _coach_state = CoachState.ERROR
        _dialogue_state = DialogueState.IDLE
    finally:
        if _coach_state != CoachState.ERROR:
            _coach_state = CoachState.IDLE
        _engine = None
        if _session_task is asyncio.current_task():
            _session_task = None
        logger.info("Live セッション終了")


def _on_subtitle(event: SubtitleEvent) -> None:
    """字幕イベントを受け取り最新字幕を更新する.

    Args:
        event: 字幕イベント（AI・ユーザー両方）。
    """
    global _latest_subtitle, _latest_ai_subtitle, _latest_user_subtitle
    _latest_subtitle = _merge_subtitle_event(_latest_subtitle, event)
    if event.is_user:
        _latest_user_subtitle = _merge_subtitle_event(_latest_user_subtitle, event)
    else:
        _latest_ai_subtitle = _merge_subtitle_event(_latest_ai_subtitle, event)


def _merge_subtitle_event(
    current: SubtitleEvent | None,
    new_event: SubtitleEvent,
) -> SubtitleEvent:
    """短時間に届く同一話者の断片字幕を結合する."""
    if current is None:
        return new_event

    # 確定済みの字幕に新しいイベントが来たら merge せず切り替え（ターン境界）
    if current.finished:
        return new_event

    same_speaker = current.is_user == new_event.is_user
    time_delta = new_event.timestamp - current.timestamp
    close_enough = 0.0 <= time_delta <= SUBTITLE_MERGE_WINDOW_SECONDS
    if not same_speaker or not close_enough:
        return new_event

    return SubtitleEvent(
        text=_merge_subtitle_text(current.text, new_event.text),
        timestamp=new_event.timestamp,
        is_user=new_event.is_user,
        finished=new_event.finished,
    )


def _merge_subtitle_text(current_text: str, new_text: str) -> str:
    """断片字幕を重複なく自然につなぐ."""
    left = current_text.strip()
    right = new_text.strip()
    if not left:
        return right
    if not right:
        return left
    if right.startswith(left):
        return right
    if left.startswith(right):
        return left

    overlap = _find_suffix_prefix_overlap(left, right)
    if overlap > 0:
        return left + right[overlap:]

    separator = _subtitle_separator(left, right)
    return f"{left}{separator}{right}"


def _find_suffix_prefix_overlap(left: str, right: str) -> int:
    """left の末尾と right の先頭の最大重複長を返す."""
    max_size = min(len(left), len(right))
    for size in range(max_size, 0, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def _subtitle_separator(left: str, right: str) -> str:
    """英単語どうしだけ空白を補い、日本語は素の連結にする."""
    if left[-1].isascii() and left[-1].isalnum() and right[0].isascii():
        if right[0].isalnum():
            return " "
    return ""


def _on_state_change(new_state: DialogueState) -> None:
    """対話状態変化を受け取りグローバル状態を更新する.

    Args:
        new_state: 新しい DialogueState。
    """
    global _dialogue_state
    _dialogue_state = new_state


# --- エントリポイント ---

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run(
        "sidecar.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=port,
        reload=False,
        log_level="info",
    )
